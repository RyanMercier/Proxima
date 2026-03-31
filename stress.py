#!/usr/bin/env python3
"""
stress.py -- Benchmark and protocol comparison.

Submits bulk transactions, mines blocks, and compares message count
and bandwidth against HotStuff and PBFT. Can run standalone (no node)
or against a live node with --node.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

import numpy as np

import blockchain
blockchain.USE_REAL_BLS = False  # hash-based mocks for speed, same message sizes

from blockchain import (
    Blockchain, make_validators, make_partial_obs, calibrate_threshold,
    vector_consensus, tree_consensus, BLSKeyPair,
)
from hotstuff import hotstuff_consensus, pbft_consensus


class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def run_standalone_benchmark(n_txs: int, n_validators: int, n_byzantine: int,
                             txs_per_block: int = 50, miss_prob: float = 0.37,
                             strategy: str = "drop_half",
                             use_tree: bool = False, branching: int = 10):
    """Full benchmark without a running node. Builds a chain, mines, compares."""
    print(f"\n{C.BOLD}STRESS TEST: {n_txs} transactions, {n_validators} validators "
          f"({n_byzantine} Byzantine){C.RESET}")
    print("=" * 64)

    # Build chain
    BLSKeyPair._counter = 1
    all_v, honest, byz = make_validators(n_validators - n_byzantine, n_byzantine, strategy)
    chain = Blockchain(all_v)

    # Create accounts
    n_accounts = 20
    accounts = []
    for i in range(n_accounts):
        name = f"Acct-{i}"
        chain.register_account(name, 100_000.0)
        accounts.append(name)

    # Submit transactions
    t_submit = time.time()
    for i in range(n_txs):
        sender = accounts[i % n_accounts]
        receiver = accounts[(i + 1) % n_accounts]
        tx = chain.make_tx(sender, receiver, 1.0, fee=0.01)
        if tx:
            chain.submit_tx(tx)
    t_submit = time.time() - t_submit
    print(f"\nSubmitting {n_txs} transactions... done ({t_submit:.1f}s)")

    # Mine blocks
    our_results = []
    block_num = 0
    t_mine_start = time.time()
    proposer_idx = 0

    while chain.mempool:
        proposer = honest[proposer_idx % len(honest)]
        proposer_idx += 1

        block = chain.propose_block(proposer)
        n_block_txs = len(block.tx_data_strings)
        threshold = calibrate_threshold(block.tx_data_strings)
        partial_obs = make_partial_obs(all_v, n_block_txs, miss_prob=miss_prob)

        if use_tree:
            result = tree_consensus(all_v, block, threshold, partial_obs, branching)
        else:
            result = vector_consensus(all_v, block, threshold, partial_obs)
        if result["finalized"]:
            chain.finalize_block(block)

        path = "tree     " if result.get("tree_mode") else (
            "fast path" if result["fast_path"] else "full proto")
        n_user_txs = n_block_txs - 1  # minus coinbase
        print(f"Block {block_num:>3}: {n_user_txs:>3} txs | {path} | "
              f"{result['total_time']:.2f}s | {result['msgs']:>5} msgs | "
              f"{result['msg_bytes'] / 1024:>5.0f} KB")

        our_results.append(result)
        result["block"] = block
        block_num += 1

    t_mine_total = time.time() - t_mine_start
    n_blocks = len(our_results)

    # Summary stats
    total_msgs = sum(r["msgs"] for r in our_results)
    total_bytes = sum(r["msg_bytes"] for r in our_results)
    avg_msgs = total_msgs / max(n_blocks, 1)
    avg_bytes = total_bytes / max(n_blocks, 1)
    fast_pct = 100 * sum(1 for r in our_results if r["fast_path"]) / max(n_blocks, 1)
    avg_time = sum(r["total_time"] for r in our_results) / max(n_blocks, 1)

    print(f"\n{C.BOLD}RESULTS{C.RESET}")
    print(f"  Blocks produced: {n_blocks}")
    print(f"  Total time: {t_mine_total:.1f}s")
    print(f"  Avg finality: {avg_time:.4f}s per block")
    print(f"  Fast path: {fast_pct:.0f}% of blocks")
    print(f"  Messages per block: {avg_msgs:.0f} avg")
    print(f"  Bandwidth per block: {avg_bytes / 1024:.0f} KB avg")

    # Count Byzantine exclusions
    byz_excluded = 0
    for r in our_results:
        for name, is_byz, strat, dist in r.get("excluded", []):
            if is_byz:
                byz_excluded += 1
    total_byz_opportunities = n_blocks * n_byzantine
    if total_byz_opportunities > 0:
        print(f"  Byzantine excluded: {byz_excluded}/{total_byz_opportunities} "
              f"({100 * byz_excluded / total_byz_opportunities:.0f}%)")

    # HotStuff comparison (same scenario)
    print(f"\n{C.BOLD}COMPARISON (same scenario):{C.RESET}")
    hs = hotstuff_consensus(n_validators, n_byzantine, txs_per_block, miss_prob)
    pbft = pbft_consensus(n_validators, n_byzantine, txs_per_block)

    print(f"  {C.CYAN}Ours:    {C.RESET} {avg_msgs:>6.0f} msgs/block, {avg_bytes / 1024:>6.0f} KB/block")
    print(f"  {C.YELLOW}HotStuff:{C.RESET} {hs['msgs']:>6} msgs/block, {hs['msg_bytes'] / 1024:>6.0f} KB/block")
    print(f"  {C.RED}PBFT:    {C.RESET} {pbft['msgs']:>6} msgs/block, {pbft['msg_bytes'] / 1024:>6.0f} KB/block")

    # Scale test
    print(f"\n{C.BOLD}SCALE TEST{C.RESET}")
    scale_points = [100, 500, 1000, 2000]
    byz_frac = n_byzantine / n_validators

    def _build_scale_chain(n, n_byz, n_hon):
        BLSKeyPair._counter = 1
        sv, sh, sb = make_validators(n_hon, n_byz, strategy)
        sc = Blockchain(sv)
        for i in range(5):
            sc.register_account(f"S-{i}", 100_000.0)
        for i in range(txs_per_block):
            tx = sc.make_tx(f"S-{i % 5}", f"S-{(i + 1) % 5}", 1.0)
            if tx:
                sc.submit_tx(tx)
        block = sc.propose_block(sh[0])
        threshold = calibrate_threshold(block.tx_data_strings)
        pobs = make_partial_obs(sv, len(block.tx_data_strings), miss_prob=miss_prob)
        return sv, block, threshold, pobs

    for n in scale_points:
        n_byz = int(n * byz_frac)
        n_hon = n - n_byz

        sv, block, threshold, pobs = _build_scale_chain(n, n_byz, n_hon)
        flat = vector_consensus(sv, block, threshold, pobs)

        sv2, block2, threshold2, pobs2 = _build_scale_chain(n, n_byz, n_hon)
        tree = tree_consensus(sv2, block2, threshold2, pobs2, branching)

        hs_r = hotstuff_consensus(n, n_byz, txs_per_block, miss_prob)

        print(f"  N={n:<5} {C.CYAN}Flat:{C.RESET} {flat['msgs']:>6} msgs  "
              f"{C.GREEN}Tree:{C.RESET} {tree['msgs']:>6} msgs  "
              f"{C.YELLOW}HotStuff:{C.RESET} {hs_r['msgs']:>6} msgs")


def run_remote_stress(node_url: str, n_txs: int):
    """Submit bulk transactions to a running node and trigger mining."""
    node = node_url.rstrip("/")

    def post(path, data):
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{node}{path}", data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def get(path):
        with urllib.request.urlopen(f"{node}{path}", timeout=10) as resp:
            return json.loads(resp.read())

    # Register test accounts
    accounts = [f"Stress-{i}" for i in range(10)]
    for name in accounts:
        post("/register", {"name": name, "balance": 100_000.0})

    print(f"Submitting {n_txs} transactions to {node}...")
    t0 = time.time()

    # Batch submit
    batch = []
    for i in range(n_txs):
        batch.append({
            "sender": accounts[i % len(accounts)],
            "receiver": accounts[(i + 1) % len(accounts)],
            "amount": 1.0,
        })
    resp = post("/stress/submit", {"transactions": batch})
    print(f"Submitted {resp.get('submitted', 0)} in {time.time() - t0:.1f}s")

    # Mine blocks until mempool is empty
    print("Mining blocks...")
    block_results = []
    while True:
        mp = get("/mempool")
        if mp.get("size", 0) == 0:
            break
        result = post("/stress/mine", {"max_txs": 50})
        if "error" in result:
            break
        path = "fast path" if result.get("fast_path") else "full proto"
        print(f"  Block {result.get('block_height', '?')}: {result.get('n_txs', 0)} txs | "
              f"{path} | {result['msgs']} msgs | {result['msg_bytes'] / 1024:.0f} KB")
        block_results.append(result)

    if block_results:
        avg_msgs = sum(r["msgs"] for r in block_results) / len(block_results)
        avg_bytes = sum(r["msg_bytes"] for r in block_results) / len(block_results)
        fast = sum(1 for r in block_results if r.get("fast_path")) / len(block_results) * 100
        print(f"\n{len(block_results)} blocks | {avg_msgs:.0f} msgs avg | "
              f"{avg_bytes / 1024:.0f} KB avg | {fast:.0f}% fast path")


def main():
    parser = argparse.ArgumentParser(description="Stress test and benchmark")
    parser.add_argument("--txs", type=int, default=1000, help="Number of transactions")
    parser.add_argument("--validators", type=int, default=100)
    parser.add_argument("--byzantine", type=int, default=30)
    parser.add_argument("--txs-per-block", type=int, default=50)
    parser.add_argument("--miss-prob", type=float, default=0.37)
    parser.add_argument("--strategy", default="drop_half")
    parser.add_argument("--tree", action="store_true",
                        help="Use tree-structured consensus")
    parser.add_argument("--branching", type=int, default=10)
    parser.add_argument("--node", default=None,
                        help="If set, run stress test against a live node instead of standalone")
    args = parser.parse_args()

    if args.node:
        run_remote_stress(args.node, args.txs)
    else:
        run_standalone_benchmark(
            args.txs, args.validators, args.byzantine,
            args.txs_per_block, args.miss_prob, args.strategy,
            args.tree, args.branching,
        )


if __name__ == "__main__":
    main()
