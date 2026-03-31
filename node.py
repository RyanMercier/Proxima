#!/usr/bin/env python3
"""
node.py -- Blockchain node server.

HTTP API for submitting transactions and querying state. Runs miner
threads in the background that pull from the mempool and run consensus.
Start this first, then connect wallets.
"""

import argparse
import json
import threading
import time
import sys
import numpy as np
from flask import Flask, request, jsonify

from blockchain import (
    Blockchain, Validator, BLSKeyPair, Block, Transaction, CoinbaseTx,
    make_validators, make_partial_obs, calibrate_threshold,
    vector_consensus, tree_consensus, block_reward,
)

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def pf(*args, **kwargs):
    """Print with flush for real-time output in threaded context."""
    print(*args, **kwargs, flush=True)


def print_block_result(block: Block, result: dict, chain):
    """Colored block summary for the server terminal."""
    n_user_txs = sum(1 for tx in block.transactions if isinstance(tx, Transaction))
    header = f"{C.BOLD}{C.CYAN}[BLOCK #{block.height}]{C.RESET} Proposed by {block.proposer_name} ({n_user_txs} txs + coinbase)"
    pf(f"\n{header}")

    cluster_size = result["cluster_size"]
    n_total = cluster_size + len(result["excluded"])

    if result.get("tree_mode"):
        # Tree consensus output
        nl = result["n_levels"]
        br = result["branching"]
        pf(f"  Tree: {nl} levels, branching={br}")
        stats = result.get("level_stats", [])
        if stats:
            s0 = stats[0]
            pf(f"  Level 0: {s0['groups']} leaves, "
               f"{s0['excluded']} excluded, {s0['passed']} passed filter")
        for s in stats[1:]:
            pf(f"  Level {s['level']}: {s['groups']} nodes, "
               f"{s.get('msgs_this_level', 0)} summaries (76 bytes each)")
        status = f"{C.GREEN}FINALIZED{C.RESET}" if result["finalized"] else f"{C.RED}FAILED{C.RESET}"
        pf(f"  Phase 2: {result['n_commits']} BLS commits, agg sig 96 bytes | {status}")
        pf(f"  {result['msgs']:,} msgs | {result['msg_bytes'] / 1024:.1f} KB | {result['total_time']:.3f}s")

    elif result["fast_path"]:
        variance = result["cluster_variance"]
        path = f"{C.GREEN}FAST PATH{C.RESET}"
        pf(f"  Phase 1: {cluster_size}/{n_total} vectors | variance={variance:.4f} | {path}")
        pf(f"  {C.GREEN}Finalized in 1 round{C.RESET} | {result['msgs']} msgs | {result['msg_bytes'] / 1024:.1f} KB")

    else:
        variance = result["cluster_variance"]
        pf(f"  Phase 1: {cluster_size}/{n_total} vectors | variance={variance:.4f}")

        excluded = result.get("excluded", [])
        if excluded:
            exc_parts = []
            for name, is_byz, strategy, dist in excluded:
                if is_byz:
                    exc_parts.append(f"{C.RED}{name}(d={dist:.1f}, {strategy}){C.RESET}")
                else:
                    exc_parts.append(f"{C.YELLOW}{name}(d={dist:.1f}){C.RESET}")
            pf(f"  Excluded: {', '.join(exc_parts)}")

        if result["sync_pushed"] > 0:
            for vname, n_miss in result["sync_details"]:
                pf(f"  {C.DIM}Pushed {n_miss} tx to {vname} (bloom diff){C.RESET}")

        status = f"{C.GREEN}FINALIZED{C.RESET}" if result["finalized"] else f"{C.RED}FAILED{C.RESET}"
        pf(f"  Phase 2: {result['n_commits']} BLS commits | agg sig 96 bytes | {status}")
        pf(f"  {result['msgs']} msgs | {result['msg_bytes'] / 1024:.1f} KB | {result['total_time']:.3f}s")

    # Balance updates for named accounts
    balances = []
    for addr, name in chain.state.names.items():
        bal = chain.state.bal(addr)
        if bal > 0 and not name.startswith("Miner") and not name.startswith("Byz"):
            balances.append(f"{name}={bal:.2f}")
    if balances:
        pf(f"  {C.DIM}Balances: {' | '.join(balances)}{C.RESET}")


# ---------------------------------------------------------------------------
# Miner loop
# ---------------------------------------------------------------------------

def miner_loop(chain: Blockchain, interval: float, miss_prob: float,
               stop_event: threading.Event, use_tree: bool = False,
               branching: int = 10):
    """Background thread: propose and finalize blocks when mempool is not empty."""
    miner_idx = 0
    honest_validators = [v for v in chain.validators if not v.is_byzantine]

    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break

        if not chain.mempool:
            continue

        proposer = honest_validators[miner_idx % len(honest_validators)]
        miner_idx += 1

        block = chain.propose_block(proposer)
        n_txs = len(block.tx_data_strings)
        partial_obs = make_partial_obs(chain.validators, n_txs, miss_prob=miss_prob)
        threshold = calibrate_threshold(block.tx_data_strings)

        if use_tree:
            result = tree_consensus(chain.validators, block, threshold,
                                    partial_obs, branching)
        else:
            result = vector_consensus(chain.validators, block, threshold, partial_obs)
        if result["finalized"]:
            chain.finalize_block(block)
        result["block"] = block
        chain.consensus_log.append(result)

        print_block_result(block, result, chain)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(chain: Blockchain) -> Flask:
    app = Flask(__name__)

    @app.route("/tx", methods=["POST"])
    def submit_tx():
        data = request.get_json(force=True)
        sender = data.get("sender", "")
        receiver = data.get("receiver", "")
        amount = float(data.get("amount", 0))

        if not chain.addr_for(sender):
            return jsonify({"error": f"unknown sender: {sender}"}), 400
        if not chain.addr_for(receiver):
            return jsonify({"error": f"unknown receiver: {receiver}"}), 400

        s_addr = chain.addr_for(sender)
        bal = chain.state.bal(s_addr)
        if bal < amount + 0.01:
            return jsonify({"error": f"insufficient balance: {bal:.2f}"}), 400

        tx = chain.make_tx(sender, receiver, amount)
        if tx is None:
            return jsonify({"error": "failed to create transaction"}), 400

        chain.submit_tx(tx)
        pf(f"{C.DIM}  TX: {sender} -> {receiver} {amount:.2f} PROX ({tx.tx_hash[:12]}...){C.RESET}")
        return jsonify({"tx_hash": tx.tx_hash, "status": "pending"})

    @app.route("/balance/<name>")
    def get_balance(name):
        addr = chain.addr_for(name)
        if not addr:
            return jsonify({"error": f"unknown account: {name}"}), 404
        bal = chain.state.bal(addr)
        pending_out = sum(
            tx.amount + tx.fee for tx in chain.mempool if tx.sender == addr
        )
        return jsonify({
            "name": name,
            "balance": bal,
            "pending_out": pending_out,
            "available": bal - pending_out,
        })

    @app.route("/chain")
    def get_chain():
        return jsonify({
            "height": chain.height,
            "tip": chain.tip[:16] + "...",
            "supply": chain.state.supply,
            "mempool_size": len(chain.mempool),
        })

    @app.route("/block/<int:height>")
    def get_block(height):
        if height < 0 or height >= chain.height:
            return jsonify({"error": "block not found"}), 404
        return jsonify(chain.chain[height].to_dict())

    @app.route("/mempool")
    def get_mempool():
        return jsonify({
            "size": len(chain.mempool),
            "transactions": [tx.to_dict() for tx in chain.mempool],
        })

    @app.route("/validators")
    def get_validators():
        return jsonify({
            "count": len(chain.validators),
            "validators": [v.to_dict() for v in chain.validators],
        })

    @app.route("/register", methods=["POST"])
    def register():
        data = request.get_json(force=True)
        name = data.get("name", "")
        balance = float(data.get("balance", 0))
        if not name:
            return jsonify({"error": "name required"}), 400
        if chain.addr_for(name):
            return jsonify({"name": name, "status": "already registered"})
        addr = chain.register_account(name, balance)
        pf(f"{C.DIM}  Registered: {name} (balance={balance:.2f}){C.RESET}")
        return jsonify({"name": name, "address": addr[:16] + "...", "balance": balance})

    @app.route("/status")
    def status():
        last = chain.consensus_log[-1] if chain.consensus_log else None
        return jsonify({
            "height": chain.height,
            "mempool_size": len(chain.mempool),
            "validators": len(chain.validators),
            "supply": chain.state.supply,
            "last_consensus": {
                "finalized": last["finalized"],
                "fast_path": last.get("fast_path", False),
                "msgs": last["msgs"],
                "time": last["total_time"],
            } if last else None,
        })

    @app.route("/history/<name>")
    def get_history(name):
        history = chain.get_history(name)
        pending = [
            tx.to_dict() for tx in chain.mempool
            if tx.sender_name == name or tx.receiver_name == name
        ]
        return jsonify({"name": name, "history": history, "pending": pending})

    # Stress test endpoints
    @app.route("/stress/submit", methods=["POST"])
    def stress_submit():
        """Bulk tx submission. Used by stress.py."""
        data = request.get_json(force=True)
        txs = data.get("transactions", [])
        submitted = 0
        for t in txs:
            tx = chain.make_tx(t["sender"], t["receiver"], t["amount"])
            if tx:
                chain.submit_tx(tx)
                submitted += 1
        return jsonify({"submitted": submitted})

    @app.route("/stress/mine", methods=["POST"])
    def stress_mine():
        """Mine one block now, bypassing the miner loop timer."""
        data = request.get_json(force=True)
        n_txs = int(data.get("max_txs", 50))
        miss_prob = float(data.get("miss_prob", 0.37))

        if not chain.mempool:
            return jsonify({"error": "mempool empty"}), 400

        # Take up to n_txs from mempool
        proposer = [v for v in chain.validators if not v.is_byzantine][0]
        block = chain.propose_block(proposer)
        threshold = calibrate_threshold(block.tx_data_strings)
        partial_obs = make_partial_obs(chain.validators, len(block.tx_data_strings),
                                       miss_prob=miss_prob)

        result = vector_consensus(chain.validators, block, threshold, partial_obs)
        if result["finalized"]:
            chain.finalize_block(block)
        result["block_height"] = block.height
        result["n_txs"] = len(block.transactions)
        # Remove non-serializable items
        result.pop("distances", None)
        chain.consensus_log.append(result)
        return jsonify(result)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Blockchain node server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8545)
    parser.add_argument("--honest", type=int, default=4, help="Number of honest validators")
    parser.add_argument("--byzantine", type=int, default=1, help="Number of Byzantine validators")
    parser.add_argument("--byzantine-strategy", default="drop_half",
                        choices=["drop_half", "random_vector", "replace_one_tx",
                                 "mimic_honest", "coalition"])
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between mining attempts")
    parser.add_argument("--miss-prob", type=float, default=0.37,
                        help="Fraction of honest validators with partial observation")
    parser.add_argument("--init-accounts", nargs="*", default=["Alice", "Bob"],
                        help="Pre-fund these accounts with 1000 PROX each")
    parser.add_argument("--init-balance", type=float, default=1000.0)
    parser.add_argument("--tree", action="store_true",
                        help="Use tree-structured consensus instead of flat")
    parser.add_argument("--branching", type=int, default=10,
                        help="Tree branching factor (validators per leaf group)")
    args = parser.parse_args()

    # Build validators
    all_v, honest, byzantine = make_validators(
        args.honest, args.byzantine, args.byzantine_strategy
    )
    chain = Blockchain(all_v)

    # Pre-fund accounts
    for name in args.init_accounts:
        chain.register_account(name, args.init_balance)

    # Print startup info
    print(f"\n{C.BOLD}Node running on http://localhost:{args.port}{C.RESET}")
    print(f"Validators: ", end="")
    parts = []
    for v in all_v:
        if v.is_byzantine:
            parts.append(f"{C.RED}{v.name} (BYZANTINE: {v.strategy}){C.RESET}")
        else:
            parts.append(f"{C.GREEN}{v.name} (honest){C.RESET}")
    print(", ".join(parts))

    if args.init_accounts:
        accts = ", ".join(f"{n}={args.init_balance:.0f}" for n in args.init_accounts)
        print(f"Accounts: {accts}")
    mode = f"tree (branching={args.branching})" if args.tree else "flat"
    print(f"Block interval: {args.interval}s | Miss probability: {args.miss_prob} | Mode: {mode}")
    print(f"Waiting for transactions...\n")

    # Start miner thread
    stop_event = threading.Event()
    miner = threading.Thread(
        target=miner_loop,
        args=(chain, args.interval, args.miss_prob, stop_event,
              args.tree, args.branching),
        daemon=True,
    )
    miner.start()

    # Start Flask
    app = create_app(chain)
    try:
        # Suppress Flask request logging to keep terminal clean
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Shutting down...{C.RESET}")
        stop_event.set()


if __name__ == "__main__":
    main()
