#!/usr/bin/env python3
"""Test each Byzantine strategy with 20 txs to verify exclusion behavior."""

import blockchain
blockchain.USE_REAL_BLS = False

from blockchain import (
    BLSKeyPair, Blockchain, make_validators, calibrate_threshold,
    vector_consensus,
)

strategies = ["drop_half", "random_vector", "replace_one_tx", "mimic_honest", "coalition"]
n_txs = 20
n_honest = 4
n_byz = 1

for strategy in strategies:
    BLSKeyPair._counter = 1
    all_v, honest, byz = make_validators(n_honest, n_byz, strategy)
    chain = Blockchain(all_v)
    for i in range(5):
        chain.register_account(f"A{i}", 100_000.0)
    for i in range(n_txs):
        tx = chain.make_tx(f"A{i % 5}", f"A{(i+1) % 5}", 1.0)
        if tx:
            chain.submit_tx(tx)

    block = chain.propose_block(honest[0])
    threshold = calibrate_threshold(block.tx_data_strings)

    # No partial observation so we isolate the Byzantine effect
    result = vector_consensus(all_v, block, threshold, partial_obs={})

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Strategy: {strategy}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Fast path: {result['fast_path']}")
    print(f"Cluster size: {result['cluster_size']}")

    for name, is_byz, strat, dist in result.get("excluded", []):
        tag = "BYZANTINE" if is_byz else "honest"
        print(f"  EXCLUDED: {name} ({tag}, {strat}) distance={dist:.2f}")

    if not result.get("excluded"):
        for v in all_v:
            if v.is_byzantine:
                d = result["distances"][v.name]
                print(f"  IN CLUSTER: {v.name} (BYZANTINE, {v.strategy}) distance={d:.2f}")

    print(f"Finalized: {result['finalized']}")
    print(f"Messages: {result['msgs']} | Bandwidth: {result['msg_bytes']/1024:.1f} KB")

# Also test with small block (3 txs) to show the threshold problem
print(f"\n{'=' * 60}")
print("SMALL BLOCK TEST (3 txs) -- drop_half")
print("This shows why you need 10+ txs for clear Byzantine exclusion")
print("=" * 60)

BLSKeyPair._counter = 1
all_v, honest, byz = make_validators(4, 1, "drop_half")
chain = Blockchain(all_v)
chain.register_account("Alice", 100_000.0)
chain.register_account("Bob", 100_000.0)

for i in range(2):
    tx = chain.make_tx("Alice", "Bob", 10.0)
    if tx:
        chain.submit_tx(tx)

block = chain.propose_block(honest[0])
threshold = calibrate_threshold(block.tx_data_strings)
result = vector_consensus(all_v, block, threshold, partial_obs={})

print(f"Threshold: {threshold:.2f}")
print(f"Cluster size: {result['cluster_size']}")
for v in all_v:
    if v.is_byzantine:
        d = result["distances"][v.name]
        status = "EXCLUDED" if d >= threshold else "IN CLUSTER"
        print(f"  {status}: {v.name} distance={d:.2f} (threshold={threshold:.2f})")
print(f"With only 3 txs (2 user + coinbase), dropping 1 keeps the Byzantine close enough.")
print(f"Use --interval 10 and submit 10+ txs before the first block for a clear demo.")
