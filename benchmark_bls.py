#!/usr/bin/env python3
"""
benchmark_bls.py -- Measures BLS aggregation, distance computation, bloom ops.

Outputs measured (mock) timings and projected (blst) production numbers.
Saves results to benchmark_results.json for use by visualize.py.
"""

import json
import math
import time
import hashlib
import numpy as np

import blockchain as _bc
_bc.USE_REAL_BLS = False

from blockchain import (
    BLSKeyPair, BloomFilter, N_DIMS,
    tx_to_vector, compute_vector,
)


# Production BLS constants from blst benchmarks (Ethereum Lighthouse/Prysm)
BLST_SIGN_MS = 0.1
BLST_AGG_ADD_MS = 0.05
BLST_AGG_VERIFY_MS = 1.5

# Other ops (negligible at scale but included for completeness)
DIST_CHECK_MS = 0.002
BLOOM_CHECK_MS = 0.01
VECTOR_COMPUTE_MS = 0.01

# Network RTT for latency model
RTT_LOCAL_MS = 1
RTT_REGIONAL_MS = 80
RTT_GLOBAL_MS = 200


def median_time(fn, runs=11):
    """Run fn() multiple times, return median duration in ms."""
    times = []
    for _ in range(runs):
        t0 = time.monotonic()
        fn()
        times.append((time.monotonic() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench_bls_aggregation(n_sigs):
    """Time: aggregate n_sigs mock BLS signatures."""
    sigs = [hashlib.sha384(f"sig:{i}".encode()).digest() for i in range(n_sigs)]
    def aggregate():
        result = sigs[0]
        for s in sigs[1:]:
            result = hashlib.sha384(b"agg:" + result + s).digest()
        return result
    return median_time(aggregate)


def bench_distance_checks(n):
    """Time: n Euclidean distance checks in 8D."""
    ref = np.random.rand(N_DIMS)
    vectors = np.random.rand(n, N_DIMS)
    def check():
        for i in range(n):
            np.linalg.norm(vectors[i] - ref)
    return median_time(check)


def bench_bloom_checks(n, n_txs=20):
    """Time: check n bloom filters against a full set of n_txs."""
    full_set = [hashlib.sha256(f"tx-{i}".encode()).hexdigest() for i in range(n_txs)]
    blooms = []
    for i in range(n):
        bf = BloomFilter(n_txs)
        # Each bloom is missing 1-2 items
        for j, tx in enumerate(full_set):
            if j != i % n_txs:
                bf.add(tx)
        blooms.append(bf)
    def check():
        for bf in blooms:
            bf.missing_from(full_set)
    return median_time(check)


def projected_flat_processing(n, byz_frac=0.3, partial_frac=0.37):
    """Projected processing time (ms) for flat aggregator using blst constants."""
    n_honest = int(n * (1 - byz_frac))
    n_partial = int(n_honest * partial_frac)
    # Phase 1: distance check every validator + bloom check partial ones
    phase1 = n * DIST_CHECK_MS + n_partial * BLOOM_CHECK_MS
    # Phase 2: aggregate n_honest BLS signatures + verify
    phase2 = n_honest * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    return phase1 + phase2


def projected_tree_processing(n, branching=10, byz_frac=0.3, partial_frac=0.37):
    """Projected processing time (ms) on the critical path for tree."""
    n_honest = int(n * (1 - byz_frac))
    n_leaves = math.ceil(n / branching)
    n_levels = 1
    nodes = n_leaves
    while nodes > 1:
        nodes = math.ceil(nodes / branching)
        n_levels += 1

    # Leaf: distance check + bloom check + BLS aggregate for branching validators
    honest_per_leaf = int(branching * (1 - byz_frac))
    partial_per_leaf = int(honest_per_leaf * partial_frac)
    leaf_time = (branching * DIST_CHECK_MS +
                 partial_per_leaf * BLOOM_CHECK_MS +
                 honest_per_leaf * BLST_AGG_ADD_MS +
                 BLST_AGG_VERIFY_MS)

    # Internal levels: aggregate branching child signatures + verify
    internal_time = branching * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    total = leaf_time + (n_levels - 1) * internal_time
    return total


def projected_hotstuff_processing(n, byz_frac=0.3, partial_frac=0.37):
    """Projected processing time (ms) for HotStuff (3 rounds of BLS agg)."""
    n_honest = int(n * (1 - byz_frac))
    n_partial = int(n_honest * partial_frac)
    # Retransmit: sequential, n_partial requests
    retransmit = n_partial * 0.1
    # Each of 3 rounds: aggregate N votes
    per_round = n * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    return retransmit + 3 * per_round


def projected_flat_bls_only(n, byz_frac=0.3):
    """Just the BLS aggregation part for flat."""
    n_honest = int(n * (1 - byz_frac))
    return n_honest * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS


def projected_tree_bls_only(n, branching=10, byz_frac=0.3):
    """Just the BLS aggregation on the critical path for tree."""
    honest_per_leaf = int(branching * (1 - byz_frac))
    n_leaves = math.ceil(n / branching)
    n_levels = 1
    nodes = n_leaves
    while nodes > 1:
        nodes = math.ceil(nodes / branching)
        n_levels += 1
    leaf_agg = honest_per_leaf * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    internal_agg = branching * BLST_AGG_ADD_MS + BLST_AGG_VERIFY_MS
    return leaf_agg + (n_levels - 1) * internal_agg


def main():
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    branching = 10
    results = {"sizes": sizes, "branching": branching, "mock": {}, "projected": {}}

    # ---- Mock BLS measurements ----
    print("BLS AGGREGATION BENCHMARK")
    print("=" * 56)
    print()
    print("Mock BLS (measured on this machine):")

    mock_flat = []
    mock_tree = []
    for n in sizes:
        flat_ms = bench_bls_aggregation(min(n, 10000))
        if n > 10000:
            # Extrapolate linearly for very large N
            flat_ms = flat_ms * (n / 10000)
        tree_ms = bench_bls_aggregation(branching)
        ratio = flat_ms / max(tree_ms, 0.001)
        mock_flat.append(flat_ms)
        mock_tree.append(tree_ms)
        print(f"  N={n:>7,}: flat_agg={flat_ms:>8.1f}ms  tree_agg={tree_ms:>6.2f}ms  ratio={ratio:>7.0f}x")

    results["mock"]["flat_agg_ms"] = mock_flat
    results["mock"]["tree_agg_ms"] = mock_tree

    # ---- Projected production BLS ----
    print()
    print("Production BLS (projected from blst benchmarks):")

    proj_flat_bls = []
    proj_tree_bls = []
    for n in sizes:
        fb = projected_flat_bls_only(n)
        tb = projected_tree_bls_only(n, branching)
        ratio = fb / max(tb, 0.001)
        proj_flat_bls.append(fb)
        proj_tree_bls.append(tb)
        print(f"  N={n:>7,}: flat_agg={fb:>9.1f}ms  tree_agg={tb:>6.1f}ms  ratio={ratio:>7.1f}x")

    results["projected"]["flat_bls_ms"] = proj_flat_bls
    results["projected"]["tree_bls_ms"] = proj_tree_bls

    # ---- Full processing time (production projected) ----
    print()
    print("Full processing time (distance + bloom + BLS, production projected):")

    proj_flat_full = []
    proj_tree_full = []
    proj_hs_full = []
    for n in sizes:
        ff = projected_flat_processing(n)
        tf = projected_tree_processing(n, branching)
        hf = projected_hotstuff_processing(n)
        proj_flat_full.append(ff)
        proj_tree_full.append(tf)
        proj_hs_full.append(hf)
        print(f"  N={n:>7,}: flat={ff:>9.1f}ms  tree={tf:>6.1f}ms  HotStuff={hf:>9.1f}ms")

    results["projected"]["flat_full_ms"] = proj_flat_full
    results["projected"]["tree_full_ms"] = proj_tree_full
    results["projected"]["hotstuff_full_ms"] = proj_hs_full

    # ---- Latency model (network + processing) ----
    print()
    print("Total latency model (network RTT + processing, production projected):")

    latency_flat = []
    latency_tree = []
    latency_hs = []

    for n in sizes:
        # Flat: 2 rounds * (global RTT + processing) + finality broadcast
        flat_proc = projected_flat_processing(n)
        flat_lat = 2 * RTT_GLOBAL_MS + flat_proc + RTT_GLOBAL_MS

        # Tree: leaf (local) + internal levels (regional) + root (global) * 2 for P1+P2
        tree_proc = projected_tree_processing(n, branching)
        n_leaves = math.ceil(n / branching)
        n_levels = 1
        nodes = n_leaves
        while nodes > 1:
            nodes = math.ceil(nodes / branching)
            n_levels += 1
        tree_lat = (RTT_LOCAL_MS + (n_levels - 2) * RTT_REGIONAL_MS +
                    RTT_GLOBAL_MS + tree_proc) * 2

        # HotStuff: retransmit + 3 rounds * (global RTT + processing)
        hs_proc = projected_hotstuff_processing(n)
        hs_lat = RTT_GLOBAL_MS + 3 * RTT_GLOBAL_MS + hs_proc

        latency_flat.append(flat_lat)
        latency_tree.append(tree_lat)
        latency_hs.append(hs_lat)
        print(f"  N={n:>7,}: flat={flat_lat:>9.1f}ms  tree={tree_lat:>8.1f}ms  HotStuff={hs_lat:>9.1f}ms")

    results["projected"]["latency_flat_ms"] = latency_flat
    results["projected"]["latency_tree_ms"] = latency_tree
    results["projected"]["latency_hotstuff_ms"] = latency_hs

    # Save for visualize.py
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
