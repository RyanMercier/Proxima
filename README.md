# Proxima: Distance-Preserving Digests for BFT Consensus

MEng capstone project, University of Connecticut. Advisor: Dr. Joe Johnson.

Every BFT consensus protocol uses collision-resistant hashes to compare validator state. Collision resistance destroys distance: validators agreeing on 19 of 20 transactions produce unrelated hashes, indistinguishable from validators sharing nothing. This forces three constraints across the BFT literature: validators must synchronize state before voting, agreement quality cannot be measured until votes are counted, and hierarchical committees must be large enough for independent BFT.

Proxima replaces collision-resistant hashes with distance-preserving transaction digests. SHA-512 output is split into 8 segments, summed across transactions, producing an 8D vector where Euclidean distance is proportional to disagreement. This single primitive removes all three constraints: agreement is measurable in one round, tree groups need only 10 validators (vs Ethereum's 128), and cross-shard consistency costs 128 bytes per shard pair instead of per-transaction coordination.

## Project Structure

```
blockchain.py          Core protocol: digests, bloom filters, BLS, consensus
                        - tx_to_vector(): SHA-512 -> 8D vector
                        - compute_vector(): commutative sum of tx vectors
                        - BloomFilter: set membership for sync
                        - BLSKeyPair: BLS signatures (py-ecc or mock)
                        - vector_consensus(): flat two-phase protocol
                        - tree_consensus(): hierarchical tree protocol
                        - Blockchain, State, Block, Transaction classes
                        - calibrate_threshold(): Monte Carlo threshold tuning
                        - MessageCounter: per-message-type accounting

hotstuff.py            HotStuff + PBFT simulation for fair comparison
                        - hotstuff_consensus(): 3-phase leader-based BFT
                        - pbft_consensus(): O(N^2) classic PBFT counts
                        - Same MessageCounter as blockchain.py for like-for-like

node.py                HTTP server with background miner threads
                        - Flask API on port 8545
                        - Colored terminal output showing consensus details
                        - Supports --tree mode with configurable branching

wallet.py              Interactive REPL client
                        - Connects to node, sends txs, checks balances
                        - No dependencies beyond stdlib + readline

stress.py              Benchmark and protocol comparison
                        - Standalone mode (no node needed)
                        - Flat vs tree vs HotStuff vs PBFT
                        - Scale test from N=100 to N=2000

benchmark_bls.py       BLS aggregation timing
                        - Mock BLS measured on local machine
                        - Production blst numbers projected from constants
                        - Saves benchmark_results.json

cross_shard_sim.py     Cross-shard verification comparison
                        - 2PC vs receipt-based vs digest comparison
                        - Propagation sensitivity sweep
                        - Multi-shard scaling analysis
                        - Bloom filter accuracy verification
                        - Saves cross_shard_results.json

visualize.py           Publication figures (8 PNGs)
                        - Reads cached JSON from benchmark_bls.py + cross_shard_sim.py

test_byzantine.py      Byzantine strategy verification
                        - Tests all 5 strategies (drop_half, random_vector,
                          replace_one_tx, mimic_honest, coalition)

requirements.txt       pip dependencies
```

Roughly 2,000 lines of Python total.

## Setup

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda activate kuramoto
pip install py-ecc bitarray matplotlib numpy flask
```

Dependencies:
- **py-ecc**: BLS signatures (Ethereum Foundation library). Used in demo mode with few validators. Benchmarks use hash-based mocks for speed.
- **bitarray**: Bloom filter bit arrays
- **numpy**: Vector math, distance computation
- **matplotlib**: Figure generation
- **flask**: HTTP server

## Reproducing the Paper

To regenerate every benchmark and figure from scratch:

```bash
pip install -r requirements.txt
python benchmark_bls.py        # generates benchmark_results.json
python cross_shard_sim.py      # generates cross_shard_results.json
python visualize.py            # generates figures/fig1..fig8
```

Total runtime: roughly 3 minutes on a laptop. The figures land in `figures/`.

## Interactive Demo

The demo runs across 3+ terminals. Start the node first, then connect wallets.

### Terminal 1: Start the node

```bash
python node.py --honest 4 --byzantine 1 --byzantine-strategy drop_half --interval 10
```

Arguments:
- `--honest N`: number of honest validators (default 4)
- `--byzantine N`: number of Byzantine validators (default 1)
- `--byzantine-strategy`: one of `drop_half`, `random_vector`, `replace_one_tx`, `mimic_honest`, `coalition`
- `--interval S`: seconds between mining attempts (default 5, use 10 for demo)
- `--tree`: enable tree-structured consensus
- `--branching N`: tree branching factor (default 10)
- `--port N`: HTTP port (default 8545)
- `--init-accounts`: pre-funded accounts (default: Alice Bob)
- `--init-balance`: starting balance (default 1000)
- `--miss-prob`: fraction of honest validators with partial observation (default 0.37)

### Terminal 2: Alice's wallet

```bash
python wallet.py --name Alice --node http://localhost:8545
```

Submit 10+ transactions before the first block mines (important for visible Byzantine exclusion):

```
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
send Bob 10
```

### Terminal 3: Bob's wallet

```bash
python wallet.py --name Bob --node http://localhost:8545
```

```
balance
send Alice 5
history
status
```

### What to watch for

The node terminal shows colored output per block:

```
[BLOCK #0] Proposed by Miner-0 (10 txs + coinbase)
  Phase 1: 4/5 vectors | variance=0.0000 | FAST PATH
  Excluded: Byz-0(d=14.6, drop_half)
  Finalized in 1 round | 13 msgs | 0.6 KB
  Balances: Alice=900.00 | Bob=1100.00
```

- Green = finalized, Red = Byzantine excluded
- Byzantine distance (~14) vs threshold (~4.5) shows clear separation
- Fast path means all honest validators agreed in one round

### Wallet commands

| Command | Description |
|---|---|
| `send <name> <amount>` | Send PROX coins |
| `balance` | Check balance (shows pending) |
| `history` | Transaction history |
| `status` | Chain height, mempool, last consensus |
| `mempool` | Pending transactions |
| `chain` | Chain info (height, supply) |
| `block <height>` | Block details |
| `help` | List commands |
| `quit` | Exit |

### Tree mode demo

```bash
python node.py --honest 9 --byzantine 3 --tree --branching 4 --interval 10
```

Output shows tree levels:

```
[BLOCK #3] Proposed by Miner-0 (20 txs + coinbase)
  Tree: 3 levels, branching=4
  Level 0: 3 leaves, 3 excluded, 9 passed filter
  Level 1: 1 node
  Phase 2: 9 BLS commits, aggregate sig 96 bytes
  FINALIZED | 42 msgs | 3.8 KB
```

## Benchmarks

### Stress test

Runs a full benchmark without a live node. Compares Proxima flat/tree against HotStuff and PBFT.

```bash
# Flat mode
python stress.py --txs 500 --validators 100 --byzantine 30

# Tree mode at scale
python stress.py --txs 500 --validators 1000 --byzantine 300 --tree

# Against a live node
python stress.py --txs 500 --node http://localhost:8545
```

Arguments:
- `--txs N`: total transactions to submit
- `--validators N`: validator count
- `--byzantine N`: Byzantine validator count
- `--txs-per-block N`: transactions per block (default 50)
- `--miss-prob F`: partial observation rate (default 0.37)
- `--strategy`: Byzantine strategy (default drop_half)
- `--tree`: enable tree mode
- `--node URL`: run against live node instead of standalone

### BLS benchmark

Measures the BLS aggregation bottleneck and projects production blst numbers.

```bash
python benchmark_bls.py
```

At N=100K with single-core BLS: flat aggregation takes 3,960ms on the critical path. Tree aggregation takes 9.9ms (each leaf processes ~7 signatures, internal nodes process ~10 child aggregates). Multi-core BLS (16-way) brings flat down to ~220ms and HotStuff's three rounds down to ~940ms. The tree gains nothing from extra cores because each leaf already processes under a millisecond of BLS work, but it retains its critical-path advantage at any core count.

Saves `benchmark_results.json`.

### Cross-shard simulation

Analytical comparison of cross-shard verification methods.

```bash
python cross_shard_sim.py
```

Compares 2PC, NEAR-style receipts, and digest comparison across propagation rates. At 95% propagation: digest comparison uses 5,052 messages versus 404,000 for 2PC and 101,000 for receipts (99% and 95% reduction respectively). Saves `cross_shard_results.json`.

### Byzantine strategy test

Tests all 5 attack strategies and shows exclusion behavior.

```bash
python test_byzantine.py
```

| Strategy | Excluded by Phase 1? | Distance | Caught by Phase 2? |
|---|---|---|---|
| drop_half | Yes | ~14 | N/A (already excluded) |
| random_vector | Yes | ~10-18 | N/A |
| coalition | Yes | ~16 | N/A |
| replace_one_tx | No | ~1.5 | Yes (invalid BLS commit) |
| mimic_honest | No | ~1.2 | Yes (invalid BLS commit) |

Phase 1 catches large deviations. Phase 2 catches small ones (and any adversarial transaction crafted to land within the threshold; see Section 8.4 of the paper).

## Publication Figures

```bash
python visualize.py
```

Generates 8 PNGs in `figures/`. Takes 2-3 minutes.

| Figure | What it shows |
|---|---|
| fig1 | SHA-256 Hamming distance (flat) vs 8D vector distance (proportional) |
| fig2 | Two-phase Proxima protocol vs HotStuff three-round structure |
| fig3 | Cross-shard overhead: digest vs 2PC vs receipts across propagation rates |
| fig4 | Fast path probability heatmap (miss rate x Byzantine fraction) |
| fig5 | Messages + bandwidth scaling: Tree, Flat, HotStuff, PBFT |
| fig6 | Tree level breakdown: where messages live (leaves dominate) |
| fig7 | BLS bottleneck: processing time flat vs tree vs HotStuff (blst projected) |
| fig8 | Latency model: network RTT + BLS processing, breakdown at N=100K |

Figures 7 and 8 use projected blst constants, not measured py-ecc times. Figure 3 uses analytical message counts from `cross_shard_sim.py`.

## API Reference

The node exposes JSON endpoints on port 8545:

| Endpoint | Method | Description |
|---|---|---|
| `/tx` | POST | Submit transaction `{"sender": "Alice", "receiver": "Bob", "amount": 50}` |
| `/balance/<name>` | GET | Balance + pending info |
| `/chain` | GET | Height, tip hash, total supply |
| `/block/<height>` | GET | Block details with transactions |
| `/mempool` | GET | Pending transactions |
| `/validators` | GET | Validator list, Byzantine status |
| `/register` | POST | Register account `{"name": "Carol", "balance": 1000}` |
| `/status` | GET | Node status, last consensus result |
| `/history/<name>` | GET | Transaction history |
| `/stress/submit` | POST | Bulk transaction submission |
| `/stress/mine` | POST | Mine one block immediately |

## Protocol Overview

### The Primitive

Each transaction is hashed with SHA-512 (64 bytes), split into 8 segments of 8 bytes, each mapped to [0, 1). The result is an 8D vector per transaction. A validator's digest is the commutative sum of its transaction vectors.

Three properties hashes do not have:
1. **Proportional distance**: missing 1 tx = distance ~1.6, missing 10 = distance ~14.5
2. **Exact summarization**: weighted mean of N digests is exact, not an approximation
3. **Set difference identification**: bloom filter diff finds exactly which txs are missing

### Flat Consensus (vector_consensus)

1. Validators send digest (64 bytes) + bloom filter (25 bytes) to aggregator
2. Aggregator measures distance from reference, clusters within threshold
3. Bloom diff pushes missing txs to incomplete validators
4. If variance near zero: fast path, finalize in 1 round
5. Otherwise Phase 2: BLS commits, aggregate signature (96 bytes) + bitmap, 2/3 required

### Tree Consensus (tree_consensus)

Same two phases, routed through a tree of branching factor B (default 10).

- Leaves filter by distance (no per-group BFT), compute weighted mean, send 76-byte summary up
- Internal nodes aggregate child summaries (weighted mean of means is exact)
- Phase 2: BLS commits route up through tree, each node aggregates (BLS is associative)
- Finality proof broadcasts back down to all validators

Key insight: leaves never "fail." A leaf with 5 Byzantine out of 10 excludes the 5 and reports the mean of the remaining 5. No group vote needed. A leaf entirely Byzantine produces a fabricated summary that the root's distance check still rejects.

## Key Numbers

### Message complexity (30% Byzantine, 37% partial observation)

| Validators | Proxima Tree | Proxima Flat | HotStuff | PBFT |
|---|---|---|---|---|
| 1,000 | 2,990 | 3,348 | 6,518 | 1,999,000 |
| 10,000 | 30,042 | 33,600 | 65,180 | 199,990,000 |
| 100,000 | 300,245 | 335,803 | 651,800 | ~20 billion |

Proxima Tree is 2.2x fewer messages than HotStuff. PBFT is included for context (O(N^2)).

### Projected finality latency at N=100,000

|  | Proxima Tree | Proxima Flat | HotStuff |
|---|---|---|---|
| BLS processing (1 core) | 9.9 ms | 3,960 ms | 17,595 ms |
| Network RTT (model) | 892 ms | 600 ms | 800 ms |
| Total finality (1 core) | 902 ms | 4,561 ms | 18,395 ms |
| BLS only (16 cores) | ~10 ms | ~220 ms | ~940 ms |

Multi-core BLS narrows the gap considerably but does not eliminate it. The tree retains a critical-path BLS advantage at any core count because each leaf only aggregates ~7 signatures (already under a millisecond). The structural advantages (message complexity, smaller hierarchical groups, fast-path finality) are core-count-independent.

### Cross-shard overhead (1000 cross-shard txs, 100 validators per shard)

| Method | Messages | Bandwidth |
|---|---|---|
| 2PC | 404,000 | 37,750 KB |
| Receipt (NEAR) | 101,000 | 12,625 KB |
| Digest (95% propagation) | 5,052 | 987 KB |

99% reduction versus 2PC, 95% versus receipts. Cost scales with conflicts (5%) rather than with total cross-shard volume.

## Security

**Safety (proved):** fewer than N/3 Byzantine cannot cause conflicting finalization. Phase 1 clustering cannot affect safety (wrong inclusion = invalid BLS sig, wrong exclusion = reduced liveness). Tree routing cannot affect safety (BLS aggregation is associative, Byzantine nodes can suppress but not forge).

**Liveness (bounded):** by Hoeffding's inequality, the probability that more than N/3 honest validators are excluded is at most exp(-0.22N), below 10^-9 at N=100 and below 10^-95 at N=1000. Rotating aggregator under partial synchrony ensures progress after GST.

**Adversarial transaction construction:** Phase 1 distance filtering provides no cryptographic security. A Byzantine adversary can craft a transaction set whose digest lands within the threshold using k-list search (~12 bits of work at our parameters). This is fine: Phase 1 is a filter, not a safety mechanism. Phase 2 still requires a valid BLS signature on the honest block hash, which the Byzantine cannot produce. The cost of a successful Phase 1 spoof is one wasted Phase 2 round, indistinguishable from any other Byzantine signature withholding.

**Byzantine tolerance:** 100% success 0-33%, 0% at 35%+, matching the standard BFT bound.
