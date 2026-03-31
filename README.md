# Proxima: Locality-Sensitive BFT Consensus

MEng capstone project. A BFT consensus protocol that uses distance-preserving transaction vectors instead of collision-resistant hashes. SHA-256 destroys distance between similar inputs (by design), but consensus benefits from preserving it. When validators agree on 19/20 transactions, their vectors are close. When they disagree on half, they are far apart. This lets validators vote immediately without synchronizing state first.

Includes an interactive blockchain demo (multiple terminals, real accounts, real transactions), a tree-structured hierarchical mode, a HotStuff/PBFT comparison, and publication figures.

## Setup

Requires Python 3.10+ and the following packages:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda activate kuramoto
pip install py-ecc bitarray matplotlib numpy flask
```

## Quick Start

### 1. Start the node

```bash
python node.py --honest 4 --byzantine 1 --byzantine-strategy drop_half --interval 10
```

This starts the blockchain server on port 8545 with 4 honest validators and 1 Byzantine that drops half its transactions. The 10-second interval gives you time to submit transactions before the first block mines.

Pre-funded accounts: Alice and Bob, 1000 PROX each.

### 2. Open wallets (separate terminals)

```bash
# Terminal 2
python wallet.py --name Alice --node http://localhost:8545

# Terminal 3
python wallet.py --name Bob --node http://localhost:8545
```

### 3. Send transactions

In Alice's wallet, submit 10+ transactions before the first block mines (this makes Byzantine exclusion clearly visible):

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

In Bob's wallet:

```
balance
send Alice 5
history
```

Watch the node terminal. With 10+ transactions, the Byzantine validator gets excluded (distance ~14 vs threshold ~4.5). Colored output shows green for finalized, red for Byzantine with its distance printed.

### 4. Wallet commands

```
send <name> <amount>    Send coins
balance                 Check balance
history                 Transaction history
status                  Chain height, mempool, last consensus
mempool                 Pending transactions
block <height>          View block details
chain                   Chain info
help                    List commands
quit                    Exit
```

## Tree Mode

Proxima supports hierarchical tree-structured consensus. Validators are grouped into leaves of size `branching`. Each leaf filters Byzantine validators by distance, computes a weighted mean, and sends a 76-byte summary upstream. Internal nodes aggregate child summaries. Phase 2 BLS commits route through the same tree.

The tree does not need per-leaf BFT. Distance filtering works at the individual level, not the group level. A leaf with 5 Byzantine out of 10 just excludes the 5 and reports the mean of the remaining 5.

```bash
# Tree mode with 12 validators, branching factor 4
python node.py --honest 9 --byzantine 3 --tree --branching 4 --interval 10
```

The server terminal shows tree levels:

```
[BLOCK #3] Proposed by Miner-0 (20 txs + coinbase)
  Tree: 3 levels, branching=4
  Level 0: 3 leaves, 3 excluded, 9 passed filter
  Level 1: 1 node
  Phase 2: 9 BLS commits, aggregate sig 96 bytes
  FINALIZED | 42 msgs | 3.8 KB
```

## Stress Test

Runs a full benchmark without a live node. Compares Proxima (flat and tree) against HotStuff and PBFT.

```bash
# Flat mode, 100 validators
python stress.py --txs 500 --validators 100 --byzantine 30

# Tree mode, 1000 validators
python stress.py --txs 500 --validators 1000 --byzantine 300 --tree

# Against a live node
python stress.py --txs 500 --node http://localhost:8545
```

Output includes per-block results, a summary table, protocol comparison, and a scale test from N=100 to N=2000.

## BLS Benchmark

Measures BLS aggregation time (the bottleneck at scale) and projects production numbers using blst constants from Ethereum client benchmarks.

```bash
python benchmark_bls.py
```

Outputs measured mock timings plus projected blst production numbers. Saves `benchmark_results.json` for the figures. Key result: at N=100K, flat BLS aggregation takes 3.5s on the critical path while tree takes 10ms.

## Byzantine Strategy Test

Tests all 5 Byzantine strategies and shows exclusion behavior.

```bash
python test_byzantine.py
```

Strategies:
- **drop_half**: drops every other transaction. Excluded at distance ~14.
- **random_vector**: fabricates a random vector. Excluded at distance ~10-18.
- **coalition**: only keeps first half of transactions. Excluded at distance ~16.
- **replace_one_tx**: swaps one transaction. Passes Phase 1 distance filter (distance ~1.5), caught by Phase 2 BLS commit.
- **mimic_honest**: changes one transaction slightly. Same as above.

Phase 1 catches large deviations. Phase 2 (BLS hash commits) catches small deviations that slip through the distance filter.

## Publication Figures

Generates 7 PNGs in `figures/`.

```bash
python visualize.py
```

1. **Distance-preserving vs distance-destroying** -- SHA-256 Hamming distance (flat, no structure) vs 8D vector Euclidean distance (proportional to disagreement)
2. **Scale comparison** -- Messages and bandwidth for Proxima Tree, Proxima Flat, HotStuff, and PBFT from N=100 to N=10,000
3. **Byzantine sweep** -- Security and message counts as Byzantine fraction increases from 0% to 45%
4. **Fast path heatmap** -- 1-round finality rate across miss probability and Byzantine fraction
5. **Tree level breakdown** -- Stacked bars showing where messages go in the tree (most stay at the leaves)
6. **BLS aggregation bottleneck** -- Processing time at the aggregator (tree stays constant, flat grows linearly)
7. **Latency model** -- Total finality latency (network RTT + BLS processing) across a 3-region deployment

Figures 6-7 use projected blst production numbers, not measured py-ecc times.

## Protocol Summary

### Flat consensus (vector_consensus)

1. Each validator computes an 8D transaction vector (SHA-512 split into 8 segments) and a bloom filter (~25 bytes) of its transaction set.
2. Aggregator receives vectors, measures Euclidean distance from the reference (proposed block's full vector). Validators beyond the threshold are excluded.
3. Aggregator diffs bloom filters to push missing transactions to incomplete validators.
4. If cluster variance is near zero, finalize in 1 round (fast path). Otherwise run Phase 2.
5. Phase 2: cluster members send BLS-signed hash commits. Aggregator produces an aggregate signature (96 bytes) + signer bitmap. Finality requires 2/3 matching.

### Tree consensus (tree_consensus)

Same two phases, routed through a tree of branching factor 10.

- Phase 1 (bottom-up): leaves filter by distance, compute weighted mean, send 76-byte summaries upstream. Internal nodes aggregate child summaries. Root checks global mean.
- Phase 2 (top-down then up): root broadcasts "collect commits". Validators send BLS commits up through the tree. Each node aggregates child signatures (BLS aggregation is associative). Root produces final aggregate. Finality proof broadcasts back down to all validators.

### Key numbers (N=100, 30% Byzantine, 37% partial observation)

| Protocol | Messages | Bandwidth |
|----------|----------|-----------|
| Proxima Flat | ~336 | ~31 KB |
| Proxima Tree | ~336 | ~31 KB |
| HotStuff | ~650 | ~100 KB |
| PBFT | ~19,900 | ~2,488 KB |

Tree and flat converge at small N. The tree advantage shows at N=1000+ where message routing through the tree reduces aggregator load.

## File Structure

```
blockchain.py        Core: vectors, bloom, BLS, consensus (flat + tree), chain state
hotstuff.py          HotStuff and PBFT simulation for comparison
node.py              HTTP server, miner threads, colored terminal output
wallet.py            Interactive REPL client
stress.py            Benchmark: bulk txs, scale test, protocol comparison
visualize.py         Publication figures (7 PNGs)
benchmark_bls.py     BLS aggregation timing benchmark
test_byzantine.py    Byzantine strategy verification
requirements.txt     pip dependencies
```

## API Endpoints

The node exposes a JSON API on port 8545:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tx` | POST | Submit a transaction (sender, receiver, amount) |
| `/balance/<name>` | GET | Account balance |
| `/chain` | GET | Chain height, tip hash, total supply |
| `/block/<height>` | GET | Block details |
| `/mempool` | GET | Pending transactions |
| `/validators` | GET | Validator list and status |
| `/register` | POST | Register a new account |
| `/status` | GET | Node status, last consensus result |
| `/history/<name>` | GET | Transaction history for an account |

## Dependencies

- **py-ecc**: Ethereum Foundation BLS implementation (used for real BLS in demo mode, mocked in benchmarks)
- **bitarray**: Bloom filter bit arrays
- **numpy**: Vector math, distance computation
- **matplotlib**: Figure generation
- **flask**: HTTP server for the node
