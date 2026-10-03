# Proxima: Agreement Observability for Byzantine Consensus

[![tests](https://github.com/RyanMercier/Proxima/actions/workflows/tests.yml/badge.svg)](https://github.com/RyanMercier/Proxima/actions/workflows/tests.yml)

MEng capstone project, University of Connecticut. Advisor: Dr. Joe Johnson.
MIT licensed. Cite via [CITATION.cff](CITATION.cff).

BFT protocols compare validator state with collision-resistant hashes, which
say *whether* two views differ but not *by how much*. Proxima studies what a
consensus protocol gains from measuring how much, and what it must never do
with that measurement.

- **Branch `master`**: the artifact for the BLOCKCHAIN'26 paper
  ("Distance-Preserving Digests: A Primitive for BFT Consensus",
  [arXiv:2605.15329](https://arxiv.org/abs/2605.15329)).
- **Branch `v2`** (this branch): the journal version, which corrects several
  claims of the conference paper (see "Corrections" below) and is the code
  behind `paper_v2/proxima_v2.tex`.

## What v2 contains

| Module | What it does |
|---|---|
| `dpmh.py` | Ledger digest = **LtHash** (uniform Z_q^1024, binding) + **Rademacher sketch** (`{-1,+1}^512` sums, AMS estimator of the symmetric difference). Salts from height + previous block hash. `adversarial_balance()` demonstrates the deflation attack. Prefix accumulators, fork localization, prune accounting. |
| `bft.py` | The base protocol: single-slot two-phase locking BFT with view change, plus the one-round fast path at quorum `ceil((n+3f+1)/2)`. Adversarial fuzzer (equivocating leaders, double voting, lying NEW-VIEWs, partial delivery) and a counterexample showing the `2f+1` fast path is unsafe. |
| `blockchain.py` | Ledger, validators, and per-block message/byte accounting of that protocol: flat and tree modes, fetch-by-id for stragglers, verified signatures (real BLS12-381 or a verifiable mock), sketches on prepare votes for observability, readiness sensing, partition healing. |
| `reconcile.py` | IBLT set reconciliation with per-session salted short ids; the decoder only knows its own items. |
| `conservation.py` | Cross-shard conservation: corridor-keyed LtHash accumulators indexed by debit height, exact detection of dropped / replayed / minted / misrouted credits, group-testing localization, multiset decode. |
| `hotstuff.py` | Baselines under one shared cost model: HotStuff, HotStuff-2, chained HotStuff, Kauri (message model), PBFT; bitmap or threshold certificates; leader ingress. |
| `test_v2.py` | 59 executable checks of every behavioural claim in the paper. |
| `visualize_v2_paper.py` | Produces every figure in `paper_v2/figures/` and every number in `paper_v2/results.json` (bit-reproducible). |
| `node.py`, `wallet.py`, `stress.py` | Interactive demo node, wallet REPL, and standalone benchmark. |

## Reproduce

Python 3.11+.

```bash
pip install -r requirements.txt        # or requirements-lock.txt for exact versions
python test_v2.py                      # 59 checks, about 15 s
python test_byzantine.py
python bft.py                          # safety fuzzing summary
python conservation.py                 # conservation demo
python visualize_v2_paper.py           # all figures + results.json, about 90 s
```

To build the paper (needs a LaTeX install with IEEEtran):

```bash
cd paper_v2 && pdflatex proxima_v2 && bibtex proxima_v2 && pdflatex proxima_v2 && pdflatex proxima_v2
```

## Key results (all from `paper_v2/results.json`)

- **Estimation is advisory.** The sketch is unbiased for sets fixed before the
  salt, but greedy side-choosing holds the estimate near 14 at n = 512 for any
  true difference from 200 to 1600. Estimates gate scheduling, sync, and
  monitoring only, and every consumer fails soft.
- **Binding needs LtHash.** A `{-1,+1}` sum never wraps modulo q for realistic
  sets, so a mod-q collision is an integer collision and the modulus adds no
  security. The ledger digest binds with LtHash.
- **Safe fast path.** 0 conflicting decisions in 50,000 fuzzed executions at
  quorum `ceil((n+3f+1)/2)` (with vote histories in NEW-VIEW); the `2f+1`
  rule of the conference paper decides conflicting values in about 4% of
  executions. At n = 3f+1 the safe fast quorum is all n.
- **Cost.** Under one cost model the augmented protocol matches HotStuff-2 in
  messages and costs about 46% more bytes at N = 1000; chained HotStuff is
  cheapest. The contribution is observability, not efficiency.
- **Conservation.** Exact detection and localization of cross-shard faults
  with no false alarms under honest settlement lag; cheaper than per-corridor
  batch roots only on dense corridor graphs (from 34 shards on a full mesh).

## Corrections to the conference paper

1. The one-round fast path at 2N/3 signatures is unsafe across view changes.
2. The liveness bound `exp(-2N(1/3-p)^2)` ignores Byzantine abstainers.
3. The 2x message advantage over HotStuff came from unequal accounting
   (certificates not delivered to all validators, different retransmission
   costs, no chained HotStuff).
4. The aggregator-chosen reference gave the aggregator discretion; the
   reference is now D(B) of the proposal.
5. Byzantine validators were never tree leaf leaders (they were placed last).

Details and evidence: Section "Corrections to the Earlier Version" of
`paper_v2/proxima_v2.tex`.

## Interactive demo

```bash
python node.py --honest 4 --byzantine 1 --byzantine-strategy equivocate --interval 10
python wallet.py --name Alice --node http://localhost:8545    # in another terminal
```

Strategies: `drop_half`, `random_vector`, `replace_one_tx`, `mimic_honest`,
`coalition`, `equivocate`, `abstain`. The node prints, per block, the
sketch flags (observability only), id fetches by stragglers, fast or slow
path, valid signature count, messages, bytes, and leader ingress. The wallet
supports `send`, `balance`, `history`, `status`, `mempool`, `chain`, `block`.
