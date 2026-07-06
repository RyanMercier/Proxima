# Proxima v2: Work Plan

Companion to `docs/v2_formal_foundations.md` (the technical content) and the
reviewer critique that motivated it. This file is the *plan*: what to build,
in what order, with dependencies, effort, risks, and the decisions that shape
scope. The camera-ready (BLOCKCHAIN'26, submitted) is frozen; v2 is a new
paper, not an edit of it.

---

## 0. Thesis shift

v1 pitch: "replace collision-resistant hashes with distance-preserving
digests." v2 pitch: **separate agreement estimation from agreement security,
using one linear-algebraic object that is simultaneously a collision-binding
homomorphic accumulator and an unbiased symmetric-difference estimator.**
Proxima becomes the flagship instantiation; the primitive, the impossibility
result, and the composition theorem become the backbone; conservation
sharding and digest-space sync become co-equal applications.

Working title candidates (decide in G): "Agreement-Aware BFT," "Set
Reconciliation as a Consensus Primitive," "Distance-Preserving Multiset
Hashes for Consensus." The title must stop selling the primitive as the
security object; safety comes entirely from signatures.

---

## 1. Strategic decisions (make these first; they gate scope)

These are forks, not tasks. Each changes how much of the plan runs.

- **D1. One paper or two?** The cross-shard conservation result (Workstream D)
  is strong enough to stand alone. Option (a): one flagship paper, primitive +
  Proxima + conservation as co-equal apps. Option (b): split, with the
  primitive+consensus paper first and a conservation-sharding paper second.
  Recommendation: build D as a self-contained module regardless, decide at
  draft time based on length.
- **D2. Testbed: yes or no?** A real 50-200 node, 3-region deployment converts
  Table IV from projection to measurement and is, per the critique, "the
  difference between a workshop paper and a conference paper." Biggest single
  lift in the plan (Workstream E5). Recommendation: yes if targeting AFT /
  Financial Cryptography / a systems venue; optional if targeting IEEE ICBC.
- **D3. Cryptanalysis depth.** The SIS binding claim (Workstream B4) needs
  real cryptanalysis at the exact parameters, or an honest "reduction shape +
  open problem" framing, or a cryptographer co-author. This is the highest
  technical risk. Recommendation: inherit the LtHash analysis, present a
  reduction sketch, and state the exact-parameter cryptanalysis as open unless
  a collaborator joins.
- **D4. Venue + timeline.** AFT and Financial Cryptography reward the
  theory+systems combination; IEEE ICBC is faster and lighter. Venue sets the
  bar for D2 and B4.

---

## 2. Workstreams

Effort tags are rough: S = days, M = 1-2 weeks, L = 3+ weeks. Dependencies
reference other task IDs.

### A. Cryptographic core (the new primitive)

New module `dpmh.py`, replacing the `tx_to_vector` / `compute_vector` /
`N_DIMS = 8` construction in `blockchain.py`.

- **A1 [M]** Implement the Rademacher construction: `v_h(tx) = sgn(SHA-512(
  domain || height || tx))` in {-1,+1}^n, integer coordinate sums, stored as
  signed words. Two domains: round-salted (q = 2^16, n = 512 default) and
  ledger-fixed (q = 2^32). Reference impl is in v2_formal_foundations.md S1.
  Deterministic integer arithmetic only, no floats on any code path that
  feeds consensus (closes the float-nondeterminism bug class).
- **A2 [S]** Domain separation by round/height so precomputed collision sets
  cannot be replayed across blocks. Enforce multiplicity bound mu = 1 within a
  block at validation (a block is a set).
- **A3 [S]** Test vectors as executable checks: d = 1 must give ||ΔD||² = n
  exactly; d = 2 must match the moments of 4·Bin(n, 1/2); d = 3 vs
  n + 8·Bin(n, 1/4). These double as the correctness proof for A1.
- **A4 [S]** Parameter profiles table (n = 512 default, n = 128 light,
  accumulator n = 512 q = 2^32) as config, wired so figures can sweep them.
- **A5 [S]** Formal definition writeup: a (d_max, ε, δ)-distance-preserving
  multiset hash with homomorphism; saturation at d_max = q/2. (Feeds B.)

### B. Theory / proofs

- **B1 [S]** Lemma 1 (unbiased estimator: E||ΔD||² = nd, Var(d_hat) =
  2d(d-1)/n, RSE ≤ sqrt(2/n)) + Lemma 2 (Hoeffding concentration). Straight
  from AMS/Tug-of-War specialized to multiset indicators; cite Alon-Matias-
  Szegedy. Depends A1.
- **B2 [S]** Closed-form threshold τ² = 3n with the honest-exclusion (d ≤ 2)
  and Byzantine-inclusion (d = 4) tail bounds. **Delete the Monte Carlo
  calibration** (old III-C); simulation becomes validation, not definition.
- **B3 [S]** Liveness theorem with the analytic false-exclusion term
  p_miss·e^{-n/8} replacing v1's empirical 5e-3, and the "exclusion is a
  scheduling event, not a participation event" reframing (excluded validators
  rejoin Phase 2 after sync). The Hoeffding step gets cleaner because the k²
  mean-offset term vanishes with zero-mean coordinates.
- **B4 [L, RISK]** Collision-resistance of the accumulator: reduce
  collision-finding to ±1-matrix SIS in the ROM; give first-cut Wagner k-tree
  and lattice cost estimates under the weight/multiplicity caps; inherit the
  LtHash parameter analysis. Do **not** claim a security level without
  dedicated cryptanalysis (see D3). This is the item most likely to need
  outside help.
- **B5 [M]** Impossibility theorem (proximity non-bindability): any
  distance-preserving hash cannot make proximity authoritative; hence
  proximity may gate scheduling/sync/optimism but never safety. Frame as the
  public-coin multiset-hash analogue of Hardt-Woodruff (linear sketches not
  adaptively robust). This is the theorem that makes the estimation/security
  separation *forced*, not stylistic. Depends B1.
- **B6 [M]** Composition theorem (safety preservation): any BFT protocol whose
  commit predicate is "2N/3 signatures on H(B)" plus an estimation layer that
  gates only scheduling/sync/reporting inherits safety verbatim (simulate the
  estimation layer internally; commit messages identical). The v1 fast-path
  correction is the corollary — **already implemented in the camera-ready**,
  so this is writeup, not code. Depends B5.
- **B7 [M]** Conservation soundness theorem + O(f log S) localization
  correctness (bisection over a linear invariant is exact). Depends D.
- **B8 [S]** Adversarial-deflation analysis (near-antipodal grinding: factor-2
  at ~2^47/block, factor-4 at ~2^104) and the two-domain design consequence
  it forces. Depends A2.

### C. Protocol improvements (simulator)

Extend `vector_consensus` / `tree_consensus` in `blockchain.py`.

- **C1 [M]** Readiness sensing: proposer watches gossiped digest-cloud
  variance and proposes when it drops below threshold, turning the fast-path
  probability from weather ((1-p_miss)^N) into a controlled quantity. New
  simulation knob; feeds E4.
- **C2 [M]** Robust reference selection: replace the aggregator's-own-state
  reference with a Byzantine-robust location estimate (coordinate-wise median
  / trimmed mean / geometric median, breakdown ≥ 1/3), verifiable from signed
  Phase 1 digests. Import Krum / Yin et al. Feeds E3, B (robustness note).
- **C3 [L]** Two-syndrome reconciliation: **replace the bloom filter
  everywhere** (Phase 1 sync, cross-shard resolution, fork healing) with
  estimate-then-decode — DPMH d_hat sizes the divergence, then minisketch at
  capacity ~d_hat (or rateless IBLT when d_hat is unreliable) recovers the
  exact differing transactions. Deletes the bloom false-positive analysis
  (old VIII-E) and the "bloom FP → exclusion" liveness chain. Needs a
  minisketch/PinSketch binding or a pure-Python IBLT. Feeds E6.
- **C4 [M]** Fork localization via cumulative accumulators: binary search on
  range digests P[b]-P[a], O(log H) rounds, each probe also yielding d_hat.
  State-sync as the same machinery. Feeds E-fork figure.
- **C5 [M]** Partition healing: bimodal cloud → exact cluster centroids by
  linearity → inter-camp d_hat → one reconciliation exchange. Subsumes the
  v1 fast-path variance check as the unimodal special case. Feeds E7.

### D. Cross-shard conservation (the differentiated application)

Extend `cross_shard_sim.py` (has `two_phase_commit_cost`, `receipt_cost`,
`digest_cost`, `multi_shard_overhead`). Build as a self-contained module so
it can spin out per D1.

- **D1x [M]** Conservation model: each cross-shard tx contributes +v(t) to
  source outbound O_ij and +v(t) to dest inbound I_ij (ledger domain); beacon
  checks the global invariant Σ O_i = Σ I_j per block, O(S) signed ~KB
  objects. Compare against **batched** 2PC and **batched** receipts on
  per-block verification metadata and failure-path cost (not raw counts vs
  unbatched 2PC — that comparison will get flagged).
- **D2x [M]** Group-testing localization: beacon bisects the edge set on
  invariant failure, O(f log S) signed digest requests, then the implicated
  pair runs C3 reconciliation. Correctness is B7.
- **D3x [S]** Pending-set digest / window alignment (from v2 foundations S14):
  P_ij = O_ij - I_ij *is* the ledger digest of the in-flight set on that
  corridor; its norm meters in-flight volume continuously, and the invariant
  is checked as "P_ij decays to zero within W blocks," not per-block equality
  (credits necessarily lag debits by network latency). Add per-corridor
  aging → alarm.
- **D4x [S]** Corridor-local syndrome (from S14): maintain the minisketch
  BCH syndrome of the in-flight set alongside P_ij (both linear), so on alarm
  the destination decodes the dangling transactions with no extra exchange.
- **D5x [S]** Global ledger digest G += ΣΔ_i per block (one ~2KB object for
  the whole system's applied history) with verifiable prune accounting
  (D(before)-D(after) = D(P)). Contrast Merkle roots (no subtraction, no
  cross-shard addition, no O(1) range extraction).
- **D6x [S]** Write the semantics caveat plainly: this is detection +
  settlement verification, **not** locking/atomicity. Pair with optimistic
  execution + W-block revert, or credit-on-match (receipt-equivalent latency,
  constant-size metadata). The contribution is verification and
  fraud-localization cost.

### E. Evaluation / experiments

Regenerate through `visualize.py` (has figure1-8 + compact variants);
baselines in `hotstuff.py`.

- **E1 [S]** Fig 1 rerun with zero-mean coordinates: box plots collapse onto
  nd with the Lemma-1 variance; overlay the analytic curve; **add the
  substitution case** (v1's blind spot) showing it now reads as 2j. Depends A1.
- **E2 [S]** Threshold figure: exact d = 1..4 distributions vs τ² = 3n,
  log-scale tails, analytic bounds overlaid. Depends B2.
- **E3 [M]** Robust reference: adversarial aggregator biasing the reference,
  v1 vs geometric-median; honest-exclusion rate vs Byzantine fraction. Depends
  C2.
- **E4 [M]** Fast path: speculative-signing trigger vs v1 variance trigger
  under one in-cluster Byzantine (**shows the griefing fix**); fast-path rate
  vs p_miss, sensed vs fixed-interval proposal. Depends C1.
- **E5 [L, RISK, gated by D2]** Real multi-region testbed: 50-200 nodes on
  cheap VPSes across 3 regions, converting Table IV latency from projection to
  measurement. Also lets us measure, not assume, p_miss. Largest lift.
- **E6 [M]** Reconciliation: bloom (v1) vs minisketch vs rateless IBLT, bytes
  and rounds vs true difference size. Depends C3.
- **E7 [M]** Partition healing: 60/40 split with d disputed transactions;
  healing cost vs d and vs a PBFT-style view-change message storm. Depends C5.
- **E8 [M]** **Kauri and HotStuff-2 baselines in the message/latency tables.**
  Per the critique this is *the* comparison, not future work: the claimed edge
  over tree-HotStuff is exactly "small groups without per-group BFT," so
  without Kauri the headline numbers are unanchored, and "two phases vs three"
  halves against HotStuff-2. A simulated Kauri using the same `MessageCounter`
  is acceptable for a first submission. Also address chained/pipelined
  HotStuff (per-block amortized cost differs from a per-block comparison).
  New functions in `hotstuff.py`.
- **E9 [S]** Ground p_miss = 0.37 in published gossip-propagation
  measurements, or present everything as sensitivity curves with 0.37 as one
  point. Cheap, kills an "arbitrary constant" objection.
- **E10 [M]** Tree liveness under Byzantine relays: with 30% Byzantine and
  rotating leaf leaders, quantify the expected fraction of suppressed subtrees
  per round (Ethereum's 128 is partly about aggregation availability, not just
  per-committee safety). New analysis + simulation.

### F. Related work + positioning

- **F1 [M]** Literature pass and citation of: Clarke et al. (ASIACRYPT 2003,
  incremental multiset hashes — the construction family), Bellare-Micciancio
  (AdHash), Wagner (k-tree), LtHash (Lewi et al. 2019 — closest construction),
  Alon-Matias-Szegedy (AMS/Tug-of-War — the estimator), Hardt-Woodruff (STOC
  2013 — non-robustness), Minsky-Trachtenberg-Zippel + minisketch/Erlay
  (Naumenko et al. 2019) + Graphene (Ozisik et al. 2019) + rateless IBLT
  (Yang et al. SIGCOMM 2024) — the decoding side, Goodrich-Mitzenmacher IBLT,
  Blanchard et al. Krum + Yin et al. — robust aggregation, Zyzzyva / SBFT /
  HotStuff-2 — speculative + modern baseline. Also sweep CRDT anti-entropy /
  Dynamo Merkle-repair and "homomorphic fingerprinting."
- **F2 [S]** Verify the precise novelty claim is unclaimed before writing "to
  our knowledge new": the specific target is *an L2 distance guarantee stated
  for a collision-binding additive multiset hash*. If someone stated it, cite
  and pivot to the consensus applications (which still carry the paper). The
  defensible claim is the three-part one in v2_foundations S12: (a) one object
  = binding accumulator + unbiased symdiff estimator with a forced role
  separation, (b) quantitative public-hash deflation analysis + two-domain
  design, (c) the protocol layer (estimation-augmented BFT transformer,
  conservation sharding with O(f log S) localization, digest-space fork/
  partition healing). **Do not claim a new hash.**

### G. Writing / venue

- **G1 [S]** Resolve D1-D4.
- **G2 [M]** Draft the theory core (A5, B1-B8) as the paper's backbone.
- **G3 [M]** Draft protocol + applications (C, D) around it.
- **G4 [S]** Draft evaluation (E) once figures land.
- **G5 [S]** Move the extended-version [proxima-tr] material and the arXiv-id
  TODO forward; the v1 camera-ready cites the repo as the technical report,
  so v2 / the extended version should become that arXiv posting.

---

## 3. Critical path & suggested ordering

The critique's order of operations, expanded:

1. **A1-A4** (Rademacher core + test vectors). Everything downstream needs it.
2. **B1-B3, E1-E2** (estimator theory + reran Fig 1 + threshold figure). This
   is the quickest visible win — figures get *cleaner*, and the substitution
   blind spot closes. Do this early to de-risk the central claim.
3. **C1, E4** (readiness sensing + fast-path figure). The speculative fast
   path is already implemented; this adds the sensing knob and the
   griefing-fix figure.
4. **E8, E9, E10** (Kauri + HotStuff-2 baselines, p_miss grounding, tree
   liveness). The evaluation objections reviewers *will* raise; cheap
   relative to their impact.
5. **C3, E6** (two-syndrome reconciliation). Replaces bloom, deletes an entire
   liveness chain.
6. **D1x-D6x, B7** (conservation sharding). The differentiated application;
   self-contained so it can spin out.
7. **C2/E3, C4, C5/E7** (robust reference, fork localization, partition
   healing). Rounds out the "what estimation buys you" story.
8. **B4, B5, B6, B8, F1, F2** (the crypto theory + impossibility + composition
   + related work). B5/B6 can be written early (they don't depend on
   cryptanalysis); B4 is the long pole.
9. **E5** (testbed), if D2 = yes. Runs in parallel with writing.
10. **G** (drafting), overlapping 6-9.

Fastest path to a submittable IEEE-ICBC-tier paper: 1-5 + 8 (B5/B6) + G,
skipping the testbed and deep cryptanalysis. Fastest path to an
AFT/FC-tier paper: add 6, 9, and B4-with-a-cryptographer.

---

## 4. Risk register

- **R1 (B4, high).** Cryptanalysis at the exact parameters could reveal weaker
  security than the n·log q ≈ 8192-bit budget suggests, or simply be beyond a
  first-cut estimate. Mitigation: inherit LtHash, frame as reduction shape +
  open problem, or recruit a cryptographer (D3).
- **R2 (F2, medium).** The distance property of bounded-coordinate additive
  multiset hashes might already be written down. Mitigation: F2 before any
  novelty language; the consensus applications carry the paper regardless.
- **R3 (E5, medium).** Testbed is time/cost heavy and can slip. Mitigation:
  gate on D2; the projection-based tables already exist as a fallback.
- **R4 (C3, low-medium).** No mature Python minisketch binding; may need a
  pure-Python IBLT or a C shim. Mitigation: rateless IBLT is pure-Python
  friendly and the critique already suggests it for the unknown-capacity case.
- **R5 (publication, low).** `docs/v2_formal_foundations.md` is unpublished
  design material; pushing the public repo publishes it with a timestamp.
  Decide deliberately whether to push before the v2 preprint exists.

---

## 5. Already done in the camera-ready (do not redo)

- Speculative-signing fast path: conditional BLS commitments in Phase 1, 2N/3
  trigger, certificate byte-identical to the Phase 2 object, covered by
  Theorem 1. Implemented in `blockchain.py` and described in the paper. This
  is the Workstream-B6 corollary and the single highest-leverage fix from the
  critique.
- Honest-sufficient trigger rule (count signatures, not variance) — closes the
  one-Byzantine griefing vector by construction.
- Zyzzyva + SBFT cited (speculative lineage).
- A future-work sentence in the paper points at the three v2 directions
  (zero-mean coordinates, exact set reconciliation, conservation sharding)
  with no claims and no new citations.

So v2 starts from: primitive swap (A), the theory (B), the reconciliation and
sharding build-out (C/D), and the evaluation the reviewers will demand (E8-E10
+ testbed).
