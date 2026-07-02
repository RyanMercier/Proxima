# Formal Foundations for Proxima v2: Distance-Preserving Multiset Hashes

> Status: design notes for the extended version / follow-on paper. None of this
> is part of the BLOCKCHAIN'26 camera-ready, whose construction and numbers are
> frozen to what reviewers accepted (see paper/CUTS.md and paper/BLOCKERS.md).
> The camera-ready already implements two items from this program: the
> speculative-signing fast path (Section 10 corollary here) and the
> honest-sufficient trigger rule. Everything else below changes the reviewed
> construction (zero-mean coordinates, n = 512, mod-q domains, minisketch
> replacing bloom) and therefore belongs to v2.

Draft section replacing III-B through III-D and VIII-C/D, plus new theory material.
Notation: n is digest dimension, q the modulus, d = |A Δ B| the multiset symmetric
difference, T a transaction multiset, H a random oracle (SHA-512).

---

## 1. Construction

For round (height) h, define the per-transaction vector

    v_h(tx) = sgn(H("PXM/round" || h || tx)) in {-1, +1}^n

where sgn maps each of the first n output bits b to 2b - 1. For n = 512 this consumes
exactly one SHA-512 output. The round digest of a multiset T is

    D_h(T) = sum_{tx in T} v_h(tx)

computed coordinate-wise over the integers, stored as n signed 16-bit words (q = 2^16,
centered representatives). Because each transaction moves each coordinate by exactly 1,
|coordinate| <= |T|, so for any block with |T| <= 32767 the mod-q digest equals the
integer digest with no wraparound. Within a block, q = 2^16 is lossless, not approximate.

Two digest domains (Section 6 explains why both are needed):

- Round digests: salted by height h. Used for clustering, readiness sensing, sync.
- Ledger accumulators: fixed salt "PXM/ledger", q = 2^32 (n signed 32-bit words).
  Used for cumulative chain digests, conservation invariants, fork localization.

Homomorphism is immediate in both domains: D(A ⊎ B) = D(A) + D(B), and D(A) - D(B)
is well defined. Group summaries are transmitted as exact integer pairs
(sum of digests, total weight); means are formed by the verifier, so no floats ever
enter the protocol. Cross-platform float nondeterminism, a real consensus bug class,
is eliminated by construction.

Reference implementation (numpy):

```python
import hashlib, numpy as np

N = 512

def vec(tx: bytes, height: int, domain=b"PXM/round") -> np.ndarray:
    hbytes = hashlib.sha512(domain + height.to_bytes(8, "big") + tx).digest()
    bits = np.unpackbits(np.frombuffer(hbytes, dtype=np.uint8))[:N]
    return bits.astype(np.int16) * 2 - 1          # {-1, +1}^512

def digest(txs, height) -> np.ndarray:
    out = np.zeros(N, dtype=np.int32)
    for tx in txs:
        out += vec(tx, height)
    return out                                     # cast to int16 for the wire

def est_symdiff(d1, d2) -> float:
    delta = (d1.astype(np.int64) - d2.astype(np.int64))
    return float(np.dot(delta, delta)) / N         # unbiased estimate of |A Δ B|
```

Digest update is 512 int16 adds per transaction (two AVX-512 instructions); distance
is one O(n) dot product. Both are nanosecond-scale.

---

## 2. Distance Estimation

Throughout this section A and B are fixed before the oracle is queried (the honest,
non-adaptive regime). The adversarial regime is treated in Sections 5 and 6.

**Lemma 1 (Unbiased symmetric-difference estimator).**
Let d = |A Δ B|. Then ΔD = D(A) - D(B) is distributed as a sum of d i.i.d. uniform
{-1,+1}^n vectors (signs are absorbed by symmetry of the Rademacher distribution).
Per coordinate, S = sum of d Rademacher variables satisfies E[S] = 0, E[S^2] = d,
E[S^4] = 3d^2 - 2d. Hence

    E[ ||ΔD||^2 ] = n d,      d_hat := ||ΔD||^2 / n   is unbiased,
    Var(d_hat) = 2 d (d - 1) / n,
    relative standard error  = sqrt(2(d-1)/(d n)) <= sqrt(2/n).

At n = 512 the relative error is at most 6.25 percent; at n = 64, 17.7 percent; the
paper's current n = 8 gives 50 percent, which is why the dimension must increase.
Contrast with the v1 nonnegative construction, where E||ΔD||^2 = n(d/12) + (k_A - k_B)^2 n/4
depends on the direction of the difference: substitutions partially cancel and can sit
below honest stragglers. The zero-mean construction depends only on d. Substituting j
transactions is exactly a symmetric difference of 2j and is measured as such.

**Exact small-d distributions** (useful for the threshold table and as implementation
test vectors):

- d = 1: S^2 = 1 deterministically, so ||ΔD||^2 = n exactly, with zero variance.
  A singleton difference is certified by the exact value n.
- d = 2: ||ΔD||^2 = 4 Bin(n, 1/2). Mean 2n, support [0, 4n].
- d = 3: ||ΔD||^2 = n + 8 Bin(n, 1/4). Mean 3n.

**Lemma 2 (Concentration).** Coordinates of ΔD are independent (independent oracle
bits), and S^2 in [0, d^2]. Hoeffding gives, for any t > 0,

    Pr[ | ||ΔD||^2 - n d | >= n t ] <= 2 exp( -2 n t^2 / d^4 ).

For small d the exact binomial forms above are sharper and should be quoted.

**Threshold rule.** With honest partial observation bounded by k_max = 2 missing
transactions (relative to the reference, so d <= 2), set

    tau^2 = 3 n.

Then:

- Honest exclusion (d = 2): Pr[4 Bin(n,1/2) > 3n] = Pr[Bin(512,1/2) > 384]
  <= exp(-64) ~ 1.6e-28. For d = 1 exclusion is impossible (||ΔD||^2 = n < 3n).
- Byzantine inclusion at d = 4 (two substituted transactions): mean 4n,
  per-coordinate variance 24, a 4.6 sigma deviation, probability ~ 2e-6.
- d = 3 is genuinely ambiguous and is resolved by sync plus Phase 2, never by the
  estimator. This is inherent to any estimator with finite variance and costs at most
  one extra round; it never touches safety.

Threshold calibration is now closed form. The Monte Carlo calibration subsection
(old III-C) can be deleted entirely and replaced by the line above; the simulation
becomes a validation check rather than the definition of the threshold.

---

## 3. Liveness Bound (replaces III-D)

**Theorem (Liveness).** Suppose f < N/3 validators are Byzantine, each honest
validator independently has an incomplete view with probability p_miss and misses at
most k_max = 2 transactions when incomplete, and let delta_sync be the probability
that a flagged straggler fails to be synchronized before the Phase 2 timeout. A
validator fails to contribute a Phase 2 signature only if it is Byzantine, falsely
excluded, or unsynchronized. The per-honest-validator failure probability is

    p <= p_miss * exp(-n/8) + delta_sync .

At n = 512 the first term is p_miss * e^{-64}, i.e. negligible; liveness is governed
entirely by the sync path, as it should be. By Hoeffding over the N honest indicators,

    Pr[ liveness failure ] = Pr[ X >= N/3 ] <= exp( -2 N (1/3 - p)^2 ),

which for p <= 0.01 is below exp(-0.21 N): under 1e-9 at N = 100 and under 1e-91 at
N = 1000. Note the structural improvement over v1: the false-exclusion term moved from
5e-3 (empirical, Monte Carlo) to e^{-64} (analytic), and excluded validators are no
longer lost for the round, since after a sync push they rejoin Phase 2; exclusion is a
scheduling event, not a participation event.

---

## 4. Mod-q Variant, Accumulators, Saturation

**Round digests (q = 2^16).** Lossless for |T| <= 32767 as shown in Section 1; all of
Section 2 applies verbatim. Wire size n * 2 bytes = 1024 bytes.

**Ledger accumulators (q = 2^32).** Define P[b] = sum over blocks 1..b of the
fixed-domain block digests. Any range digest is P[b] - P[a] in O(1). Equality tests
mod q are meaningful at any chain length (binding is a mod-q property, Section 5).
Distance estimates on a difference P[b] - P[a] are exact whenever the divergent
transaction count is below q/2 = 2^31, i.e. always in practice; absolute coordinate
wraparound in P[b] itself is irrelevant because the protocol only ever subtracts
accumulators and tests equality. Wire size 2048 bytes.

**Saturation statement (for the formal definition).** The construction is a
(d_max, sqrt(2/n))-distance-preserving multiset hash with d_max = q/2 on differences:
below d_max, distance estimation holds with the Lemma 1/2 guarantees; above it,
estimates degrade but equality binding (Section 5) is unaffected. Distance has a
metering range; binding does not.

**Group summaries.** A group of m validators is summarized exactly by
(sum of D_i, sum of weights, sum of ||D_i - D_ref||^2), about 1.1 KB at n = 512. This
replaces the v1 76-byte summary; the cost of going from n = 8 to n = 512 is a 13x
larger summary that is still 100x smaller than forwarding votes. A bandwidth-sensitive
profile at n = 128 (272-byte summaries, 12.5 percent RSE, false-exclusion exp(-16))
remains available; profiles are tabulated in Section 11.

---

## 5. Equality Binding: Collision Resistance of the Accumulator

A collision is a pair of distinct bounded-multiplicity multisets A != B with
D(A) = D(B) mod q, equivalently a nonzero integer vector z with bounded entries
(multiplicities) and bounded support (transactions per block or per window) such that

    V z = 0 mod q,

where the columns of V are the oracle-derived {-1,+1}^n vectors of the ground set.
This is a Short Integer Solution instance with a +-1 matrix. Three observations:

1. **Existence vs. discovery.** Collisions exist by pigeonhole: over a ground set of
   2^40 candidate transactions, counting shows minimum-weight integer collisions of
   weight on the order of 40 to 200 transactions exist. Existence is unavoidable for
   any compressing homomorphic hash; security is the claim that finding one is
   infeasible.
2. **Attack surfaces.** Wagner's k-tree algorithm over Z_q^n needs 2^a lists to cancel
   n log2 q = 8192 constraint bits at per-list cost ~ 2^{8192/(a+1)}, but produces
   solutions of Hamming weight 2^a. The block-size cap |T| <= 2^15 forces a <= 15,
   giving per-list cost 2^512. Lattice attacks reduce to finding short vectors in the
   q-ary kernel lattice of V; with the weight and multiplicity caps the required norm
   is far below the Gaussian-heuristic shortest vector at these parameters. This is
   the same problem family analyzed for LtHash (Lewi, Kim, Maykov, Weis, "Securing
   Update Propagation with Homomorphic Hashing", 2019), whose parameter analysis at
   n = 1024, q = 2^16 should be cited and inherited rather than re-derived; our
   accumulator profile (n = 512, q = 2^32) has the same n log q budget.
3. **Honest claim for the paper.** State binding as: collision-finding reduces to a
   +-1-matrix SIS instance in the ROM; we give first-cut cost estimates for the two
   standard attacks and inherit the LtHash analysis; a dedicated cryptanalysis at
   these exact parameters is open. Do not claim a security level without it.

Per-height domain separation on round digests means any precomputed structure against
the round domain is useless across blocks; only the fixed-domain accumulator carries
long-lived binding claims, and that is exactly where SIS hardness is asserted.

---

## 6. Adversarial Deflation of Distance (and why two domains)

Distance is an estimator, not a commitment, and an adversary who can grind
transactions can make a large true difference look smaller. The cheapest strategy is
near-antipodal pairs: for v = v(tx), find tx' with v(tx') close to -v. If the two
vectors agree on a coordinates, the pair contributes ||v + v'||^2 = 4a to the squared
distance instead of the honest expectation 2n.

Cost: agreement count is Bin(n, 1/2), so Pr[a <= n/4] = exp(-n/8) = 2^{-92} at
n = 512, and a birthday search over M ground candidates yields M^2 2^{-93} usable
pairs. Concretely:

- Factor-2 deflation (pair appears as d_hat ~ 1 instead of 2): one pair per ~2^47
  hash-and-sign candidates; a 2^50 table yields about 100 such pairs.
- Factor-4 deflation (a <= n/8): 2^{-208} per pair, ~2^104 candidates for one pair.

So deflation is bounded: constant factors at exponential-grinding cost, never to zero,
and each candidate costs a signature from an adversary-controlled account plus gas.
Per-height salting forces the entire table to be rebuilt every block, converting a
one-time precomputation into an online 2^47-per-block cost, which closes the attack
for any realistic adversary.

**Design consequence (two domains).** Salting kills grinding but also kills
cross-height algebra. Hence the split in Section 1: round digests are salted (they
feed thresholds, where deflation matters and no cross-height algebra is needed);
ledger accumulators are unsalted (they feed equality and conservation checks, which
rest on SIS binding, not on distance, so deflation is irrelevant there). Each domain
gets exactly the property it needs.

**Theorem (Proximity non-bindability, impossibility).** For any multiset hash
satisfying the distance-preservation property (small symmetric difference implies
small digest distance), there is no security notion under which digest proximity
certifies set proximity against adversarially chosen sets: the adversary can always
exhibit sets at distance below any threshold tau that differ in ways the protocol
cares about, because the property itself guarantees it for small d, and grinding
extends it to larger d at quantifiable cost (above). Consequently, in any protocol
composed with such a hash, proximity may gate scheduling, synchronization, and
optimism, but never safety. This is the multiset-hash, public-coin analogue of the
Hardt-Woodruff theorem that linear sketches are not robust to adaptive adversaries
(STOC 2013); the resolution differs: rather than seeking a robust sketch, the
architecture routes all safety through signatures on collision-resistant hashes and
lets the estimator fail soft.

---

## 7. Robust Reference Selection

The v1 reference digest is the aggregator's own state, a single point of manipulation.
With digests living in R^n and group sums exact, the reference can instead be a
Byzantine-robust location estimate of the validator cloud: coordinate-wise median,
trimmed mean, or geometric median, all with breakdown point >= 1/3 (geometric median:
1/2), computed by the aggregator but verifiable by any validator from the signed
Phase 1 digests. This imports the robust aggregation toolkit from
Byzantine-tolerant federated learning (Krum, Blanchard et al. NeurIPS 2017;
coordinate-median and trimmed-mean analyses, Yin et al. ICML 2018) into consensus.
A Byzantine aggregator can still suppress messages (liveness, as before) but can no
longer bias which honest validators look like outliers. One paragraph in the protocol
section plus one simulation figure (reference manipulation with adversarial
aggregator, v1 vs robust reference) makes this concrete.

---

## 8. Cross-Shard Conservation

**Setup.** Every cross-shard transaction t from shard i to shard j contributes
+v(t) (ledger domain) to shard i's outbound accumulator O_{i->j} when the debit is
applied, and +v(t) to shard j's inbound accumulator I_{i->j} when the credit is
applied. Each shard's 2/3 BLS quorum signs its (aggregated) outbound and inbound
digests each block: O_i = sum_j O_{i->j}, I_j = sum_i I_{i->j}.

**Invariant.** Within a settlement window W,  sum_i O_i = sum_j I_j.

**Theorem (Conservation soundness).** If the invariant holds over signed digests and
the accumulator is collision-binding (Section 5), then the multiset of applied debits
equals the multiset of applied credits, except with negligible probability. Any
minted, lost, or replayed cross-shard effect requires either forging a 2/3 quorum
signature or finding a digest collision. Value conservation follows from multiset
conservation plus intra-shard execution validity (each shard's quorum attests it
applied the amounts contained in the transactions); do not value-weight the vectors
themselves, since amount-weighted coefficients would weaken the SIS instance, and the
amount is already bound inside the hashed transaction.

**Semantics caveat (state plainly in the paper).** This is detection and settlement
verification, not locking: digests order nothing. The protocol either runs optimistic
execution with a W-block revert window, or credits at the destination only after digest
match (receipt-equivalent latency, but constant-size verification metadata instead of
per-receipt data). The contribution is verification cost, O(S) signed kilobyte-scale
objects per block instead of O(S^2) pair relations or O(volume) receipts, plus fraud
localization below. The honest baseline comparison is per-block verification metadata
and failure-path cost against batched 2PC and batched receipts, not raw message counts
against unbatched 2PC.

**Proposition (Localization by group testing).** Shards publish only aggregate
O_i, I_j per block; per-edge accumulators are kept locally. If the global invariant
fails, the beacon bisects: it requests digests aggregated over half the edge set,
recursing on imbalanced halves. f faulty edges are localized in O(f log S) digest
requests, each one signed kilobyte-scale object, by linearity of the invariant.
Bisection over a linear measurement is exact, not statistical. After localization, the
implicated pair runs set reconciliation (Section 9) to identify the exact divergent
transactions, at cost proportional to the true divergence.

**Global ledger digest.** The beacon maintains G += sum_i Delta_i per block, a single
2 KB object summarizing the entire sharded system's applied transaction history,
updated by addition, prunable by subtraction with verifiable prune accounting:
a node claiming to have pruned set P proves D(before) - D(after) = D(P), and binding
makes the claim sound. Merkle roots support none of subtraction, addition across
shards, or O(1) range extraction.

---

## 9. Reconciliation: the Two-Syndrome Architecture

The digest is the syndrome of a random linear code over Z_q: hard to decode (that
hardness is exactly the SIS binding), but the weight of the error, i.e. the symmetric
difference size, is estimable from the syndrome norm (Lemma 1). BCH syndromes
(minisketch, as deployed in Bitcoin Erlay) are the complement: efficiently decodable,
recovering the exact differing elements at communication proportional to the
difference, but not collision-resistant and not distance-revealing beyond capacity.

Architecture: estimate with the hard code, decode with the easy code.

1. Compare DPMH digests; d_hat sizes the divergence (O(1) bytes, binding).
2. Run set reconciliation sized by d_hat: minisketch at capacity ~ d_hat, or rateless
   IBLT (Yang et al., SIGCOMM 2024) when d_hat is unreliable, since rateless schemes
   need no a priori capacity.
3. Push exactly the recovered transactions.

This replaces the bloom filter throughout the protocol (Phase 1 sync, cross-shard
resolution, fork healing): exact recovery, no false positives, communication
proportional to the true difference. The bloom false-positive analysis (old VIII-E)
is deleted, and the v1 liveness chain "bloom FP leads to exclusion" disappears.
Minisketch syndromes are themselves linear over GF(2^b), so cumulative BCH syndromes
subtract exactly like the accumulators do; both syndromes can be maintained per block
and range-queried in O(1).

**Fork localization.** With cumulative accumulators P[.], two nodes find their fork
point by binary search on range digests: O(log H) rounds, one accumulator each, and
each probe also yields d_hat for the half-range, so the search reports where the fork
is and how big it is, then heals it with one rateless-IBLT exchange over the
divergent suffix. Compare Bitcoin block locators, which return only identical or not
per probe.

**Partition healing.** During a partition the Phase 1 cloud is bimodal. Cluster means
are exact by linearity, so the aggregator (or any observer of the signed digests)
computes the two centroids, estimates the inter-camp divergence
d_hat = ||mu_1 - mu_2||^2 / n, and triggers one reconciliation exchange between camp
representatives, then pushes the union. Hash-based protocols observe only quorum
failure; digest-based protocols observe partition geometry and heal it at cost
proportional to the actual divergence, not to state size. This subsumes the v1
fast-path variance check as the unimodal special case.

---

## 10. Composition: Estimation-Augmented BFT

**Definition.** Let Pi be a BFT protocol whose commit predicate is "at least 2N/3
valid signatures on H(B)" for a collision-resistant H. An estimation augmentation
Pi+E adds messages and rules that may depend on DPMH digests, but whose outputs gate
only scheduling (when to propose, whom to poll first, fast-path triggering),
synchronization (which transactions to push to whom), and reporting (reputation).
The commit predicate is unchanged.

**Theorem (Safety preservation).** Pi+E satisfies the same safety property as Pi
against the same adversary class. Proof sketch: estimation messages are auxiliary;
given any adversary against Pi+E, construct the adversary against Pi that simulates
the estimation layer internally (it can, since digests are computable from public
data and require no secrets). Commit messages are identical in both executions, and
Pi's safety argument (quorum intersection on signed H(B), Theorem 1 of v1) applies
verbatim. No property of D(.) is invoked. QED.

**Corollary (Speculative fast path).** In Phase 1 each validator sends
(D_i, sigma_i = BLS_i(h, H(B_prop))), the signature conditional on its local state
matching the proposal. If the aggregator collects 2N/3 matching signatures, it emits
a standard finality certificate in one round trip. The certificate is the same object
Phase 2 would produce; the fast path is a scheduling event, covered by the theorem.
This simultaneously repairs the v1 fast-path soundness gap (a Phase 1 certificate
backed only by digests bound nothing) and the griefing vector (one Byzantine validator
inside tau could force nonzero cluster variance forever; now the trigger is
"2N/3 matching signatures collected", an honest-sufficient condition the adversary
cannot block below the corruption bound).
[Note: this corollary is implemented in the camera-ready and in blockchain.py.]

**Design rule worth stating.** Optimistic triggers must fire on honest-sufficient
conditions, never on global statistics an adversary can perturb. Variance is a
statistic; a signature count is honest-sufficient.

**Liveness and expected latency.** Liveness of Pi+E follows from Pi's liveness plus
the Section 3 bound on estimator-gated exclusion. Expected rounds improve by the
fast-path probability; with sensing (the proposer watches gossiped digest variance
and proposes when the cloud has converged), the fast-path probability is no longer
weather, (1 - p_miss)^N, but a controlled quantity: propose when observed variance
implies completion. One simulation figure: fast-path rate and latency, fixed-interval
vs sensed proposal, as a function of gossip delay.

---

## 11. Parameters

| Profile | n | q | wire size | RSE of d_hat | false excl. (d<=2, tau^2=3n) | role |
|---|---|---|---|---|---|---|
| Round, default | 512 | 2^16 | 1024 B | <= 6.3% | exp(-64) | clustering, sync, fast path |
| Round, light | 128 | 2^16 | 256 B | <= 12.5% | exp(-16) | bandwidth-constrained leaves |
| Accumulator | 512 | 2^32 | 2048 B | exact below 2^31 divergence | n/a | conservation, history, forks |

Multiplicity bound mu = 1 within a block (a block is a set); enforce at validation.
k_max = 2, tau^2 = 3n, both now analytic. All Monte Carlo in old III-C/III-D is
demoted to validation of these closed forms. Test vectors: d = 1 must give
||ΔD||^2 = n exactly; d = 2 must match 4 Bin(n, 1/2) moments.

---

## 12. Related Work To Add (and the honest novelty claim)

Must cite, will be raised by reviewers otherwise:

- Clarke, Devadas, van Dijk, Gassend, Suh (ASIACRYPT 2003): incremental multiset
  hashes including additive mod-2^k constructions. The construction family is theirs.
- Bellare, Micciancio (EUROCRYPT 1997): AdHash; Wagner (CRYPTO 2002): the k-tree
  attack that forces AdHash moduli enormous and which the vectorized form sidesteps.
- Lewi, Kim, Maykov, Weis (2019): LtHash, the deployed n = 1024, 16-bit homomorphic
  hash with lattice-based analysis. Closest construction to ours.
- Alon, Matias, Szegedy (STOC 1996): the Tug-of-War sketch; the distance property of
  Section 2 is the AMS L2 estimator specialized to multiset indicator vectors.
- Hardt, Woodruff (STOC 2013): linear sketches are not adversarially robust; our
  impossibility theorem is its public-coin multiset-hash analogue, and our composition
  theorem is the architectural answer.
- Minsky, Trachtenberg, Zippel (2003), minisketch/Erlay (Naumenko et al. 2019),
  Graphene (Ozisik et al. 2019), rateless IBLT (Yang et al. SIGCOMM 2024): the
  decoding side of the two-syndrome architecture, and Erlay as the existence proof
  that set reconciliation belongs in blockchain protocols.
- Blanchard et al. (2017), Yin et al. (2018): Byzantine-robust aggregation, imported
  for reference selection.
- Zyzzyva (Kotla et al. 2007), SBFT (Gueta et al. 2019), HotStuff-2 (Malkhi, Nayak
  2023): speculative execution and the modern two-phase baseline.

The defensible novelty claim, stated precisely: the construction family is known
(Clarke et al., LtHash) and the estimation property is known in streaming (AMS). What
is new is (a) the formal identification that one object can simultaneously be a
collision-binding homomorphic accumulator and an unbiased symmetric-difference
estimator, with a clean separation of which protocol roles each property may serve
(the impossibility theorem makes the separation forced, not stylistic); (b) the
quantitative adversarial-deflation analysis in the public-hash model with the
two-domain salting design it implies; and (c) the protocol layer: the
estimation-augmented BFT transformer with safety preservation, conservation sharding
with O(f log S) fraud localization, and digest-space fork and partition healing.
Write exactly this; do not claim a new hash.

---

## 13. Experiment Checklist (new and changed figures)

1. Fig 1 rerun with zero-mean coordinates: box plots collapse onto nd with the
   Lemma 1 variance; overlay the analytic curve. Add the substitution case
   (v1's blind spot) showing it now reads as 2j.
2. Threshold figure: exact d = 1..4 distributions vs tau^2 = 3n, log-scale tails,
   analytic bounds overlaid.
3. Robust reference: adversarial aggregator biasing the reference, v1 vs
   geometric-median reference; honest-exclusion rate vs fraction Byzantine.
4. Fast path: speculative-signing trigger vs v1 variance trigger under one in-cluster
   Byzantine (shows the griefing fix); fast-path rate vs p_miss, sensed vs
   fixed-interval proposal.
5. Conservation sharding: S shards, injected divergences, verification bytes per
   block vs batched-2PC and batched-receipt baselines; localization probes vs f.
6. Reconciliation: bloom (v1) vs minisketch vs rateless IBLT, bytes and rounds vs
   true difference size.
7. Partition healing: 60/40 split with d disputed transactions; healing cost vs d
   and vs PBFT-style view-change message storm.
8. Kauri and HotStuff-2 baselines in Tables III and IV (the comparison reviewers
   will demand; simulated Kauri with the same message counter is acceptable for a
   first submission).

---

## 14. Additional notes (session review, July 2026)

Observations added while integrating these notes with the camera-ready:

- **Camera-ready status.** Items already shipped in the accepted version: the
  Section 10 corollary (speculative fast path with conditional BLS commitments,
  2N/3 trigger, certificate identical to the Phase 2 object) and the
  honest-sufficient trigger rule, both in the paper and in blockchain.py, with
  Zyzzyva/SBFT cited. Everything else here changes reviewed numbers and waits
  for v2.
- **Pending-set digest for free.** In the conservation setup, the per-corridor
  difference P_ij = O_{i->j} - I_{i->j} is not merely an alarm signal: it IS the
  ledger-domain digest of the in-flight transaction set on that corridor. Its
  norm therefore meters in-flight volume per corridor continuously (not just on
  violation), and an aging rule (alarm only when a pending digest persists
  unchanged for more than W blocks) separates ordinary settlement lag from
  faults. Window alignment comes free: the invariant should be checked as
  "P_ij decays to zero within W", not as per-block equality, since credits
  necessarily lag debits by network latency.
- **Estimator for the corridor, decoder for the corridor.** Because minisketch
  syndromes are also linear, each corridor can maintain the BCH syndrome of the
  same in-flight set alongside P_ij; on alarm, the destination decodes exactly
  the dangling transactions without a further exchange. The two-syndrome
  architecture applies corridor-locally, not only pairwise after bisection.
- **Literature check additions.** Besides Section 12: Invertible Bloom Lookup
  Tables (Goodrich-Mitzenmacher 2011) carry additive count/keySum fields that
  are themselves linear multiset sketches with decode, so position against them
  explicitly; also sweep CRDT state-digest and anti-entropy literature
  (Dynamo-style Merkle repair replacements) and "homomorphic fingerprinting"
  for the distance claim. The specific claim to verify as unclaimed: an L2
  distance guarantee stated for a collision-binding additive multiset hash.
- **Publication caution.** This file is design material for an unpublished
  follow-on paper. The repository is public once pushed; pushing this file
  publishes the ideas (with a timestamp, which cuts both ways). Decide
  deliberately whether to push it before the v2 paper or arXiv preprint exists.
- **Camera-ready pointer.** The paper's future-work block now carries a one-line
  promissory pointer (zero-mean coordinates, exact set reconciliation,
  conservation-style cross-shard verification) with no claims and no new
  citations; the full development is this document.
