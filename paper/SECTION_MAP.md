# SECTION_MAP: proxima_ieee (submitted IEEEtran source)

Source: `paper/proxima_ieee_submitted.tex` (byte-identical to the arXiv package
`proxima_arxiv_submission.tar.gz` and to `Downloads/proxima_ieee (1).tex`, the
version reviewed for BLOCKCHAIN'26). Line/word counts are source lines and
words between sectioning commands (word counts include LaTeX markup, so treat
them as relative weights, not prose length).

## Front matter

| Unit | Words | Figures / tables |
|---|---|---|
| Abstract | ~218 | - |
| Keywords (IEEEkeywords) | 12 | - |

## Sections (IEEE numbering)

| IEEE Sect. | Title | Lines | Words | Figures / tables |
|---|---|---|---|---|
| I | Introduction | 10 | 338 | - |
| II | System Model | 16 | 267 | - |
| III | The Primitive: Distance-Preserving Digests | 40 | 560 | Fig. 1 (fig1_distance_comparison.png) |
| III.A | Construction | 7 | 71 | - |
| III.B | Properties | 14 | 218 | Fig. 1 |
| III.C | Threshold Calibration | 3 | 40 | - |
| III.D | Probabilistic Liveness Bound | 15 | 231 | - |
| IV | Application 1: BFT Consensus Protocol | 46 | 595 | Fig. 2 (fig9_protocol_flow.png), Table I (tab:committees) |
| IV.A | Flat Protocol | 14 | 204 | Fig. 2 |
| IV.B | Tree Protocol | 7 | 153 | - |
| IV.C | Why Phase 2 Is Necessary | 3 | 45 | - |
| IV.D | Comparison with Ethereum Committees | 24 | 193 | Table I |
| V | Application 2: Cross-Shard Consistency | 43 | 372 | Fig. 3 (fig8_cross_shard.png), Table II (tab:cross-shard) |
| V.A | The Problem | 3 | 51 | - |
| V.B | Digest-Based Verification | 5 | 61 | - |
| V.C | Analytical Comparison | 28 | 191 | Table II, Fig. 3 |
| V.D | Multi-Shard Scaling | 3 | 33 | - |
| V.E | Why This Requires Digests | 4 | 36 | - |
| VI | Application 3: Agreement Quality Measurement | 15 | 146 | Fig. 4 (fig4_fast_path.png) |
| VII | Evaluation | 91 | 852 | Fig. 5 (fig2_scale_comparison.png), Fig. 6 (fig5_tree_breakdown.png), Fig. 7 (fig6_bls_bottleneck.png), Fig. 8 (fig7_latency_model.png), Fig. 9 (fig3_byzantine_sweep.png), Table III (tab:messages), Table IV (tab:latency) |
| VII.A | Message Complexity | 35 | 192 | Table III, Fig. 5, Fig. 6 |
| VII.B | BLS Aggregation Bottleneck | 37 | 427 | Table IV, Fig. 7, Fig. 8 |
| VII.C | Caveats | 7 | 148 | - |
| VII.D | Byzantine Tolerance | 11 | 85 | Fig. 9 |
| VIII | Security Analysis | 33 | 724 | Theorem 1, Corollaries 1-2 |
| VIII.A | Safety | 18 | 173 | Theorem 1, Corollaries 1-2 |
| VIII.B | Liveness | 3 | 61 | - |
| VIII.C | Collision Resistance | 3 | 143 | - |
| VIII.D | Adversarial Transaction Construction | 3 | 192 | - |
| VIII.E | Bloom Filter False Positives | 5 | 155 | - |
| IX | Related Work | 12 | 269 | - |
| X | Implementation | 4 | 89 | - |
| XI | Limitations and Future Work | 10 | 117 | - |
| XII | Conclusion | 6 | ~110 | - |
| - | References (21 bibitems) | 26 | ~340 | - |

Figure-number-to-file map (IEEE PDF numbering, by order of figure envs in
source): Fig. 1 = fig1_distance_comparison, Fig. 2 = fig9_protocol_flow,
Fig. 3 = fig8_cross_shard, Fig. 4 = fig4_fast_path, Fig. 5 =
fig2_scale_comparison, Fig. 6 = fig5_tree_breakdown, Fig. 7 =
fig6_bls_bottleneck, Fig. 8 = fig7_latency_model, Fig. 9 =
fig3_byzantine_sweep. All nine PNGs in `paper/figures/` are the reviewed
(arXiv-package) versions.

`fig9_protocol_flow.png` (paper Fig. 2) has no generation script in the
repository; `visualize.py` produces the other eight. See BLOCKERS.md.

## Springer section mapping (after B2 relocation of Related Work)

| IEEE | Springer (llncs) |
|---|---|
| I Introduction | 1 Introduction |
| IX Related Work | 2 Related Work |
| II System Model | 3 System Model |
| III The Primitive | 4 The Primitive (4.1-4.4) |
| IV Application 1 | 5 Application 1: BFT Consensus (5.1-5.4) |
| V Application 2 | 6 Application 2: Cross-Shard Consistency (6.1-6.5) |
| VI Application 3 | 7 Application 3: Agreement Quality Measurement |
| VII Evaluation | 8 Evaluation (8.1-8.4) |
| VIII Security Analysis | 9 Security Analysis (9.1-9.5) |
| X Implementation | 10 Implementation |
| XI Limitations and Future Work | 11 Limitations and Future Work |
| XII Conclusion | 12 Conclusion |

## Starting page count (mechanical llncs port, before edits)

17 pages (`proxima_llncs.pdf`, llncs class, splncs04 bibliography, figures at
their IEEE relative widths). Confirmed with pdfinfo. Target is 10 pages
including references, so roughly 7 pages must come out of text condensation
and figure/table sizing while Phases B and C add material.
