# Bedroom_1 output-based probing evaluation

This report evaluates only the seven existing, unseeded bedroom_1 outputs. There is one output per condition, no stochastic replicates, and no new inference. All values are **DESCRIPTIVE**. Generated scenes and benchmark JSONs were preserved.

## Interpretation and provenance

L1 asks whether output changed. L2 asks how within/between output separation compares under an applicable design. Semantic evaluations ask whether supported properties changed in the expected direction. Specificity asks which structural channels changed. Validity asks whether geometry and stated relations satisfy explicit checks. Large L1 change is neither automatically good nor evidence of semantic correctness.

Only the canonical `bedroom_1/<condition>/run_qwen35_27b.json` files are used. Baseline replay/determinism-attempt directories are excluded. With one benchmark, each aggregate equals the bedroom_1 value; no cross-bedroom average is implied. API nondeterminism and one unseeded scene per condition prevent a causal separation of adjective effects from generation randomness.

| condition | status | path |
| --- | --- | --- |
| baseline | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/baseline/run_qwen35_27b.json |
| well_organized | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/well_organized/run_qwen35_27b.json |
| spacious | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/spacious/run_qwen35_27b.json |
| bright | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/bright/run_qwen35_27b.json |
| quiet | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/quiet/run_qwen35_27b.json |
| cluttered | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/cluttered/run_qwen35_27b.json |
| crowded | available | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/crowded/run_qwen35_27b.json |

## A. High-level comparison

| condition | occupancy_hamming | centroid_shift | direction_accuracy | semantic_alignment_cosine | object_preservation | collateral_change_rate | constraint_violation_delta | spatial_relation_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| well_organized | 0.84 | 3.541 | 0.6667 | 0.2823 | 1 | 0.0625 | 0.03125 | 1 |
| spacious | 0.72 | 3.533 | 0.3333 | 0.2555 | 1 | 0.07692 | 0.01562 | 1 |
| bright | 0.88 | 3.641 | N/A | N/A | 1 | N/A | 0.04688 | 1 |
| quiet | 0.72 | 3.545 | N/A | N/A | 1 | N/A | 0.03125 | 1 |
| cluttered | 0.2 | 0.3539 | 0.25 | -0.3571 | 1 | 0.07143 | 0.01562 | 1 |
| crowded | 0.68 | 3.382 | 0.3333 | -0.03754 | 1 | 0 | 0.03125 | 0.6667 |

Brightness and quietness are not directly observable: the schema has no illumination, photometry, sound level or acoustic model. Free-text references to light/comfort are not quantitative measurements and are not scored as semantic success or failure.

## B. L1 behavioral sensitivity

The existing `adv_l1_behavioral_qwen35_27b.compare` and `adjective_common_qwen35_27b.pair_metrics` run unchanged. First divergence is zero-based and compares trimmed response text; structural divergence means differing call counts/stage sequences, not object motion. Jaccard normalizes trailing numeric label suffixes; count difference uses full object-list lengths; centroids match exact label plus occurrence. Occupancy is the repository 5×5 category grid. Legality means no footprint overlap and excludes bounds.

| condition | first_divergent_call | structural_divergence | placement_changed | object_set_diff | object_count_diff | occupancy_hamming | centroid_shift | legality_flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| well_organized | 0 | False | True | 0 | 0 | 0.84 | 3.541 | 0 |
| spacious | 0 | False | True | 0 | 0 | 0.72 | 3.533 | 0 |
| bright | 0 | False | True | 0 | 0 | 0.88 | 3.641 | 0 |
| quiet | 0 | False | True | 0 | 0 | 0.72 | 3.545 | 1 |
| cluttered | 0 | False | True | 0 | 0 | 0.2 | 0.3539 | 0 |
| crowded | 0 | False | True | 0 | 0 | 0.68 | 3.382 | 0 |

## C. L2 applicability

**NOT APPLICABLE as variance, cross-benchmark separation or inference.** Exact n is one benchmark, one baseline and one adjective output for each contrast. No within-condition pair exists, so `T = mean_between - mean_within` is undefined. The existing variance implementation requires at least two matched seeds and cannot run legitimately on this design. No seeds, pseudo-replicates, permutations, p-values or confidence intervals were introduced. Its distance primitive is reused to report the sole matched distance below; this duplicates L1 and is not independent evidence.

| condition | metric | n_benchmarks | within_baseline | within_adjective | between | test_statistic | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| well_organized | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| well_organized | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| well_organized | occupancy_hamming | 1 | N/A | N/A | 0.84 | N/A | N/A |
| well_organized | centroid_shift | 1 | N/A | N/A | 3.541 | N/A | N/A |
| well_organized | legality_flip | 1 | N/A | N/A | 0 | N/A | N/A |
| spacious | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| spacious | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| spacious | occupancy_hamming | 1 | N/A | N/A | 0.72 | N/A | N/A |
| spacious | centroid_shift | 1 | N/A | N/A | 3.533 | N/A | N/A |
| spacious | legality_flip | 1 | N/A | N/A | 0 | N/A | N/A |
| bright | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| bright | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| bright | occupancy_hamming | 1 | N/A | N/A | 0.88 | N/A | N/A |
| bright | centroid_shift | 1 | N/A | N/A | 3.641 | N/A | N/A |
| bright | legality_flip | 1 | N/A | N/A | 0 | N/A | N/A |
| quiet | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| quiet | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| quiet | occupancy_hamming | 1 | N/A | N/A | 0.72 | N/A | N/A |
| quiet | centroid_shift | 1 | N/A | N/A | 3.545 | N/A | N/A |
| quiet | legality_flip | 1 | N/A | N/A | 1 | N/A | N/A |
| cluttered | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| cluttered | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| cluttered | occupancy_hamming | 1 | N/A | N/A | 0.2 | N/A | N/A |
| cluttered | centroid_shift | 1 | N/A | N/A | 0.3539 | N/A | N/A |
| cluttered | legality_flip | 1 | N/A | N/A | 0 | N/A | N/A |
| crowded | object_set_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| crowded | object_count_diff | 1 | N/A | N/A | 0 | N/A | N/A |
| crowded | occupancy_hamming | 1 | N/A | N/A | 0.68 | N/A | N/A |
| crowded | centroid_shift | 1 | N/A | N/A | 3.382 | N/A | N/A |
| crowded | legality_flip | 1 | N/A | N/A | 0 | N/A | N/A |

## D. Semantic evaluations 1–4

These are operational geometric proxies, not a validated semantic ground truth. Bboxes are `[x,y,w,h]`; room extent is `[-width/2,width/2] × [0,height]`. All footprint calculations use the same unrotated axis-aligned convention as repository geometry; rotations are separately assessed for cardinal alignment. No height, z-axis, 3D collision model or object-local front axis is invented.

- Free space: one minus exact union area of footprints clipped to the room, divided by room area. Overlap can increase apparent free space, so validity must be read alongside this proxy.
- Clearance: mean nearest-neighbor rectangle-edge gap divided by room diagonal; overlap/contact gives zero gap.
- Density: number of generated objects divided by fixed room area. Preserving inventory makes this feature invariant; a zero change fails strict directional improvement but is not an inventory failure.
- Regularity: `1/(1+CV)` of nearest-centroid distances.
- Alignment: arithmetic mean of pairwise row/column alignment rate (within 5% of room diagonal on either coordinate) and cardinal rotation alignment `1-distance_to_nearest_90°/45°`.
- Spacing consistency: `1/(1+CV)` of nearest-edge gaps. Undefined when all gaps are zero; zero contact is not automatically orderly. These correlated proxies are not independent discoveries.

Direction accuracy is the fraction of evaluatable intended features with strictly expected signed change >1e-6; zero changes are reported explicitly. Spacious expects free-space/clearance increases and density decrease; crowded reverses them. Well-organized expects regularity/alignment/spacing-consistency increases. Cluttered expects their decreases and density increase. Per-bedroom accuracy is reported per adjective, not pooled across unrelated features.

Signed relative effects are `(adjective-baseline)/(|baseline|+1e-6)`. Near-zero baselines can inflate relative effects; raw baseline/condition values and deltas are always saved. Intended-effect mean flips feature signs to their expected direction and averages evaluatable features; it is descriptive, not a composite model-quality score.

Cosine alignment uses all six feature changes, including zero expected components. Features are dimensionless except density, scaled by room area / baseline count. Zero-length or partially undefined vectors yield N/A; no success is fabricated.

| condition | semantic_status | direction_accuracy | intended_signed_effect_mean | semantic_alignment_cosine | zero_change_features | skipped_features |
| --- | --- | --- | --- | --- | --- | --- |
| well_organized | DESCRIPTIVE GEOMETRIC PROXY | 0.6667 | 0.02082 | 0.2823 | [] | [] |
| spacious | DESCRIPTIVE GEOMETRIC PROXY | 0.3333 | -0.02076 | 0.2555 | ['density'] | [] |
| bright | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION | N/A | N/A | N/A | [] | [] |
| quiet | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION | N/A | N/A | N/A | [] | [] |
| cluttered | DESCRIPTIVE GEOMETRIC PROXY | 0.25 | -0.04024 | -0.3571 | ['density'] | [] |
| crowded | DESCRIPTIVE GEOMETRIC PROXY | 0.3333 | 0.2475 | -0.03754 | ['density'] | [] |

### Detailed signed semantic effects

| condition | feature | baseline_value | adjective_value | delta | signed_relative_effect | expected_sign | direction_satisfied |
| --- | --- | --- | --- | --- | --- | --- | --- |
| well_organized | free_space | 0.7272 | 0.76 | 0.0328 | 0.0451 | N/A | N/A |
| well_organized | clearance | 0.05421 | 0.01414 | -0.04007 | -0.7391 | N/A | N/A |
| well_organized | density | 0.24 | 0.24 | 0 | 0 | N/A | N/A |
| well_organized | regularity | 0.6581 | 0.6812 | 0.0231 | 0.0351 | 1 | True |
| well_organized | alignment | 0.6333 | 0.7 | 0.06667 | 0.1053 | 1 | True |
| well_organized | spacing_consistency | 0.5422 | 0.5 | -0.04224 | -0.0779 | 1 | False |
| spacious | free_space | 0.7272 | 0.84 | 0.1128 | 0.1551 | 1 | True |
| spacious | clearance | 0.05421 | 0.04243 | -0.01179 | -0.2174 | 1 | False |
| spacious | density | 0.24 | 0.24 | 0 | 0 | -1 | False |
| spacious | regularity | 0.6581 | 0.5596 | -0.09844 | -0.1496 | N/A | N/A |
| spacious | alignment | 0.6333 | 0.8 | 0.1667 | 0.2632 | N/A | N/A |
| spacious | spacing_consistency | 0.5422 | 0.5 | -0.04224 | -0.0779 | N/A | N/A |
| bright | free_space | 0.7272 | 0.6916 | -0.0356 | -0.04895 | N/A | N/A |
| bright | clearance | 0.05421 | 0.01414 | -0.04007 | -0.7391 | N/A | N/A |
| bright | density | 0.24 | 0.24 | 0 | 0 | N/A | N/A |
| bright | regularity | 0.6581 | 0.7619 | 0.1038 | 0.1578 | N/A | N/A |
| bright | alignment | 0.6333 | 0.7333 | 0.1 | 0.1579 | N/A | N/A |
| bright | spacing_consistency | 0.5422 | 0.5 | -0.04224 | -0.0779 | N/A | N/A |
| quiet | free_space | 0.7272 | 0.7556 | 0.0284 | 0.03905 | N/A | N/A |
| quiet | clearance | 0.05421 | 0.05657 | 0.002357 | 0.04348 | N/A | N/A |
| quiet | density | 0.24 | 0.24 | 0 | 0 | N/A | N/A |
| quiet | regularity | 0.6581 | 0.7458 | 0.08773 | 0.1333 | N/A | N/A |
| quiet | alignment | 0.6333 | 0.7667 | 0.1333 | 0.2105 | N/A | N/A |
| quiet | spacing_consistency | 0.5422 | 0.5585 | 0.01624 | 0.02995 | N/A | N/A |
| cluttered | free_space | 0.7272 | 0.6328 | -0.0944 | -0.1298 | N/A | N/A |
| cluttered | clearance | 0.05421 | 0.02357 | -0.03064 | -0.5652 | N/A | N/A |
| cluttered | density | 0.24 | 0.24 | 0 | 0 | 1 | False |
| cluttered | regularity | 0.6581 | 0.653 | -0.005044 | -0.007665 | -1 | True |
| cluttered | alignment | 0.6333 | 0.7333 | 0.1 | 0.1579 | -1 | False |
| cluttered | spacing_consistency | 0.5422 | 0.5481 | 0.005817 | 0.01073 | -1 | False |
| crowded | free_space | 0.7272 | 0.788 | 0.0608 | 0.08361 | -1 | False |
| crowded | clearance | 0.05421 | 0.009428 | -0.04478 | -0.8261 | -1 | True |
| crowded | density | 0.24 | 0.24 | 0 | 0 | 1 | False |
| crowded | regularity | 0.6581 | 0.6822 | 0.02408 | 0.0366 | N/A | N/A |
| crowded | alignment | 0.6333 | 0.6333 | 0 | 0 | N/A | N/A |
| crowded | spacing_consistency | 0.5422 | 0.309 | -0.2332 | -0.4301 | N/A | N/A |

### Opposite-adjective ordering (evaluation 3)

Strict inequalities are tested separately for each feature. Density reverses the order; no distance-only substitute is used. Pair consistency in the main CSV is the fraction of applicable feature orderings satisfied.

| positive_condition | negative_condition | feature | positive_value | baseline_value | negative_value | ordering_satisfied |
| --- | --- | --- | --- | --- | --- | --- |
| spacious | crowded | free_space | 0.84 | 0.7272 | 0.788 | False |
| spacious | crowded | clearance | 0.04243 | 0.05421 | 0.009428 | False |
| spacious | crowded | density | 0.24 | 0.24 | 0.24 | False |
| well_organized | cluttered | regularity | 0.6812 | 0.6581 | 0.653 | True |
| well_organized | cluttered | alignment | 0.7 | 0.6333 | 0.7333 | False |
| well_organized | cluttered | spacing_consistency | 0.5 | 0.5422 | 0.5481 | False |

### Scene feature values

| condition | free_space | clearance | density | regularity | alignment | spacing_consistency |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7272 | 0.05421 | 0.24 | 0.6581 | 0.6333 | 0.5422 |
| well_organized | 0.76 | 0.01414 | 0.24 | 0.6812 | 0.7 | 0.5 |
| spacious | 0.84 | 0.04243 | 0.24 | 0.5596 | 0.8 | 0.5 |
| bright | 0.6916 | 0.01414 | 0.24 | 0.7619 | 0.7333 | 0.5 |
| quiet | 0.7556 | 0.05657 | 0.24 | 0.7458 | 0.7667 | 0.5585 |
| cluttered | 0.6328 | 0.02357 | 0.24 | 0.653 | 0.7333 | 0.5481 |
| crowded | 0.788 | 0.009428 | 0.24 | 0.6822 | 0.6333 | 0.309 |

## E. Specificity evaluations 5–8

Object-set Jaccard here intentionally supports the existing L1 object-set difference; it is not an independent result. Inventory recall additionally checks multiplicities against baseline and required benchmark assets. Matching uses exact label plus occurrence, so duplicate-object permutations can inflate per-object displacement; no physical instance identity is claimed.

CCR and SSR use equal-weight changed channels, not arbitrary sums of distances. Relevant channels for geometry-observable adjectives are each shared object’s position, bbox, rotation and anchor. Collateral channels are inventory-multiplicity changes, a total-count-change indicator, object-type changes and room-dimension changes. Inventory and total-count signals overlap intentionally and are disclosed. Free-text descriptions are excluded because they can convey the intervention without structural movement. Geometry channels can be relevant but semantically wrong; SSR is a relevance-channel proxy, not proof that a fraction of L1 is causally explained.

`CCR = collateral/(all changed channels+1e-6)`; `SSR = relevant/all changed channels`. No structural changes yields N/A, not a fabricated perfect score. Bright/quiet relevance is unsupported, so CCR and SSR are N/A even though inventory preservation remains assessable.

| condition | object_set_jaccard | baseline_inventory_recall | required_inventory_recall | collateral_change_rate | semantic_structural_ratio | mean_displacement | median_displacement | max_displacement | fraction_objects_affected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| well_organized | 1 | 1 | 1 | 0.0625 | 0.9375 | 3.541 | 3.324 | 4.201 | 1 |
| spacious | 1 | 1 | 1 | 0.07692 | 0.9231 | 3.533 | 3.701 | 3.858 | 1 |
| bright | 1 | 1 | 1 | N/A | N/A | 3.641 | 3.674 | 3.932 | 1 |
| quiet | 1 | 1 | 1 | N/A | N/A | 3.545 | 3.825 | 4.119 | 1 |
| cluttered | 1 | 1 | 1 | 0.07143 | 0.9286 | 0.3539 | 0.3552 | 0.6727 | 1 |
| crowded | 1 | 1 | 1 | 0 | 1 | 3.382 | 3.643 | 4.272 | 1 |

The complete `per_object_results.csv/json` records shared/added/removed labels, occurrences, position displacement, 4D bbox L2 change, wrapped rotation difference, anchor/type changes and affected flags. Normalized structural change is the Euclidean norm of displacement/room-diagonal, bbox-change/room-diagonal, rotation/180 and anchor-change indicator. It is a localization diagnostic, not a combined quality score.

## F. Scene validity and relations (evaluations 9–10)

CVR is violated atomic checks / all evaluated atomic checks. Checks include scene structure, required category counts, unexpected inventory, required per-object fields, finite positive geometry, finite rotation, inside-room footprint, position/bbox-center agreement, anchor existence/uniqueness, and nonoverlap for every unordered object pair. Checks have equal weight; overlap-pair denominators grow with object count. CVR deltas compare each condition with baseline. Counts and the full check ledger are saved.

Clutter is not itself invalid: only explicit geometry/inventory/reference violations are counted. Nonoverlap follows the existing SceneCritic footprint constraint, but a shelf above a desk may be physically valid in 3D; absent height, footprint overlap is only a planar constraint diagnostic. Full 3D feasibility is not evaluated.

SRC = satisfied / evaluatable stated relations. Only explicitly mentioned unique anchor targets are used. Left/right use world x. Near uses edge gap ≤15% of room diagonal; far ≥50%; adjacent ≤5% with no positive overlap; inside requires rectangle containment. These thresholds are disclosed heuristic conventions, not calibrated semantic truth. Descriptions with unclear referents, duplicate targets, vertical above/below/on-top/under relations, or orientation-dependent front/behind/facing are retained as unevaluatable. Bare anchors are reference checks, not spatial assertions. Rotation is available, but no object-local front direction is documented. Zero evaluated relations gives N/A.

| condition | required_object_recall | constraint_violation_rate | constraint_violation_delta_from_baseline | oob_objects | overlap_pairs | src | src_delta_from_baseline | evaluated_count | unevaluated_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 0.01562 | 0 | 0 | 1 | 1 | 0 | 2 | 3 |
| well_organized | 1 | 0.04688 | 0.03125 | 2 | 1 | 1 | 0 | 2 | 2 |
| spacious | 1 | 0.03125 | 0.01562 | 0 | 2 | 1 | 0 | 2 | 2 |
| bright | 1 | 0.0625 | 0.04688 | 3 | 1 | 1 | 0 | 2 | 2 |
| quiet | 1 | 0.04688 | 0.03125 | 3 | 0 | 1 | 0 | 3 | 1 |
| cluttered | 1 | 0.03125 | 0.01562 | 1 | 1 | 1 | 0 | 2 | 2 |
| crowded | 1 | 0.04688 | 0.03125 | 1 | 2 | 0.6667 | -0.3333 | 3 | 2 |

## Cross-adjective interpretation

| adjective | sensitivity | semantic_correctness | specificity | validity |
| --- | --- | --- | --- | --- |
| well_organized | occupancy=0.840; centroid=3.541 | 67% proxy directions | inventory Jaccard=1.000 | CVR delta=+0.031; SRC=1 |
| spacious | occupancy=0.720; centroid=3.533 | 33% proxy directions | inventory Jaccard=1.000 | CVR delta=+0.016; SRC=1 |
| bright | occupancy=0.880; centroid=3.641 | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION | inventory Jaccard=1.000 | CVR delta=+0.047; SRC=1 |
| quiet | occupancy=0.720; centroid=3.545 | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION | inventory Jaccard=1.000 | CVR delta=+0.031; SRC=1 |
| cluttered | occupancy=0.200; centroid=0.354 | 25% proxy directions | inventory Jaccard=1.000 | CVR delta=+0.016; SRC=1 |
| crowded | occupancy=0.680; centroid=3.382 | 33% proxy directions | inventory Jaccard=1.000 | CVR delta=+0.031; SRC=0.6666666666666666 |

## Scientific interpretation

1. **Behavioral sensitivity:** each available adjective’s L1 results describe symbolic layout change; no change magnitude establishes semantic correctness.
2. **Statistical separation:** with one benchmark and no replicates, neither stochastic variance nor cross-benchmark separation is estimable. All p-values and T statistics are null.
3. **Semantic understanding:** intended proxy-direction rates are well_organized 67%, spacious 33%, cluttered 25%, crowded 33%. These are operational observations, not a causal claim.
4. **Specificity:** inventory preservation and per-object localization are separately reported. Relevance-channel SSR does not certify correct semantic change.
5. **Validity:** CVR and SRC expose explicit structural/reference issues; vertical relations and physical 3D feasibility remain unsupported.
6. **Strongest supported direction evidence:** well_organized has the highest observed proxy-direction rate; ties and correlated proxies prevent a unique overall ranking.
7. **Weak/unsupported concepts:** cluttered has the lowest applicable direction rate. Bright and quiet are unobservable here, not automatically model failures.
8. **Limitations:** one bedroom benchmark, one unseeded output per condition, observed API nondeterminism, no stochastic replicates, no inference, conservative textual relation extraction, correlated geometric proxies and absent photometric/acoustic/height information. Reported results do not generalize to the missing bedroom_2/bedroom_3 experiments.

## Artifacts and reproduction

`final_evaluation_results.json` integrates all categories. Detailed JSON/CSV tables are `l1_behavioral_results`, `l2_results`, `semantic_results`, `specificity_results`, `validity_results`, `per_object_results`, `scene_features`, `opposite_pair_results`, `spatial_relation_details`, `constraint_details`, and `integrated_summary`. No original generation or benchmark file is written.

```bash
python VLM_Probe/evaluate_output_probe_qwen35_27b.py
```

## Verification

25 regression tests passed. All hashed generation inputs, the benchmark, and the four inspected L1/L2 source files remained unchanged during evaluation. No inference API was imported or called by this evaluator.
