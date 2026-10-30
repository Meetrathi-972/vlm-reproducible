# Bedroom_1 adjective evaluation



Available scenes: **7/7**. Missing scenes: **0**. No inference/API calls. Input scene JSONs and original benchmarks are preserved. Old evaluation outputs remain historical; only the previous revised-evaluation directory is replaced.



## What changed

Four retained features only; fixed denominator 4; auditable structural channels and atomic violations; exact checked-in facing verifier; unsupported relations excluded; readable paginated PNGs. Existing L1 comparison function is unchanged. L2 inference is NOT APPLICABLE.



## Feature definitions

Free space = 1 - union area of clipped footprints / room area. Clearance = mean nearest rectangle-edge gap / room diagonal. Regularity = 1/(1+population CV of nearest-centroid distances). Spacing consistency = 1/(1+population CV of nearest-edge gaps); all-zero gaps remain N/A. All use existing axis-aligned footprint conventions.



## Expected-direction matrix and scientific limits

Spacious/well-organized operationally defined as more open AND more regular; crowded/cluttered as more compact AND less regular. Not universal adjective entailments. Cross-feature scores are limited proxies.

Four-feature cosine uses raw dimensionless feature-change vectors and the four expected signs, yielding N/A for undefined/zero-length vectors. Signed relative effects remain delta/(abs(baseline)+1e-6) and can inflate near a zero baseline. Direction Accuracy is correct/4, never divided by the number of available features. Zero delta is incorrect (strict 1e-6 threshold). An undefined retained feature makes the full score N/A rather than silently reducing the denominator. Bright/quiet receive no geometric semantic score.



| adjective | free_space | clearance | regularity | spacing_consistency |
| --- | --- | --- | --- | --- |
| spacious | + | + | + | + |
| well_organized | + | + | + | + |
| crowded | - | - | - | - |
| cluttered | - | - | - | - |



## Direction accuracy

| benchmark | adjective | correct_features | denominator | direction_accuracy | semantic_direction_cosine | intended_signed_effect_mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bedroom_1 | well_organized | 2 | 4 | 0.5 | -0.1868 | -0.1842 | LIMITED operational open-and-regular proxy |
| bedroom_1 | spacious | 1 | 4 | 0.25 | -0.1271 | -0.07244 | LIMITED operational open-and-regular proxy |
| bedroom_1 | bright | N/A | N/A | N/A | N/A | N/A | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION |
| bedroom_1 | quiet | N/A | N/A | N/A | N/A | N/A | NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION |
| bedroom_1 | cluttered | 3 | 4 | 0.75 | 0.6242 | 0.173 | LIMITED operational open-and-regular proxy |
| bedroom_1 | crowded | 2 | 4 | 0.5 | 0.392 | 0.284 | LIMITED operational open-and-regular proxy |



## Baseline vs adjective features

| benchmark | adjective | feature | baseline | adjective_value | delta | expected_direction | correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bedroom_1 | well_organized | free_space | 0.7272 | 0.76 | 0.0328 | + | YES |
| bedroom_1 | well_organized | clearance | 0.05421 | 0.01414 | -0.04007 | + | NO |
| bedroom_1 | well_organized | regularity | 0.6581 | 0.6812 | 0.0231 | + | YES |
| bedroom_1 | well_organized | spacing_consistency | 0.5422 | 0.5 | -0.04224 | + | NO |
| bedroom_1 | spacious | free_space | 0.7272 | 0.84 | 0.1128 | + | YES |
| bedroom_1 | spacious | clearance | 0.05421 | 0.04243 | -0.01179 | + | NO |
| bedroom_1 | spacious | regularity | 0.6581 | 0.5596 | -0.09844 | + | NO |
| bedroom_1 | spacious | spacing_consistency | 0.5422 | 0.5 | -0.04224 | + | NO |
| bedroom_1 | bright | free_space | 0.7272 | 0.6916 | -0.0356 | N/A | N/A |
| bedroom_1 | bright | clearance | 0.05421 | 0.01414 | -0.04007 | N/A | N/A |
| bedroom_1 | bright | regularity | 0.6581 | 0.7619 | 0.1038 | N/A | N/A |
| bedroom_1 | bright | spacing_consistency | 0.5422 | 0.5 | -0.04224 | N/A | N/A |
| bedroom_1 | quiet | free_space | 0.7272 | 0.7556 | 0.0284 | N/A | N/A |
| bedroom_1 | quiet | clearance | 0.05421 | 0.05657 | 0.002357 | N/A | N/A |
| bedroom_1 | quiet | regularity | 0.6581 | 0.7458 | 0.08773 | N/A | N/A |
| bedroom_1 | quiet | spacing_consistency | 0.5422 | 0.5585 | 0.01624 | N/A | N/A |
| bedroom_1 | cluttered | free_space | 0.7272 | 0.6328 | -0.0944 | - | YES |
| bedroom_1 | cluttered | clearance | 0.05421 | 0.02357 | -0.03064 | - | YES |
| bedroom_1 | cluttered | regularity | 0.6581 | 0.653 | -0.005044 | - | YES |
| bedroom_1 | cluttered | spacing_consistency | 0.5422 | 0.5481 | 0.005817 | - | NO |
| bedroom_1 | crowded | free_space | 0.7272 | 0.788 | 0.0608 | - | NO |
| bedroom_1 | crowded | clearance | 0.05421 | 0.009428 | -0.04478 | - | YES |
| bedroom_1 | crowded | regularity | 0.6581 | 0.6822 | 0.02408 | - | NO |
| bedroom_1 | crowded | spacing_consistency | 0.5422 | 0.309 | -0.2332 | - | YES |



## CCR: exact structural changes

Relevant channels are per-object position/bbox/rotation/anchor; collateral channels are inventory multiplicities, total-count indicator, type, and room dimensions. The prior equal-weight change-channel definition is preserved. Changed free text is not a numerical channel. CCR is collateral/(total+1e-6); zero collateral gives 0 for observable geometric adjectives. Bright/quiet relevance remains N/A.



| benchmark | adjective | ccr | relevant_changes | collateral_change_details | total_monitored_changes |
| --- | --- | --- | --- | --- | --- |
| bedroom_1 | well_organized | 0.0625 | bed#1 position: [0.0, 1.1] -> [0.0, 4.25]; bed#1 bbox: [-1.0, 0.2, 2.0, 1.8] -> [-1.0, 3.5, 2.0, 1.5]; cabinet#1 position: [-2.0, 0.7] -> [-1.6, 4.0]; cabinet#1 bbox: [-2.3, 0.2, 0.6, 1.0] -> [-2.0, 3.5, 0.8, 1.0]; cabinet#2 position: [2.0, 0.7] -> [1.6, 4.0]; cabinet#2 bbox: [1.7, 0.2, 0.6, 1.0] -> [1.2, 3.5, 0.8, 1.0]; chair#1 position: [-0.7, 3.2] -> [1.3, 0.9]; chair#1 bbox: [-1.0, 2.8, 0.6, 0.8] -> [1.0, 0.5, 0.6, 0.8]; chair#1 rotation: 0.0 -> 90.0; desk#1 position: [-0.5, 4.2] -> [2.1, 0.9]; desk#1 bbox: [-1.2, 3.8, 1.4, 0.8] -> [1.5, 0.5, 1.2, 0.8]; desk#1 rotation: 0.0 -> 90.0; shelf#1 position: [-0.5, 4.7] -> [2.1, 1.4]; shelf#1 bbox: [-1.2, 4.5, 1.4, 0.4] -> [1.5, 1.3, 1.2, 0.2]; shelf#1 rotation: 0.0 -> 90.0 | shelf#1 object_type: bulk -> ornaments | 16 |
| bedroom_1 | spacious | 0.07692 | bed#1 position: [0.0, 1.1] -> [0.0, 4.5]; bed#1 bbox: [-1.0, 0.2, 2.0, 1.8] -> [-1.0, 4.2, 2.0, 0.6]; cabinet#1 position: [-2.0, 0.7] -> [-1.9, 4.4]; cabinet#1 bbox: [-2.3, 0.2, 0.6, 1.0] -> [-2.2, 4.0, 0.6, 0.8]; cabinet#2 position: [2.0, 0.7] -> [1.9, 4.4]; cabinet#2 bbox: [1.7, 0.2, 0.6, 1.0] -> [1.6, 4.0, 0.6, 0.8]; chair#1 position: [-0.7, 3.2] -> [-0.15, 0.45]; chair#1 bbox: [-1.0, 2.8, 0.6, 0.8] -> [-0.4, 0.0, 0.5, 0.9]; desk#1 position: [-0.5, 4.2] -> [0.0, 0.5]; desk#1 bbox: [-1.2, 3.8, 1.4, 0.8] -> [-1.2, 0.2, 2.4, 0.6]; shelf#1 position: [-0.5, 4.7] -> [0.0, 0.875]; shelf#1 bbox: [-1.2, 4.5, 1.4, 0.4] -> [-1.0, 0.8, 2.0, 0.15] | shelf#1 object_type: bulk -> ornaments | 13 |
| bedroom_1 | bright | N/A | N/A: relevance unobservable | None | 15 |
| bedroom_1 | quiet | N/A | N/A: relevance unobservable | None | 13 |
| bedroom_1 | cluttered | 0.07143 | bed#1 bbox: [-1.0, 0.2, 2.0, 1.8] -> [-1.2, 0.1, 2.4, 2.0]; cabinet#1 position: [-2.0, 0.7] -> [-1.9, 0.7]; cabinet#1 bbox: [-2.3, 0.2, 0.6, 1.0] -> [-2.3, 0.1, 0.8, 1.2]; cabinet#2 position: [2.0, 0.7] -> [1.9, 0.7]; cabinet#2 bbox: [1.7, 0.2, 0.6, 1.0] -> [1.5, 0.1, 0.8, 1.2]; chair#1 position: [-0.7, 3.2] -> [-0.2, 3.65]; chair#1 bbox: [-1.0, 2.8, 0.6, 0.8] -> [-0.5, 3.2, 0.6, 0.9]; chair#1 rotation: 0.0 -> 180.0; desk#1 position: [-0.5, 4.2] -> [0.0, 4.6]; desk#1 bbox: [-1.2, 3.8, 1.4, 0.8] -> [-1.2, 4.2, 2.4, 0.8]; desk#1 rotation: 0.0 -> 180.0; shelf#1 position: [-0.5, 4.7] -> [0.0, 5.05]; shelf#1 bbox: [-1.2, 4.5, 1.4, 0.4] -> [-0.8, 4.9, 1.6, 0.3] | shelf#1 object_type: bulk -> ornaments | 14 |
| bedroom_1 | crowded | 0 | bed#1 position: [0.0, 1.1] -> [-1.5, 4.5]; bed#1 bbox: [-1.0, 0.2, 2.0, 1.8] -> [-2.4, 3.5, 1.8, 2.0]; cabinet#1 position: [-2.0, 0.7] -> [-0.3, 3.9]; cabinet#1 bbox: [-2.3, 0.2, 0.6, 1.0] -> [-0.6, 3.5, 0.6, 0.8]; cabinet#2 position: [2.0, 0.7] -> [-2.1, 1.9]; cabinet#2 bbox: [1.7, 0.2, 0.6, 1.0] -> [-2.4, 1.5, 0.6, 0.8]; chair#1 position: [-0.7, 3.2] -> [-0.75, 1.25]; chair#1 bbox: [-1.0, 2.8, 0.6, 0.8] -> [-1.0, 0.8, 0.5, 0.9]; chair#1 rotation: 0.0 -> 180.0; desk#1 position: [-0.5, 4.2] -> [-0.8, 0.55]; desk#1 bbox: [-1.2, 3.8, 1.4, 0.8] -> [-1.5, 0.2, 1.4, 0.7]; shelf#1 position: [-0.5, 4.7] -> [-0.8, 1.65]; shelf#1 bbox: [-1.2, 4.5, 1.4, 0.4] -> [-1.4, 1.5, 1.2, 0.3] | None | 13 |



## CVR: auditable atomic checks

The prior atomic check set and denominators are preserved: scene structure, inventory counts, fields, geometry, rotation, bounds, position/bbox center, anchors and all pairwise footprint nonoverlap checks. These are not silently replaced with SceneCritic weighted overall quality scores. Full 3D collision feasibility is unknown; touching/close objects are not violations unless the explicit footprint test fails.



| benchmark | condition | cvr | violated_checks | total_checks |
| --- | --- | --- | --- | --- |
| bedroom_1 | baseline | 0.01562 | 1 | 64 |
| bedroom_1 | well_organized | 0.04688 | 3 | 64 |
| bedroom_1 | spacious | 0.03125 | 2 | 64 |
| bedroom_1 | bright | 0.0625 | 4 | 64 |
| bedroom_1 | quiet | 0.04688 | 3 | 64 |
| bedroom_1 | cluttered | 0.03125 | 2 | 64 |
| bedroom_1 | crowded | 0.04688 | 3 | 64 |



| benchmark | condition | subject | constraint | violation_reason |
| --- | --- | --- | --- | --- |
| bedroom_1 | baseline | desk#1 + shelf#1 | footprint_nonoverlap | ('desk', 1) bbox=[-1.2, 3.8, 1.4, 0.8]; ('shelf', 1) bbox=[-1.2, 4.5, 1.4, 0.4]; intersection area=0.14 > 1e-6 |
| bedroom_1 | well_organized | desk#1 | inside_room | x_max=2.7 exceeds room limit 2.5 |
| bedroom_1 | well_organized | shelf#1 | inside_room | x_max=2.7 exceeds room limit 2.5 |
| bedroom_1 | well_organized | desk#1 + chair#1 | footprint_nonoverlap | ('desk', 1) bbox=[1.5, 0.5, 1.2, 0.8]; ('chair', 1) bbox=[1.0, 0.5, 0.6, 0.8]; intersection area=0.08 > 1e-6 |
| bedroom_1 | spacious | desk#1 + chair#1 | footprint_nonoverlap | ('desk', 1) bbox=[-1.2, 0.2, 2.4, 0.6]; ('chair', 1) bbox=[-0.4, 0.0, 0.5, 0.9]; intersection area=0.3 > 1e-6 |
| bedroom_1 | spacious | chair#1 + shelf#1 | footprint_nonoverlap | ('chair', 1) bbox=[-0.4, 0.0, 0.5, 0.9]; ('shelf', 1) bbox=[-1.0, 0.8, 2.0, 0.15]; intersection area=0.05 > 1e-6 |
| bedroom_1 | bright | bed#1 | inside_room | y_max=5.3 exceeds room limit 5 |
| bedroom_1 | bright | desk#1 | inside_room | x_max=2.7 exceeds room limit 2.5 |
| bedroom_1 | bright | shelf#1 | inside_room | x_max=2.7 exceeds room limit 2.5 |
| bedroom_1 | bright | desk#1 + chair#1 | footprint_nonoverlap | ('desk', 1) bbox=[1.5, 0.5, 1.2, 1.5]; ('chair', 1) bbox=[1.2, 0.0, 0.6, 0.8]; intersection area=0.09 > 1e-6 |
| bedroom_1 | quiet | bed#1 | inside_room | y_max=5.5 exceeds room limit 5 |
| bedroom_1 | quiet | cabinet#1 | inside_room | y_max=5.2 exceeds room limit 5 |
| bedroom_1 | quiet | cabinet#2 | inside_room | y_max=5.2 exceeds room limit 5 |
| bedroom_1 | cluttered | shelf#1 | inside_room | y_max=5.2 exceeds room limit 5 |
| bedroom_1 | cluttered | desk#1 + shelf#1 | footprint_nonoverlap | ('desk', 1) bbox=[-1.2, 4.2, 2.4, 0.8]; ('shelf', 1) bbox=[-0.8, 4.9, 1.6, 0.3]; intersection area=0.16 > 1e-6 |
| bedroom_1 | crowded | bed#1 | inside_room | y_max=5.5 exceeds room limit 5 |
| bedroom_1 | crowded | desk#1 + chair#1 | footprint_nonoverlap | ('desk', 1) bbox=[-1.5, 0.2, 1.4, 0.7]; ('chair', 1) bbox=[-1.0, 0.8, 0.5, 0.9]; intersection area=0.05 > 1e-6 |
| bedroom_1 | crowded | chair#1 + shelf#1 | footprint_nonoverlap | ('chair', 1) bbox=[-1.0, 0.8, 0.5, 0.9]; ('shelf', 1) bbox=[-1.4, 1.5, 1.2, 0.3]; intersection area=0.1 > 1e-6 |



## SRC: all evaluated and unevaluated statements

SATISFIED and VIOLATED enter the denominator; UNEVALUATED does not. Left/right/near/far/adjacent/inside retain the previous explicit rules. Near gap <=0.15 room diagonal, far >=0.5, adjacent <=0.05 with no footprint overlap; left/right use x; inside uses rectangle containment. Only explicit unique anchor targets are accepted.



| benchmark | condition | src | evaluated_relations | satisfied_relations | violated_relations | unevaluated_relations |
| --- | --- | --- | --- | --- | --- | --- |
| bedroom_1 | baseline | 1 | 2 | 2 | 0 | 3 |
| bedroom_1 | well_organized | 1 | 2 | 2 | 0 | 2 |
| bedroom_1 | spacious | 1 | 2 | 2 | 0 | 2 |
| bedroom_1 | bright | 1 | 2 | 2 | 0 | 2 |
| bedroom_1 | quiet | 1 | 3 | 3 | 0 | 1 |
| bedroom_1 | cluttered | 1 | 2 | 2 | 0 | 2 |
| bedroom_1 | crowded | 0.5 | 4 | 2 | 2 | 2 |



| benchmark | condition | subject | relation | anchor | geometry_test | result | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bedroom_1 | baseline | cabinet#1 | left | bed | x_subject=-2 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | baseline | cabinet#2 | right | bed | x_subject=2 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | baseline | desk#1 | near | N/A |  | UNEVALUATED | Missing/ambiguous/unmentioned anchor target; no guessed referent |
| bedroom_1 | baseline | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | baseline | shelf#1 | above | desk | 2D footprint intersection=0.14; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | well_organized | cabinet#1 | left | bed | x_subject=-1.6 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | well_organized | cabinet#2 | right | bed | x_subject=1.6 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | well_organized | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | well_organized | shelf#1 | on top of | desk | 2D footprint intersection=0; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | spacious | cabinet#1 | left | bed | x_subject=-1.9 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | spacious | cabinet#2 | right | bed | x_subject=1.9 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | spacious | chair#1 | under | desk | 2D footprint intersection=0.3; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | spacious | shelf#1 | on top of | desk | 2D footprint intersection=0; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | bright | cabinet#1 | left | bed | x_subject=-1.8 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | bright | cabinet#2 | right | bed | x_subject=1.8 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | bright | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | bright | shelf#1 | above | desk | 2D footprint intersection=0; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | quiet | cabinet#1 | left | bed | x_subject=-2 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | quiet | cabinet#2 | right | bed | x_subject=2 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | quiet | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | quiet | shelf#1 | near | desk | edge gap=0.3 <= 1.061 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | cluttered | cabinet#1 | left | bed | x_subject=-1.9 < x_anchor=0 - 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | cluttered | cabinet#2 | right | bed | x_subject=1.9 > x_anchor=0 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | cluttered | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | cluttered | shelf#1 | on top of | desk | 2D footprint intersection=0.16; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |
| bedroom_1 | crowded | cabinet#1 | right | bed | x_subject=-0.3 > x_anchor=-1.5 + 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | crowded | cabinet#1 | adjacent | bed | edge gap=0 <= 0.3536; intersection=8.882e-17 <= 1e-6 | SATISFIED | Preserved prior explicit geometric rule |
| bedroom_1 | crowded | cabinet#2 | near | bed | edge gap=1.2 <= 1.061 | VIOLATED | Preserved prior explicit geometric rule |
| bedroom_1 | crowded | chair#1 | front of | desk |  | UNEVALUATED | UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added. |
| bedroom_1 | crowded | chair#1 | facing | desk | SceneCritic yaw=180; target yaw=355.9; angular error=175.9 deg <= 75 (coincident centers pass) | VIOLATED | Exact OrientationVerifier._check_faces_point / _angle_check: hard |
| bedroom_1 | crowded | shelf#1 | above | desk | 2D footprint intersection=0; no z_min/z_max or support-surface elevation | UNEVALUATED | UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier. |



## SceneCritic implementation and support

Inspected `/home/meetrathi-97/Desktop/vlm/SceneCritic/SceneCritic/SceneCritic_evaluator.py` OrientationVerifier (lines 970–1275), overlap verifiers (778–968), SceneCritic README format-adaptation warning (line 210), and pipeline rotation conventions. The pipeline says 0 degrees toward ego/camera and 90 degrees right; the checked-in verifier uses 0=-y,90=+x. The scenes have 2D centers and finite rotation angles, permitting this established yaw mapping.



| relation | scenecritic_supported | current_data_sufficient | evaluation_rule | used |
| --- | --- | --- | --- | --- |
| front of | NO explicit textual rule | NO | No textual front/behind verifier in checked-in source | NO |
| behind | NO explicit textual rule | NO | No textual front/behind verifier in checked-in source | NO |
| facing | YES (orientation facing test) | YES | Exact _check_faces_point: yaw target=(atan2(dy,dx)+90)%360; angular error <=75; coincident centers pass | YES for stated unique-target relations |
| above | NO explicit textual rule | NO | No vertical relation verifier; missing z extent | NO |
| below | NO explicit textual rule | NO | No vertical relation verifier; missing z extent | NO |
| on top of | NO explicit textual rule | NO | No contact/support verifier; missing z/support surfaces; 2D full intersection is insufficient | NO |
| under | NO explicit textual rule | NO | No vertical relation verifier; missing z extent | NO |



## Vertical intersection/contact methodology

The current saved schema has bbox=[x,y,w,h], position=[x,y], no z_min/z_max, object height or surface elevation. Above/below/under require vertical ordering, and on-top needs vertical ordering plus support-surface intersection/contact. The inspected evaluator contains no corresponding textual vertical rule. Its 2D-to-3D converter (lines 440–466) infers/defaults height and sets center z=height/2, grounding every object. Those synthetic heights and elevations do not establish actual shelf elevation or support contact, so they are not substituted for missing measured geometry. We expose 2D projection intersection where possible but DO NOT substitute it for a full 3D relation. Those statements remain UNEVALUATED; full-intersection/contact cannot be established from these inputs.



## Orientation methodology

Facing executes the original OrientationVerifier._check_faces_point, _atan2_to_yaw_convention and _angle_check through AST-only loading. Desired yaw=(atan2(dy,dx)+90) mod 360; error<=75 degrees passes, error>=150 hard-fails, intermediate error soft-fails. Center differences below 0.01 on both axes pass per original logic. We add no ontology-inferred relationships or arbitrary front/behind half-spaces: the checked-in verifier has no textual front/behind test.



## Bedroom_1 input availability

| benchmark | condition | status | source |
| --- | --- | --- | --- |
| bedroom_1 | baseline | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/baseline/run_qwen35_27b.json |
| bedroom_1 | well_organized | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/well_organized/run_qwen35_27b.json |
| bedroom_1 | spacious | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/spacious/run_qwen35_27b.json |
| bedroom_1 | bright | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/bright/run_qwen35_27b.json |
| bedroom_1 | quiet | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/quiet/run_qwen35_27b.json |
| bedroom_1 | cluttered | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/cluttered/run_qwen35_27b.json |
| bedroom_1 | crowded | AVAILABLE | /home/meetrathi-97/Desktop/vlm/VLM_Probe/outputs/qwen35_27b_adjective_probe/bedroom_1_seven_outputs/bedroom_1/crowded/run_qwen35_27b.json |



## Bedroom_1 descriptive summary

Each row describes one adjective contrast against the bedroom_1 baseline. There is one saved output per condition; these are not estimates of stochastic variability. No p-values or confidence intervals.



| adjective | n_benchmarks | direction_accuracy | ccr | cvr_delta | src |
| --- | --- | --- | --- | --- | --- |
| well_organized | 1 | 0.5 | 0.0625 | 0.03125 | 1 |
| spacious | 1 | 0.25 | 0.07692 | 0.01562 | 1 |
| bright | 1 | N/A | N/A | 0.04688 | 1 |
| quiet | 1 | N/A | N/A | 0.03125 | 1 |
| cluttered | 1 | 0.75 | 0.07143 | 0.01562 | 1 |
| crowded | 1 | 0.5 | 0 | 0.03125 | 0.5 |



## Limitations

- Only bedroom_1 is evaluated; results cannot generalize across rooms

- One output per condition; no stochastic replicates; prior API non-reproducibility

- Four-sign matrix needs disclosed operational assumptions

- No z geometry; vertical full-intersection/contact unavailable

- Facing rule is exact checked-in SceneCritic; front/behind are not implemented there



## Reproduction

```bash
/home/meetrathi-97/miniconda3/envs/dspy/bin/python VLM_Probe/revised_output_probe_qwen35_27b.py
```


## Generated files

- `01_overall_summary.png`
- `02_direction_accuracy.png`
- `03_semantic_feature_values.png`
- `04_semantic_feature_deltas.png`
- `04_semantic_feature_deltas_page_02.png`
- `04_semantic_feature_deltas_page_03.png`
- `05_ccr_summary.png`
- `06_ccr_collateral_details.png`
- `07_cvr_summary.png`
- `08_cvr_violation_details.png`
- `08_cvr_violation_details_page_02.png`
- `09_src_summary.png`
- `10_src_relation_details.png`
- `10_src_relation_details_page_02.png`
- `10_src_relation_details_page_03.png`
- `10_src_relation_details_page_04.png`
- `11_baseline_vs_adjective.png`
- `11_baseline_vs_adjective_page_02.png`
- `11_baseline_vs_adjective_page_03.png`
- `12_scenecritic_relation_support.png`
- `FINAL_EVALUATION_REPORT.md`
- `bedroom_1_descriptive_summary.csv`
- `bedroom_1_descriptive_summary.json`
- `ccr_collateral_details.csv`
- `ccr_collateral_details.json`
- `ccr_details.csv`
- `ccr_details.json`
- `cvr_details.csv`
- `cvr_details.json`
- `cvr_summary.csv`
- `cvr_summary.json`
- `direction_accuracy.csv`
- `direction_accuracy.json`
- `evaluation_report.json`
- `input_inventory.csv`
- `input_inventory.json`
- `l1_behavioral_results.csv`
- `l1_behavioral_results.json`
- `overall_summary.csv`
- `overall_summary.json`
- `per_object_results.csv`
- `per_object_results.json`
- `scenecritic_relation_support.csv`
- `scenecritic_relation_support.json`
- `semantic_feature_deltas.csv`
- `semantic_feature_deltas.json`
- `semantic_features.csv`
- `semantic_features.json`
- `src_details.csv`
- `src_details.json`
- `src_summary.csv`
- `src_summary.json`
- `artifact_manifest.json`
