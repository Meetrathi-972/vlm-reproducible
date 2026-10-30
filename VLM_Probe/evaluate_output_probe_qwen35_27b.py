"""Offline L1/L2 and semantic/specificity/validity evaluation of saved bedroom_1 scenes."""
from __future__ import annotations
import argparse
import collections
import itertools
import math
import re
import statistics
from pathlib import Path
import adjective_common_qwen35_27b as C
from adv_l1_behavioral_qwen35_27b import compare

EPS = 1e-6
UNOBSERVABLE = 'NOT DIRECTLY OBSERVABLE FROM CURRENT OUTPUT REPRESENTATION'
FEATURES = ('free_space', 'clearance', 'regularity', 'spacing_consistency')
EXPECTED = {condition: dict.fromkeys(FEATURES, sign) for condition, sign in [('spacious', 1), ('well_organized', 1), ('crowded', -1), ('cluttered', -1)]}


def canon(label):
    return re.sub(r'_\d+$', '', label.lower().strip())


def indexed(objects):
    counts = collections.Counter()
    result = {}
    for obj in objects:
        label = obj['label']
        counts[label] += 1
        result[(label, counts[label])] = obj
    return result


def corners(obj):
    x, y, w, h = obj['bbox']
    return x, y, x + w, y + h


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def rect_gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def union_area(rectangles):
    """Exact axis-aligned sweep; input rectangles are already clipped to room."""
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        length, end = 0.0, None
        for low, high in intervals:
            if end is None or low > end:
                length += high-low
                end = high
            elif high > end:
                length += high-end
                end = high
        area += (right-left)*length
    return area


def consistency(values):
    if not values:
        return None
    mean = statistics.mean(values)
    if mean <= EPS:
        # Uniform zero edge gaps reflects contacts/overlap, not orderly spacing.
        return None
    return 1/(1+statistics.pstdev(values)/mean)


def features(run):
    objects = run['scene']['objects']
    room = run['benchmark_copy']['room_dimension']
    width, height = room['room_dimension_x'], room['room_dimension_y']
    extent = (-width/2, 0, width/2, height)
    diagonal = math.hypot(width, height)
    boxes = [corners(obj) for obj in objects]
    clipped = [(max(b[0], extent[0]), max(b[1], 0), min(b[2], extent[2]), min(b[3], height)) for b in boxes]
    clipped = [b for b in clipped if b[2] > b[0] and b[3] > b[1]]
    gaps = [min(rect_gap(a, b) for j, b in enumerate(boxes) if i!=j) for i, a in enumerate(boxes)] if len(boxes)>1 else []
    positions = [obj['position'] for obj in objects]
    nearest = [min(math.dist(a, b) for j, b in enumerate(positions) if i!=j) for i, a in enumerate(positions)] if len(positions)>1 else []
    return dict(free_space=1-union_area(clipped)/(width*height),
        clearance=C.finite_mean([g/diagonal for g in gaps]),
        regularity=consistency(nearest),
        spacing_consistency=consistency(gaps), room_area=width*height,
        room_diagonal=diagonal, object_count=len(objects))


def object_profile(a, b):
    A, B = indexed(a['scene']['objects']), indexed(b['scene']['objects'])
    diagonal = features(a)['room_diagonal']
    rows = []
    for key in sorted(A.keys() | B.keys()):
        x, y = A.get(key), B.get(key)
        row = dict(label=key[0], occurrence=key[1], status='shared' if x and y else 'removed' if x else 'added',
            position_shift=None, normalized_position_shift=None, bbox_l2_change=None,
            normalized_bbox_l2_change=None, rotation_change_degrees=None, anchor_changed=None,
            object_type_changed=None, affected=None, structural_change_norm=None,
            baseline_position=x.get('position') if x else None, adjective_position=y.get('position') if y else None,
            baseline_bbox=x.get('bbox') if x else None, adjective_bbox=y.get('bbox') if y else None,
            baseline_rotation=x.get('rotation_angle') if x else None, adjective_rotation=y.get('rotation_angle') if y else None,
            baseline_anchor=x.get('anchor') if x else None, adjective_anchor=y.get('anchor') if y else None)
        if x and y:
            shift = math.dist(x['position'],y['position'])
            bbox = math.dist(x['bbox'],y['bbox'])
            rx, ry = x.get('rotation_angle'), y.get('rotation_angle')
            rotation = abs((ry-rx+180)%360-180) if isinstance(rx,(int,float)) and isinstance(ry,(int,float)) else None
            row.update(position_shift=shift, normalized_position_shift=shift/diagonal,
                bbox_l2_change=bbox, normalized_bbox_l2_change=bbox/diagonal,
                rotation_change_degrees=rotation, anchor_changed=x.get('anchor')!=y.get('anchor'),
                object_type_changed=x.get('object_type')!=y.get('object_type'),
                affected=shift>EPS or bbox>EPS or (rotation or 0)>EPS or x.get('anchor')!=y.get('anchor'))
            row['structural_change_norm']=math.sqrt((shift/diagonal)**2+(bbox/diagonal)**2+((rotation or 0)/180)**2+int(row['anchor_changed']))
        rows.append(row)
    return rows


def relations(run):
    """Conservative, anchor-targeted extraction; no inferred z/front orientation."""
    objects = run['scene']['objects']
    diagonal = features(run)['room_diagonal']
    result = []
    for key, obj in indexed(objects).items():
        text = obj.get('description', '').lower()
        anchor = obj.get('anchor')
        candidates = [o for o in objects if anchor and (o['label']==anchor or canon(o['label'])==canon(anchor))]
        anchor_mentioned = bool(anchor and re.search(r'\b'+re.escape(canon(anchor))+r'\b',text))
        tests = []
        if re.search(r'\bleft (?:of|side)\b',text):tests.append(('left',lambda o,t:o['position'][0]<t['position'][0]-EPS))
        if re.search(r'\bright (?:of|side)\b',text):tests.append(('right',lambda o,t:o['position'][0]>t['position'][0]+EPS))
        if re.search(r'\bnear\b',text):tests.append(('near',lambda o,t:rect_gap(corners(o),corners(t))<=.15*diagonal))
        if re.search(r'\bfar from\b',text):tests.append(('far',lambda o,t:rect_gap(corners(o),corners(t))>=.5*diagonal))
        if re.search(r'\b(?:next to|adjacent)\b',text):tests.append(('adjacent',lambda o,t:rect_gap(corners(o),corners(t))<=.05*diagonal and intersection(corners(o),corners(t))<=EPS))
        if re.search(r'\binside\b',text):tests.append(('inside',lambda o,t:all((corners(o)[0]>=corners(t)[0]-EPS,corners(o)[1]>=corners(t)[1]-EPS,corners(o)[2]<=corners(t)[2]+EPS,corners(o)[3]<=corners(t)[3]+EPS))))
        for name, test in tests:
            evaluatable = len(candidates)==1 and anchor_mentioned and candidates[0] is not obj
            result.append(dict(label=key[0],occurrence=key[1],relation=name,target=anchor,description=obj.get('description'),
                evaluatable=evaluatable,satisfied=bool(test(obj,candidates[0])) if evaluatable else None,
                reason='Unique explicit anchor target' if evaluatable else 'Missing/ambiguous/unmentioned anchor target; no guessed referent'))
        for pattern, name in [(r'\b(?:above|below|on top of|under)\b','vertical'),(r'\b(?:in front of|behind|facing)\b','front_back')]:
            if re.search(pattern,text):
                result.append(dict(label=key[0],occurrence=key[1],relation=name,target=anchor,description=obj.get('description'),evaluatable=False,satisfied=None,
                    reason='No z/elevation or documented object-local front/back axis in the saved representation'))
    evaluated = [r for r in result if r['evaluatable']]
    return dict(src=C.finite_mean([int(r['satisfied']) for r in evaluated]),evaluated_count=len(evaluated),unevaluated_count=len(result)-len(evaluated)),result


def validity(run):
    objs=run['scene']['objects']; room=run['benchmark_copy']['room_dimension']
    w,h=room['room_dimension_x'],room['room_dimension_y']; extent=(-w/2,0,w/2,h)
    required=run['benchmark_copy']['assets']; actual=collections.Counter(canon(o['label']) for o in objs)
    checks=[]
    def add(kind,item,ok):checks.append(dict(check=kind,item=item,violated=not bool(ok)))
    add('scene_structure','objects_list',isinstance(objs,list))
    for label,count in required.items():add('required_object_count',label,actual[canon(label)]>=count)
    add('unexpected_inventory','categories',set(actual)<=set(map(canon,required)))
    mandatory={'label','object_type','anchor','bbox','position','rotation_angle','description','concise_description'}
    for key,obj in indexed(objs).items():
        item=f'{key[0]}#{key[1]}'
        add('required_fields',item,mandatory<=set(obj))
        try:C.check_objects([obj]);geometry=True
        except (ValueError,TypeError):geometry=False
        add('valid_geometry',item,geometry)
        rot=obj.get('rotation_angle');add('finite_rotation',item,isinstance(rot,(int,float)) and not isinstance(rot,bool) and math.isfinite(rot))
        if geometry:
            b=corners(obj);add('inside_room',item,b[0]>=extent[0]-EPS and b[1]>=-EPS and b[2]<=extent[2]+EPS and b[3]<=h+EPS)
            add('position_matches_bbox_center',item,math.dist(obj['position'],[(b[0]+b[2])/2,(b[1]+b[3])/2])<=EPS)
        anchor=obj.get('anchor')
        matches=[o for o in objs if anchor and (o['label']==anchor or canon(o['label'])==canon(anchor))]
        add('anchor_exists',item,anchor is None or any(o is not obj for o in matches))
        add('anchor_unambiguous',item,anchor is None or len(matches)==1)
    for (ka,a),(kb,b) in itertools.combinations(indexed(objs).items(),2):
        add('footprint_nonoverlap',f'{ka}|{kb}',intersection(corners(a),corners(b))<=EPS)
    rel, details=relations(run)
    return dict(constraint_violation_rate=sum(c['violated'] for c in checks)/len(checks),violated_checks=sum(c['violated'] for c in checks),total_checks=len(checks),
        missing_required_objects=sum(max(0,n-actual[canon(k)]) for k,n in required.items()),
        required_object_recall=sum(min(n,actual[canon(k)]) for k,n in required.items())/sum(required.values()),
        oob_objects=sum(c['violated'] for c in checks if c['check']=='inside_room'),
        overlap_pairs=sum(c['violated'] for c in checks if c['check']=='footprint_nonoverlap'),**rel),checks,details


def semantic(base, adv, condition):
    f,g=features(base),features(adv); signs=EXPECTED.get(condition)
    deltas={k:g[k]-f[k] if g[k] is not None and f[k] is not None else None for k in FEATURES}
    effects={k:deltas[k]/(abs(f[k])+EPS) if deltas[k] is not None else None for k in FEATURES}
    valid=[k for k in (signs or {}) if deltas[k] is not None]
    hits={k:int(deltas[k]*signs[k]>EPS) for k in valid}
    # All four observable features form the vector; zero expected components penalize unrelated geometric movement.
    vector=[deltas[k] for k in FEATURES] if all(deltas[k] is not None for k in FEATURES) else None
    expected=[(signs or {}).get(k,0) for k in FEATURES]
    cosine=None
    if vector is not None and signs:
        den=math.sqrt(sum(v*v for v in vector)*sum(v*v for v in expected))
        if den>EPS:cosine=sum(x*y for x,y in zip(vector,expected))/den
    return dict(semantic_status='DESCRIPTIVE GEOMETRIC PROXY' if signs else UNOBSERVABLE,
        direction_accuracy=sum(hits.values())/4 if signs and len(valid)==4 else None,
        direction_denominator=4 if signs else None,
        evaluatable_direction_features=len(valid),direction_feature_results=hits,
        feature_delta=deltas,normalized_signed_effect=effects,
        intended_signed_effect_mean=C.finite_mean([signs[k]*effects[k] for k in valid]) if signs else None,
        semantic_direction_cosine=cosine,feature_change_vector=vector,expected_direction_vector=expected if signs else None,
        zero_change_features=[k for k in valid if abs(deltas[k])<=EPS],
        skipped_features=[k for k in (signs or {}) if deltas[k] is None])


def specificity(base,adv,condition,profile):
    A=collections.Counter(canon(o['label']) for o in base['scene']['objects']);B=collections.Counter(canon(o['label']) for o in adv['scene']['objects'])
    union=set(A)|set(B); shared=set(A)&set(B)
    required=base['benchmark_copy']['assets']; denominator=sum(required.values())
    common=[r for r in profile if r['status']=='shared']
    relevant=sum((r['position_shift'] or 0)>EPS for r in common)+sum((r['bbox_l2_change'] or 0)>EPS for r in common)+sum((r['rotation_change_degrees'] or 0)>EPS for r in common)+sum(bool(r['anchor_changed']) for r in common)
    identity=sum(abs(A[k]-B[k]) for k in union)
    count=int(sum(A.values())!=sum(B.values()))
    type_changes=sum(bool(r['object_type_changed']) for r in common)
    room=int(base['benchmark_copy']['room_dimension']!=adv['benchmark_copy']['room_dimension'])
    collateral=identity+count+type_changes+room
    total=relevant+collateral
    observable=condition in EXPECTED
    return dict(object_set_jaccard=len(shared)/len(union) if union else 1.,
        baseline_inventory_recall=sum(min(A[k],B[k]) for k in A)/sum(A.values()) if A else None,
        required_inventory_recall=sum(min(n,B[canon(k)]) for k,n in required.items())/denominator,
        relevant_changed_channels=relevant if observable else None,collateral_changed_channels=collateral if observable else None,
        total_changed_channels=total,collateral_change_rate=collateral/(total+EPS) if observable and total else None,
        semantic_structural_ratio=relevant/total if observable and total else None,
        ratio_status='DESCRIPTIVE relevance-channel proxy, not proof of semantic explanation' if observable else UNOBSERVABLE,
        shared_objects=len(common),mean_displacement=C.finite_mean([r['position_shift'] for r in common]),
        median_displacement=statistics.median([r['position_shift'] for r in common]) if common else None,
        max_displacement=max((r['position_shift'] for r in common),default=None),
        mean_structural_change=C.finite_mean([r['structural_change_norm'] for r in common]),
        median_structural_change=statistics.median([r['structural_change_norm'] for r in common]) if common else None,
        max_structural_change=max((r['structural_change_norm'] for r in common),default=None),
        fraction_objects_affected=C.finite_mean([int(r['affected']) for r in common]))


def table(rows,columns):
    def cell(value):
        if value is None:return 'N/A'
        if isinstance(value,float):return f'{value:.4g}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(cell(r.get(k)) for k in columns)+' |' for r in rows]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=C.OUTPUT_ROOT/'bedroom_1_seven_outputs')
    args=parser.parse_args();root=args.input_root.resolve();out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    # Replace only generated evaluation products; never clear scenes or prior scientific outputs.
    stems=('l1_behavioral_results','l2_results','semantic_results','specificity_results','validity_results','per_object_results','scene_features','opposite_pair_results','spatial_relation_details','constraint_details','integrated_summary','semantic_feature_results')
    for name in ('final_evaluation_report.md','final_evaluation_results.json',*[stem+ext for stem in stems for ext in ('.csv','.json')]):
        path=out/name
        if path.exists():path.unlink()
    tracked={str(p):C.digest(p) for p in root.rglob('*') if p.is_file() and out not in p.parents}
    benchmark_path=C.BENCHMARK_ROOT/'bedroom/bedroom_1.json';tracked[str(benchmark_path)]=C.digest(benchmark_path)
    source_paths=[C.ROOT/'adv_l1_behavioral_qwen35_27b.py',C.ROOT/'adv_l1_variance_qwen35_27b.py',C.ROOT/'adjective_common_qwen35_27b.py',C.ROOT/'probe_metrics/adv_l1_variance.py']
    tracked.update({str(p):C.digest(p) for p in source_paths})
    runs={};availability=[]
    original=C.read_json(benchmark_path)
    for condition in C.CONDITIONS:
        p=root/'bedroom_1'/condition/'run_qwen35_27b.json'
        if not p.exists():
            availability.append(dict(condition=condition,status='MISSING',path=str(p)));continue
        run=C.read_json(p)
        if run['model']!=C.MODEL or run['provider']!='deepinfra':raise ValueError('Unexpected inference provenance')
        if run['benchmark_copy']!=C.modified_benchmark(original,condition):raise ValueError('Intervention benchmark differs')
        if tuple(c['stage'] for c in run['calls'])!=C.STAGES or run['refinement']!=C.REFINEMENT:raise ValueError('Unexpected generation procedure')
        if any('seed' in c.get('api_response',{}).get('request',{}) for c in run['calls']):raise ValueError('Seeded run outside current design')
        runs[condition]=run;availability.append(dict(condition=condition,status='available',path=str(p),sha256=C.digest(p)))
    metadata=dict(scope='bedroom_1 only, seven conditions, one output per condition',benchmark='bedroom_1',
        evaluation_implementation_sha256=C.digest(__file__),
        regression_command="python -m unittest discover -s VLM_Probe -p 'test_*qwen35_27b.py'",
        n_benchmarks=1,stochastic_replicates=0,api_calls=0,statistical_interpretation='DESCRIPTIVE',
        aggregate='Arithmetic mean over available benchmark values; with one benchmark, aggregate equals bedroom_1.',
        observational_limit='API nondeterminism was observed previously; one unseeded sample per condition cannot isolate causal adjective effects.',
        epsilon=EPS,unsupported_adjectives={'bright':UNOBSERVABLE,'quiet':UNOBSERVABLE},input_inventory=availability)
    scene_rows=[];valid_rows=[];constraints=[];relation_details=[];valid={}
    for condition,run in runs.items():
        scene_rows.append(dict(benchmark='bedroom_1',condition=condition,**features(run)))
        v,checks,relations_=validity(run);valid[condition]=v
        valid_rows.append(dict(benchmark='bedroom_1',condition=condition,**v,
            constraint_violation_delta_from_baseline=None,src_delta_from_baseline=None))
        constraints.extend(dict(benchmark='bedroom_1',condition=condition,**c) for c in checks)
        relation_details.extend(dict(benchmark='bedroom_1',condition=condition,**r) for r in relations_)
    if 'baseline' in valid:
        for row in valid_rows:
            row['constraint_violation_delta_from_baseline']=row['constraint_violation_rate']-valid['baseline']['constraint_violation_rate']
            row['src_delta_from_baseline']=row['src']-valid['baseline']['src'] if row['src'] is not None and valid['baseline']['src'] is not None else None
    l1=[];l2=[];sem=[];specs=[];objects=[];opposites=[];integrated=[];feature_rows=[]
    for condition in list(C.CONDITIONS)[1:]:
        prefix=dict(benchmark='bedroom_1',condition=condition)
        if 'baseline' not in runs or condition not in runs:
            l1.append(dict(**prefix,status='MISSING',**{k:None for k in (*C.METRICS,'first_divergent_call','first_divergent_stage','structural_divergence','placement_changed','baseline_call_count','adjective_call_count','baseline_object_count','adjective_object_count','baseline_legality','adjective_legality','baseline_oob','adjective_oob')}))
            continue
        a,b=runs['baseline'],runs[condition]
        l1.append(dict(**prefix,status='available',**compare(a,b)))
        profile=object_profile(a,b);objects.extend(dict(**prefix,**r) for r in profile)
        sem.append(dict(**prefix,**semantic(a,b,condition)))
        fa,fb=features(a),features(b)
        for feature in FEATURES:
            sign=EXPECTED.get(condition,{}).get(feature)
            delta=fb[feature]-fa[feature] if fa[feature] is not None and fb[feature] is not None else None
            feature_rows.append(dict(**prefix,feature=feature,baseline_value=fa[feature],adjective_value=fb[feature],delta=delta,
                signed_relative_effect=delta/(abs(fa[feature])+EPS) if delta is not None else None,expected_sign=sign,
                direction_satisfied=delta*sign>EPS if sign is not None and delta is not None else None,
                semantic_status='DESCRIPTIVE intended geometric feature' if sign else 'Observed geometric collateral/unspecified feature; not a semantic target'))
        specs.append(dict(**prefix,**specificity(a,b,condition,profile)))
        for metric,value in C.pair_metrics(a,b).items():
            l2.append(dict(**prefix,metric=metric,n_benchmarks=1,n_baseline=1,n_adjective=1,
                within_baseline=None,within_adjective=None,between=value,test_statistic=None,p_value=None,
                status='NOT APPLICABLE: no within-condition or cross-benchmark observations',
                between_interpretation='Single matched-benchmark distance only; equivalent to L1 distance, not an independent L2 result'))
    for positive,negative,keys in [('spacious','crowded',{'free_space':1,'clearance':1}),('well_organized','cluttered',{'regularity':1,'spacing_consistency':1})]:
        if not all(c in runs for c in ('baseline',positive,negative)):continue
        f,g,h=(features(runs[c]) for c in (positive,'baseline',negative))
        for key,sign in keys.items():
            applicable=all(v[key] is not None for v in (f,g,h))
            opposites.append(dict(benchmark='bedroom_1',positive_condition=positive,negative_condition=negative,feature=key,
                positive_value=f[key],baseline_value=g[key],negative_value=h[key],expected_order='positive > baseline > negative' if sign==1 else 'positive < baseline < negative',
                ordering_satisfied=(sign*(f[key]-g[key])>EPS and sign*(g[key]-h[key])>EPS) if applicable else None,
                status='DESCRIPTIVE' if applicable else 'NOT APPLICABLE: unobservable/degenerate feature'))
    for condition in list(C.CONDITIONS)[1:]:
        l=next((r for r in l1 if r['condition']==condition),{});s=next((r for r in sem if r['condition']==condition),{});q=next((r for r in specs if r['condition']==condition),{});v=next((r for r in valid_rows if r['condition']==condition),{})
        pair=[r['ordering_satisfied'] for r in opposites if condition in (r['positive_condition'],r['negative_condition']) and r['ordering_satisfied'] is not None]
        integrated.append(dict(condition=condition,n_benchmarks=1,status=l.get('status','MISSING'),
            structural_divergence=l.get('structural_divergence'),object_set_diff=l.get('object_set_diff'),occupancy_hamming=l.get('occupancy_hamming'),centroid_shift=l.get('centroid_shift'),legality_flip=l.get('legality_flip'),
            l2_statistic=None,l2_within=None,l2_between=None,p_value=None,l2_status='NOT APPLICABLE',
            direction_accuracy=s.get('direction_accuracy'),intended_signed_effect_mean=s.get('intended_signed_effect_mean'),opposite_pair_consistency=C.finite_mean([int(x) for x in pair]),semantic_direction_cosine=s.get('semantic_direction_cosine'),semantic_status=s.get('semantic_status','MISSING'),
            object_preservation=q.get('object_set_jaccard'),collateral_change_rate=q.get('collateral_change_rate'),semantic_structural_ratio=q.get('semantic_structural_ratio'),
            constraint_violation_delta=v.get('constraint_violation_delta_from_baseline'),spatial_relation_consistency=v.get('src'),spatial_relation_delta=v.get('src_delta_from_baseline')))
    categories=dict(l1_behavioral_results=l1,l2_results=l2,semantic_results=sem,specificity_results=specs,validity_results=valid_rows,per_object_results=objects,scene_features=scene_rows,opposite_pair_results=opposites,spatial_relation_details=relation_details,constraint_details=constraints,integrated_summary=integrated,semantic_feature_results=feature_rows)
    for stem,rows in categories.items():C.write_summary(out,stem,rows,metadata)
    result=dict(metadata=metadata,results=categories,input_hashes=tracked,
        l2_applicability={'fully_applicable':False,'n_benchmarks':1,'n_outputs_per_condition':1,'stochastic_replicates':0,'computed':'One baseline/adjective distance per metric, labeled descriptively; no within-condition distances, T, permutations, p-values or confidence intervals','required_for_cross_benchmark_descriptive_T':'At least two distinct completed benchmark instances per condition; room-normalization comparability and inventory confounds must be stated','required_for_inference':'A justified randomized/exchangeable assignment design, not treating benchmarks as API samples'})
    C.write_json(out/'final_evaluation_results.json',result)
    text=make_report(metadata,availability,categories)
    (out/'final_evaluation_report.md').write_text(text)
    assert all(C.digest(p)==sha for p,sha in tracked.items()),'An input was altered during offline evaluation'
    print('Offline evaluation complete:',out)
    print('Scenes:',len(runs),'; L1 contrasts:',sum(r['status']=='available' for r in l1),'; object rows:',len(objects),'; API calls: 0')


def make_report(metadata,availability,c):
    summary=c['integrated_summary'];sem=c['semantic_results'];valid=c['validity_results'];spec=c['specificity_results']
    lines=['# Bedroom_1 output-based probing evaluation','',
        'This report evaluates only the seven existing, unseeded bedroom_1 outputs. There is one output per condition, no stochastic replicates, and no new inference. All values are **DESCRIPTIVE**. Generated scenes and benchmark JSONs were preserved.','',
        '## Interpretation and provenance','',
        'L1 asks whether output changed. L2 asks how within/between output separation compares under an applicable design. Semantic evaluations ask whether supported properties changed in the expected direction. Specificity asks which structural channels changed. Validity asks whether geometry and stated relations satisfy explicit checks. Large L1 change is neither automatically good nor evidence of semantic correctness.','',
        'Only the canonical `bedroom_1/<condition>/run_qwen35_27b.json` files are used. Baseline replay/determinism-attempt directories are excluded. With one benchmark, each aggregate equals the bedroom_1 value; no cross-bedroom average is implied. API nondeterminism and one unseeded scene per condition prevent a causal separation of adjective effects from generation randomness.','',
        table(availability,['condition','status','path']),'',
        '## A. High-level comparison','',table(summary,['condition','occupancy_hamming','centroid_shift','direction_accuracy','semantic_direction_cosine','object_preservation','collateral_change_rate','constraint_violation_delta','spatial_relation_consistency']),'',
        'Brightness and quietness are not directly observable: the schema has no illumination, photometry, sound level or acoustic model. Free-text references to light/comfort are not quantitative measurements and are not scored as semantic success or failure.','',
        '## B. L1 behavioral sensitivity','',
        'The existing `adv_l1_behavioral_qwen35_27b.compare` and `adjective_common_qwen35_27b.pair_metrics` run unchanged. First divergence is zero-based and compares trimmed response text; structural divergence means differing call counts/stage sequences, not object motion. Jaccard normalizes trailing numeric label suffixes; count difference uses full object-list lengths; centroids match exact label plus occurrence. Occupancy is the repository 5×5 category grid. Legality means no footprint overlap and excludes bounds.','',
        table(c['l1_behavioral_results'],['condition','first_divergent_call','structural_divergence','placement_changed','object_set_diff','object_count_diff','occupancy_hamming','centroid_shift','legality_flip']),'',
        '## C. L2 applicability','',
        '**NOT APPLICABLE as variance, cross-benchmark separation or inference.** Exact n is one benchmark, one baseline and one adjective output for each contrast. No within-condition pair exists, so `T = mean_between - mean_within` is undefined. The existing variance implementation requires at least two matched seeds and cannot run legitimately on this design. No seeds, pseudo-replicates, permutations, p-values or confidence intervals were introduced. Its distance primitive is reused to report the sole matched distance below; this duplicates L1 and is not independent evidence.','',
        table(c['l2_results'],['condition','metric','n_benchmarks','within_baseline','within_adjective','between','test_statistic','p_value']),'',
        '## D. Semantic evaluations 1–4','',
        'These are operational geometric proxies, not a validated semantic ground truth. Bboxes are `[x,y,w,h]`; room extent is `[-width/2,width/2] × [0,height]`. All footprint calculations use the same unrotated axis-aligned convention as repository geometry; rotations do not enter semantic features. No height, z-axis, 3D collision model or object-local front axis is invented.','',
        '- Free space: one minus room-clipped footprint union area divided by room area.\n- Clearance: mean nearest-edge gap divided by room diagonal.\n- Regularity: 1/(1+CV) of nearest-centroid distances.\n- Spacing consistency: 1/(1+CV) of nearest-edge gaps; all-zero gaps yield N/A.','',
        'Direction scoring uses exactly four signs and a denominator of four. Missing values yield N/A. Signs define an open-and-regular operational layout assumption: spacious/well-organized ++++; crowded/cluttered ----. This is limited and not a semantic entailment. Zero change is incorrect. Relative effects use delta/(abs(baseline)+1e-6).','',
        table(sem,['condition','semantic_status','direction_accuracy','intended_signed_effect_mean','semantic_direction_cosine','zero_change_features','skipped_features']),'',
        '### Detailed signed semantic effects','',table(c['semantic_feature_results'],['condition','feature','baseline_value','adjective_value','delta','signed_relative_effect','expected_sign','direction_satisfied']),'',
        '### Opposite-adjective ordering (evaluation 3)','',
        'Strict inequalities are tested separately for each feature. no distance-only substitute is used. Pair consistency in the main CSV is the fraction of applicable feature orderings satisfied.','',
        table(c['opposite_pair_results'],['positive_condition','negative_condition','feature','positive_value','baseline_value','negative_value','ordering_satisfied']),'',
        '### Scene feature values','',table(c['scene_features'],['condition',*FEATURES]),'',
        '## E. Specificity evaluations 5–8','',
        'Object-set Jaccard here intentionally supports the existing L1 object-set difference; it is not an independent result. Inventory recall additionally checks multiplicities against baseline and required benchmark assets. Matching uses exact label plus occurrence, so duplicate-object permutations can inflate per-object displacement; no physical instance identity is claimed.','',
        'CCR and SSR use equal-weight changed channels, not arbitrary sums of distances. Relevant channels for geometry-observable adjectives are each shared object’s position, bbox, rotation and anchor. Collateral channels are inventory-multiplicity changes, a total-count-change indicator, object-type changes and room-dimension changes. Inventory and total-count signals overlap intentionally and are disclosed. Free-text descriptions are excluded because they can convey the intervention without structural movement. Geometry channels can be relevant but semantically wrong; SSR is a relevance-channel proxy, not proof that a fraction of L1 is causally explained.','',
        '`CCR = collateral/(all changed channels+1e-6)`; `SSR = relevant/all changed channels`. No structural changes yields N/A, not a fabricated perfect score. Bright/quiet relevance is unsupported, so CCR and SSR are N/A even though inventory preservation remains assessable.','',
        table(spec,['condition','object_set_jaccard','baseline_inventory_recall','required_inventory_recall','collateral_change_rate','semantic_structural_ratio','mean_displacement','median_displacement','max_displacement','fraction_objects_affected']),'',
        'The complete `per_object_results.csv/json` records shared/added/removed labels, occurrences, position displacement, 4D bbox L2 change, wrapped rotation difference, anchor/type changes and affected flags. Normalized structural change is the Euclidean norm of displacement/room-diagonal, bbox-change/room-diagonal, rotation/180 and anchor-change indicator. It is a localization diagnostic, not a combined quality score.','',
        '## F. Scene validity and relations (evaluations 9–10)','',
        'CVR is violated atomic checks / all evaluated atomic checks. Checks include scene structure, required category counts, unexpected inventory, required per-object fields, finite positive geometry, finite rotation, inside-room footprint, position/bbox-center agreement, anchor existence/uniqueness, and nonoverlap for every unordered object pair. Checks have equal weight; overlap-pair denominators grow with object count. CVR deltas compare each condition with baseline. Counts and the full check ledger are saved.','',
        'Clutter is not itself invalid: only explicit geometry/inventory/reference violations are counted. Nonoverlap follows the existing SceneCritic footprint constraint, but a shelf above a desk may be physically valid in 3D; absent height, footprint overlap is only a planar constraint diagnostic. Full 3D feasibility is not evaluated.','',
        'SRC = satisfied / evaluatable stated relations. Only explicitly mentioned unique anchor targets are used. Left/right use world x. Near uses edge gap ≤15% of room diagonal; far ≥50%; adjacent ≤5% with no positive overlap; inside requires rectangle containment. These thresholds are disclosed heuristic conventions, not calibrated semantic truth. Descriptions with unclear referents, duplicate targets, vertical above/below/on-top/under relations, or orientation-dependent front/behind/facing are retained as unevaluatable. Bare anchors are reference checks, not spatial assertions. Rotation is available, but no object-local front direction is documented. Zero evaluated relations gives N/A.','',
        table(valid,['condition','required_object_recall','constraint_violation_rate','constraint_violation_delta_from_baseline','oob_objects','overlap_pairs','src','src_delta_from_baseline','evaluated_count','unevaluated_count']),'',
        '## Cross-adjective interpretation','']
    interpretation=[]
    for row in summary:
        interpretation.append(dict(adjective=row['condition'],sensitivity=f"occupancy={row['occupancy_hamming']:.3f}; centroid={row['centroid_shift']:.3f}" if row['centroid_shift'] is not None else 'N/A',
            semantic_correctness=f"{row['direction_accuracy']:.0%} proxy directions" if row['direction_accuracy'] is not None else UNOBSERVABLE,
            specificity=f"inventory Jaccard={row['object_preservation']:.3f}" if row['object_preservation'] is not None else 'N/A',
            validity=f"CVR delta={row['constraint_violation_delta']:+.3f}; SRC={row['spatial_relation_consistency']}" if row['constraint_violation_delta'] is not None else 'N/A'))
    lines += [table(interpretation,['adjective','sensitivity','semantic_correctness','specificity','validity']),'','## Scientific interpretation','']
    applicable=[r for r in summary if r['direction_accuracy'] is not None]
    strongest=max(applicable,key=lambda r:r['direction_accuracy']) if applicable else None
    weakest=min(applicable,key=lambda r:r['direction_accuracy']) if applicable else None
    lines += ['1. **Behavioral sensitivity:** each available adjective’s L1 results describe symbolic layout change; no change magnitude establishes semantic correctness.',
        '2. **Statistical separation:** with one benchmark and no replicates, neither stochastic variance nor cross-benchmark separation is estimable. All p-values and T statistics are null.',
        '3. **Semantic understanding:** intended proxy-direction rates are '+', '.join(f"{r['condition']} {r['direction_accuracy']:.0%}" for r in applicable)+'. These are operational observations, not a causal claim.',
        '4. **Specificity:** inventory preservation and per-object localization are separately reported. Relevance-channel SSR does not certify correct semantic change.',
        '5. **Validity:** CVR and SRC expose explicit structural/reference issues; vertical relations and physical 3D feasibility remain unsupported.',
        f"6. **Strongest supported direction evidence:** {strongest['condition'] if strongest else 'none estimable'} has the highest observed proxy-direction rate; ties and correlated proxies prevent a unique overall ranking.",
        f"7. **Weak/unsupported concepts:** {weakest['condition'] if weakest else 'none estimable'} has the lowest applicable direction rate. Bright and quiet are unobservable here, not automatically model failures.",
        '8. **Limitations:** one bedroom benchmark, one unseeded output per condition, observed API nondeterminism, no stochastic replicates, no inference, conservative textual relation extraction, correlated geometric proxies and absent photometric/acoustic/height information. Reported results do not generalize to the missing bedroom_2/bedroom_3 experiments.','',
        '## Artifacts and reproduction','',
        '`final_evaluation_results.json` integrates all categories. Detailed JSON/CSV tables are `l1_behavioral_results`, `l2_results`, `semantic_results`, `specificity_results`, `validity_results`, `per_object_results`, `scene_features`, `opposite_pair_results`, `spatial_relation_details`, `constraint_details`, and `integrated_summary`. No original generation or benchmark file is written.','',
        '```bash\npython VLM_Probe/evaluate_output_probe_qwen35_27b.py\n```','']
    return '\n'.join(lines)


if __name__=='__main__':
    main()
