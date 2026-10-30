"""Offline revised four-feature evaluation with auditable change/violation/relation tables."""
import argparse
import collections
import copy
import csv
import itertools
import json
import math
import re
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace
import adjective_common_qwen35_27b as C
import evaluate_output_probe_qwen35_27b as E
from adv_l1_behavioral_qwen35_27b import compare

SOURCE=C.ROOT.parent/'SceneCritic/SceneCritic/SceneCritic_evaluator.py'
FEATURES=E.FEATURES
EPS=E.EPS
SIGN_MATRIX=E.EXPECTED


def orientation_verifier():
    # Compile exactly the checked-in verifier; do not import the inference/evaluator main module.
    cls=C.load_definitions(SOURCE,{'OrientationVerifier'},{'RuleChecker':object,'math':math})[0].OrientationVerifier
    return cls()


def save(root,stem,rows):
    C.write_json(root/(stem+'.json'),rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/(stem+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def changes(a,b):
    profile=E.object_profile(a,b)
    A=collections.Counter(E.canon(o['label']) for o in a['scene']['objects'])
    B=collections.Counter(E.canon(o['label']) for o in b['scene']['objects'])
    relevant=[];collateral=[]
    for p in profile:
        if p['status']!='shared':continue
        name=f"{p['label']}#{p['occurrence']}"
        for key,old,new,delta in [('position',p['baseline_position'],p['adjective_position'],p['position_shift']),('bbox',p['baseline_bbox'],p['adjective_bbox'],p['bbox_l2_change']),('rotation',p['baseline_rotation'],p['adjective_rotation'],p['rotation_change_degrees']),('anchor',p['baseline_anchor'],p['adjective_anchor'],int(p['anchor_changed']))]:
            if delta is not None and delta>EPS:relevant.append(dict(subject=name,channel=key,before=old,after=new))
        if p['object_type_changed']:
            x=E.indexed(a['scene']['objects'])[(p['label'],p['occurrence'])]
            y=E.indexed(b['scene']['objects'])[(p['label'],p['occurrence'])]
            collateral.append(dict(subject=name,channel='object_type',before=x.get('object_type'),after=y.get('object_type')))
    for label in sorted(A.keys()|B.keys()):
        # Preserve existing absolute-multiplicity channel weighting.
        for i in range(abs(A[label]-B[label])):collateral.append(dict(subject=label,channel='inventory_multiplicity',before=A[label],after=B[label]))
    if sum(A.values())!=sum(B.values()):collateral.append(dict(subject='scene',channel='total_object_count',before=sum(A.values()),after=sum(B.values())))
    x=a['benchmark_copy']['room_dimension'];y=b['benchmark_copy']['room_dimension']
    if x!=y:collateral.append(dict(subject='room',channel='room_dimensions',before=x,after=y))
    return relevant,collateral


def violation_details(run,checks):
    objects=E.indexed(run['scene']['objects']);room=run['benchmark_copy']['room_dimension']
    extent=[-room['room_dimension_x']/2,0,room['room_dimension_x']/2,room['room_dimension_y']]
    ledger=[]
    for check in checks:
        if not check['violated']:continue
        item=check['item'];kind=check['check'];obj=next((o for (label,n),o in objects.items() if f'{label}#{n}'==item),None)
        reason=''
        if kind=='inside_room' and obj:
            box=E.corners(obj);reasons=[]
            for key,value,bound,violated in [('x_min',box[0],extent[0],box[0]<extent[0]-EPS),('y_min',box[1],extent[1],box[1]<extent[1]-EPS),('x_max',box[2],extent[2],box[2]>extent[2]+EPS),('y_max',box[3],extent[3],box[3]>extent[3]+EPS)]:
                if violated:reasons.append(f'{key}={value:.4g} exceeds room limit {bound:.4g}')
            reason='; '.join(reasons)
        elif kind=='footprint_nonoverlap':
            import ast
            ka,kb=(ast.literal_eval(part) for part in item.split('|'))
            x,y=objects[ka],objects[kb]
            reason=f"{ka} bbox={x['bbox']}; {kb} bbox={y['bbox']}; intersection area={E.intersection(E.corners(x),E.corners(y)):.6g} > 1e-6"
            item=f'{ka[0]}#{ka[1]} + {kb[0]}#{kb[1]}'
        elif kind=='required_object_count':
            actual=sum(E.canon(o['label'])==E.canon(item) for o in objects.values());reason=f"Required >= {run['benchmark_copy']['assets'][item]}; observed {actual}"
        elif obj:
            if kind=='position_matches_bbox_center':reason=f"position={obj['position']}; bbox={obj['bbox']}; center mismatch > 1e-6"
            elif kind.startswith('anchor'):reason=f"anchor={obj.get('anchor')!r}; candidate labels={[o['label'] for o in objects.values() if obj.get('anchor') and E.canon(o['label'])==E.canon(obj['anchor'])]}"
            elif kind=='required_fields':reason=f"Missing fields={sorted({'label','object_type','anchor','bbox','position','rotation_angle','description','concise_description'}-set(obj))}"
            else:reason=f"bbox={obj.get('bbox')}; position={obj.get('position')}; rotation={obj.get('rotation_angle')}"
        else:reason=f'Atomic check {kind} failed for {item}'
        ledger.append(dict(subject=item,constraint=kind,violation_reason=reason))
    return ledger


def relation_details(run):
    # Existing conservative text extraction and explicit left/right/near rules stay unchanged.
    _,previous=E.relations(run)
    objs=run['scene']['objects'];diag=E.features(run)['room_diagonal'];idx=E.indexed(objs)
    verifier=orientation_verifier();rows=[]
    expanded=[]
    for entry in previous:
        if entry['relation']=='front_back':
            for name,pattern in [('front of',r'\bin front of\b'),('behind',r'\bbehind\b'),('facing',r'\bfacing\b')]:
                if re.search(pattern,entry['description'].lower()):
                    item=dict(entry);item['relation']=name;expanded.append(item)
        else:expanded.append(entry)
    for entry in expanded:
        subject=idx[(entry['label'],entry['occurrence'])];anchor=entry['target']
        targets=[o for o in objs if anchor and (o['label']==anchor or E.canon(o['label'])==E.canon(anchor))]
        relation=entry['relation'];text=entry['description'];target=targets[0] if len(targets)==1 else None
        if relation=='front_back':
            relation='facing' if re.search(r'\bfacing\b',text.lower()) else 'behind' if 'behind' in text.lower() else 'front of'
        if relation=='vertical':
            relation=next((word for word in ('on top of','above','below','under') if word in text.lower()),'vertical')
        row=dict(subject=f"{entry['label']}#{entry['occurrence']}",relation=relation,anchor=anchor,textual_statement=text,
            geometry_test='',expected=relation,actual=None,result='UNEVALUATED',reason=entry['reason'],
            subject_bbox=subject.get('bbox'),anchor_bbox=target.get('bbox') if target else None)
        if entry['evaluatable'] and target:
            gap=E.rect_gap(E.corners(subject),E.corners(target));area=E.intersection(E.corners(subject),E.corners(target))
            sx,sy=subject['position'];tx,ty=target['position']
            test={'left':f'x_subject={sx:.4g} < x_anchor={tx:.4g} - 1e-6','right':f'x_subject={sx:.4g} > x_anchor={tx:.4g} + 1e-6',
                'near':f'edge gap={gap:.4g} <= {0.15*diag:.4g}', 'far':f'edge gap={gap:.4g} >= {0.5*diag:.4g}',
                'adjacent':f'edge gap={gap:.4g} <= {0.05*diag:.4g}; intersection={area:.4g} <= 1e-6',
                'inside':f'subject corners={E.corners(subject)} inside anchor corners={E.corners(target)}'}
            row.update(geometry_test=test[relation],result='SATISFIED' if entry['satisfied'] else 'VIOLATED',actual=relation if entry['satisfied'] else 'not '+relation,reason='Preserved prior explicit geometric rule')
        elif relation=='facing' and target:
            # "facing it" resolves to the unique declared anchor. No implicit ontology relations are added.
            sx,sy=subject['position'];tx,ty=target['position'];yaw=subject.get('rotation_angle')
            if isinstance(yaw,(int,float)) and math.isfinite(yaw):
                obj=SimpleNamespace(x=sx,y=sy,yaw=yaw)
                answer=verifier._check_faces_point(obj,tx,ty)
                angle=verifier._atan2_to_yaw_convention(ty-sy,tx-sx)
                difference=verifier._angle_diff(yaw,angle)
                row.update(geometry_test=f'SceneCritic yaw={yaw:g}; target yaw={angle:.4g}; angular error={difference:.4g} deg <= 75 (coincident centers pass)',
                    actual='facing' if answer=='pass' else 'not facing',result='SATISFIED' if answer=='pass' else 'VIOLATED',reason=f'Exact OrientationVerifier._check_faces_point / _angle_check: {answer}')
        elif relation in ('above','below','on top of','under','vertical'):
            # Report projection information, but NEVER turn 2D intersection into vertical proof.
            intersection=E.intersection(E.corners(subject),E.corners(target)) if target else None
            row.update(geometry_test=f'2D footprint intersection={format(intersection, ".4g") if intersection is not None else "N/A"}; no z_min/z_max or support-surface elevation',reason='UNEVALUATED: no vertical extent; full 3D projection + contact cannot be established. Checked-in evaluator has no textual vertical/support verifier.')
        elif relation in ('front of','behind'):
            row.update(reason='UNEVALUATED: checked-in OrientationVerifier verifies facing directions, not textual front/behind half-spaces. No arbitrary proxy added.')
        rows.append(row)
    evaluated=[r for r in rows if r['result']!='UNEVALUATED']
    return rows,dict(src=sum(r['result']=='SATISFIED' for r in evaluated)/len(evaluated) if evaluated else None,evaluated_relations=len(evaluated),satisfied_relations=sum(r['result']=='SATISFIED' for r in evaluated),violated_relations=sum(r['result']=='VIOLATED' for r in evaluated),unevaluated_relations=len(rows)-len(evaluated))


def png_tables(root,name,rows,columns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed readable font sizes and logical row chunks; expand row heights for wrapped content.
    groups=[rows[i:i+9] for i in range(0,len(rows),9)] or [[]]
    for page,data in enumerate(groups,1):
        widths=[1.9 if k in ('benchmark','condition','adjective','subject','relation','feature','result') else 3.5 if any(t in k for t in ('detail','reason','statement','test','changes','rule')) else 2 for k in columns]
        total=sum(widths);font=10
        def value(v):
            if v is None:return 'N/A'
            if isinstance(v,float):return f'{v:.4f}'
            if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
            return str(v).replace('_',' ')
        cells=[[textwrap.fill(value(row.get(key)),width=max(12,int(width*9))) for key,width in zip(columns,widths)] for row in data]
        lines=[max((cell.count('\n')+1 for cell in row),default=1) for row in cells]
        heights=[.65]+[max(.60,.23*n+.20) for n in lines]
        fig,ax=plt.subplots(figsize=(max(12,total),sum(heights)+1.4));ax.axis('off')
        title=name.replace('_',' ').removesuffix('.png').title()
        ax.set_title(title+(f' — page {page}/{len(groups)}' if len(groups)>1 else ''),fontsize=15,fontweight='bold',loc='left',pad=18)
        labels=[textwrap.fill(k.replace('_',' ').title(),width=max(12,int(w*9))) for k,w in zip(columns,widths)]
        table=ax.table(cellText=cells or [['No available rows']+['']*(len(columns)-1)],colLabels=labels,colWidths=[w/total for w in widths],cellLoc='left',loc='center')
        table.auto_set_font_size(False);table.set_fontsize(font)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#d2dce6');cell.set_linewidth(.5)
            cell.set_height(heights[min(i,len(heights)-1)]/(sum(heights)+.7))
            if i==0:cell.set_facecolor('#17324d');cell.set_text_props(color='white',weight='bold')
            else:
                cell.set_facecolor('#eef3f8' if i%2 else '#ffffff')
                text=cell.get_text().get_text()
                if text=='VIOLATED':cell.set_text_props(color='#af2020',weight='bold')
                elif text=='SATISFIED':cell.set_text_props(color='#14643a',weight='bold')
                elif text=='UNEVALUATED':cell.set_text_props(color='#775c15',weight='bold')
        fig.text(.01,.01,'DESCRIPTIVE • no inference • N/A is distinct from zero',fontsize=10,color='#42566c')
        filename=name if page==1 else name.replace('.png',f'_page_{page:02d}.png')
        fig.savefig(root/filename,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-png',action='store_true');args=parser.parse_args()
    out=C.OUTPUT_ROOT/'bedroom_only/revised_evaluation'
    if out.exists():
        if out.is_symlink():raise ValueError('Refusing linked evaluation output')
        shutil.rmtree(out)  # Clear only the previous revised evaluation, never generated inputs.
    out.mkdir(parents=True)
    originals={str(p):C.digest(p) for p in C.BENCHMARK_ROOT.glob('bedroom/*.json')}
    originals.update({str(p):C.digest(p) for p in C.OUTPUT_ROOT.rglob('*.json') if out not in p.parents})
    originals[str(SOURCE)]=C.digest(SOURCE)
    runs={};inventory=[]
    for bedroom in ('bedroom_1',):
        for condition in C.CONDITIONS:
            # Prefer the coherent completed seven-condition run; never mix determinism replays into samples.
            candidates=[C.OUTPUT_ROOT/'bedroom_1_seven_outputs'/bedroom/condition/'run_qwen35_27b.json',C.OUTPUT_ROOT/'bedroom_only'/bedroom/condition/'run_qwen35_27b.json']
            path=next((p for p in candidates if p.exists()),None)
            if path:
                r=C.read_json(path)
                benchmark=C.read_json(C.BENCHMARK_ROOT/f'bedroom/{bedroom}.json')
                assert r['benchmark_copy']==C.modified_benchmark(benchmark,condition)
                assert r['refinement']==C.REFINEMENT and tuple(c['stage'] for c in r['calls'])==C.STAGES
                assert not any('seed' in c['api_response']['request'] for c in r['calls'])
                runs[(bedroom,condition)]=r
            inventory.append(dict(benchmark=bedroom,condition=condition,status='AVAILABLE' if path else 'MISSING',source=str(path) if path else None))
    support=[]
    for name,exists,sufficient,rule in [('front of',False,False,'No textual front/behind verifier in checked-in source'),('behind',False,False,'No textual front/behind verifier in checked-in source'),('facing',True,True,'Exact _check_faces_point: yaw target=(atan2(dy,dx)+90)%360; angular error <=75; coincident centers pass'),('above',False,False,'No vertical relation verifier; missing z extent'),('below',False,False,'No vertical relation verifier; missing z extent'),('on top of',False,False,'No contact/support verifier; missing z/support surfaces; 2D full intersection is insufficient'),('under',False,False,'No vertical relation verifier; missing z extent')]:
        support.append(dict(relation=name,scenecritic_supported='YES (orientation facing test)' if exists else 'NO explicit textual rule',current_data_sufficient='YES' if sufficient else 'NO',evaluation_rule=rule,used='YES for stated unique-target relations' if sufficient else 'NO',source=str(SOURCE)))
    feat=[];directions=[];semantic_rows=[];ccr=[];collateral_rows=[];cvr=[];violations=[];src=[];srcrows=[];l1=[];overall=[];profiles=[]
    for (bedroom,condition),run in runs.items():
        f=E.features(run);feat.append(dict(benchmark=bedroom,condition=condition,**{k:f[k] for k in FEATURES}))
        v,checks,_=E.validity(run);details=violation_details(run,checks)
        cvr.append(dict(benchmark=bedroom,condition=condition,cvr=v['constraint_violation_rate'],violated_checks=v['violated_checks'],total_checks=v['total_checks'],violation_details='; '.join(d['subject']+' — '+d['constraint']+' — '+d['violation_reason'] for d in details) or 'None'))
        violations.extend(dict(benchmark=bedroom,condition=condition,baseline_or_intervention='BASELINE' if condition=='baseline' else 'ADJECTIVE',**d) for d in details)
        rel,summary=relation_details(run);src.append(dict(benchmark=bedroom,condition=condition,**summary));srcrows.extend(dict(benchmark=bedroom,condition=condition,**r) for r in rel)
    for bedroom in ('bedroom_1',):
        for condition in list(C.CONDITIONS)[1:]:
            a=runs.get((bedroom,'baseline'));b=runs.get((bedroom,condition));prefix=dict(benchmark=bedroom,adjective=condition)
            if not a or not b:
                overall.append(dict(**prefix,status='MISSING',direction_accuracy=None,ccr=None,cvr_delta=None,src=None,occupancy_hamming=None,centroid_shift=None));continue
            l=compare(a,b);l1.append(dict(**prefix,**l));f,g=E.features(a),E.features(b)
            signs=SIGN_MATRIX.get(condition);correct=0;undefined=False
            for key in FEATURES:
                delta=g[key]-f[key] if f[key] is not None and g[key] is not None else None
                good=delta*signs[key]>EPS if signs and delta is not None else None
                if good:correct+=1
                if delta is None:undefined=True
                semantic_rows.append(dict(**prefix,feature=key,baseline=f[key],adjective_value=g[key],delta=delta,expected_direction=('+' if signs[key]>0 else '-') if signs else 'N/A',correct='YES' if good else 'NO' if good is not None else 'N/A',signed_relative_effect=delta/(abs(f[key])+EPS) if delta is not None else None))
            accuracy=correct/4 if signs and not undefined else None
            semantic_score=E.semantic(a,b,condition)
            directions.append(dict(**prefix,semantic_direction_cosine=semantic_score['semantic_direction_cosine'],intended_signed_effect_mean=semantic_score['intended_signed_effect_mean'],correct_features=correct if signs else None,denominator=4 if signs else None,direction_accuracy=accuracy,status='LIMITED operational open-and-regular proxy' if signs and not undefined else 'N/A: undefined feature' if signs else E.UNOBSERVABLE))
            p=E.object_profile(a,b);profiles.extend(dict(**prefix,**r) for r in p)
            relevant,collateral=changes(a,b);total=len(relevant)+len(collateral)
            ratio=len(collateral)/(total+EPS) if signs and total else 0. if signs and not collateral else None
            def display(items):return '; '.join(f"{i['subject']} {i['channel']}: {i['before']} -> {i['after']}" for i in items) or 'None'
            ccr.append(dict(**prefix,ccr=ratio,relevant_changes=display(relevant) if signs else 'N/A: relevance unobservable',collateral_changes=display(collateral) if signs else 'N/A: relevance unobservable',collateral_change_details=display(collateral),relevant_change_count=len(relevant) if signs else None,collateral_change_count=len(collateral) if signs else None,total_monitored_changes=total))
            for item in collateral:collateral_rows.append(dict(**prefix,classification='COLLATERAL' if signs else 'UNCLASSIFIED',**item))
            v=next(r for r in cvr if r['benchmark']==bedroom and r['condition']==condition);base=next(r for r in cvr if r['benchmark']==bedroom and r['condition']=='baseline')
            ss=next(r for r in src if r['benchmark']==bedroom and r['condition']==condition)
            overall.append(dict(**prefix,status='AVAILABLE',direction_accuracy=accuracy,ccr=ratio,cvr_delta=v['cvr']-base['cvr'],src=ss['src'],occupancy_hamming=l['occupancy_hamming'],centroid_shift=l['centroid_shift']))
    for item in inventory:
        if item['status']!='MISSING':continue
        bedroom,condition=item['benchmark'],item['condition']
        feat.append(dict(benchmark=bedroom,condition=condition,**dict.fromkeys(FEATURES)))
        cvr.append(dict(benchmark=bedroom,condition=condition,cvr=None,violated_checks=None,total_checks=0,violation_details='MISSING SCENE: no checks performed'))
        src.append(dict(benchmark=bedroom,condition=condition,src=None,evaluated_relations=None,satisfied_relations=None,violated_relations=None,unevaluated_relations=None))
        if condition!='baseline':
            prefix=dict(benchmark=bedroom,adjective=condition)
            directions.append(dict(**prefix,semantic_direction_cosine=None,intended_signed_effect_mean=None,correct_features=None,denominator=4 if condition in SIGN_MATRIX else None,direction_accuracy=None,status='MISSING SCENE'))
            ccr.append(dict(**prefix,ccr=None,relevant_changes='MISSING SCENE',collateral_changes='MISSING SCENE',collateral_change_details='MISSING SCENE: no comparison performed',relevant_change_count=None,collateral_change_count=None,total_monitored_changes=None))
            for key in FEATURES:
                sign=SIGN_MATRIX.get(condition,{}).get(key)
                semantic_rows.append(dict(**prefix,feature=key,baseline=None,adjective_value=None,delta=None,expected_direction=('+' if sign>0 else '-') if sign else 'N/A',correct='MISSING SCENE',signed_relative_effect=None))
    aggregates=[]
    for condition in list(C.CONDITIONS)[1:]:
        rows=[r for r in overall if r['adjective']==condition and r['status']=='AVAILABLE']
        aggregates.append(dict(adjective=condition,n_benchmarks=len(rows),**{key:C.finite_mean([r[key] for r in rows]) for key in ('direction_accuracy','ccr','cvr_delta','src','occupancy_hamming','centroid_shift')}))
    tables={'overall_summary':overall,'direction_accuracy':directions,'semantic_features':feat,'semantic_feature_deltas':semantic_rows,'ccr_details':ccr,'ccr_collateral_details':collateral_rows,'cvr_summary':cvr,'cvr_details':violations,'src_summary':src,'src_details':srcrows,'scenecritic_relation_support':support,'input_inventory':inventory,'l1_behavioral_results':l1,'per_object_results':profiles,'bedroom_1_descriptive_summary':aggregates}
    for stem,rows in tables.items():save(out,stem,rows)
    report=dict(scope='bedroom_1 only: seven conditions; existing outputs only',benchmarks=['bedroom_1'],requested_scenes=7,scenes_available=len(runs),scenes_missing=7-len(runs),features=list(FEATURES),expected_directions=SIGN_MATRIX,denominator=4,l2='NOT APPLICABLE; no inferential analysis',api_calls=0,
        expectation_assumption='Spacious/well-organized operationally defined as more open AND more regular; crowded/cluttered as more compact AND less regular. Not universal adjective entailments. Cross-feature scores are limited proxies.',
        source_hash=C.digest(SOURCE),source=str(SOURCE),tables=tables,limitations=['Only bedroom_1 is evaluated; results cannot generalize across rooms','One output per condition; no stochastic replicates; prior API non-reproducibility','Four-sign matrix needs disclosed operational assumptions','No z geometry; vertical full-intersection/contact unavailable','Facing rule is exact checked-in SceneCritic; front/behind are not implemented there'],preserved_input_hashes=originals)
    C.write_json(out/'evaluation_report.json',report)
    report_text=['# Bedroom_1 adjective evaluation','',f"Available scenes: **{len(runs)}/7**. Missing scenes: **{7-len(runs)}**. No inference/API calls. Input scene JSONs and original benchmarks are preserved. Old evaluation outputs remain historical; only the previous revised-evaluation directory is replaced.",'',
        '## What changed','Four retained features only; fixed denominator 4; auditable structural channels and atomic violations; exact checked-in facing verifier; unsupported relations excluded; readable paginated PNGs. Existing L1 comparison function is unchanged. L2 inference is NOT APPLICABLE.','',
        '## Feature definitions','Free space = 1 - union area of clipped footprints / room area. Clearance = mean nearest rectangle-edge gap / room diagonal. Regularity = 1/(1+population CV of nearest-centroid distances). Spacing consistency = 1/(1+population CV of nearest-edge gaps); all-zero gaps remain N/A. All use existing axis-aligned footprint conventions.','',
        '## Expected-direction matrix and scientific limits',report['expectation_assumption'],'Four-feature cosine uses raw dimensionless feature-change vectors and the four expected signs, yielding N/A for undefined/zero-length vectors. Signed relative effects remain delta/(abs(baseline)+1e-6) and can inflate near a zero baseline. Direction Accuracy is correct/4, never divided by the number of available features. Zero delta is incorrect (strict 1e-6 threshold). An undefined retained feature makes the full score N/A rather than silently reducing the denominator. Bright/quiet receive no geometric semantic score.','',
        E.table([dict(adjective=k,**{f:'+' if v[f]>0 else '-' for f in FEATURES}) for k,v in SIGN_MATRIX.items()],['adjective',*FEATURES]),'',
        '## Direction accuracy',E.table(directions,['benchmark','adjective','correct_features','denominator','direction_accuracy','semantic_direction_cosine','intended_signed_effect_mean','status']),'',
        '## Baseline vs adjective features',E.table(semantic_rows,['benchmark','adjective','feature','baseline','adjective_value','delta','expected_direction','correct']),'',
        '## CCR: exact structural changes','Relevant channels are per-object position/bbox/rotation/anchor; collateral channels are inventory multiplicities, total-count indicator, type, and room dimensions. The prior equal-weight change-channel definition is preserved. Changed free text is not a numerical channel. CCR is collateral/(total+1e-6); zero collateral gives 0 for observable geometric adjectives. Bright/quiet relevance remains N/A.','',E.table(ccr,['benchmark','adjective','ccr','relevant_changes','collateral_change_details','total_monitored_changes']),'',
        '## CVR: auditable atomic checks','The prior atomic check set and denominators are preserved: scene structure, inventory counts, fields, geometry, rotation, bounds, position/bbox center, anchors and all pairwise footprint nonoverlap checks. These are not silently replaced with SceneCritic weighted overall quality scores. Full 3D collision feasibility is unknown; touching/close objects are not violations unless the explicit footprint test fails.','',E.table(cvr,['benchmark','condition','cvr','violated_checks','total_checks']),'',E.table(violations,['benchmark','condition','subject','constraint','violation_reason']),'',
        '## SRC: all evaluated and unevaluated statements','SATISFIED and VIOLATED enter the denominator; UNEVALUATED does not. Left/right/near/far/adjacent/inside retain the previous explicit rules. Near gap <=0.15 room diagonal, far >=0.5, adjacent <=0.05 with no footprint overlap; left/right use x; inside uses rectangle containment. Only explicit unique anchor targets are accepted.','',E.table(src,['benchmark','condition','src','evaluated_relations','satisfied_relations','violated_relations','unevaluated_relations']),'',E.table(srcrows,['benchmark','condition','subject','relation','anchor','geometry_test','result','reason']),'',
        '## SceneCritic implementation and support',f'Inspected `{SOURCE}` OrientationVerifier (lines 970–1275), overlap verifiers (778–968), SceneCritic README format-adaptation warning (line 210), and pipeline rotation conventions. The pipeline says 0 degrees toward ego/camera and 90 degrees right; the checked-in verifier uses 0=-y,90=+x. The scenes have 2D centers and finite rotation angles, permitting this established yaw mapping.','',E.table(support,['relation','scenecritic_supported','current_data_sufficient','evaluation_rule','used']),'',
        '## Vertical intersection/contact methodology','The current saved schema has bbox=[x,y,w,h], position=[x,y], no z_min/z_max, object height or surface elevation. Above/below/under require vertical ordering, and on-top needs vertical ordering plus support-surface intersection/contact. The inspected evaluator contains no corresponding textual vertical rule. Its 2D-to-3D converter (lines 440–466) infers/defaults height and sets center z=height/2, grounding every object. Those synthetic heights and elevations do not establish actual shelf elevation or support contact, so they are not substituted for missing measured geometry. We expose 2D projection intersection where possible but DO NOT substitute it for a full 3D relation. Those statements remain UNEVALUATED; full-intersection/contact cannot be established from these inputs.','',
        '## Orientation methodology','Facing executes the original OrientationVerifier._check_faces_point, _atan2_to_yaw_convention and _angle_check through AST-only loading. Desired yaw=(atan2(dy,dx)+90) mod 360; error<=75 degrees passes, error>=150 hard-fails, intermediate error soft-fails. Center differences below 0.01 on both axes pass per original logic. We add no ontology-inferred relationships or arbitrary front/behind half-spaces: the checked-in verifier has no textual front/behind test.','',
        '## Bedroom_1 input availability',E.table(inventory,['benchmark','condition','status','source']),'',
        '## Bedroom_1 descriptive summary','Each row describes one adjective contrast against the bedroom_1 baseline. There is one saved output per condition; these are not estimates of stochastic variability. No p-values or confidence intervals.','',E.table(aggregates,['adjective','n_benchmarks','direction_accuracy','ccr','cvr_delta','src']),'',
        '## Limitations',*['- '+s for s in report['limitations']],'',
        '## Reproduction','```bash\n/home/meetrathi-97/miniconda3/envs/dspy/bin/python VLM_Probe/revised_output_probe_qwen35_27b.py\n```','']
    (out/'FINAL_EVALUATION_REPORT.md').write_text('\n\n'.join(report_text))
    specs=[('01_overall_summary.png',overall,['benchmark','adjective','status','direction_accuracy','ccr','cvr_delta','src']),('02_direction_accuracy.png',directions,['benchmark','adjective','correct_features','denominator','direction_accuracy','status']),('03_semantic_feature_values.png',feat,['benchmark','condition',*FEATURES]),('04_semantic_feature_deltas.png',semantic_rows,['benchmark','adjective','feature','delta','expected_direction','correct']),('05_ccr_summary.png',ccr,['benchmark','adjective','ccr','relevant_change_count','collateral_change_count','total_monitored_changes']),('06_ccr_collateral_details.png',collateral_rows,['benchmark','adjective','subject','channel','before','after']),('07_cvr_summary.png',cvr,['benchmark','condition','cvr','violated_checks','total_checks']),('08_cvr_violation_details.png',violations,['benchmark','condition','subject','constraint','violation_reason']),('09_src_summary.png',src,['benchmark','condition','src','evaluated_relations','violated_relations','unevaluated_relations']),('10_src_relation_details.png',srcrows,['benchmark','condition','subject','relation','anchor','geometry_test','result','reason']),('11_baseline_vs_adjective.png',semantic_rows,['benchmark','adjective','feature','baseline','adjective_value','delta','expected_direction','correct']),('12_scenecritic_relation_support.png',support,['relation','scenecritic_supported','current_data_sufficient','evaluation_rule','used'])]
    if not args.skip_png:
        for name,rows,cols in specs:png_tables(out,name,rows,cols)
    assert all(C.digest(p)==sha for p,sha in originals.items()),'Original benchmark or generated output changed'
    files=sorted(p.name for p in out.iterdir())
    with (out/'FINAL_EVALUATION_REPORT.md').open('a') as file:
        file.write('\n## Generated files\n\n'+'\n'.join('- `'+name+'`' for name in files+['artifact_manifest.json'])+'\n')
    C.write_json(out/'artifact_manifest.json',dict(files=files+['artifact_manifest.json'],api_calls=0,inputs_unchanged=True))
    print('Revised output directory:',out);print('Available:',len(runs),'/7; missing:',7-len(runs));print('Artifacts:',len(files)+1)


if __name__=='__main__':main()
