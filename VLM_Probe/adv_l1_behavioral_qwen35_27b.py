"""Baseline versus each adjective, for every category and matched seed."""
from __future__ import annotations

import argparse
from pathlib import Path
import adjective_common_qwen35_27b as C


def compare(a, b):
    ca, cb = a['calls'], b['calls']
    first = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x['output_text'].strip() != y['output_text'].strip()), None)
    structure = len(ca) != len(cb) or [x['stage'] for x in ca] != [x['stage'] for x in cb]
    if first is None and len(ca) != len(cb):
        first = min(len(ca), len(cb))
    placement_a = next(x for x in ca if x['stage'] == 'placement')
    placement_b = next(x for x in cb if x['stage'] == 'placement')
    return dict(C.pair_metrics(a, b), first_divergent_call=first, first_divergent_stage=ca[first]['stage'] if first is not None and first < len(ca) else (cb[first]['stage'] if first is not None and first < len(cb) else None), structural_divergence=structure, placement_changed=placement_a['output_text'].strip() != placement_b['output_text'].strip(), baseline_call_count=len(ca), adjective_call_count=len(cb), baseline_object_count=len(a['scene']['objects']), adjective_object_count=len(b['scene']['objects']), baseline_legality=a['labels']['legality'], adjective_legality=b['labels']['legality'], baseline_oob=a['labels']['oob'], adjective_oob=b['labels']['oob'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    manifest, groups = C.load_runs(args.input_dir)
    rows = []
    for category in manifest['categories']:
        for condition in manifest['conditions']:
            if condition == 'baseline':
                continue
            for seed in manifest['seeds']:
                a, b = groups[(category, 'baseline')].get(seed), groups[(category, condition)].get(seed)
                if a is None or b is None:
                    benchmark = next(s['benchmark'] for s in manifest['benchmarks'] if s['category'] == category)
                    missing = {key: None for key in (*C.METRICS, 'first_divergent_call', 'first_divergent_stage', 'structural_divergence', 'placement_changed', 'baseline_call_count', 'adjective_call_count', 'baseline_object_count', 'adjective_object_count', 'baseline_legality', 'adjective_legality', 'baseline_oob', 'adjective_oob')}
                    rows.append(dict(benchmark=benchmark, category=category, condition=condition, baseline_condition='baseline', seed=seed, status='unavailable_failed_generation', **missing))
                    continue
                rows.append(dict(benchmark=a['benchmark'], category=category, condition=condition, baseline_condition='baseline', seed=seed, status='success', **compare(a, b)))
    C.write_summary(args.output_dir or args.input_dir / 'results', 'behavioral_summary', rows, {'mock': manifest['mock'], 'experiment_identity': manifest['identity'], 'object_set_normalization': 'lowercase and strip trailing _N', 'centroid_matching': 'exact label + occurrence order; null when no shared labels', 'first_divergent_call_indexing': 'zero-based', 'occupancy': 'repository 5x5 category grid; strict >10% cell coverage, largest intersection wins', 'legality': 'no overlap, independent of bounds'})
    print(f'Wrote {len(rows)} baseline/adjective comparisons; mock={manifest["mock"]}')


if __name__ == '__main__':
    main()
