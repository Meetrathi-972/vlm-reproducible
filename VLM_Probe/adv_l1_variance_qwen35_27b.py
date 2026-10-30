"""Repository distance statistic with paired exact permutations and exploratory unrestricted permutations."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import adjective_common_qwen35_27b as C


def summarize(base, adjective, alpha):
    n = len(base)
    if n < 2 or len(adjective) != n:
        raise ValueError('Variance analysis requires at least two matched seeds')
    runs = base + adjective
    matrices = {metric: [[None] * (2*n) for _ in runs] for metric in C.METRICS}
    for i, j in itertools.combinations(range(2*n), 2):
        for metric, value in C.pair_metrics(runs[i], runs[j]).items():
            matrices[metric][i][j] = matrices[metric][j][i] = value
    rows = []
    for metric, matrix in matrices.items():
        primary = C.permutation_test(matrix, n, paired=True)
        exploratory = C.permutation_test(matrix, n, paired=False)
        if primary['p_value'] is None:
            verdict = 'not estimable'
        elif primary['min_achievable_p'] > alpha:
            verdict = 'underpowered by design'
        elif primary['p_value'] <= alpha:
            verdict = 'effect (unadjusted paired permutation p<=alpha)'
        else:
            verdict = 'within noise'
        rows.append(dict(metric=metric, within_baseline=C.finite_mean([matrix[i][j] for i, j in itertools.combinations(range(n), 2)]), within_adjective=C.finite_mean([matrix[i][j] for i, j in itertools.combinations(range(n, 2*n), 2)]), between=C.finite_mean([matrix[i][j] for i in range(n) for j in range(n, 2*n)]), matched_seed_distance=C.finite_mean([matrix[i][i+n] for i in range(n)]), **primary, verdict=verdict, n_seeds=n, alpha=alpha, test_type='paired_exact_seed_swaps', permutation_method='exact within-seed condition swaps', exploratory_unrestricted_statistic=exploratory['test_statistic'], exploratory_unrestricted_p_value=exploratory['p_value'], exploratory_unrestricted_permutations=exploratory['number_of_permutations'], exploratory_unrestricted_min_p=exploratory['min_achievable_p']))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error('alpha must be between 0 and 1')
    manifest, groups = C.load_runs(args.input_dir)
    if len(manifest['seeds']) < 2:
        parser.error('Variance analysis requires at least two matched seeds')
    if len(manifest['seeds']) > 10:
        parser.error('Exact unrestricted enumeration is capped at 10 seeds; split or revise the design explicitly')
    rows = []
    for category in manifest['categories']:
        for condition in manifest['conditions']:
            if condition == 'baseline':
                continue
            paired_seeds = [seed for seed in manifest['seeds'] if seed in groups[(category, 'baseline')] and seed in groups[(category, condition)]]
            benchmark = next(s['benchmark'] for s in manifest['benchmarks'] if s['category'] == category)
            if len(paired_seeds) < 2:
                for metric in C.METRICS:
                    rows.append(dict(benchmark=benchmark, category=category, condition=condition, baseline_condition='baseline', metric=metric, within_baseline=None, within_adjective=None, between=None, matched_seed_distance=None, test_statistic=None, p_value=None, number_of_permutations=0, min_achievable_p=None, verdict='not estimable: fewer than two complete seed pairs', n_seeds=len(paired_seeds), alpha=args.alpha, test_type='paired_exact_seed_swaps', permutation_method='exact within-seed condition swaps', exploratory_unrestricted_statistic=None, exploratory_unrestricted_p_value=None, exploratory_unrestricted_permutations=0, exploratory_unrestricted_min_p=None, complete_paired_seeds=' '.join(map(str, paired_seeds))))
                continue
            base = [groups[(category, 'baseline')][seed] for seed in paired_seeds]
            adjective = [groups[(category, condition)][seed] for seed in paired_seeds]
            rows.extend(dict(benchmark=benchmark, category=category, condition=condition, baseline_condition='baseline', **row, complete_paired_seeds=' '.join(map(str, paired_seeds))) for row in summarize(base, adjective, args.alpha))
    C.write_summary(args.output_dir or args.input_dir / 'results', 'variance_summary', rows, {'mock': manifest['mock'], 'experiment_identity': manifest['identity'], 'statistic': 'mean_between_group_distance - mean_within_group_distance', 'primary_test': 'Exact seed-blocked swaps; preserves the paired experimental design', 'exploratory_test': 'Original repository unrestricted relabeling; exchangeability across seeded pairs is not guaranteed', 'minimum_p_five_pairs': 0.0625, 'minimum_p_five_unrestricted': 2/252, 'multiple_testing': 'p-values are unadjusted across categories, adjectives and metrics; exploratory only', 'missing_centroids': 'null pair distances excluded, matching original NaN handling; inspect object-set metrics too'})
    print(f'Wrote {len(rows)} variance rows; mock={manifest["mock"]}. Five paired seeds cannot attain p<=0.05 with this symmetric statistic.')


if __name__ == '__main__':
    main()
