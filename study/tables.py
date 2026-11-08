"""Export readable selected tables from recorded summaries and the fixed design.

This formats existing quantities; it does not refit, recalibrate or change the
cohort. Historical timing summaries describe one host and are not byte targets
for future hosts. Run after the root reproduction with its provider input copy.
"""
from pathlib import Path
import argparse
import csv
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'transport_scan': 'Transport (ours)', 'transport_lp': 'Transport LP',
    'reconciled_box': 'Reconciled box', 'reconciled_ellipse': 'Reconciled ellipsoid',
    'aggregate_global': 'Aggregate global', 'aggregate_spatial': 'Aggregate spatial',
    'label_free': 'Label-free transport', 'coarse_only': 'Coarse-only',
    'coherent_point': 'Coherent point',
}


def write_table(destination, name, header, rows):
    """Write one ordered table using portable CSV and explicit column labels."""
    with (destination/name).open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def export(source, destination):
    """Format fixed comparisons, costs, extension, composition and cohort sizes."""
    destination.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(source/'primary/study-summary.csv')
    for run, filename in [('primary', 'allocation-comparison.csv'),
                          ('southern_stress', 'geographic-stress.csv')]:
        rows = []
        methods = [m for m in LABELS if m != 'transport_lp' and
                   (run == 'primary' or m not in ['coarse_only', 'coherent_point'])]
        for domain in ['census', 'egrid']:
            frame = summary[(summary.domain == domain) & (summary.run == run)].set_index('method')
            for method in methods:
                row = frame.loc[method]
                n = int(row.test_maps)
                whole = '--' if pd.isna(row.whole_map_coverage) else f'{round(row.whole_map_coverage*n)}/{n}'
                rows.append([domain, LABELS[method], whole, f'{int(row.catalog_successes)}/{n}',
                             f'{row.width_ratio:.4f}', f'{row.interval_score:.4f}'])
        write_table(destination, filename,
                    ['domain', 'method', 'allocation_maps_covered', 'catalogues_covered',
                     'mean_normalised_width', 'mean_interval_score'], rows)

    extended = pd.read_csv(source/'extension/study-summary.csv')
    rows = []
    for domain in ['census', 'egrid']:
        frame = extended[extended.domain == domain].set_index(['method', 'subset'])
        for method in ['transport_scan', 'reconciled_box', 'reconciled_ellipse',
                       'aggregate_global', 'aggregate_spatial', 'coarse_only']:
            added, union = frame.loc[(method, 'new'), :], frame.loc[(method, 'union'), :]
            label = LABELS[method]
            if method.startswith('aggregate_'):
                label += ' (fixed/expanded)'
                for subset in ['new', 'union']:
                    fixed, expanded = frame.loc[(method, subset), :], frame.loc[(method+'_expanded', subset), :]
                    assert all(fixed[k] == expanded[k] for k in
                               ['test_maps', 'catalog_successes', 'width_ratio', 'interval_score'])
            rows.append([domain, label, f'{int(added.catalog_successes)}/{int(added.test_maps)}',
                         f'{int(union.catalog_successes)}/{int(union.test_maps)}',
                         f'{added.width_ratio:.4f}', f'{added.interval_score:.4f}'])
    write_table(destination, 'query-extension.csv',
                ['domain', 'method', 'added_catalogues_covered', 'union_catalogues_covered',
                 'added_mean_normalised_width', 'added_mean_interval_score'], rows)

    composition = pd.read_csv(source/'composition/summary.csv')
    rows = [[LABELS[r.baseline], int(r.matched_maps), f'{r.selection_component:.4f}',
             f'{r.within_map_component:.4f}', f'{r.observed_unmatched_change:.4f}',
             f'{r.paired_change_median:.4f}'] for r in composition.itertuples()
            if r.domain == 'egrid' and r.baseline in ['reconciled_box', 'reconciled_ellipse']]
    write_table(destination, 'matched-map-decomposition.csv',
                ['baseline', 'matched_maps', 'selection', 'within_map', 'total', 'median'], rows)

    costs = pd.read_csv(source/'primary/cost-summary.csv')
    rows = [[r.domain, LABELS[r.method], f'{r.calibration_score_seconds_including_setup:.3g}',
             f'{r.test_inference_setup_seconds:.3g}', f'{r.endpoint_seconds:.3g}',
             f'{r.test_reference_score_diagnostic_seconds:.3g}'] for r in costs.itertuples()
            if r.run == 'primary' and r.method not in ['coarse_only', 'coherent_point']]
    write_table(destination, 'component-costs.csv',
                ['domain', 'method', 'calibration_seconds_including_setup',
                 'test_setup_seconds', 'query_seconds', 'test_reference_diagnostic_seconds'], rows)

    # Recover cohort sizes from provider inputs using the existing selection.
    # Only aggregate counts enter this table; provider records remain local.
    from design import load_frames
    domains, _, audits, _ = load_frames()
    protocol = json.loads((ROOT/'protocol.json').read_text())['splits']
    values = {}
    for domain in ['census', 'egrid']:
        split = protocol[domain]
        row = summary[(summary.domain == domain) & (summary.run == 'primary') &
                      (summary.method == 'transport_scan')].iloc[0]
        values[domain] = [audits['census']['blocks'] if domain == 'census' else
                          audits['egrid']['retained_numerical_CONUS_cohort'],
                          'block group / tract' if domain == 'census' else 'county / patch',
                          len(domains[domain]),
                          ' / '.join(str(len(split['primary'][k])) for k in ['train', 'calibration', 'test']),
                          int(row.queries), len(split['southern_stress']['test'])]
    names = ['fine_reference_records', 'source_and_map', 'eligible_maps',
             'primary_training_calibration_test', 'primary_test_queries', 'southern_test_maps']
    write_table(destination, 'cohort-summary.csv', ['quantity', 'census', 'energy'],
                [[name, values['census'][i], values['egrid'][i]] for i, name in enumerate(names)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'run')
    parser.add_argument('--output', type=Path, default=ROOT/'run/tables')
    args = parser.parse_args()
    export(args.source, args.output)
