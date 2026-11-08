"""Reproduce the fixed comparisons and precision diagnostics in a fresh run.

All inputs are pinned by content; this does not choose another population,
split, predictor, parameter grid or criterion. Existing runs are never replaced.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SUMMARY_FILES = (
    'primary/study-summary.csv', 'primary/registered-regime-summary.csv',
    'primary/score-repartition-sensitivity.csv', 'precision/summary.csv',
    'crossing/summary.csv', 'extension/study-summary.csv', 'composition/summary.csv',
)
RTOL, ATOL = 1e-7, 1e-9
COUNT_COLUMNS = {
    'maps', 'test_maps', 'queries', 'catalog_successes', 'matched_maps', 'full_maps',
    'repartitions', 'exact_saturated_queries', 'recorded_useful_queries',
    'certified_useful_queries', 'certified_broad_queries', 'requires_sort_queries',
    'exact_saturation', 'upper_strictly_tighter', 'certified_useful',
    'certified_broad', 'requires_sort_for_classification', 'narrower_to_wider',
    'wider_to_narrower',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(root, name):
    """Resolve one manifest member without escaping its root or using symlinks."""
    relative = PurePosixPath(name)
    if (not relative.parts or relative.is_absolute() or '..' in relative.parts or
            '\\' in name or relative.as_posix() != name):
        raise ValueError('Unsafe manifest path: ' + name)
    root = Path(root).resolve()
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Manifest path uses a symlink: ' + name)
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or escaping manifest file: ' + name)
    return path


def verify_source():
    """Bind the exact exported files without relying on a Git installation."""
    expected = json.loads(checked_file(ROOT, 'checksums.json').read_text())
    for name, value in expected.items():
        path = checked_file(ROOT, name)
        if digest(path) != value:
            raise ValueError('Source checksum mismatch: ' + name)


def compare_summary(reference, actual):
    """Check ordered scientific summaries, allowing only small float differences.

    Labels, counts, missingness, row order and schema remain exact. Relative
    1e-7 / absolute 1e-9 tolerances cover platform arithmetic, not recalibration,
    changed discrete decisions or statistical resampling. See docs/method.md.
    """
    with Path(reference).open(newline='') as stream:
        expected = list(csv.reader(stream))
    with Path(actual).open(newline='') as stream:
        observed = list(csv.reader(stream))
    if not expected or not observed or expected[0] != observed[0] or len(expected) != len(observed):
        raise AssertionError('Summary schema or row count differs: ' + str(actual))
    header = expected[0]
    if len(set(header)) != len(header):
        raise AssertionError('Duplicate summary columns')
    for row_number, (left, right) in enumerate(zip(expected[1:], observed[1:]), 2):
        if len(left) != len(header) or len(right) != len(header):
            raise AssertionError('Malformed summary row: ' + str(row_number))
        for column, a, b in zip(header, left, right):
            if a == b:
                continue
            exact = (column in COUNT_COLUMNS or column.endswith(('_positive', '_negative', '_ties'))
                     or re.fullmatch(r'[+-]?\d+', a) is not None)
            try:
                x, y = float(a), float(b)
            except ValueError:
                equal = False
            else:
                equal = (not exact and ((math.isnan(x) and math.isnan(y)) or x == y or
                         (math.isfinite(x) and math.isfinite(y) and
                          math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL))))
            if not equal:
                raise AssertionError(f'{actual}: row {row_number}, {column}: {b!r} != {a!r}')
    return {'rows': len(expected)-1, 'columns': len(header), 'status': 'passed'}


def verify_summaries(out):
    """Compare the selected substantive findings; exclude historical timings."""
    return {name: compare_summary(checked_file(ROOT/'reference', name), checked_file(out, name))
            for name in SUMMARY_FILES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exact', action='store_true',
                        help='Also require historical dense-output byte hashes (optional diagnostic)')
    args = parser.parse_args(argv)
    verify_source()
    out = ROOT / 'run'
    if out.exists():
        raise FileExistsError('Use a fresh source copy; run/ already exists')
    start = time.perf_counter()
    out.mkdir()
    (out / 'primary').mkdir()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ROOT/'study'), MPLCONFIGDIR=str(out/'mpl-cache'),
               XDG_CACHE_HOME=str(out/'xdg-cache'))
    # Verification creates only the provenance view expected by the extraction
    # loader. Exact originals must already exist; no network occurs here.
    commands = [[sys.executable, 'acquire.py'],
                *[[sys.executable, 'study/'+s+'.py'] for s in
                  ['records','catalog','compare','summarize','diagnostics','extension','composition','figures']]]
    for i, command in enumerate(commands):
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (out/f'step-{i}.log').write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(f'{command} failed; see run/step-{i}.log\n'+(run.stdout+run.stderr)[-2500:])
        print('Completed', ' '.join(command[1:]), flush=True)
    summaries = verify_summaries(out)
    checks = None
    if args.exact:
        hashes = json.loads(checked_file(ROOT, 'reference/checksums.json').read_text())
        checks = {name: digest(checked_file(out, name))==value for name,value in hashes.items()}
        if not all(checks.values()):
            raise AssertionError({n:p for n,p in checks.items() if not p})
    derived = json.loads(checked_file(ROOT, 'data/derived-hashes.json').read_text())
    inputs = Path(env.get('ALLOCATION_INPUTS', ROOT/'data/cache'))
    if not all(digest(checked_file(inputs, n))==h for n,h in derived.items()):
        raise AssertionError('Derived input mismatch')
    receipt = dict(status='passed', scientific_summaries=summaries,
                   summary_tolerance=dict(rtol=RTOL, atol=ATOL, counts_labels_missingness='exact'),
                   exact_non_timing_outputs=checks,
                   derived_input_hashes=derived, source_checksum=digest(ROOT/'checksums.json'),
                   elapsed_seconds=time.perf_counter()-start, python=sys.version,
                   scope='Fixed-cohort computation and diagnostics; no independent population validation. Timings may differ.')
    (out/'reproduction-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
