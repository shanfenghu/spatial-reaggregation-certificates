"""Reproduce the fixed comparisons and precision diagnostics in a fresh run.

All inputs are pinned by content; this does not choose another population,
split, predictor, parameter grid or criterion. Existing runs are never replaced.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


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


def main():
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
    hashes = json.loads(checked_file(ROOT, 'reference/checksums.json').read_text())
    checks = {name: digest(checked_file(out, name))==value for name,value in hashes.items()}
    if not all(checks.values()):
        raise AssertionError({n:p for n,p in checks.items() if not p})
    derived = json.loads(checked_file(ROOT, 'data/derived-hashes.json').read_text())
    inputs = Path(env.get('ALLOCATION_INPUTS', ROOT/'data/cache'))
    if not all(digest(checked_file(inputs, n))==h for n,h in derived.items()):
        raise AssertionError('Derived input mismatch')
    receipt = dict(status='passed', exact_non_timing_outputs=checks,
                   derived_input_hashes=derived, source_checksum=digest(ROOT/'checksums.json'),
                   elapsed_seconds=time.perf_counter()-start, python=sys.version,
                   scope='Fixed-cohort computation and diagnostics; no independent population validation. Timings may differ.')
    (out/'reproduction-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
