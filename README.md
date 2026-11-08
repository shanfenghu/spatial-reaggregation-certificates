# Conserved allocation query bounds

Compute exact binary spatial-query ranges over continuous allocations that
preserve known source totals. A shared transport-cost budget restricts movement
within each source. The examples compare allocation containment with coverage
of registered query answers, including broad ranges, missed answers and costs.

Use Python 3.11 and uv on Linux or macOS and allow 2 GiB RAM and 1 GiB free disk.
No accelerator or account is needed. Dependencies and original data require
network access initially; all tests and computations then run offline.

```sh
uv venv .venv --python python3.11
uv pip install --python .venv/bin/python --require-hashes -r requirements.txt
PYTHONPATH=study .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python acquire.py --download
.venv/bin/python reproduce.py
```

See [data/README.md](data/README.md) for exact input versions, checksums,
attribution and restoration from an existing retained copy. Provider originals and prepared records are obtained from their providers,
not distributed in Git or release assets. Exhaustive generated outcomes stay local. A checksum or
download URL alone is not a backup of the exact inputs.

The reproduction refuses an existing `run/`; use a new source copy to repeat it.
It reconstructs the 1,668 registered masks, executes the fixed primary and
southern comparisons, checks endpoints against LPs and conic residuals, computes
1,000 fixed-score repartitions, and reproduces the changed-radius workload,
matched-map decomposition and before-sort precision diagnostics. It checks
seven substantive summary CSVs with relative 1e-7 / absolute 1e-9 float
tolerances, exact counts and labels, derived input hashes and independently
recomputed metrics. These tolerances address platform arithmetic; the fixed
splits and random seeds are unchanged. Historical dense-output byte checks are
available separately with `reproduce.py --exact`, as an optional diagnostic. Timings vary with the host and are
not deployment-performance guarantees. Allow several minutes after setup.

`run/figures/` contains the constructed two-source example, primary width/loss
comparison and precision regimes. `run/diagnostic-plots/` contains additional
diagnostics. Every selected reference summary has a generation route documented
in [docs/method.md](docs/method.md); exhaustive outcomes are regenerated locally.
The tests include independent LP oracles, free/absent crossings, proportional-plan
bounds, rank/conic/box controls, matched-map weighting and acquisition failures.

## Selected scientific outputs

The actual selected figures, readable tables and key numerical summaries are
included. Provider-input exclusions do not exclude these original findings.

- Allocation/query illustration: [PDF](figures/allocation-query-precision.pdf), [PNG](figures/allocation-query-precision.png).
- Width and miss-penalised score comparison: [PDF](figures/interval-comparison.pdf), [PNG](figures/interval-comparison.png).
- Budget compatibility and precision regimes: [PDF](figures/precision-regimes.pdf), [PNG](figures/precision-regimes.png).
- [Eight selected table CSVs and their generation routes](tables/README.md).
- [Full-precision scientific summaries and recorded costs](reference/), including
  primary/geographic comparisons, registered regimes, repartitions, crossing and
  precision certificates, changed radii and matched-map composition.

The root reproduction renders the three figures through `study/figures.py` from
the reproduced results. To format the selected computed tables afterwards:

```sh
PYTHONPATH=study .venv/bin/python study/tables.py
```

This writes `run/tables/`. The included files preserve the selected recorded
displays; numerical relationships and plotted geometry define correspondence,
while timing observations and graphics metadata can vary by environment.

The Census task concerns released counts assigned to internal points; the energy
task concerns a numerical-reporting inventory cohort. Neither represents exact
resident locations, a complete physical inventory or physical exposure. Rank
coverage concerns one uniformly selected held-out map in a fixed frame; it does
not establish coverage under geographic selection or new-population validity.
See [NOTICE.md](NOTICE.md) for agency terms; the existing MIT licence covers the
software and documentation, not a new licence for agency data.
