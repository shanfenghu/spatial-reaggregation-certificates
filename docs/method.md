# Model and reproduction routes

One map has candidate points in disjoint sources, each with a known nonnegative
total. `study/engine.py:geometry` expects a reset zero-based pandas frame with
`source`, `truth`, `weight`, `x`, `y` columns. `truth` supplies source sums and a
diagnostic reference; redistributing it within a source at fixed total leaves
the predictor and endpoints unchanged. `x/y` are projected kilometres.
Invalid or zero-total weights trigger a uniform allocation within the source.
Positive map mass and a positive within-source distance scale are required;
zero-mass and single-point identities are checked separately by the workflow.

```python
import sys
sys.path.insert(0, 'study')
import numpy as np
import pandas as pd
from engine import geometry, endpoint_scan
frame = pd.DataFrame(dict(source=['a','a','b','b'], x=[0.,1.,0.,1.],
    y=[1.,1.,0.,0.], truth=[40.,60.,20.,0.], weight=[1.,1.,1.,1.]))
g = geometry(frame)
q = np.array([1,0,0,0])
assert abs(endpoint_scan(g,q,1/6,False)[0]-30.) < 1e-9
assert abs(endpoint_scan(g,q,1/6,True)[0]-70.) < 1e-9
```

Mass fractions use the known map total M; distances use the largest within-source
distance D. The budget is dimensionless, not a common physical displacement
tolerance. Binary membership must follow candidate order. With continuous mass
and unlimited receivers, cheapest finite crossings can be chosen independently;
sorting and fractional filling attain both endpoints. Indivisible mass, finite
receiver capacities, uncertain source totals or extra overlapping margins change
the problem. Construction/storage uses E=sum(source size squared) arcs; extraction
plus sorting costs O(E+n log n). Calibration still solves transport LPs.
The endpoint APIs reject fractional/nonfinite or mis-sized query indicators
before conversion and check each LP coupling for finite/nonnegative flow,
row conservation, cost and objective agreement. Zero and infinite budgets are
supported. Numerical witness checks supplement rather than prove optimality.
Standalone and comparison LP entry points consistently use one HiGHS thread;
importing comparison helpers after a standalone solve remains supported.

`study/precision.py` bounds the residual uncrossed mass using a feasible
proportional crossing plan. The comparison bound in `crossing_bounds.py` uses
minimum/maximum positive costs. Free crossings fill first and absent destinations
have infinite cost. These are deterministic diagnostics, not confidence bands,
new calibration or improved endpoints. Unequal-cost slack and unresolved
classification remain possible.

The fixed rank rule uses k=ceil((m+1)(1-alpha)), with infinity when k>m.
It covers one uniformly selected map marginally over a fixed frame's random
assignment. Allocation containment implies all binary-query answers on that
same support; registered-catalogue coverage alone does not protect later queries.
Reconciled joint sets also support projections. Southern selection has no
automatic rank guarantee. Repartitions reuse fixed scores, not new populations.

Run the root reproduction command for these routes, in order:

- `records.py` extracts the three fixed derived inputs and preserves blanks.
- `catalog.py` reconstructs every registered mask and alias; `protocol.json`
  supplies fixed splits, grids, scoring and weighting, without selecting a new design.
- `compare.py` executes both primary and southern comparisons with the same
  predictor/information. `summarize.py` independently recomputes point/reference/
  coarse answers and catalogue metrics. These regenerate `reference/primary/`
  summaries and the exhaustive primary outcomes.
- `diagnostics.py` computes both certificates for all 1,627 recorded test queries,
  reproducing `reference/crossing/summary.csv` and `reference/precision/summary.csv`.
- `extension.py` changes radii to 1.5/6 miles at existing centres, with frozen
  original fits and separately declared expanded-catalogue controls. It regenerates
  `reference/extension/study-summary.csv` and its detailed outcomes.
- `composition.py` separates map selection from paired changes using equal map
  weights and original normalisers; it regenerates `reference/composition/summary.csv`.
- `figures.py` renders the two-source example, width/score comparison and certificate
  regimes from the reproduced quantities. Graphics metadata may differ between
  runs; geometry and numerical encodings determine correspondence.

`reference/primary/cost-summary.csv` and
`reference/crossing/recorded-component-accounting.csv` preserve historical
single-host timing observations. Fresh timings are written under `run/primary/`;
they should vary rather than match historical seconds. Shared loading, extraction,
geometry and training are counted once. Diagnostic scoring requires test references
and is distinct from inference. Faster endpoints do not establish a deployment
speedup; the box can be faster on this workload.
The small `registration-costs.json`, `input-costs.json` and extension
`cost-components.json` records retain the corresponding original measurements;
fresh registration and extraction times are recorded separately during a run.

Default reproduction compares seven scientific summary CSVs in `reference/`
(primary results, registered regimes, score repartitions, precision, crossing,
changed-radius results and matched-map composition). Floating quantities use
rtol=1e-7 and atol=1e-9; labels, counts, missingness, schema and ordered rows stay
exact. This is small relative to reported effects and the four-decimal summary
precision, while allowing platform arithmetic. For example, a width ratio of
0.7942 tolerates about 8e-8, far below its difference from 1. No statistical
resampling tolerance is needed: assignments and seeds remain fixed. Changed
coverage counts or substantive effects fail rather than being absorbed as noise.

Exact source/input checks bind provenance, the query catalogue binds the fixed
design, and internal feasibility checks retain their correctness tolerances.
`reproduce.py --exact` additionally compares historical dense non-timing bytes
using `reference/checksums.json`. This optional diagnostic can differ across
platforms without changing scientific findings; investigate mismatches rather
than editing expected hashes. Timings and graphics metadata are not compared.
Independent arithmetic
uses rtol1e-11/atol1e-8 in quantity units. Coverage classification allows 2e-7 M
for queries and 1e-10 for map-score comparisons, without recalibrating thresholds.
Conic/LP feasibility uses 2e-7 normalised mass. These internal checks are retained alongside the summary comparison; they do
not establish validity for a new population.
