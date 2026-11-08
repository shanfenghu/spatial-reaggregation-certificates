# Selected tables and numerical findings

These are our scientific outputs and reference definitions. They are included
alongside the complete small summary CSVs in [reference/](../reference/).
They are not copies of provider input records.

- [Method comparison](method-comparison.csv): covered object, constraints and computation.
- [Core notation](notation.csv): symbols, domains and meanings.
- [Cohort and assignments](cohort-summary.csv): records, eligible maps and fixed partitions.
- [Allocation comparison](allocation-comparison.csv): primary counts, width and score.
- [Geographic stress](geographic-stress.csv): southern counts, width and score.
- [Query extension](query-extension.csv): added/union counts, width and score.
- [Matched-map decomposition](matched-map-decomposition.csv): selection and paired changes.
- [Component costs](component-costs.csv): historical calibration, setup, query and diagnostic seconds.

Widths/scores average queries within maps, then maps. Counts refer to whole
allocations or complete catalogues as labelled; they are not equal-coverage
comparisons. Added queries reuse maps and do not establish new-population
coverage. The coherent point row has no interval assurance. Timing observations
come from one recorded host and should vary on other hosts. See
[the method](../docs/method.md) for the assumptions and adverse findings.

After the root setup, provider acquisition and full reproduction, regenerate
the six computed CSVs without refitting:

```sh
PYTHONPATH=study .venv/bin/python study/tables.py
```

This writes `run/tables/` from `run/primary/study-summary.csv`,
`run/primary/cost-summary.csv`, `run/extension/study-summary.csv`,
`run/composition/summary.csv`, fixed assignments and aggregate counts reconstructed
by the existing cohort loader. It refuses an existing destination. To format the
included historical summaries instead, with acquired/extracted inputs available:

```sh
PYTHONPATH=study .venv/bin/python study/tables.py --source reference --output run/reference-tables
```

`method-comparison.csv` and `notation.csv` are authored reference definitions,
retained directly rather than estimated from data. The comparison uses matched
predictors, totals, geometry and covariates; allocation sets impose nonnegativity
and known source sums. Projection applies on the same candidate support when
the set contains its reference allocation. Here n counts locations and E counts
stored within-source pairs; geometry construction and calibration LPs are
additional work.

Required scholarly credit for these adaptations: Blanchet and Murthy (2019),
[Quantifying Distributional Model Risk via Optimal Transport](https://doi.org/10.1287/moor.2018.0936);
Gao and Kleywegt (2023),
[Distributionally Robust Stochastic Optimization with Wasserstein Distance](https://doi.org/10.1287/moor.2022.1275);
Chen, Kuhn and Wiesemann (2024),
[Data-Driven Chance Constrained Programs over Wasserstein Balls](https://doi.org/10.1287/opre.2022.2330).
The reconciled sets adapt the joint set in Principato et al.,
[Conformal Prediction for Hierarchical Data, version 3, 27 October 2025](https://arxiv.org/html/2411.13479v3).
Agency data retain the credit and terms in [NOTICE.md](../NOTICE.md).
