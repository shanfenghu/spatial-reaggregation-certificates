# Provider inputs and preparation

Obtain the three public inputs directly from their original providers. Neither
original files nor prepared provider records are distributed in this repository
or its release assets. Only source metadata, checksums, fixed analysis settings,
minimal synthetic fixtures and our small computational summaries are included.

## Sources, versions and credit

- U.S. Census Bureau, **2020 TIGER/Line tabulation-block shapefiles, Rhode Island**:
  [product and documentation](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2020.html),
  [original ZIP](https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_44_tabblock20.zip).
  Use `tl_2020_44_tabblock20.zip` (28,351,631 bytes), including its original
  metadata. The analysis uses released 2020 block population counts (`POP20`),
  land area weights (`ALAND20`) and internal points, with block-group source
  totals inside census-tract maps. Counts assigned to internal points
  do not represent exact resident locations.
- U.S. Environmental Protection Agency, **eGRID2023 revision 2 (2025)**,
  Office of Atmospheric Protection, Clean Air and Power Division:
  [product and documentation](https://www.epa.gov/egrid/detailed-data),
  [original workbook](https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx).
  Use `egrid2023_data_rev2.xlsx` (21,213,301 bytes), released 12 June 2025,
  describing reporting year 2023. Extract `PLNT23` field codes for plant identity,
  county, coordinates, fuel, nameplate capacity, annual NOx and generation;
  extract `ST23` state NOx totals for inventory diagnostics. Retain nonnegative
  numerical NOx reports in the contiguous-US, county-resolved cohort. Preserve
  missing reports as missing, never physical zero. This cohort is not a complete
  physical emissions inventory. Query centres are fossil-fuel plants with at
  least 1 MW nameplate capacity; connected disjoint county patches have 3–5 counties.
- U.S. Census Bureau, **2023 County Adjacency File**:
  [product and documentation](https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.2023.html),
  [original text](https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2023.txt).
  Use `county_adjacency2023.txt` (1,116,106 bytes) for the fixed adjacency graph
  and unique same-state county-name resolution.

The exact SHA-256 values, official URLs and formal attribution are recorded in
[input-sources.json](input-sources.json). The Census archive metadata permits
use with acknowledgement and retains boundary/accuracy and trademark
qualifications. The [Census public-access policy](https://www2.census.gov/foia/ds_policies/ds027.pdf)
describes the federal-data copyright basis. The
[eGRID FAQ, question 25](https://www.epa.gov/egrid/frequent-questions-about-egrid)
identifies eGRID as public domain and permits commercial and non-commercial use.
See [NOTICE.md](../NOTICE.md) for required credit and qualifications. MIT covers
our software and documentation; it does not relicense agency data.

## Download, extract and analyse

Follow the [root setup](../README.md) with Python 3.11 and the hash-locked
dependencies. Allow 2 GiB RAM and 1 GiB free disk; the three downloads total
50,681,038 bytes. No account or accelerator is needed. From the repository root:

```sh
.venv/bin/python acquire.py --download
PYTHONPATH=study .venv/bin/python study/records.py
.venv/bin/python reproduce.py
```

The downloader retrieves only absent files from the provider URLs into ignored
`data/cache/` and writes `acquisition.json`. It checks exact sizes and hashes;
mismatches, unexpected redirects, symlinks and partial downloads fail visibly.
Existing mismatched files are never replaced. `records.py` writes
`census-block-records.csv`, `plant-records.csv`, `state-inventory-records.csv`
and `schema-extraction.json` locally. The full reproduction repeats extraction
safely and then reconstructs the fixed catalogues and analyses; allow several
minutes after setup. No prepared records need to be obtained from this repository.

For a separately retained original bundle, use a writable working copy:

```sh
export ALLOCATION_INPUTS=/absolute/path/to/exact-inputs
.venv/bin/python acquire.py
PYTHONPATH=study .venv/bin/python study/records.py
.venv/bin/python reproduce.py
```

Leave frozen originals untouched. If a provider URL changes, recover the same
version from your own retained storage or obtain that version from the provider;
do not silently substitute newer inputs or change expected hashes. A URL or
checksum alone does not preserve the inputs. Do not commit or upload downloaded
or prepared provider files as release assets.

## Selection and verification

Extraction checks the fixed DBF schema and internal points against same-record
bounding boxes (not full polygon containment). Workbook extraction uses the
field-code header and preserves blanks. Loaders construct the 60 eligible Census
maps and 369 eligible energy maps. `protocol.json` fixes random assignments,
predictors, grids and weighting; `extension.json` fixes changed query radii and
`composition.json` fixes matched-map contrasts. No new population, fit selection
or split selection is implied by reproduction.

Provider and derived-record hashes protect version/selection provenance, and the
registered query catalogue is checked exactly. Scientific summary comparisons
use the numerical tolerances explained in [docs/method.md](../docs/method.md).
Timings and graphics metadata may vary. The optional `reproduce.py --exact`
diagnostic additionally checks historical dense-output hashes; matching those
bytes is not the default scientific reproduction requirement.
