"""Fixed-cohort loaders and continuous source-conserving query primitives."""
import os
from guards import checked_query, check_witness
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import platform
import re
import resource
import time
import warnings

import numpy as np
import pandas as pd
import pyproj
from pyproj import Geod, Transformer
import scipy
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("ALLOCATION_INPUTS", ROOT / "data/cache"))
OUT = ROOT / "run/primary"
FOSSIL = {"COAL", "GAS", "OIL", "OFSL"}
GEOD = Geod(ellps="WGS84")
PROJECT = Transformer.from_crs(4326, 5070, always_xy=True)


def _solve_lp(*args, **kwargs):
    """Keep HiGHS' shared process scheduler consistent across entry points.

    A first automatic-thread solve prevents later explicit serial solves in the
    same process. Use one thread here as in the comparison helpers, regardless
    of whether those helpers have been imported yet.
    """
    options = dict(kwargs.pop("options", {}), threads=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected", category=OptimizeWarning)
        return linprog(*args, options=options, **kwargs)


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def norm_name(value):
    """Case/punctuation normalization retaining city/county distinction."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def county_metadata():
    rows = pd.read_csv(BASE / "county_adjacency2023.txt", sep="|", dtype=str)
    names = dict(zip(rows["County GEOID"], rows["County Name"]))
    graph = {key: set() for key in names}
    aliases = defaultdict(set)
    for key, label in names.items():
        name, state = label.rsplit(", ", 1)
        # Removing a geographic suffix is permitted only for a UNIQUE state match.
        simple = re.sub(r" (County|Parish|Borough|Municipality|Census Area|city)$", "", name, flags=re.I)
        for alias in [name, simple]:
            aliases[(state, norm_name(alias))].add(key)
    for key, neighbor in zip(rows["County GEOID"], rows["Neighbor GEOID"]):
        if key != neighbor:
            graph[key].add(neighbor)
    assert all(a in graph[b] for a in graph for b in graph[a])
    return names, graph, aliases


def make_patches(graph):
    """Fixed connected triples; attach small leftovers only when size <= 5."""
    excluded_states = {"02", "15", "60", "66", "69", "72", "78"}
    remaining = {key for key in graph if key[:2] not in excluded_states}
    patches, leftovers = [], []
    while remaining:
        nodes = {min(remaining)}
        while len(nodes) < 3:
            frontier = set().union(*(graph[x] for x in nodes)) & (remaining - nodes)
            if not frontier:
                break
            nodes.add(min(frontier))
        remaining -= nodes
        (patches if len(nodes) >= 3 else leftovers).append(nodes)
    unattached = []
    for nodes in leftovers:
        options = [i for i, patch in enumerate(patches) if len(patch) + len(nodes) <= 5
                   and any(graph[x] & patch for x in nodes)]
        if options:
            patches[min(options)].update(nodes)
        else:
            unattached.append(sorted(nodes))
    used = set()
    for patch in patches:
        assert not used & patch
        used |= patch
        reached = {min(patch)}
        while True:
            updated = reached | (set().union(*(graph[x] for x in reached)) & patch)
            if updated == reached:
                break
            reached = updated
        assert reached == patch
    return sorted(patches, key=lambda p: min(p)), unattached


def inspect_plants(names, aliases):
    raw = pd.read_csv(BASE / "plant-records.csv", dtype={"ORISPL": str, "FIPSST": str, "FIPSCNTY": str})
    assert raw.ORISPL.is_unique and raw.YEAR.eq(2023).all()
    assert (raw.PLNOXAN.dropna() >= 0).all()
    assert (raw.NAMEPCAP.dropna() >= 0).all()
    assert raw.LAT.between(-90, 90).all() and raw.LON.between(-180, 180).all()
    resolutions, exclusions, county_ids, name_mismatches = [], [], [], []
    for _, row in raw.iterrows():
        key = None
        if pd.notna(row.FIPSCNTY):
            key = str(int(float(row.FIPSST))).zfill(2) + str(int(float(row.FIPSCNTY))).zfill(3)
        if key not in names:
            matches = aliases.get((row.PSTATABB, norm_name(row.CNTYNAME)), set())
            if len(matches) == 1:
                new_key = next(iter(matches))
                resolutions.append({"ORISPL": row.ORISPL, "original_code": key,
                                    "county_name": row.CNTYNAME, "resolved_code": new_key})
                key = new_key
            else:
                key = None
        elif pd.notna(row.CNTYNAME):
            matches = aliases.get((row.PSTATABB, norm_name(row.CNTYNAME)), set())
            if matches and key not in matches:
                name_mismatches.append({"ORISPL": row.ORISPL, "reported_code": key,
                                        "reported_name": row.CNTYNAME, "name_codes": sorted(matches)})
        county_ids.append(key)
        reasons = []
        if row.PSTATABB in ["AK", "HI", "PR"]:
            reasons.append("outside_CONUS_frame")
        if pd.isna(row.PLNOXAN):
            reasons.append("NOx_not_numerical")
        if key is None:
            reasons.append("unresolved_2023_county_identity")
        if not (-125 <= row.LON <= -66 and 24 <= row.LAT <= 50.5):
            reasons.append("outside_declared_CONUS_coordinate_envelope")
        if reasons:
            exclusions.append({"ORISPL": row.ORISPL, "reasons": ";".join(reasons),
                               "PSTATABB": row.PSTATABB, "NOx": row.PLNOXAN})
    raw["source"] = county_ids
    bad = {x["ORISPL"] for x in exclusions}
    records = raw[~raw.ORISPL.isin(bad)].copy()
    records = records.rename(columns={"ORISPL": "id", "LAT": "lat", "LON": "lon", "PLNOXAN": "truth", "NAMEPCAP": "weight"})
    projection_start = time.perf_counter()
    records["x"], records["y"] = PROJECT.transform(records.lon.to_numpy(), records.lat.to_numpy())
    records[["x", "y"]] /= 1000.0
    projection_seconds = time.perf_counter() - projection_start
    # Query centers depend on coordinates, fuel and capacity, not NOx reporting/error.
    centers = raw[raw.PLFUELCT.isin(FOSSIL) & raw.NAMEPCAP.ge(1)
                  & raw.LON.between(-125, -66) & raw.LAT.between(24, 50.5)].copy()
    centers = centers.sort_values("ORISPL", key=lambda s: s.astype(int))
    state = pd.read_csv(BASE / "state-inventory-records.csv")
    known = raw.groupby("PSTATABB").PLNOXAN.sum(min_count=1)
    comparisons = [{"state": r.PSTATABB, "known_plant_sum": float(known[r.PSTATABB]),
                    "native_state_total": float(r.STNOXAN),
                    "difference": float(known[r.PSTATABB] - r.STNOXAN)} for _, r in state.iterrows()]
    write_csv("county-code-resolutions.csv", resolutions)
    write_csv("plant-exclusions.csv", exclusions)
    write_csv("county-name-code-mismatches.csv", name_mismatches)
    write_csv("state-accounting-controls.csv", comparisons)
    return records, centers, {"all_plants": len(raw), "unique_ORISPL": raw.ORISPL.nunique(),
                              "all_numerical_NOx": int(raw.PLNOXAN.notna().sum()),
                              "blank_NOx": int(raw.PLNOXAN.isna().sum()),
                              "blank_county_code": int(raw.FIPSCNTY.isna().sum()),
                              "code_resolutions": len(resolutions), "name_code_mismatches": len(name_mismatches),
                              "excluded_rows": len(exclusions), "retained_numerical_CONUS_cohort": len(records),
                              "retained_NOx_short_tons": float(records.truth.sum()),
                              "negative_numerical_NOx": int((raw.PLNOXAN < 0).sum()),
                              "projection_seconds": projection_seconds,
                              "state_accounting_max_abs_difference": max(abs(r["difference"]) for r in comparisons),
                              "interpretation": "Numerical-reporting cohort; blank inventory values not converted to physical zeros."}


def inspect_census():
    f = pd.read_csv(BASE / "census-block-records.csv", dtype={k: str for k in
                    ["GEOID20", "STATEFP20", "COUNTYFP20", "TRACTCE20", "BLOCKCE20"]})
    assert f.GEOID20.is_unique
    assert f.STATEFP20.eq("44").all()
    assert (f.GEOID20 == f.STATEFP20 + f.COUNTYFP20 + f.TRACTCE20 + f.BLOCKCE20).all()
    assert f[["POP20", "HOUSING20", "ALAND20", "AWATER20"]].notna().all().all()
    assert (f[["POP20", "HOUSING20", "ALAND20", "AWATER20"]] >= 0).all().all()
    f["id"], f["source"], f["map"] = f.GEOID20, f.GEOID20.str[:12], f.GEOID20.str[:11]
    projection_start = time.perf_counter()
    lon, lat = Transformer.from_crs(4269, 4326, always_xy=True).transform(f.INTPTLON20.to_numpy(), f.INTPTLAT20.to_numpy())
    f["lon"], f["lat"] = lon, lat
    assert f.lat.between(41, 43).all() and f.lon.between(-73, -70).all()
    f["x"], f["y"] = PROJECT.transform(lon, lat)
    f[["x", "y"]] /= 1000.0
    projection_seconds = time.perf_counter() - projection_start
    f["truth"], f["weight"] = f.POP20.astype(float), f.ALAND20.astype(float)
    return f, {"blocks": len(f), "tracts": f["map"].nunique(), "block_groups": f.source.nunique(),
               "released_population": int(f.POP20.sum()), "zero_population_blocks": int(f.POP20.eq(0).sum()),
               "nonnegative_complete_fields": True, "ID_composition_failures": 0,
               "projection_seconds": projection_seconds,
               "point_support": "NAD83 internal points transformed to WGS84; same-record SHP bounding-box check passed in extraction."}


def crossing_query(frame, centers):
    """First geometry-only EPA-style 3-mile query that cuts a positive source."""
    candidates = centers[centers.LAT.between(frame.lat.min() - .1, frame.lat.max() + .1)
                         & centers.LON.between(frame.lon.min() - .15, frame.lon.max() + .15)]
    # This conservative envelope exceeds 3 miles throughout the declared CONUS latitudes.
    for _, center in candidates.iterrows():
        _, _, meters = GEOD.inv(np.full(len(frame), center.LON), np.full(len(frame), center.LAT),
                                frame.lon.to_numpy(), frame.lat.to_numpy())
        query = np.asarray(meters) <= 3 * 1609.344
        for _, source in frame.groupby("source"):
            positions = frame.index.get_indexer(source.index)
            if source.truth.sum() > 0 and query[positions].any() and not query[positions].all():
                return query.astype(int), {"ORISPL": str(center.ORISPL), "lat": float(center.LAT),
                                            "lon": float(center.LON), "radius_m": 3 * 1609.344}
    return None, None


def inventory_maps(domain, frame, groups, centers):
    inventory, eligible = [], []
    for key, rows in groups:
        rows = rows.copy().reset_index(drop=True)
        size = sum(len(s) ** 2 for _, s in rows.groupby("source"))
        total = float(rows.truth.sum())
        query, center = crossing_query(rows, centers) if total > 0 else (None, None)
        passed = total > 0 and size <= 50000 and query is not None
        entry = {"domain": domain, "map": key, "source_units": rows.source.nunique(),
                 "fine_locations": len(rows), "coarse_total": total, "transport_arcs_upper_bound": size,
                 "zero_mass_control": total == 0, "crossing_query": query is not None,
                 "pilot_size_eligible": size <= 50000, "eligible": passed,
                 "query_center_ORISPL": None if center is None else center["ORISPL"]}
        inventory.append(entry)
        if passed:
            eligible.append((key, rows, query, center))
    return inventory, eligible


def geometry(frame):
    """Sparse transport structure with an explicit source-identity invariant."""
    start = time.perf_counter()
    mass = frame.truth.to_numpy(float)
    total = float(mass.sum())
    predicted = np.zeros(len(frame))
    source_totals, fallbacks = {}, []
    edges_u, edges_v, distances = [], [], []
    for key, source in frame.groupby("source", sort=True):
        positions = frame.index.get_indexer(source.index)
        b = float(source.truth.sum())
        weights = source.weight.to_numpy(float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            weights = np.ones(len(source))
            if b > 0:
                fallbacks.append(str(key))
        predicted[positions] = b * weights / weights.sum()
        source_totals[str(key)] = b
        edges_u.append(np.repeat(positions, len(positions)))
        edges_v.append(np.tile(positions, len(positions)))
        distances.append(cdist(source[["x", "y"]], source[["x", "y"]]).ravel())
    u, v, distance = np.concatenate(edges_u), np.concatenate(edges_v), np.concatenate(distances)
    scale = float(distance.max())
    assert total > 0 and scale > 0
    n, m = len(frame), len(u)
    row = coo_matrix((np.ones(m), (u, np.arange(m))), shape=(n, m)).tocsr()
    col = coo_matrix((np.ones(m), (v, np.arange(m))), shape=(n, m)).tocsr()
    p, truth, cost = predicted / total, mass / total, distance / scale
    assert frame.source.to_numpy()[u].tolist() == frame.source.to_numpy()[v].tolist()
    balance = max(abs(predicted[g.index.to_numpy()].sum() - g.truth.sum())
                  for _, g in frame.groupby("source"))
    assert balance <= 1e-10 * total
    return {"u": u, "v": v, "row": row, "col": col, "p": p, "truth": truth,
            "cost": cost, "total": total, "scale_km": scale, "source_totals": source_totals,
            "fallback_sources": fallbacks, "source_balance_abs": float(balance),
            "seconds": time.perf_counter() - start}


def score(geo):
    start = time.perf_counter()
    matrix = vstack([geo["row"], geo["col"]], format="csr")
    rhs = np.r_[geo["p"], geo["truth"]]
    fit = _solve_lp(geo["cost"], A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs")
    assert fit.success, fit.message
    residual = float(np.max(np.abs(matrix @ fit.x - rhs)))
    dual_error = abs(float(fit.eqlin.marginals @ rhs - fit.fun))
    assert residual < 1e-7 and dual_error < 1e-7
    return float(max(0., fit.fun)), time.perf_counter() - start, residual, dual_error


def endpoint_scan(geo, query, budget, upper):
    """Return an attainable binary endpoint in mass units and elapsed seconds.

    Budget is nonnegative (infinity allowed); upper selects the upper endpoint.
    Continuous mass and unrestricted same-source receivers are essential.
    """
    query = checked_query(geo, query, budget)
    start = time.perf_counter()
    losing, gaining = (0, 1) if upper else (1, 0)
    allowed = (query[geo["u"]] == losing) & (query[geo["v"]] == gaining)
    delta = np.full(len(query), np.inf)
    np.minimum.at(delta, geo["u"][allowed], geo["cost"][allowed])
    moves = sorted((float(delta[k]), float(geo["p"][k])) for k in range(len(query)) if np.isfinite(delta[k]))
    remaining, gain = budget, 0.
    for distance, capacity in moves:
        moved = capacity if distance == 0 else min(capacity, max(0., remaining) / distance)
        remaining -= moved * distance
        gain += moved
    value = float(geo["p"] @ query) + (gain if upper else -gain)
    return geo["total"] * value, time.perf_counter() - start


def endpoint_lp(geo, query, budget, upper):
    """Return the identical-model LP endpoint, time and maximum primal residual.

    The solver coupling is checked as a feasible endpoint witness.
    """
    query = checked_query(geo, query, budget)
    start = time.perf_counter()
    objective = query[geo["v"]] * (-1. if upper else 1.)
    fit = _solve_lp(objective, A_eq=geo["row"], b_eq=geo["p"], A_ub=geo["cost"][None, :] if np.isfinite(budget) else None,
                  b_ub=[budget] if np.isfinite(budget) else None, bounds=(0, None), method="highs")
    assert fit.success, fit.message
    residual = max(float(np.max(np.abs(geo["row"] @ fit.x - geo["p"]))),
                   max(0., float(geo["cost"] @ fit.x - budget)))
    dual = float(fit.eqlin.marginals @ geo["p"] + (fit.ineqlin.marginals[0] * budget if np.isfinite(budget) else 0.))
    assert residual < 1e-7 and abs(dual - fit.fun) < 1e-7
    check_witness(geo, query, budget, fit.x, float(-fit.fun if upper else fit.fun))
    return geo["total"] * float(-fit.fun if upper else fit.fun), time.perf_counter() - start, residual
