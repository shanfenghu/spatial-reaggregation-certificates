"""Reproduce the fixed changed-radius workload; no new population or tuning."""
from pathlib import Path
import hashlib
import json
import platform
import resource
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

import compare as study
import design as design
import engine as engine

V = Path(__file__).resolve().parents[1]
R = V / "run/primary"
OUT = V / "run/extension"
PROTOCOL = V / "extension.json"
METHODS = ["transport_scan", "reconciled_box", "reconciled_ellipse",
           "aggregate_global", "aggregate_spatial", "aggregate_global_expanded",
           "aggregate_spatial_expanded", "coarse_only"]


def digest(path):
    """Bind retained bytes, rather than download a replacement input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_union(frame, registered, centers, radii):
    """Deduplicate original and changed-radius masks using only known geometry.

    Positive source sums are the same released inputs as in the original task.
    Membership aliases retain both center and radius; zero-width masks do not
    receive undefined normalized losses or additional statistical weight.
    """
    found = {}
    for row in registered.itertuples():
        found[row.mask_hex] = {"mask_hex": row.mask_hex, "original": True,
                              "radii": {3.0}, "aliases": []}
    ids = sorted({str(x) for value in registered.all_center_ids
                  for x in json.loads(value)}, key=int)
    lookup = centers.set_index("ORISPL")
    positive = [g.index.to_numpy() for _, g in frame.groupby("source") if g.truth.sum() > 0]
    controls = {"center_radius_pairs": 0, "noncrossing_pairs": 0,
                "original_mask_pairs": 0, "duplicate_added_pairs": 0}
    for center_id in ids:
        center = lookup.loc[center_id]
        _, _, meters = engine.GEOD.inv(np.full(len(frame), center.LON),
                                      np.full(len(frame), center.LAT),
                                      frame.lon.to_numpy(), frame.lat.to_numpy())
        for radius in radii:
            controls["center_radius_pairs"] += 1
            mask = np.asarray(meters) <= radius * 1609.344
            if not any(mask[g].any() and not mask[g].all() for g in positive):
                controls["noncrossing_pairs"] += 1
                continue
            key = np.packbits(mask, bitorder="little").tobytes().hex()
            if key in found:
                controls["original_mask_pairs" if found[key]["original"] else "duplicate_added_pairs"] += 1
            else:
                found[key] = {"mask_hex": key, "original": False, "radii": set(), "aliases": []}
            found[key]["radii"].add(radius)
            found[key]["aliases"].append({"center": center_id, "radius_miles": radius})
    return sorted(found.values(), key=lambda row: row["mask_hex"]), controls


def fixed_spatial_model(frames, catalog, ids, fit):
    """Reconstruct sufficient training quantities without selecting new parameters."""
    positions, scores = [], []
    for key in ids:
        frame = frames[key]
        rows = catalog[catalog["map"] == key]
        queries = [np.unpackbits(np.frombuffer(bytes.fromhex(x), dtype=np.uint8),
                                 bitorder="little")[:len(frame)] for x in rows.mask_hex]
        c = study.case(frame, queries)
        positions.append(c["centroid"])
        scores.append(float(np.max(np.abs(c["reference"] - c["prediction"]) / (c["hi"] - c["lo"]))))
    return {"training_positions": np.asarray(positions), "training_catalog_scores": np.asarray(scores),
            "bandwidth_km": fit["spatial_bandwidth_km"]}


def endpoints(c, method, fit, expanded, W, scale):
    """Use exactly the original endpoint rules with explicit catalog permissions."""
    if method == "transport_scan":
        lo = [engine.endpoint_scan(c["geo"], q, fit["radii"][method], False)[0] / c["geo"]["total"] for q in c["q"]]
        hi = [engine.endpoint_scan(c["geo"], q, fit["radii"][method], True)[0] / c["geo"]["total"] for q in c["q"]]
    elif method == "reconciled_box":
        lo, hi, _, _ = study.box_bounds(c, fit["radii"][method])
    elif method == "reconciled_ellipse":
        values = [(study.ellipse_endpoint(c, W, fit["radii"][method], q),
                   study.ellipse_endpoint(c, W, fit["radii"][method], -q)) for q in c["q"]]
        lo, hi = [a[0] for a, _ in values], [-b[0] for _, b in values]
        residual = max(max(a[1], b[1]) for a, b in values)
        gap = max(max(a[2], b[2]) for a, b in values)
        return np.maximum(c["lo"], lo), np.minimum(c["hi"], hi), residual, gap
    elif method == "coarse_only":
        lo, hi = c["lo"], c["hi"]
    else:
        base = method.removesuffix("_expanded")
        radius = expanded[base] if method.endswith("_expanded") else fit["radii"][base]
        half = radius * (c["hi"] - c["lo"]) * (scale if base == "aggregate_spatial" else 1.)
        lo, hi = c["prediction"] - half, c["prediction"] + half
    return np.maximum(c["lo"], lo), np.minimum(c["hi"], hi), 0., 0.


def main():
    """Preserve prior evidence and refuse to replace an earlier outcome run."""
    assert not OUT.exists(), "Never overwrite an outcome run"
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    # Source inventory and pinned inputs are checked by reproduce.py.
    protected = {str(p.relative_to(V)): digest(p) for p in R.iterdir() if p.is_file()}

    OUT.mkdir()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="query-extension-loader-") as temp:
        design.OUT = Path(temp)
        domains, centers, _, _ = design.load_frames()
    loading_seconds = time.perf_counter() - start
    config = json.loads(design.PROTOCOL.read_text())
    catalog = pd.read_csv(R / "registered-query-catalog.csv", dtype={"map": str, "query_id": str, "mask_hex": str})
    recorded = pd.read_csv(R / "query-results.csv", dtype={"map": str, "query_id": str}).query("run == 'primary'")
    whole = pd.read_csv(R / "map-results.csv", dtype={"map": str}).query("run == 'primary'")
    fits = {row["domain"]: row for row in json.loads((R / "fitted-models.json").read_text()) if row["run"] == "primary"}
    query_rows, map_rows, catalog_rows, cost_rows, control_rows, calibration_rows, metadata = [], [], [], [], [], [], []
    max_original_error = max_scan_lp = max_box_lp = max_conic = max_gap = 0.
    original_checks = lp_checks = 0
    for domain, frames in domains.items():
        fit = fits[domain]
        split = config["splits"][domain]["primary"]
        dc = catalog[catalog.domain == domain]
        start = time.perf_counter()
        model = fixed_spatial_model(frames, dc, split["train"], fit)
        reconstruction_seconds = time.perf_counter() - start
        cases, memberships = {}, {}
        construction_seconds = preparation_seconds = 0.
        for key in sorted(split["calibration"] + split["test"]):
            f = frames[key]
            start = time.perf_counter()
            union, controls = query_union(f, dc[dc["map"] == key], centers, protocol["queries"]["new_radii_miles"])
            construction_seconds += time.perf_counter() - start
            queries = [np.unpackbits(np.frombuffer(bytes.fromhex(x["mask_hex"]), dtype=np.uint8),
                                     bitorder="little")[:len(f)] for x in union]
            start = time.perf_counter()
            cases[key] = study.case(f, queries)
            preparation_seconds += time.perf_counter() - start
            memberships[key] = union
            control_rows.append({"domain": domain, "map": key, **controls})
            for row in union:
                catalog_rows.append({"domain": domain, "map": key,
                                     "query_id": hashlib.sha256(bytes.fromhex(row["mask_hex"])).hexdigest()[:12],
                                     "mask_hex": row["mask_hex"], "original": row["original"],
                                     "radii_miles": json.dumps(sorted(row["radii"])), "aliases": json.dumps(row["aliases"])})
        # The expanded controls declare a larger catalog before selecting a rank;
        # this uses original calibration vectors, never test outcomes.
        cal_times, expanded = {}, {}
        for method in ["aggregate_global", "aggregate_spatial"]:
            start = time.perf_counter()
            values = []
            for key in split["calibration"]:
                c = cases[key]
                scale = study.spatial_scale(c, model) if method == "aggregate_spatial" else 1.
                value = float(np.max(np.abs(c["reference"] - c["prediction"]) / (c["hi"] - c["lo"])) / scale)
                values.append(value)
                calibration_rows.append({"domain": domain, "map": key, "method": method, "score": value})
            cal_times[method] = time.perf_counter() - start
            expanded[method] = values
        start = time.perf_counter()
        expanded = {method: study.quantile(values, .1) for method, values in expanded.items()}
        threshold_seconds = time.perf_counter() - start
        assert all(expanded[m] >= fit["radii"][m] - 1e-10 for m in expanded)
        domain_lp_queries = 0
        for count, key in enumerate(sorted(split["test"])):
            c, union = cases[key], memberships[key]
            start = time.perf_counter(); W, _ = study.covariance(c, fit["rho"], fit["ell"])
            ellipse_setup = time.perf_counter() - start
            start = time.perf_counter(); scale = study.spatial_scale(c, model)
            spatial_setup = time.perf_counter() - start
            for method in METHODS:
                start = time.perf_counter()
                lo, hi, residual, gap = endpoints(c, method, fit, expanded, W, scale)
                seconds = time.perf_counter() - start
                max_conic, max_gap = max(max_conic, residual), max(max_gap, gap)
                assert np.all(lo <= hi + study.TOL)
                covered = (lo - study.TOL <= c["reference"]) & (c["reference"] <= hi + study.TOL)
                ratio = (hi - lo) / (c["hi"] - c["lo"])
                loss = (hi - lo + 20 * np.maximum(lo - c["reference"], 0) + 20 * np.maximum(c["reference"] - hi, 0)) / (c["hi"] - c["lo"])
                if method in ["transport_scan", "reconciled_box", "reconciled_ellipse"]:
                    event = whole[(whole.domain == domain) & (whole["map"] == key) & (whole.method == method)].whole_map_covered.iloc[0]
                    assert not event or covered.all(), (domain, key, method)
                cost_rows.append({"domain": domain, "map": key, "method": method, "queries": len(union),
                                  "setup_seconds": ellipse_setup if method == "reconciled_ellipse" else spatial_setup if "spatial" in method else 0.,
                                  "endpoint_seconds": seconds})
                for subset in ["original", "new", "union", "radius_1.5", "radius_6"]:
                    selected = np.array([row["original"] if subset == "original" else not row["original"] if subset == "new"
                                         else True if subset == "union" else float(subset.removeprefix("radius_")) in row["radii"] for row in union])
                    if not selected.any():
                        continue
                    map_rows.append({"domain": domain, "map": key, "method": method, "subset": subset,
                                     "queries": int(selected.sum()), "catalog_covered": bool(covered[selected].all()),
                                     "width_ratio": float(ratio[selected].mean()), "interval_score": float(loss[selected].mean()),
                                     "useful_fraction": float(np.mean(ratio[selected] < .95)),
                                     "saturated_fraction": float(np.mean(ratio[selected] >= .999999))})
                for j, row in enumerate(union):
                    qid = hashlib.sha256(bytes.fromhex(row["mask_hex"])).hexdigest()[:12]
                    query_rows.append({"domain": domain, "map": key, "query_id": qid, "method": method,
                                       "original": row["original"], "lower": lo[j] * c["geo"]["total"], "upper": hi[j] * c["geo"]["total"],
                                       "reference": c["reference"][j] * c["geo"]["total"], "point": c["prediction"][j] * c["geo"]["total"],
                                       "coarse_lower": c["lo"][j] * c["geo"]["total"], "coarse_upper": c["hi"][j] * c["geo"]["total"],
                                       "covered": bool(covered[j]), "width_ratio": ratio[j], "interval_score": loss[j]})
                    if row["original"] and method not in ["aggregate_global_expanded", "aggregate_spatial_expanded"]:
                        old = recorded[(recorded.domain == domain) & (recorded["map"] == key) & (recorded.query_id == qid) & (recorded.method == method)].iloc[0]
                        error = max(abs(lo[j] - old.lower / c["geo"]["total"]), abs(hi[j] - old.upper / c["geo"]["total"]))
                        max_original_error = max(max_original_error, error)
                        assert error < study.TOL
                        original_checks += 1
            # Deterministic bounded independent controls, never chosen by error.
            for j, row in enumerate(union):
                if row["original"] or domain_lp_queries >= 10:
                    continue
                q = c["q"][j]
                blo, bhi, lower, upper = study.box_bounds(c, fit["radii"]["reconciled_box"])
                for upper_side in [False, True]:
                    scan = engine.endpoint_scan(c["geo"], q, fit["radii"]["transport_scan"], upper_side)[0]
                    exact = engine.endpoint_lp(c["geo"], q, fit["radii"]["transport_scan"], upper_side)[0]
                    max_scan_lp = max(max_scan_lp, abs(scan - exact) / c["geo"]["total"])
                    sign = -1. if upper_side else 1.
                    exact_box = study.lp(sign * q, A_eq=c["A"], b_eq=c["b"], bounds=list(zip(lower, upper))).fun * sign
                    max_box_lp = max(max_box_lp, abs(exact_box - (bhi[j] if upper_side else blo[j])))
                    lp_checks += 2
                domain_lp_queries += 1
            if (count + 1) % 25 == 0 or count + 1 == len(split["test"]):
                print(domain, count + 1, "test maps evaluated", flush=True)
        metadata.append({"domain": domain, "expanded_radii": expanded, "original_radii": fit["radii"],
                         "expanded_calibration_score_seconds": cal_times, "expanded_joint_threshold_seconds": threshold_seconds,
                         "shared_query_construction_seconds": construction_seconds,
                         "shared_case_and_reference_preparation_seconds": preparation_seconds,
                         "frozen_spatial_reconstruction_seconds": reconstruction_seconds})
    maps = pd.DataFrame(map_rows)
    summaries = []
    for (domain, method, subset), g in maps.groupby(["domain", "method", "subset"], sort=False):
        summaries.append({"domain": domain, "method": method, "subset": subset, "test_maps": len(g),
                          "catalog_successes": int(g.catalog_covered.sum()), "queries": int(g.queries.sum()),
                          **{x: float(g[x].mean()) for x in ["width_ratio", "interval_score", "useful_fraction", "saturated_fraction"]}})
    for name, rows in [("query-results", query_rows), ("map-results", map_rows), ("query-catalog", catalog_rows),
                       ("query-costs", cost_rows), ("construction-controls", control_rows),
                       ("expanded-calibration-scores", calibration_rows), ("study-summary", summaries)]:
        pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    assert max(max_original_error, max_scan_lp, max_box_lp, max_conic, max_gap) < study.TOL
    assert all(digest(V / name) == sha for name, sha in protected.items())
    result = {"status": "passed", "protocol_sha256": digest(PROTOCOL), "driver_sha256": digest(__file__),
              "source_tree": "source checksums recorded by the reproduction driver",
              "domain_runs": metadata, "shared_loading_seconds": loading_seconds, "elapsed_seconds": time.perf_counter() - started,
              "original_query_interval_reproductions": original_checks, "maximum_original_endpoint_error": max_original_error,
              "independent_LP_endpoint_checks": lp_checks, "max_scan_LP_error": max_scan_lp, "max_box_LP_error": max_box_lp,
              "max_conic_residual": max_conic, "max_conic_gap": max_gap,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == "Darwin" else 1024),
              "protected_sha256": protected, "interpretation": protocol["limits"]}
    (OUT / "run-metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: x for k, x in result.items() if k != "protected_sha256"}, indent=2))


if __name__ == "__main__":
    main()
