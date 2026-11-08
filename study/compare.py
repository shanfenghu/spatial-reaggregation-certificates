"""Run the registered smaller comparative study, without changing its design.

All outcomes are assigned point references under controlled withholding. The
primary split targets random maps in a fixed eligible frame. The southern split
is an extrapolation stress test with no inherited rank-coverage guarantee.
"""
import argparse
import json
import math
import platform
import resource
import time
import warnings

import clarabel
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csc_matrix, coo_matrix, eye, vstack
from scipy.spatial.distance import cdist, pdist

import engine as prior
from design import OUT, PROTOCOL, ROOT, digest, load_frames

METHODS = ["transport_scan", "transport_lp", "reconciled_box", "reconciled_ellipse",
           "aggregate_global", "aggregate_spatial", "label_free", "coarse_only", "coherent_point"]
SCORES = ["transport_scan", "reconciled_box", "reconciled_ellipse",
          "aggregate_global", "aggregate_spatial", "label_free"]
TOL = 2e-7


def lp(objective, A_eq=None, b_eq=None, A_ub=None, b_ub=None, bounds=(0, None), **unused):
    """Serial HiGHS solve with normalized primal and dual checks."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected", category=OptimizeWarning)
        fit = linprog(objective, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"threads": 1, "primal_feasibility_tolerance": 1e-9,
                                                "dual_feasibility_tolerance": 1e-9})
    assert fit.success, fit.message
    residual = 0.
    dual = 0.
    if A_eq is not None:
        residual = max(residual, np.max(np.abs(A_eq @ fit.x - b_eq)))
        dual += fit.eqlin.marginals @ b_eq
    if A_ub is not None:
        residual = max(residual, np.max(A_ub @ fit.x - b_ub))
        dual += fit.ineqlin.marginals @ b_ub
    if isinstance(bounds, tuple):
        low = np.full(len(objective), bounds[0])
        high = np.full(len(objective), np.inf if bounds[1] is None else bounds[1])
    else:
        low, high = np.asarray(bounds).T
    residual = max(residual, np.max(low - fit.x), np.max(fit.x - high))
    dual += fit.lower.marginals @ low
    finite = np.isfinite(high)
    dual += fit.upper.marginals[finite] @ high[finite]
    assert residual <= TOL and abs(dual - fit.fun) <= TOL, (residual, dual, fit.fun)
    return fit


prior.linprog = lp


def coarse_bounds(A, b, queries):
    lower, upper = [], []
    for q in queries:
        counts = np.asarray(A @ q).ravel()
        sizes = np.asarray(A.sum(axis=1)).ravel()
        lower.append(float(b[counts == sizes].sum()))
        upper.append(float(b[counts > 0].sum()))
    return np.asarray(lower), np.asarray(upper)


def case(frame, queries):
    """Shared geometry, coherent predictor and query preparation, charged once."""
    start = time.perf_counter()
    geo = prior.geometry(frame)
    n = len(frame)
    groups = [g.index.to_numpy() for _, g in frame.groupby("source", sort=True)]
    A = coo_matrix((np.ones(n), (np.repeat(np.arange(len(groups)), [len(g) for g in groups]),
                               np.concatenate(groups))), shape=(len(groups), n)).tocsc()
    b = np.asarray(A @ geo["truth"]).ravel()
    q = np.asarray(queries, dtype=float)
    low, high = coarse_bounds(A, b, q)
    assert (high > low).all()
    prediction = q @ geo["p"]
    reference = q @ geo["truth"]
    delta = geo["truth"] - geo["p"]
    s = np.sqrt(geo["p"] / n + 1. / n**2)
    distance = cdist(frame[["x", "y"]], frame[["x", "y"]]) / geo["scale_km"]
    same_source = frame.source.to_numpy()[:, None] == frame.source.to_numpy()[None, :]
    elapsed = time.perf_counter() - start
    return {"frame": frame, "geo": geo, "A": A, "b": b, "q": q, "lo": low, "hi": high,
            "prediction": prediction, "reference": reference, "delta": delta, "s": s,
            "distance": distance, "same_source": same_source, "n": n,
            "centroid": np.array([frame.x.mean(), frame.y.mean()]), "groups": groups,
            "geometry_seconds": elapsed, "query_count": len(q)}


def global_geometry(c):
    start = time.perf_counter()
    n = c["n"]
    u, v = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    row = coo_matrix((np.ones(n*n), (u, np.arange(n*n))), shape=(n, n*n)).tocsc()
    col = coo_matrix((np.ones(n*n), (v, np.arange(n*n))), shape=(n, n*n)).tocsc()
    g = {"u": u, "v": v, "row": row, "col": col, "cost": c["distance"].ravel(),
         "p": c["geo"]["p"], "truth": c["geo"]["truth"]}
    return g, time.perf_counter() - start


def covariance(c, rho, ell):
    """Positive definite source-block exponential kernel with a fixed nugget."""
    correlation = rho * np.exp(-c["distance"] / ell) * c["same_source"]
    correlation[np.diag_indices(c["n"])] += 1. - rho
    C = c["s"][:, None] * correlation * c["s"][None, :]
    L = cholesky(C, lower=True)
    W = solve_triangular(L, np.eye(c["n"]), lower=True)
    return W, float(2 * np.log(np.diag(L)).sum())


def fit_model(cases, ids, config):
    started = time.perf_counter()
    selections = []
    for rho in config["reconciled_ellipse"]["rho_grid"]:
        for ell in config["reconciled_ellipse"]["ell_over_D_grid"]:
            values = []
            for key in ids:
                c = cases[key]
                W, logdet = covariance(c, rho, ell)
                quadratic = float(np.sum((W @ c["delta"])**2))
                values.append(math.log(max(quadratic, 1e-12) / c["n"]) + logdet / c["n"])
            selections.append({"rho": rho, "ell": ell, "training_profile_loss": float(np.mean(values))})
    best = min(selections, key=lambda r: r["training_profile_loss"])
    positions = np.array([cases[k]["centroid"] for k in ids])
    training_scores = np.array([np.max(np.abs(cases[k]["reference"] - cases[k]["prediction"])
                                     / (cases[k]["hi"] - cases[k]["lo"])) for k in ids])
    distances = cdist(positions, positions)
    positive = pdist(positions)
    positive = positive[positive > 0]
    base = float(np.median(positive)) if len(positive) else 1.
    bandwidths = []
    for factor in config["aggregate_spatial"]["factor_grid"]:
        weights = np.exp(-.5 * (distances / (base * factor))**2)
        np.fill_diagonal(weights, 0.)
        denominator = weights.sum(axis=1)
        estimate = (weights @ training_scores) / np.maximum(denominator, 1e-300)
        estimate[denominator == 0] = np.median(training_scores)
        estimate = np.maximum(.05, estimate)
        bandwidths.append({"factor": factor, "training_LOO_MAE": float(np.mean(np.abs(training_scores - estimate)))})
    factor = min(bandwidths, key=lambda r: r["training_LOO_MAE"])["factor"]
    return {"rho": best["rho"], "ell": best["ell"], "training_positions": positions,
            "training_catalog_scores": training_scores, "bandwidth_km": base * factor,
            "ellipse_grid": selections, "bandwidth_grid": bandwidths,
            "training_seconds": time.perf_counter() - started}


def spatial_scale(c, model):
    weights = np.exp(-.5 * (np.linalg.norm(model["training_positions"] - c["centroid"], axis=1)
                            / model["bandwidth_km"])**2)
    estimate = float(weights @ model["training_catalog_scores"] / weights.sum()) if weights.sum() > 0 else float(np.median(model["training_catalog_scores"]))
    return max(.05, estimate)


def scores(c, model):
    values, times, setup_times = {}, {}, {}
    start = time.perf_counter()
    values["transport_scan"], _, _, _ = prior.score(c["geo"])
    times["transport_scan"] = time.perf_counter() - start
    start = time.perf_counter()
    values["reconciled_box"] = float(np.max(np.abs(c["delta"] / c["s"])))
    times["reconciled_box"] = time.perf_counter() - start
    start = time.perf_counter()
    W, _ = covariance(c, model["rho"], model["ell"])
    setup_times["reconciled_ellipse"] = time.perf_counter() - start
    values["reconciled_ellipse"] = float(np.linalg.norm(W @ c["delta"]))
    times["reconciled_ellipse"] = time.perf_counter() - start
    start = time.perf_counter()
    values["aggregate_global"] = float(np.max(np.abs(c["reference"] - c["prediction"]) / (c["hi"] - c["lo"])))
    times["aggregate_global"] = time.perf_counter() - start
    start = time.perf_counter()
    scale = spatial_scale(c, model)
    setup_times["aggregate_spatial"] = time.perf_counter() - start
    values["aggregate_spatial"] = values["aggregate_global"] / scale
    times["aggregate_spatial"] = time.perf_counter() - start + times["aggregate_global"]
    start = time.perf_counter()
    g, build_seconds = global_geometry(c)
    setup_times["label_free"] = build_seconds
    fit = lp(g["cost"], A_eq=vstack([g["row"], g["col"]], format="csc"), b_eq=np.r_[g["p"], g["truth"]])
    values["label_free"] = max(0., float(fit.fun))
    times["label_free"] = time.perf_counter() - start
    return values, times, W, scale, g, setup_times


def box_bounds(c, radius):
    low = np.maximum(0., c["geo"]["p"] - radius * c["s"])
    high = c["geo"]["p"] + radius * c["s"]
    lower, upper = [], []
    for query in c["q"]:
        l, u = 0., 0.
        for k, positions in enumerate(c["groups"]):
            inside = positions[query[positions] == 1]
            outside = positions[query[positions] == 0]
            l += max(low[inside].sum(), c["b"][k] - high[outside].sum())
            u += min(high[inside].sum(), c["b"][k] - low[outside].sum())
        lower.append(l)
        upper.append(u)
    return np.array(lower), np.array(upper), low, high


def ellipse_endpoint(c, W, radius, q):
    """SOCP projection, returning a dual-certified outer objective bound."""
    if radius <= 1e-14:
        return float(q @ c["geo"]["p"]), 0., 0.
    n, k = c["n"], len(c["b"])
    A = vstack([c["A"], -eye(n, format="csc"), csc_matrix((1, n)), -csc_matrix(W)], format="csc")
    b = np.r_[c["b"], np.zeros(n), radius, -W @ c["geo"]["p"]]
    cones = [clarabel.ZeroConeT(k), clarabel.NonnegativeConeT(n), clarabel.SecondOrderConeT(n+1)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 200
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    fit = clarabel.DefaultSolver(csc_matrix((n, n)), q, A, b, cones, settings).solve()
    assert str(fit.status) in ["Solved", "AlmostSolved"], str(fit.status)
    x, z = np.asarray(fit.x), np.asarray(fit.z)
    primal = max(np.max(np.abs(c["A"] @ x - c["b"])), max(0., -x.min()),
                 max(0., np.linalg.norm(W @ (x - c["geo"]["p"])) - radius))
    stationarity = np.max(np.abs(A.T @ z + q))
    cone_error = max(0., -z[k:k+n].min(), np.linalg.norm(z[k+n+1:]) - z[k+n])
    dual = float(-b @ z)
    gap = abs(float(q @ x) - dual)
    assert max(primal, stationarity, cone_error, gap) < TOL, (primal, stationarity, cone_error, gap)
    return dual, float(max(primal, stationarity, cone_error)), gap


def infer(c, model, radii, W, scale, g):
    bounds, costs, checks = {}, {}, {"transport_error": 0., "box_error": 0., "conic_residual": 0., "conic_gap": 0.}
    for method in METHODS:
        start = time.perf_counter()
        if method in ["transport_scan", "transport_lp"]:
            endpoint = prior.endpoint_scan if method == "transport_scan" else prior.endpoint_lp
            lo, hi = [], []
            for q in c["q"]:
                lo.append(endpoint(c["geo"], q, radii["transport_scan"], False)[0] / c["geo"]["total"])
                hi.append(endpoint(c["geo"], q, radii["transport_scan"], True)[0] / c["geo"]["total"])
        elif method == "reconciled_box":
            lo, hi, _, _ = box_bounds(c, radii[method])
        elif method == "reconciled_ellipse":
            lo, hi = [], []
            for q in c["q"]:
                l, residual, gap = ellipse_endpoint(c, W, radii[method], q)
                u, residual2, gap2 = ellipse_endpoint(c, W, radii[method], -q)
                lo.append(l)
                hi.append(-u)
                checks["conic_residual"] = max(checks["conic_residual"], residual, residual2)
                checks["conic_gap"] = max(checks["conic_gap"], gap, gap2)
        elif method in ["aggregate_global", "aggregate_spatial"]:
            width = (c["hi"] - c["lo"]) * radii[method] * (scale if method == "aggregate_spatial" else 1.)
            lo, hi = c["prediction"] - width, c["prediction"] + width
        elif method == "label_free":
            matrix = vstack([g["row"], c["A"] @ g["col"]], format="csc")
            rhs = np.r_[g["p"], c["b"]]
            lo, hi = [], []
            for q in c["q"]:
                values = [lp(sign*q[g["v"]], A_eq=matrix, b_eq=rhs,
                             A_ub=g["cost"][None, :], b_ub=[radii[method]]).fun * sign for sign in [1., -1.]]
                lo.append(values[0])
                hi.append(values[1])
        elif method == "coarse_only":
            lo, hi = c["lo"], c["hi"]
        else:
            lo, hi = c["prediction"], c["prediction"]
        lo, hi = np.maximum(c["lo"], lo), np.minimum(c["hi"], hi)
        assert (lo <= hi + TOL).all()
        bounds[method] = (lo, hi)
        costs[method] = time.perf_counter() - start
    checks["transport_error"] = float(max(np.max(np.abs(bounds["transport_scan"][j] - bounds["transport_lp"][j])) for j in [0, 1]))
    assert checks["transport_error"] < TOL
    start = time.perf_counter()
    _, _, low, high = box_bounds(c, radii["reconciled_box"])
    for j, q in enumerate(c["q"]):
        for sign, end in [(1., 0), (-1., 1)]:
            fit = lp(sign*q, A_eq=c["A"], b_eq=c["b"], bounds=list(zip(low, high)))
            checks["box_error"] = max(checks["box_error"], abs(sign*fit.fun - bounds["reconciled_box"][end][j]))
    assert checks["box_error"] < TOL
    costs["box_LP_verification"] = time.perf_counter() - start
    return bounds, costs, checks


def quantile(values, alpha):
    k = math.ceil((len(values)+1)*(1.-alpha))
    return float(np.sort(values)[k-1]) if k <= len(values) else float("inf")


def wilson(success, n):
    z = 1.959963984540054
    p = success / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return center-half, center+half


def evaluate_run(domain, run, cases, split, config, catalog_ids):
    model = fit_model(cases, split["train"], config)
    selected = split["calibration"] + split["test"]
    scores_by_id, cached, score_rows = {}, {}, []
    for key in selected:
        values, times, W, scale, g, setup_times = scores(cases[key], model)
        scores_by_id[key] = values
        cached[key] = (W, scale, g)
        for method in SCORES:
            score_rows.append({"domain": domain, "run": run, "map": key,
                               "part": "calibration" if key in split["calibration"] else "test_diagnostic",
                               "method": method, "score": values[method], "seconds": times[method],
                               "inference_setup_seconds_included": setup_times.get(method, 0.)})
    start = time.perf_counter()
    radii = {method: quantile([scores_by_id[k][method] for k in split["calibration"]], config["alpha"]) for method in SCORES}
    threshold_seconds = time.perf_counter() - start
    train_n_median = float(np.median([cases[k]["n"] for k in split["train"]]))
    train_s_median = float(np.median([len(cases[k]["b"]) for k in split["train"]]))
    rows, map_rows, cost_rows, check_rows = [], [], [], []
    for count, key in enumerate(split["test"]):
        c = cases[key]
        W, scale, g = cached[key]
        bounds, costs, checks = infer(c, model, radii, W, scale, g)
        check_rows.append({"domain": domain, "run": run, "map": key, **checks})
        for method in METHODS:
            lo, hi = bounds[method]
            covered = (lo - TOL <= c["reference"]) & (c["reference"] <= hi + TOL)
            score_name = "transport_scan" if method == "transport_lp" else method
            map_covered = scores_by_id[key][score_name] <= radii[score_name] + 1e-10 if score_name in SCORES[:3] + ["label_free"] else None
            if map_covered:
                assert covered.all(), (domain, run, key, method)
            coarse_width = c["hi"] - c["lo"]
            ratio = (hi - lo) / coarse_width
            midpoint = (lo + hi) / 2
            interval_score = (hi-lo + 2/config["alpha"] * np.maximum(lo-c["reference"], 0)
                              + 2/config["alpha"] * np.maximum(c["reference"]-hi, 0)) / coarse_width
            record = {"domain": domain, "run": run, "map": key, "method": method,
                      "whole_map_covered": map_covered, "catalog_covered": bool(covered.all()),
                      "query_coverage": float(covered.mean()), "width_ratio": float(ratio.mean()),
                      "useful_fraction": float(np.mean(ratio < .95)), "saturated_fraction": float(np.mean(ratio >= .999999)),
                      "normalized_point_MAE": float(np.mean(np.abs(c["reference"] - c["prediction"]) / coarse_width)),
                      "point_MAE_units": float(np.mean(np.abs(c["reference"] - c["prediction"])) * c["geo"]["total"]),
                      "midpoint_MAE_units": float(np.mean(np.abs(c["reference"]-midpoint))*c["geo"]["total"]),
                      "interval_score": float(interval_score.mean()), "query_count": len(c["q"]),
                      "fine_locations": c["n"], "sources": len(c["b"]),
                      "fine_regime": "small" if c["n"] <= train_n_median else "large",
                      "source_regime": "few" if len(c["b"]) <= train_s_median else "many",
                      "catalog_regime": "single" if len(c["q"]) == 1 else "multiple"}
            map_rows.append(record)
            for j in range(len(c["q"])):
                rows.append({"domain": domain, "run": run, "map": key, "query_id": catalog_ids[(domain, key)][j],
                             "method": method, "lower": lo[j]*c["geo"]["total"], "upper": hi[j]*c["geo"]["total"],
                             "reference": c["reference"][j]*c["geo"]["total"], "point": c["prediction"][j]*c["geo"]["total"],
                             "coarse_lower": c["lo"][j]*c["geo"]["total"], "coarse_upper": c["hi"][j]*c["geo"]["total"],
                             "covered": bool(covered[j]), "width_ratio": ratio[j]})
        for method, seconds in costs.items():
            cost_rows.append({"domain": domain, "run": run, "map": key, "method": method,
                              "stage": "verification" if method == "box_LP_verification" else "query_inference",
                              "seconds": seconds, "queries": len(c["q"]), "fine_locations": c["n"]})
        if (count+1) % 25 == 0 or count+1 == len(split["test"]):
            print(f"{domain} {run}: {count+1}/{len(split['test'])} test maps checked", flush=True)
    randomization = []
    if run == "primary":
        rng = np.random.default_rng(config["seed"] + 1)
        permutations = [rng.permutation(len(selected)) for _ in range(1000)]
        m = len(split["calibration"])
        for method in SCORES:
            values = np.array([scores_by_id[k][method] for k in selected])
            coverage = []
            for order in permutations:
                radius = quantile(values[order[:m]], config["alpha"])
                coverage.append(float(np.mean(values[order[m:]] <= radius + 1e-10)))
            randomization.append({"domain": domain, "method": method, "event": "whole_map" if method in SCORES[:3]+["label_free"] else "catalog_score",
                                  "repartitions": 1000, "mean": float(np.mean(coverage)),
                                  "q05": float(np.quantile(coverage, .05)), "q95": float(np.quantile(coverage, .95))})
    metadata = {"domain": domain, "run": run, "radii": radii, "rho": model["rho"], "ell": model["ell"],
                "spatial_bandwidth_km": model["bandwidth_km"], "training_seconds": model["training_seconds"],
                "threshold_seconds": threshold_seconds, "ellipse_training_grid": model["ellipse_grid"],
                "bandwidth_training_grid": model["bandwidth_grid"], "train_n_median": train_n_median,
                "train_source_median": train_s_median}
    return rows, map_rows, cost_rows, check_rows, score_rows, randomization, metadata


def summarize(maps, queries):
    summary = []
    for (domain, run, method), group in maps.groupby(["domain", "run", "method"], sort=False):
        covered = int(group.catalog_covered.sum())
        low, high = wilson(covered, len(group))
        q = queries[(queries.domain == domain) & (queries.run == run) & (queries.method == method)]
        summary.append({"domain": domain, "run": run, "method": method, "test_maps": len(group),
                        "catalog_successes": covered, "catalog_coverage": covered/len(group),
                        "descriptive_Wilson_low": low, "descriptive_Wilson_high": high,
                        "whole_map_coverage": float(group.whole_map_covered.dropna().mean()) if group.whole_map_covered.notna().any() else None,
                        "queries": len(q), "micro_query_coverage": float(q.covered.mean()),
                        **{column: float(group[column].mean()) for column in ["query_coverage", "width_ratio", "useful_fraction", "saturated_fraction", "normalized_point_MAE", "point_MAE_units", "midpoint_MAE_units", "interval_score"]}})
    return summary


def self_check():
    frame = pd.DataFrame({"source": ["a", "a", "b"], "x": [0., 1., 5.], "y": [0., 0., 0.],
                          "truth": [1., 1., 0.], "weight": [1., 1., 0.]})
    c = case(frame, [np.array([1., 0., 0.])])
    for radius in [0., .1, 1.]:
        W = np.eye(3)
        bound, residual, gap = ellipse_endpoint(c, W, radius, np.array([1., 0., 0.]))
        expected = max(0., .5-radius/math.sqrt(2))
        assert abs(bound-expected) < TOL
        lo, hi, lower, upper = box_bounds(c, radius)
        for sign, expected in [(1., lo[0]), (-1., hi[0])]:
            fit = lp(sign*c["q"][0], A_eq=c["A"], b_eq=c["b"], bounds=list(zip(lower, upper)))
            assert abs(sign*fit.fun-expected) < TOL
    assert quantile(np.arange(19), .1) == 17
    assert math.isinf(quantile([1.], .1))
    radii = {name: .2 for name in SCORES}
    g, _ = global_geometry(c)
    initial, _, _ = infer(c, None, radii, np.eye(3), .5, g)
    # Changing hidden fine outcomes while holding released totals fixed must not
    # change any endpoint. Only test diagnostics may depend on the hidden vector.
    frame.loc[:1, "truth"] = [.5, 1.5]
    altered = case(frame, [np.array([1., 0., 0.])])
    other_g, _ = global_geometry(altered)
    changed, _, _ = infer(altered, None, radii, np.eye(3), .5, other_g)
    for name in METHODS:
        assert np.allclose(initial[name], changed[name], atol=TOL)
    print("Analytic conic, box-LP, finite-rank and test-outcome-isolation controls passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    started = time.perf_counter()
    assert not (OUT / "study-summary.csv").exists(), "Refuse to overwrite comparative evidence"
    protocol_sha = digest(PROTOCOL)
    config = json.loads(PROTOCOL.read_text())
    catalog_path = OUT / "registered-query-catalog.csv"
    assert digest(catalog_path) == config["query_catalog_sha256"]
    parsing_start = time.perf_counter()
    domains, _, audits, controls = load_frames()
    parsing_seconds = time.perf_counter() - parsing_start
    catalog = pd.read_csv(catalog_path, dtype={"map": str, "query_id": str, "mask_hex": str})
    cases, ids, geometries = {}, {}, []
    for domain, frames in domains.items():
        cases[domain] = {}
        for key, frame in frames.items():
            registered = catalog[(catalog.domain == domain) & (catalog["map"] == key)]
            q = [np.unpackbits(np.frombuffer(bytes.fromhex(value), dtype=np.uint8), bitorder="little")[:len(frame)] for value in registered.mask_hex]
            c = case(frame, q)
            cases[domain][key] = c
            ids[(domain, key)] = registered.query_id.tolist()
            geometries.append({"domain": domain, "map": key, "seconds": c["geometry_seconds"],
                               "fine_locations": c["n"], "queries": len(q),
                               "within_source_arcs": len(c["geo"]["u"]), "global_arcs": c["n"]**2,
                               "source_balance_abs": c["geo"]["source_balance_abs"]})
    all_rows = [[] for _ in range(7)]
    for domain in ["census", "egrid"]:
        for run in ["primary", "southern_stress"]:
            print("Starting", domain, run, flush=True)
            result = evaluate_run(domain, run, cases[domain], config["splits"][domain][run], config, ids)
            for i in range(6):
                all_rows[i].extend(result[i])
            all_rows[6].append(result[6])
            # Keep recoverable complete run evidence if a later run fails.
            for name, records in zip(["query-results", "map-results", "query-costs", "numerical-checks", "map-scores", "score-repartition-sensitivity"], all_rows[:6]):
                pd.DataFrame(records).to_csv(OUT / (name+".csv"), index=False)
    query_frame, map_frame = pd.DataFrame(all_rows[0]), pd.DataFrame(all_rows[1])
    summary = summarize(map_frame, query_frame)
    pd.DataFrame(summary).to_csv(OUT / "study-summary.csv", index=False)
    pd.DataFrame(geometries).to_csv(OUT / "shared-geometry-costs.csv", index=False)
    (OUT / "fitted-models.json").write_text(json.dumps(all_rows[6], indent=2)+"\n")
    metadata = {"protocol_sha256": protocol_sha, "driver_sha256": digest(__file__),
                "source_tree": "source checksums recorded by the reproduction driver",
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "clarabel": clarabel.__version__},
                "elapsed_seconds": time.perf_counter()-started, "input_frame_parsing_projection_seconds": parsing_seconds,
                "projection_seconds": sum(a["projection_seconds"] for a in audits.values()),
                "catalog_generation_seconds": json.loads((OUT / "registration-costs.json").read_text())["elapsed_seconds"],
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=="Darwin" else 1024),
                "linear_algebra_threads": 1, "status": "completed registered bounded comparative study",
                "warning": "Wilson ranges use iid working assumptions, not valid dependence-robust or extrapolation confidence. Repartitions reuse one finite population. Southern stress has no exchangeable calibration guarantee."}
    assert digest(PROTOCOL) == protocol_sha
    (OUT / "run-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
