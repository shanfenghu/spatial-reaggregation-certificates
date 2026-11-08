"""Verify and display the registered results; no fitting or new outcome selection."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from design import OUT, PROTOCOL, ROOT, digest, load_frames

ORDER = ["transport_scan", "reconciled_box", "reconciled_ellipse", "aggregate_global",
         "aggregate_spatial", "label_free", "coarse_only"]
LABELS = ["Transport (ours)", "Reconciled box", "Reconciled ellipsoid", "Aggregate global",
          "Aggregate spatial", "Label-free transport", "Coarse-only"]
COLORS = ["#176b87", "#7c689b", "#ac79a5", "#b46c28", "#d59d42", "#648c7b", "#b9bfc5"]
FIG = ROOT / "run/diagnostic-plots"


def verify():
    metadata = json.loads((OUT / "run-metadata.json").read_text())
    assert digest(PROTOCOL) == metadata["protocol_sha256"]
    assert digest(ROOT / "study/compare.py") == metadata["driver_sha256"]
    config = json.loads(PROTOCOL.read_text())
    assert digest(OUT / "registered-query-catalog.csv") == config["query_catalog_sha256"]
    query = pd.read_csv(OUT / "query-results.csv", dtype={"map": str, "query_id": str})
    maps = pd.read_csv(OUT / "map-results.csv", dtype={"map": str})
    catalog = pd.read_csv(OUT / "registered-query-catalog.csv", dtype=str)
    domains, _, _, controls = load_frames()
    identities, zero_single = 0, 0
    unique = query.drop_duplicates(["domain", "map", "query_id"])
    for row in unique.itertuples():
        frame = domains[row.domain][row.map]
        registered = catalog[(catalog.domain == row.domain) & (catalog["map"] == row.map) & (catalog.query_id == row.query_id)].iloc[0]
        mask = np.unpackbits(np.frombuffer(bytes.fromhex(registered.mask_hex), dtype=np.uint8), bitorder="little")[:len(frame)]
        prediction = np.zeros(len(frame))
        low, high = 0., 0.
        for _, source in frame.groupby("source"):
            pos = source.index.to_numpy()
            total = float(source.truth.sum())
            weights = source.weight.to_numpy(float)
            if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                weights = np.ones(len(source))
            prediction[pos] = total * weights / weights.sum()
            if mask[pos].all():
                low += total
            if mask[pos].any():
                high += total
        expected = [float(frame.truth.to_numpy() @ mask), float(prediction @ mask), low, high]
        actual = [row.reference, row.point, row.coarse_lower, row.coarse_upper]
        assert np.allclose(expected, actual, rtol=1e-11, atol=1e-8), (row.domain, row.map, expected, actual)
        identities += 1
    for domain, control in controls.items():
        for entry in control["all_inventory"]:
            if entry["zero_mass_control"] or entry["fine_locations"] == 1:
                assert entry["coarse_total"] >= 0
                zero_single += 1
    recomputations = 0
    for (domain, run, key, method), group in query.groupby(["domain", "run", "map", "method"]):
        registered_n = len(catalog[(catalog.domain == domain) & (catalog["map"] == key)])
        assert len(group) == registered_n
        assert set(group.query_id) == set(catalog[(catalog.domain == domain) & (catalog["map"] == key)].query_id)
        mass = domains[domain][key].truth.sum()
        covered = (group.lower-2e-7*mass <= group.reference) & (group.reference <= group.upper+2e-7*mass)
        assert (covered == group.covered).all()
        row = maps[(maps.domain == domain) & (maps.run == run) & (maps["map"] == key) & (maps.method == method)].iloc[0]
        assert bool(covered.all()) == bool(row.catalog_covered)
        assert np.isclose(group.width_ratio.mean(), row.width_ratio)
        assert (group.lower >= group.coarse_lower-1e-8).all() and (group.upper <= group.coarse_upper+1e-8).all()
        recomputations += 1
    result = {"status": "passed", "independent_point_reference_coarse_checks": identities,
              "map_catalog_metric_recomputations": recomputations,
              "zero_mass_or_single_location_control_maps": zero_single,
              "protocol_sha256": metadata["protocol_sha256"], "driver_sha256": metadata["driver_sha256"],
              "verifier_sha256": digest(__file__),
              "note": "Zero/single-point controls are native-total identities, outside the crossing-query test frame. No claims about physical zeros from missing inventory rows."}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["protected_sha256"]}), flush=True)
    return query, maps


def figures(summary, maps):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    panels = [("census", "primary", "Census · random split"), ("egrid", "primary", "Energy · random split"),
              ("census", "southern_stress", "Census · southern stress"), ("egrid", "southern_stress", "Energy · southern stress")]
    for ax, (domain, run, title) in zip(axes.flat, panels):
        rows = summary[(summary.domain == domain) & (summary.run == run)].set_index("method").loc[ORDER]
        y = np.arange(len(ORDER))
        ax.barh(y, rows.width_ratio, color=COLORS, height=.62)
        ax.set_yticks(y, LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.68)
        ax.set_xticks([0, .25, .5, .75, 1.])
        ax.axvline(1, color="#aaa", lw=.8, ls="--")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        for j, row in enumerate(rows.itertuples()):
            ax.text(1.04, j, f"{int(row.catalog_successes)}/{int(row.test_maps)}", va="center", fontsize=9)
            map_count = f"{round(row.whole_map_coverage*row.test_maps)}/{int(row.test_maps)}" if pd.notna(row.whole_map_coverage) else "—"
            ax.text(1.38, j, map_count, va="center", fontsize=9)
        ax.text(1.04, -1.0, "Catalog", fontsize=9, color="#444")
        ax.text(1.38, -1.0, "Map score", fontsize=9, color="#444")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#eee", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean interval width / coarse-only width")
    fig.suptitle("Conserved spatial transfer: useful width depends on the domain", x=.035, ha="left", fontsize=15, fontweight="bold")
    fig.text(.035, .015, "Widths average queries within each map, then maps. Coverage counts require all registered queries in a map.\n"
             "Transport LP answers match the scan and are omitted here. Aggregate methods target this finite catalog; allocation sets support all queries.\n"
             "Southern stress has no extrapolation coverage guarantee. Samples share one map frame per domain; small Census counts have low precision.", fontsize=9, color="#444")
    fig.tight_layout(rect=(0, .09, 1, .95), h_pad=2.8, w_pad=2.0)
    for extension in ["png", "svg"]:
        fig.savefig(FIG / ("coverage-and-width."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for ax, domain, title in zip(axes, ["census", "egrid"], ["Census: 21 test maps", "Energy: 269 test maps"]):
        group = maps[(maps.domain == domain) & (maps.run == "primary")]
        pivot = group.pivot(index="map", columns="method", values="width_ratio")
        transport = group[group.method == "transport_scan"].set_index("map")
        for regime, marker, color in [("small", "o", "#176b87"), ("large", "^", "#c28239")]:
            ids = transport.index[transport.fine_regime == regime]
            ax.scatter(pivot.loc[ids, "reconciled_ellipse"], pivot.loc[ids, "transport_scan"],
                       s=30, marker=marker, c=color, alpha=.7, edgecolors="white", linewidths=.35,
                       label="Support ≤ training median" if regime == "small" else "Support > training median")
        ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Reconciled ellipsoid width / coarse width",
               ylabel="Transport width / coarse width", title=title)
        ax.set_aspect("equal")
        ax.grid(color="#eee")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Paired widths show domain and support-size differences", x=.04, ha="left", fontsize=13, fontweight="bold")
    fig.text(.04, .01, "Each point is one map's mean registered-query width. Below the diagonal, transport is narrower.\n"
             "Support thresholds were fixed using training maps; coverage and miss penalties must be assessed alongside widths.", fontsize=9, color="#444")
    fig.tight_layout(rect=(0, .07, 1, .93))
    for extension in ["png", "svg"]:
        fig.savefig(FIG / ("width-regimes."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    queries, maps = verify()
    summary = pd.read_csv(OUT / "study-summary.csv")
    regimes = []
    for variable in ["fine_regime", "source_regime", "catalog_regime"]:
        for (domain, run, method, value), g in maps.groupby(["domain", "run", "method", variable]):
            regimes.append({"domain": domain, "run": run, "method": method, "regime": variable, "value": value,
                            "maps": len(g), "catalog_successes": int(g.catalog_covered.sum()),
                            "width_ratio": float(g.width_ratio.mean()), "saturated_fraction": float(g.saturated_fraction.mean()),
                            "interval_score": float(g.interval_score.mean())})
    pd.DataFrame(regimes).to_csv(OUT / "registered-regime-summary.csv", index=False)
    score_cost = pd.read_csv(OUT / "map-scores.csv")
    query_cost = pd.read_csv(OUT / "query-costs.csv")
    costs = []
    for row in summary.itertuples():
        method = "transport_scan" if row.method == "transport_lp" else row.method
        same = score_cost[(score_cost.domain == row.domain) & (score_cost.run == row.run) & (score_cost.method == method)]
        calibration = same[same.part == "calibration"]
        test = same[same.part == "test_diagnostic"]
        inference = query_cost[(query_cost.domain == row.domain) & (query_cost.run == row.run) & (query_cost.method == row.method)].seconds.sum()
        costs.append({"domain": row.domain, "run": row.run, "method": row.method,
                      "calibration_score_seconds_including_setup": float(calibration.seconds.sum()),
                      "test_inference_setup_seconds": float(test.inference_setup_seconds_included.sum()),
                      "endpoint_seconds": float(inference),
                      "test_reference_score_diagnostic_seconds": float((test.seconds-test.inference_setup_seconds_included).sum()),
                      "note": "Shared frame/catalog geometry and fitted-model training are reported separately; transport scan and LP share the same calibration, not two separate fits."})
    pd.DataFrame(costs).to_csv(OUT / "cost-summary.csv", index=False)
    findings = []
    for domain in ["census", "egrid"]:
        group = maps[(maps.domain == domain) & (maps.run == "primary")]
        transport = group[group.method == "transport_scan"]
        pivot = group.pivot(index="map", columns="method", values="width_ratio")
        findings.append({"domain": domain, "all_catalog_saturated_maps": int((transport.saturated_fraction >= .999999).sum()),
                         "test_maps": len(transport), "transport_narrower_than_ellipse_maps": int((pivot.transport_scan < pivot.reconciled_ellipse-1e-6).sum()),
                         "transport_narrower_than_box_maps": int((pivot.transport_scan < pivot.reconciled_box-1e-6).sum()),
                         "mean_transport_minus_label_free_width": float((pivot.transport_scan-pivot.label_free).mean())})
    (OUT / "comparative-findings.json").write_text(json.dumps(findings, indent=2)+"\n")
    figures(summary, maps)
    print(json.dumps(findings, indent=2), flush=True)


if __name__ == "__main__":
    main()
