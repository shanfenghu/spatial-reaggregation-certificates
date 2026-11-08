"""Fixed-weight matched-map width decomposition on reproduced outcomes."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[1]
PROTOCOL = V / "composition.json"
OUT = V / "run/composition"
BASELINES = ["reconciled_box", "reconciled_ellipse", "aggregate_global", "aggregate_spatial"]
TIE = 1e-6


def digest(p):
    """Bind retained record versions rather than regenerate previous results."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize_pair(original, added, baseline):
    """Separate map selection from a paired workload change at fixed map weights.

    Inputs have one row per map and method columns, holding within-map mean
    normalized widths. All maps with added masks are retained. Leave-one-out
    ranges describe influence, without pretending the spatial maps are iid.
    """
    assert original.index.is_unique and added.index.is_unique
    assert set(added.index) <= set(original.index) and len(added) >= 2
    ids = sorted(added.index)
    full = original.transport_scan - original[baseline]
    pairs = pd.DataFrame({"original_transport": original.loc[ids, "transport_scan"],
                          "original_baseline": original.loc[ids, baseline],
                          "added_transport": added.loc[ids, "transport_scan"],
                          "added_baseline": added.loc[ids, baseline]})
    pairs["original_contrast"] = pairs.original_transport - pairs.original_baseline
    pairs["added_contrast"] = pairs.added_transport - pairs.added_baseline
    pairs["paired_change"] = pairs.added_contrast - pairs.original_contrast
    n = len(pairs)
    summary = {"baseline": baseline, "matched_maps": n, "full_maps": len(original),
               "full_original_contrast": float(full.mean()),
               **{x: float(pairs[x].mean()) for x in pairs.columns},
               "selection_component": float(pairs.original_contrast.mean() - full.mean()),
               "within_map_component": float(pairs.paired_change.mean()),
               "observed_unmatched_change": float(pairs.added_contrast.mean() - full.mean()),
               "paired_change_median": float(pairs.paired_change.median()),
               "paired_change_q25": float(pairs.paired_change.quantile(.25)),
               "paired_change_q75": float(pairs.paired_change.quantile(.75)),
               "narrower_to_wider": int(((pairs.original_contrast < -TIE) & (pairs.added_contrast > TIE)).sum()),
               "wider_to_narrower": int(((pairs.original_contrast > TIE) & (pairs.added_contrast < -TIE)).sum())}
    for subset in ["original_contrast", "added_contrast", "paired_change"]:
        values = pairs[subset]
        summary[subset + "_positive"] = int((values > TIE).sum())
        summary[subset + "_negative"] = int((values < -TIE).sum())
        summary[subset + "_ties"] = int((abs(values) <= TIE).sum())
        omitted = (values.sum() - values) / (n - 1)
        summary[subset + "_leave_one_out_min"] = float(omitted.min())
        summary[subset + "_leave_one_out_max"] = float(omitted.max())
    assert abs(summary["observed_unmatched_change"] - summary["selection_component"] - summary["within_map_component"]) < 1e-12
    return pairs, summary


def main():
    """Compute the registered contrasts once, leaving previous evidence intact."""
    assert not OUT.exists(), "Never overwrite a diagnostic outcome"
    protocol = json.loads(PROTOCOL.read_text())
    # Input checksums are verified by the reproduction driver.
    r = V / "run/extension"
    q = pd.read_csv(r / "query-results.csv", dtype={"map": str, "query_id": str})
    recorded = pd.read_csv(r / "map-results.csv", dtype={"map": str})
    old = pd.read_csv(V / "run/primary/map-results.csv", dtype={"map": str}).query("run == 'primary'")
    # Saved query normalizers are preserved; independently recover width ratios
    # before comparing already defined original and added masks.
    ratio = (q.upper - q.lower) / (q.coarse_upper - q.coarse_lower)
    assert np.allclose(q.width_ratio, ratio, atol=1e-12, rtol=1e-10)
    q["width_ratio"] = ratio
    summaries, pair_rows = [], []
    for domain in ["census", "egrid"]:
        frame = q[q.domain == domain]
        original = frame[frame.original].groupby(["map", "method"]).width_ratio.mean().unstack()
        added = frame[~frame.original].groupby(["map", "method"]).width_ratio.mean().unstack()
        for method in ["transport_scan"] + BASELINES:
            expected = old[(old.domain == domain) & (old.method == method)].set_index("map").width_ratio
            assert np.allclose(original[method].sort_index(), expected.sort_index(), atol=1e-10)
            for subset, values in [("original", original), ("new", added)]:
                saved = recorded[(recorded.domain == domain) & (recorded.method == method) & (recorded.subset == subset)].set_index("map").width_ratio
                assert np.allclose(values[method].sort_index(), saved.sort_index(), atol=1e-10)
        for baseline in BASELINES:
            pairs, result = summarize_pair(original, added, baseline)
            result["domain"] = domain
            summaries.append(result)
            for key, row in pairs.iterrows():
                pair_rows.append({"domain": domain, "map": key, "baseline": baseline, **row.to_dict()})
    OUT.mkdir()
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "paired-map-contrasts.csv", index=False)
    # Input checksums are verified by the reproduction driver.
    metadata = {"status": "passed", "protocol_sha256": digest(PROTOCOL), "driver_sha256": digest(__file__),
                "source_tree": "source checksums recorded by the reproduction driver",
                 "query_rows_checked": len(q),
                "comparisons": len(summaries), "paired_map_comparisons": len(pair_rows),
                "interpretation": "Prespecified diagnostic on reused outcomes; fixed map-macro weights and query normalizers; exact accounting, not causal effects or independent confirmation."}
    (OUT / "verification.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
