"""Render numerical comparisons from the reproduced results."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[1]
ORDER = ["transport_scan", "reconciled_box", "reconciled_ellipse",
         "aggregate_global", "aggregate_spatial", "label_free", "coarse_only"]
LABELS = ["Transport (ours)", "Reconciled box", "Reconciled ellipsoid",
          "Aggregate global", "Aggregate spatial", "Label-free transport", "Coarse-only"]
COLORS = ["#176b87", "#7c689b", "#ac79a5", "#b46c28", "#d59d42", "#648c7b", "#b9bfc5"]

def digest(path):
    """Return SHA256 for retained input/output identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(destination):
    """Render existing map-macro width/score summaries without changing outcomes."""
    destination.mkdir(parents=True, exist_ok=False)
    source = V / "run/primary/study-summary.csv"
    before = digest(source)
    data = pd.read_csv(source)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
    records = []
    for ax, domain, title in zip(axes, ["census", "egrid"],
        ["(a) Census: 21 test maps", "(b) Energy: 269 test maps"]):
        rows = data[(data.domain == domain) & (data.run == "primary")].set_index("method").loc[ORDER]
        assert rows.width_ratio.between(0, 1).all()
        assert (rows.interval_score >= rows.width_ratio - 1e-12).all()
        y = np.arange(len(ORDER))
        ax.barh(y, rows.width_ratio, color=COLORS, height=.5)
        # Score = width + nonnegative miss penalty, with the same normaliser and
        # map-macro weights. Connecting these quantities exposes this accounting;
        # it is not an uncertainty whisker or an equal-coverage comparison.
        ax.hlines(y, rows.width_ratio, rows.interval_score, color="#252525", lw=1., zorder=3)
        ax.scatter(rows.interval_score, y, marker="D", s=25, color="#202020",
                   edgecolors="white", linewidths=.5, zorder=4)
        ax.set_yticks(y, LABELS)
        ax.set_xlim(0, 1.42)
        ax.set_xticks([0, .25, .5, .75, 1.], ["0", ".25", ".5", ".75", "1"])
        ax.axvline(1, color="#777", lw=.8, ls="--")
        ax.set_title(title, loc="left", fontsize=12.5, weight="bold")
        for j, (method, row) in enumerate(rows.iterrows()):
            ax.text(1.12, j, f"{int(row.catalog_successes)}/{int(row.test_maps)}",
                    va="center", fontsize=12)
            records.append({"domain": domain, "method": method,
                "width_ratio": float(row.width_ratio), "interval_score": float(row.interval_score),
                "catalog_successes": int(row.catalog_successes), "test_maps": int(row.test_maps)})
        ax.text(1.12, -.8, "Catalogue", fontsize=12, color="#444", va="center")
        if domain == "egrid":
            value = rows.loc["transport_scan", "interval_score"]
            ax.annotate(f"{value:.4f}", xy=(value, 0), xytext=(.72, -.65),
                fontsize=12, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": "#333", "lw": .7,
                            "shrinkA": 3, "shrinkB": 4})
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#eee")
        ax.set_axisbelow(True)
    axes[0].set_ylim(6.65, -1.2)
    fig.supxlabel("Coarse-width-normalised width and interval score", fontsize=12, y=.12)
    fig.legend(handles=[Patch(facecolor="#80888e", label="Width (bar)"),
        Line2D([], [], marker="D", color="#202020", linestyle="none", markersize=5,
               label="Miss-penalised score (diamond)")],
        loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=.255, right=.985, top=.87, bottom=.27, wspace=.18)
    for ext in ["pdf", "png"]:
        fig.savefig(destination / f"interval-comparison.{ext}", bbox_inches="tight", dpi=220,
            metadata={"Title": "Interval width and miss-penalised score", "Author": ""} if ext == "pdf" else None)
    plt.close(fig)
    assert before == digest(source)
    (destination / "provenance.json").write_text(json.dumps({
        "scope": "Display of existing primary width/score/count summaries; no new outcomes",
        "input_sha256": {str(source.relative_to(V)): before}, "recipe_sha256": digest(__file__),
        "rows": records, "unchanged": "Methods/order/colours/width values/catalogue counts",
        "added_encoding": "Existing interval score diamonds; width-to-score miss-penalty connectors; reference1",
        "output_sha256": {f"interval-comparison.{ext}": digest(destination / f"interval-comparison.{ext}") for ext in ["pdf", "png"]}}, indent=2)+"\n")
    print(destination)
