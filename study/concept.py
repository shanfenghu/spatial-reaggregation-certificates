"""Construct and verify a two-source allocation and query-range example."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

V = Path(__file__).resolve().parents[1]
OUT = V / "run/figures/concept"
BLUE, ORANGE, PURPLE = "#23798a", "#ad5c14", "#765ba0"


def bounds(total, budget):
    """Exact left-point query range for one two-point source, distance 1 km."""
    half = total / 2
    return max(0.0, half - budget), min(total, half + budget)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    predictor = np.array([50.0, 50.0, 10.0, 10.0])
    worlds = [np.array([40., 60., 20., 0.]), np.array([60., 40., 0., 20.])]
    records = []
    for world in worlds:
        assert world[:2].sum() == 100 and world[2:].sum() == 20
        # Each two-point source has one independent net transfer at unit distance.
        cost = abs(world[0] - predictor[0]) + abs(world[2] - predictor[2])
        assert cost == 20
        assert world[[0, 2]].sum() == 60
        records.append({"allocation": world.tolist(), "cost_mass_km": cost,
                        "registered_q": float(world[[0, 2]].sum()),
                        "new_r": float(world[0]), "new_s": float(world[2])})
    assert bounds(100, 20) == (30, 70) and bounds(20, 20) == (0, 20)
    checks = []
    # The positive crossing costs are all one: the existing cost-range certificate
    # is exact, H+ = H- = (total/2 - budget)+, on these two selected queries.
    for total in [100, 20]:
        for budget in [0, 2, 5, 10, 20, 35, 50, 55]:
            lo, hi = bounds(total, budget)
            h = max(total / 2 - budget, 0)
            assert abs((total - (hi - lo)) - 2*h) < 1e-12
            checks.append({"source_total": total, "budget_mass_km": budget,
                           "lower": lo, "upper": hi,
                           "width_fraction": (hi-lo)/total,
                           "certified_narrowing_fraction": 2*h/total})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(7.6, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.55], wspace=.42,
                          left=.025, right=.985, top=.84, bottom=.23)
    for i, world in enumerate(worlds):
        ax = fig.add_subplot(gs[i])
        for y, total in [(.15, 100), (-1.15, 20)]:
            ax.add_patch(Rectangle((.12, y), 1.98, 1.0, fc="#f5f5f3", ec="#656565", lw=.9))
            label_y = y+1.12 if total == 100 else y-.19
            ax.text(1.11, label_y, f"{'A' if total == 100 else 'B'}: total {total}",
                    ha="center", fontsize=12)
        ax.add_patch(Rectangle((.25, -1.08), .65, 2.16, fc=BLUE, alpha=.16,
                               ec=BLUE, lw=1.4))
        ax.add_patch(Rectangle((.25, .24), .65, .78, fill=False, ec=ORANGE,
                               lw=1.7, ls="--"))
        ax.add_patch(Rectangle((.25, -1.06), .65, .78, fill=False, ec=PURPLE,
                               lw=1.7, ls=":"))
        for (x, y), mass in zip([(.57, .65), (1.58, .65), (.57, -.65), (1.58, -.65)], world):
            ax.scatter(x, y, s=31, color="#303434", zorder=4)
            ax.text(x, y-.28, f"{mass:g}", ha="center", fontsize=14, weight="bold")
        ax.set(xlim=(0, 2.25), ylim=(-1.48, 1.52))
        ax.set_axis_off()
        ax.set_title(f"({'ab'[i]}) Allocation {i+1}", loc="left", fontsize=13.2,
                     weight="bold", pad=18)
        ax.text(.0, -.13, "q: 60", transform=ax.transAxes, color=BLUE,
                fontsize=14, weight="bold")
        ax.text(.42, -.13, f"r: {world[0]:g}", transform=ax.transAxes,
                color=ORANGE, fontsize=14, weight="bold")
        ax.text(.0, -.25, f"s: {world[2]:g}", transform=ax.transAxes,
                color=PURPLE, fontsize=14, weight="bold")

    ax = fig.add_subplot(gs[2])
    grid = np.sort(np.unique(np.r_[np.linspace(0, 55, 101), 10, 20, 50]))
    for total, color, style in [(100, ORANGE, "--"), (20, PURPLE, ":")]:
        width = np.array([bounds(total, b)[1]-bounds(total, b)[0] for b in grid])/total
        ax.plot(grid, width, color=color, ls=style, lw=2.4)
    ax.axvline(20, color="#737777", lw=1, ls=(0, (3, 3)))
    ax.scatter([20, 20], [.4, 1], color=[ORANGE, PURPLE], s=28, zorder=5)
    ax.text(23, .23, "r: 60% narrower\n50×1 to saturate", color=ORANGE, fontsize=12)
    ax.text(23, 1.015, "s: coarse-only\n10×1 to saturate", color=PURPLE, fontsize=12)
    ax.set(xlim=(0, 56), ylim=(0, 1.17), xticks=[0, 10, 20, 50],
           yticks=[0, .4, 1], xlabel="Budget (mass-km)",
           ylabel="Interval width / coarse width")
    ax.set_yticklabels(["0", "0.4", "1"])
    ax.tick_params(labelsize=12)
    ax.set_title("(c) Both admitted\nat budget 20", loc="left",
                 fontsize=14, weight="bold", pad=12)
    ax.grid(axis="y", color="#e7e7e7", lw=.6)
    ax.set_axisbelow(True)
    fig.text(.025, .015, "Blue q: registered    Orange dashed r, purple dotted s: new queries",
             fontsize=12)
    fig.savefig(OUT/"allocation-query-precision.pdf", bbox_inches="tight",
                metadata={"Title": "Allocation containment and query precision",
                          "Author": "", "Subject": "Constructed illustration"})
    fig.savefig(OUT/"allocation-query-precision.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    record = {"status": "constructed illustration; no fitted budget or empirical sample",
              "environment": {"python": sys.version.split()[0],
                              "numpy": np.__version__, "matplotlib": matplotlib.__version__},
              "randomness": "none",
              "predictor": predictor.tolist(), "source_totals": [100, 20],
              "within_source_distance_km": 1, "normalization_total_mass": 120,
              "worlds": records, "illustrative_budget_mass_km": 20,
              "query_thresholds_mass_km": {"r": 50, "s": 10},
              "checks": checks,
              "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "outputs": {f"allocation-query-precision.{ext}": hashlib.sha256(
                  (OUT/f"allocation-query-precision.{ext}").read_bytes()).hexdigest()
                          for ext in ["pdf", "png"]}}
    (OUT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Illustrative worlds, endpoint ranges and 16 certificate cases verified; Example rendered.")


if __name__ == "__main__":
    main()
