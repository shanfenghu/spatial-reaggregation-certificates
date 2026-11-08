"""Render numerical comparisons from the reproduced results."""
from pathlib import Path
import argparse, hashlib, json, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
V=Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(fig,destination,name):
    for ext in ["pdf","png"]:
        fig.savefig(destination/f"{name}.{ext}",bbox_inches="tight",dpi=220,
                    metadata={"Title":name,"Author":""} if ext=="pdf" else None)
    plt.close(fig)
def main(destination):
    destination.mkdir(parents=True,exist_ok=False)
    inputs=[V/"run/precision/query-diagnostics.csv",
            V/"run/crossing/query-diagnostics.csv",
            V/"run/precision/summary.csv"]
    before={str(p.relative_to(V)):digest(p) for p in inputs}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    queries = pd.read_csv(inputs[1]).query("run == 'primary'")
    diagnosis = pd.read_csv(inputs[2]).query("run == 'primary'").rename(columns={"certified_useful":"certified_useful_queries","certified_broad":"certified_broad_queries","requires_sort_for_classification":"requires_sort_queries"}).set_index("domain")
    assert len(queries) == 1194
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    for domain, label, marker, color in [("census", "Census", "o", "#176b87"),
                                         ("egrid", "Energy", "^", "#b06e27")]:
        group = queries[queries.domain == domain]
        axes[0].scatter(group.ratio_tau_to_saturation, 100 * group.observed_narrowing,
                        s=18, alpha=.85, marker=marker, c=color, edgecolors="none", label=label)
    axes[0].axvline(1, color="#555", lw=.9, ls="--")
    axes[0].axhline(5, color="#777", lw=.7, ls=":")
    axes[0].set(xscale="log", xlabel="Budget / saturation budget", ylabel="Query narrowing (%)",
                ylim=(-2, 70))
    axes[0].set_title("(a) Budget compatibility", loc="left", fontsize=12.5)
    axes[0].set_xticks([1, 100, 10000], ["1", "100", "10,000"])
    axes[0].legend(frameon=False, fontsize=11.5)
    left = np.zeros(2)
    categories = ["certified_useful_queries", "certified_broad_queries", "requires_sort_queries"]
    assert np.array_equal(diagnosis.loc[["census", "egrid"], categories].sum(axis=1),
                          diagnosis.loc[["census", "egrid"], "queries"])
    for key, label, color, hatch in [
        (categories[0], "Narrow", "#297e93", ""),
        (categories[1], "Broad", "#c6cbce", "///"),
        (categories[2], "Unresolved", "#dbad6b", "..")]:
        values = diagnosis.loc[["census", "egrid"], key].to_numpy()
        pct = 100 * values / diagnosis.loc[["census", "egrid"], "queries"].to_numpy()
        axes[1].barh([0, 1], pct, left=left, color=color, hatch=hatch,
                     edgecolor="#555", linewidth=.5, height=.5, label=label)
        for i in range(2):
            if pct[i] > 10:
                axes[1].text(left[i] + pct[i]/2, i, str(values[i]), ha="center", va="center",
                             fontsize=11.5, color="white" if key == categories[0] else "black")
        left += pct
    axes[1].set(yticks=[0, 1], yticklabels=["Census\n36 queries", "Energy\n1,158 queries"],
                xlim=(0, 100), xlabel="Registered queries (%)")
    axes[1].set_title("(b) Before-sort certificate", loc="left", fontsize=12.5)
    axes[1].invert_yaxis()
    # One common category legend belongs beneath the paired diagnostic, keeping
    # the plot panels open and avoiding a small legend embedded in the bars.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=11.5, columnspacing=1.)
    fig.tight_layout(rect=(0, .08, 1, 1), w_pad=1.5)
    save(fig, destination, "precision-regimes")

    assert before=={str(p.relative_to(V)):digest(p) for p in inputs}
    provenance={"scope":"Prespecified post-study certificate refinement; no query/endpoint/rank changes",
                "input_sha256":before,"recipe_sha256":digest(Path(__file__)),
                "primary_classifications":diagnosis.loc[["census","egrid"],categories+["queries"]].to_dict("index"),
                "output_sha256":{p.name:digest(p) for p in destination.glob("precision-regimes.*")}}
    (destination/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(destination)
