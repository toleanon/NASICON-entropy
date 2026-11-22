#!/usr/bin/env python3
"""Plot the constrained analytical model from saved fit results."""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import FIGURES_DIR

PAL = {
    "teal": "#70C6CC",
    "pink": "#F7C1C1",
    "purple": "#A670AD",
    "red": "#EB606D",
    "orange": "#F7C38A",
    "blue": "#5AACE0",
    "gray": "#7F7F7F",
    "text": "#2B2B2B",
    "grid": "#E0E0E0",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})


def plot(data):
    # Four-panel analytical-model figure.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.08,
                        wspace=0.25, hspace=0.24)

    # (a) Experimental trend.
    ax = axes[0, 0]
    mixed = data["is_mixed"]
    ax.scatter(data["sx_all"][~mixed] / data["r_gas"], data["y_all"][~mixed],
               s=6, color=PAL["gray"], alpha=0.22, label="Pure phase")
    for label, marker, color, value in (
        ("Rhombohedral", "o", PAL["red"], 1),
        ("Monoclinic", "s", PAL["blue"], 0),
    ):
        mask = (data["struct_all"] == value) & mixed
        if mask.any():
            ax.scatter(data["sx_all"][mask] / data["r_gas"], data["y_all"][mask],
                       s=16, marker=marker, facecolor=color, edgecolor="white",
                       linewidth=0.3, alpha=0.85, label=label)
    bins = np.linspace(data["sx_all"].min() / data["r_gas"],
                       data["sx_all"].max() / data["r_gas"], 7)
    table = pd.DataFrame({
        "bin": pd.cut(data["sx_all"] / data["r_gas"], bins, duplicates="drop"),
        "y": data["y_all"],
    }).groupby("bin", observed=True)["y"].agg(["mean", "sem"]).reset_index()
    table["mid"] = [interval.mid for interval in table["bin"]]
    ax.errorbar(table["mid"], table["mean"], yerr=table["sem"],
                fmt="o-", color=PAL["purple"], ms=3.5, lw=1.0,
                capsize=2, label="Binned mean +/- SEM")
    ax.set_xlabel(r"$S_X/R$")
    ax.set_ylabel(r"$\log_{10}\sigma$ (S cm$^{-1}$)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    # (b) Fitted curve.
    ax = axes[0, 1]
    ax.plot(data["sx_grid"] / data["r_gas"], data["quad_curve"],
            color=PAL["purple"], lw=1.5,
            label=f"Inverted-U fit ($R^2$ = {data['r2_quad']:.3f})")
    ax.axhline(data["base_line"], color=PAL["teal"], lw=1.5, ls="--",
               label=f"Base linear fit ($R^2$ = {data['r2_base']:.3f})")
    ax.set_xlabel(r"$S_X/R$")
    ax.set_ylabel(r"$\log_{10}\sigma$ (S cm$^{-1}$)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    # (c) Nested model comparison with 50-split inset.
    ax = axes[1, 0]
    labels = ["Base (B)\n4 descriptors",
              r"+ Linear $S_{\mathrm{X}}$ (L)" + "\n5 descriptors",
              r"+ Quadratic $S_{\mathrm{X}}$ (Q)" + "\n6 terms"]
    values = [data["r2_base"], data["r2_linear"], data["r2_quad"]]
    x = np.arange(3)
    ax.plot(x, values, marker="o", color=PAL["teal"], lw=1.0, ms=4, zorder=3)
    ax.scatter(x[-1], values[-1], marker="*", s=90, color=PAL["orange"],
               edgecolor="black", zorder=4)
    for index in range(2):
        delta = values[index + 1] - values[index]
        ax.text((x[index] + x[index + 1]) / 2,
                (values[index] + values[index + 1]) / 2 + 0.002,
                f"+{delta:.3f}", ha="center", va="bottom",
                fontsize=7, color=PAL["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Test $R^2$")
    ax.set_ylim(0.76, 0.82)
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    split_base = data["split_base"]
    split_linear = data["split_linear"]
    split_quad = data["split_quad"]
    delta_linear = split_linear - split_base
    delta_quad = split_quad - split_base
    med_linear = np.median(delta_linear)
    med_quad = np.median(delta_quad)
    q1_linear, q3_linear = np.percentile(delta_linear, [25, 75])
    q1_quad, q3_quad = np.percentile(delta_quad, [25, 75])

    inset = ax.inset_axes([0.50, 0.09, 0.46, 0.40])
    inset.axhline(0, color=PAL["gray"], lw=0.6, alpha=0.7)
    inset.errorbar([0, 1, 2], [0, med_linear, med_quad],
                   yerr=[[0, med_linear - q1_linear, med_quad - q1_quad],
                         [0, q3_linear - med_linear, q3_quad - med_quad]],
                   fmt="o-", color=PAL["teal"], lw=1.0, ms=3.2, capsize=2)
    inset.text(1, med_linear + 0.025, f"+{med_linear:.3f}",
               ha="center", va="bottom", fontsize=6.5, color=PAL["teal"])
    inset.text(2, med_quad + 0.035, f"+{med_quad:.3f}",
               ha="center", va="bottom", fontsize=6.5, color=PAL["teal"])
    inset.text(0.05, 0.92, f"p = {data['p_quadratic']:.1e}",
               transform=inset.transAxes, ha="left", va="top", fontsize=6.5)
    inset.set_xlim(-0.35, 2.35)
    inset.set_ylim(0, max(0.12, q3_quad * 1.25))
    inset.set_xticks([0, 1, 2])
    inset.set_xticklabels(["B", "L", "Q"], fontsize=6.5)
    inset.set_ylabel(r"$\Delta R^2$ vs base", fontsize=6.5)
    inset.set_title("50 splits, paired median", fontsize=6.5, pad=2)
    inset.tick_params(axis="both", labelsize=6, length=2)
    inset.grid(True, axis="y", alpha=0.3)

    # (d) Parity.
    ax = axes[1, 1]
    ax.scatter(data["y_train"], data["pred_train"], s=14,
               facecolor=PAL["pink"], edgecolor="white", linewidth=0.3,
               alpha=0.85, label=f"Train ($R^2$ = {data['r2_quad_train']:.3f})")
    ax.scatter(data["y_test"], data["pred_test"], s=24,
               facecolor=PAL["red"], edgecolor="white", linewidth=0.3,
               alpha=0.9, label=f"Test ($R^2$ = {data['r2_quad']:.3f})")
    lower = min(data["y_train"].min(), data["y_test"].min()) - 0.2
    upper = max(data["y_train"].max(), data["y_test"].max()) + 0.2
    ax.plot([lower, upper], [lower, upper], "k--", lw=0.7)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"True $\log_{10}\sigma$")
    ax.set_ylabel(r"Predicted $\log_{10}\sigma$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    for axis in axes.ravel():
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")

    fig.savefig(FIGURES_DIR / "Analytical_Model.pdf")
    fig.savefig(FIGURES_DIR / "Analytical_Model.png")
    plt.close(fig)


def main():
    # Load the fit results produced by 01_fit_formula.py.
    data = joblib.load(FIGURES_DIR / "analytical_plot_data.pkl")
    plot(data)
    print(f"Written: {FIGURES_DIR / 'Analytical_Model.png'}")


if __name__ == "__main__":
    main()
