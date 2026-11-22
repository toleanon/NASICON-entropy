#!/usr/bin/env python3
"""Create the high-throughput screening figures from saved results."""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import FIGURES_DIR, MODEL_DIR, SCREENING_FIGURES, SCREENING_RESULTS

R_GAS = 8.314
FEATURE_COLS = ["Na_ratio", "Δr_MX", "Δχ_XM", "Sinter_intensity", "struct", "S_M", "S_X"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(SCREENING_FIGURES / f"{name}.png")
    fig.savefig(SCREENING_FIGURES / f"{name}.pdf")
    plt.close(fig)


def plot_space(full, top30, mlip):
    x_all = (full["S_M"] + full["S_X"]) / R_GAS
    x_top = (top30["S_M"] + top30["S_X"]) / R_GAS
    x_mlip = (mlip["S_M"] + mlip["S_X"]) / R_GAS

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.scatter(x_all, full["pred_log10S"], s=8, color="#999999", alpha=0.35,
               edgecolor="none", label=f"All candidates (n={len(full)})")
    ax.scatter(x_top, top30["pred_log10S"], s=18, color="#5AACE0",
               edgecolor="black", linewidth=0.25, label="Stratified Top30")
    ax.scatter(x_mlip, mlip["pred_log10S"], s=55, marker="*", color="#A670AD",
               edgecolor="black", linewidth=0.3, zorder=5, label="M1-M8")
    for _, row in mlip.iterrows():
        offset = {"M4": (-20, -10), "M5": (5, -13), "M6": (8, -12), "M7": (6, 5)}.get(
            row["MLIP_ID"], (4, 3)
        )
        ax.annotate(row["MLIP_ID"], ((row["S_M"] + row["S_X"]) / R_GAS,
                                     row["pred_log10S"]),
                    xytext=offset, textcoords="offset points", fontsize=6)
    ax.axhline(-3.5, color="#EB606D", linestyle="--", linewidth=0.8,
               label="Performance threshold")
    ax.set_xlabel("Screening site-sum, $S_{\\mathrm{total}}/R$")
    ax.set_ylabel("Predicted $\\log_{10}(\\sigma)$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="#E5E7EA", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "Screening_Space")


def plot_scores(mlip):
    metrics = [
        ("rank_perf", "Performance"),
        ("score_synthesizability", "Synthesizability"),
        ("score_entropy_coverage", "Entropy\ncoverage"),
        ("rank_novel", "Novelty"),
    ]
    values = mlip[[key for key, _ in metrics]].to_numpy(dtype=float)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(mlip)))

    fig, ax = plt.subplots(figsize=(4.8, 4.6), subplot_kw={"polar": True})
    fig.subplots_adjust(left=0.16, right=0.84, top=0.90, bottom=0.25)
    for index, (_, row) in enumerate(mlip.iterrows()):
        vals = values[index].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[index], linewidth=1.1,
                marker="o", markersize=2.5, label=row["MLIP_ID"])
        ax.fill(angles, vals, color=colors[index], alpha=0.04)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in metrics])
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, frameon=False, columnspacing=1.0, handletextpad=0.3)
    save(fig, "Score_Profiles")


def plot_entropy(full, top30, mlip):
    bins = np.linspace(0, 0.75, 36)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist((full["S_X"] / R_GAS), bins=bins, density=True, color="#999999",
            alpha=0.35, label="All candidates")
    ax.hist((top30["S_X"] / R_GAS), bins=bins, density=True, color="#5AACE0",
            alpha=0.55, label="Stratified Top30")
    for _, row in mlip.iterrows():
        ax.axvline(row["S_X"] / R_GAS, color="#A670AD", linewidth=0.6, alpha=0.5)
    ax.scatter(mlip["S_X"] / R_GAS, np.full(len(mlip), -0.15),
               marker="|", s=40, color="#A670AD", label="M1-M8")
    ax.set_xlabel("$S_X/R$")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=-0.25)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E7EA", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "XSite_Entropy")


def plot_shap(mlip):
    model_path = FIGURES_DIR / "best_model_7d.pkl"
    scaler_path = MODEL_DIR / "processed" / "scaler_7d.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Train the model stage before plotting SHAP.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    row = mlip[mlip["MLIP_ID"] == "M2"]
    if row.empty:
        raise ValueError("M2 not found in MLIP_Candidates.csv")

    raw = row[FEATURE_COLS].to_numpy(dtype=float)
    scaled = scaler.transform(raw)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(scaled)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[0])

    explanation = shap.Explanation(
        values=values[0],
        base_values=expected,
        data=scaled[0],
        feature_names=FEATURE_COLS,
    )
    shap.waterfall_plot(explanation, show=False, max_display=len(FEATURE_COLS))
    fig = plt.gcf()
    save(fig, "SHAP_Profile_M2")


def main():
    SCREENING_FIGURES.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(SCREENING_RESULTS / "full_candidates.csv")
    top30 = pd.read_csv(SCREENING_RESULTS / "Top30_Stratified.csv")
    mlip = pd.read_csv(SCREENING_RESULTS / "MLIP_Candidates.csv")

    plot_space(full, top30, mlip)
    plot_scores(mlip)
    plot_entropy(full, top30, mlip)
    plot_shap(mlip)
    print(f"Written: {SCREENING_FIGURES}")


if __name__ == "__main__":
    main()
