#!/usr/bin/env python3
"""Create the manuscript figures from the saved model results."""

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

from paths import FIGURES_DIR

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)


def plot_performance(results):
    order = ["RF", "XGB", "LGBM", "GBDT", "SVR", "Stacking"]
    data = results.pivot(index="Model", columns="Feature Set", values="Test R2")
    data = data.reindex([model for model in order if model in data.index])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(data))
    width = 0.36
    for offset, feature_set, color in ((-width / 2, "5d", "#8EC6E6"),
                                       (width / 2, "7d", "#F2A36B")):
        if feature_set in data:
            ax.bar(x + offset, data[feature_set], width, label=feature_set, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index)
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E7EA", linewidth=0.5)
    save(fig, "Model_Performance_Bars")


def plot_parity(plot_data):
    y = np.asarray(plot_data["y_test"])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for key, label, color in (
        ("pred_5d", "5d", "#8EC6E6"),
        ("pred_7d", "7d", "#F2A36B"),
        ("pred_stacking", "Stacking", "#A670AD"),
    ):
        ax.scatter(y, plot_data[key], s=22, alpha=0.75, label=label, color=color)
    limits = [min(y.min(), -2.2), max(y.max(), -0.2)]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlabel("Experimental $\\log_{10}(\\sigma)$")
    ax.set_ylabel("Predicted $\\log_{10}(\\sigma)$")
    ax.legend(frameon=False)
    save(fig, "Model_Parity")


def plot_importance(plot_data):
    importance = plot_data["feature_importance"].sort_values("Importance")
    weights = plot_data["stacking_weights"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].barh(importance["Feature"], importance["Importance"], color="#2E86AB")
    axes[0].set_xlabel("Feature importance")
    axes[1].barh(weights["Base Model"], weights["Weight"], color="#5D3A9B")
    axes[1].set_xlabel("Stacking weight")
    for ax in axes:
        ax.grid(True, axis="x", color="#E5E7EA", linewidth=0.5)
    save(fig, "Model_Importance")


def plot_delta(delta):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(delta))
    width = 0.36
    ax.bar(x - width / 2, delta["Delta_CV_R2"], width, label="CV", color="#8EC6E6")
    ax.bar(x + width / 2, delta["Delta_Test_R2"], width, label="Test", color="#F2A36B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(delta["Model"])
    ax.set_ylabel("$\\Delta R^2$ (7d - 5d)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E7EA", linewidth=0.5)
    save(fig, "Model_Entropy_Delta")


def plot_repeated(repeated):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(repeated))
    width = 0.36
    ax.bar(x - width / 2, repeated["5d_Test_R2_mean"], width,
           yerr=repeated["5d_Test_R2_std"], capsize=3, label="5d", color="#8EC6E6")
    ax.bar(x + width / 2, repeated["7d_Test_R2_mean"], width,
           yerr=repeated["7d_Test_R2_std"], capsize=3, label="7d", color="#F2A36B")
    ax.set_xticks(x)
    ax.set_xticklabels(repeated["Model"])
    ax.set_ylabel("Test $R^2$")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E5E7EA", linewidth=0.5)
    save(fig, "Model_Repeated_Splits")


def plot_shap(plot_data):
    values = plot_data["shap_values"]
    features = plot_data["shap_data"]
    names = plot_data["features_7d"]

    shap.summary_plot(values, features, feature_names=names, plot_type="bar", show=False)
    fig = plt.gcf()
    save(fig, "Model_SHAP_Bar")

    shap.summary_plot(values, features, feature_names=names, show=False)
    fig = plt.gcf()
    save(fig, "Model_SHAP_Beeswarm")


def main():
    # Load the saved model results and draw the manuscript figures.
    data = joblib.load(FIGURES_DIR / "plot_data.pkl")
    plot_performance(data["results"])
    plot_parity(data)
    plot_importance(data)
    plot_delta(data["delta"])
    plot_repeated(data["repeated"])
    plot_shap(data)
    print(f"Figures written: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
