#!/usr/bin/env python3
"""Create the MLIP validation figures from summaries and optional local trajectories."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import MLIP_FIGURES, MLIP_RESULTS, MLIP_TRAJECTORIES

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

PAL = {
    "teal": "#70C6CC",
    "purple": "#A670AD",
    "red": "#EB606D",
    "orange": "#F7C38A",
    "blue": "#5AACE0",
    "gray": "#7F7F7F",
    "grid": "#E5E7EA",
}


def save(fig, name):
    fig.savefig(MLIP_FIGURES / f"{name}.png")
    fig.savefig(MLIP_FIGURES / f"{name}.pdf")
    plt.close(fig)


def find_trajectory(m_id, temperature):
    patterns = [f"{m_id}_T{temperature}K_r0.traj", f"{m_id}_T{temperature}K*.traj",
                f"{m_id}_T{temperature}K*.xyz"]
    for pattern in patterns:
        matches = sorted(MLIP_TRAJECTORIES.rglob(pattern))
        if matches:
            return matches[0]
    return None


def label_points(ax, frame, x_col, y_col):
    offsets = {"M4": (-18, -10), "M6": (6, -12), "M7": (6, 5), "M8": (6, -10)}
    for _, row in frame.iterrows():
        dx, dy = offsets.get(row["MLIP_ID"], (5, 5))
        ax.annotate(row["MLIP_ID"], (row[x_col], row[y_col]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6)


def plot_summary(summary, stability):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    ax = axes[0, 0]
    ax.scatter(stability["Vol_change_pct"], stability["E_rel_meV"],
               s=42, color=PAL["blue"], edgecolor="black", linewidth=0.3)
    label_points(ax, stability, "Vol_change_pct", "E_rel_meV")
    ax.set_xlabel("Volume change (%)")
    ax.set_ylabel("Relative energy (meV atom$^{-1}$)")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    low = min(summary["pred_log10S"].min(), summary["MD_log10S_RT"].min()) - 0.2
    high = max(summary["pred_log10S"].max(), summary["MD_log10S_RT"].max()) + 0.2
    ax.plot([low, high], [low, high], "k--", linewidth=0.7)
    ax.scatter(summary["pred_log10S"], summary["MD_log10S_RT"],
               s=46, color=PAL["purple"], edgecolor="black", linewidth=0.3)
    label_points(ax, summary, "pred_log10S", "MD_log10S_RT")
    ax.set_xlabel("ML prediction, $\\log_{10}\\sigma$")
    ax.set_ylabel("MLIP-MD, $\\log_{10}\\sigma$")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    ax.scatter(summary["S_X_R"], summary["Ea_eV"],
               s=46, color=PAL["orange"], edgecolor="black", linewidth=0.3)
    label_points(ax, summary, "S_X_R", "Ea_eV")
    ax.set_xlabel("$S_X/R$")
    ax.set_ylabel("Activation energy (eV)")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    ax.scatter(summary["S_X_R"], summary["MD_log10S_RT"],
               s=46, color=PAL["teal"], edgecolor="black", linewidth=0.3)
    label_points(ax, summary, "S_X_R", "MD_log10S_RT")
    ax.set_xlabel("$S_X/R$")
    ax.set_ylabel("MLIP-MD, $\\log_{10}\\sigma$")
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "MLIP_Summary")


def plot_arrhenius(fits):
    candidates = sorted(fits["MLIP_ID"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6))
    axes = axes.ravel()
    for index, m_id in enumerate(candidates):
        ax = axes[index]
        subset = fits[fits["MLIP_ID"] == m_id]
        method = subset["method"].iloc[0]
        ax.scatter(subset["x"], subset["y"], s=16, color=PAL["blue"],
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot(subset["x"], subset["y_fit"], "k--", linewidth=0.8)
        ax.set_title(f"{m_id} ({method})", fontsize=7)
        ax.set_xlabel("$1/(k_BT)$ (eV$^{-1}$)", fontsize=7)
        ax.set_ylabel("ln $D$ or ln $\\sigma$", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, color=PAL["grid"], linewidth=0.4)
    for ax in axes[len(candidates):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "Arrhenius_Fits")


def plot_msd():
    targets = ["M4", "M6", "M8"]
    temperature = 1000
    series = {}
    for m_id in targets:
        path = find_trajectory(m_id, temperature)
        if path is None:
            continue
        atoms = read(path, index=":")
        structures = [AseAtomsAdaptor.get_structure(frame) for frame in atoms]
        analyzer = DiffusionAnalyzer.from_structures(
            structures, specie="Na", temperature=temperature,
            time_step=100, step_skip=1, avg_nsteps=200, min_obs=30,
        )
        series[m_id] = np.asarray(analyzer.msd)
    if not series:
        print("[skip] MSD: no local trajectories")
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    colors = {"M4": PAL["red"], "M6": PAL["purple"], "M8": PAL["gray"]}
    for m_id, msd in series.items():
        time = np.arange(len(msd)) * 0.1
        ax.plot(time, msd, color=colors[m_id], linewidth=1.2, label=m_id)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Na MSD (Å$^2$)")
    ax.legend(frameon=False)
    ax.grid(True, color=PAL["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "MLIP_MSD")


def plot_waterfall(summary):
    frame = summary.sort_values("S_X_R").reset_index(drop=True)
    fig = plt.figure(figsize=(8.4, 3.8))
    ax1 = fig.add_axes([0.08, 0.15, 0.38, 0.75])
    ax2 = fig.add_axes([0.52, 0.06, 0.44, 0.88], projection="3d")

    x = np.arange(len(frame))
    ax1.scatter(x, frame["pred_log10S"], marker="X", s=45, color=PAL["gray"],
                edgecolor="black", linewidth=0.4, label="ML prediction")
    ax1.scatter(x, frame["MD_log10S_RT"], marker="P", s=45, color=PAL["purple"],
                edgecolor="black", linewidth=0.4, label="MLIP-MD")
    ax1.set_xticks(x)
    ax1.set_xticklabels(frame["MLIP_ID"], fontsize=7)
    ax1.set_ylabel("$\\log_{10}\\sigma$ (S cm$^{-1}$)")
    ax1.legend(frameon=False, fontsize=6)
    ax1.grid(True, color=PAL["grid"], linewidth=0.4)

    verts, face, edge = [], [], []
    for index, row in frame.iterrows():
        sx = row["S_X_R"]
        ea = row["Ea_eV"]
        local_x = np.linspace(sx - 0.06, sx + 0.06, 80)
        local_z = ea * np.exp(-((local_x - sx) / 0.03) ** 2)
        verts.append(list(zip(local_x, local_z)))
        color = PAL["red"] if row["MLIP_ID"] in ("M4", "M8") else (
            PAL["purple"] if row["MLIP_ID"] == "M6" else PAL["orange"]
        )
        face.append(mcolors.to_rgba(color, 0.55))
        edge.append(mcolors.to_rgba(color, 1.0))
    poly = PolyCollection(verts, facecolors=face, edgecolors=edge, linewidths=1.0)
    ax2.add_collection3d(poly, zs=range(len(frame)), zdir="y")
    ax2.set_xlabel("$S_X/R$")
    ax2.set_zlabel("$E_a$ (eV)")
    ax2.set_yticks(range(len(frame)))
    ax2.set_yticklabels(frame["MLIP_ID"], fontsize=6)
    ax2.set_xlim(frame["S_X_R"].min() - 0.08, frame["S_X_R"].max() + 0.08)
    ax2.set_ylim(-0.5, len(frame) - 0.5)
    ax2.set_zlim(0.08, 0.25)
    ax2.view_init(elev=24, azim=-55)
    ax2.grid(False)

    fig.savefig(MLIP_FIGURES / "MLIP_Waterfall.png")
    fig.savefig(MLIP_FIGURES / "MLIP_Waterfall.pdf")
    plt.close(fig)


def plot_density():
    targets = ["M6", "M8"]
    temperature = 1200
    maps = {}
    for m_id in targets:
        path = find_trajectory(m_id, temperature)
        if path is None:
            continue
        atoms = read(path, index=":")
        positions = []
        for frame in atoms:
            symbols = np.array(frame.get_symbols())
            positions.append(frame.get_positions()[symbols == "Na"][:, :2])
        if positions:
            maps[m_id] = np.vstack(positions)
    if not maps:
        print("[skip] Density: no local trajectories")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, m_id in zip(axes, targets):
        if m_id not in maps:
            ax.axis("off")
            continue
        hist, xedges, yedges = np.histogram2d(
            maps[m_id][:, 0], maps[m_id][:, 1], bins=60
        )
        ax.imshow(hist.T, origin="lower", aspect="auto",
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap="magma")
        ax.set_title(m_id, fontsize=8)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
    fig.tight_layout()
    save(fig, "MLIP_Density")


def plot_displacement():
    temperature = 800
    datasets = {}
    for m_id in ("M2", "M3"):
        path = find_trajectory(m_id, temperature)
        if path is None:
            continue
        atoms = read(path, index=":")
        initial = atoms[0].get_positions()
        final = atoms[-1].get_positions()
        symbols = np.array(atoms[0].get_symbols())
        displacement = np.linalg.norm(final[symbols == "Na"] - initial[symbols == "Na"], axis=1)
        datasets[m_id] = displacement
    if not datasets:
        print("[skip] Displacement: no local trajectories")
        return

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    bins = np.linspace(0, 8, 32)
    for m_id, values in datasets.items():
        ax.hist(values, bins=bins, density=True, alpha=0.45, label=m_id)
    ax.set_xlabel("Na displacement (Å)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.5)
    fig.tight_layout()
    save(fig, "MLIP_Displacement")


def plot_vector_map():
    temperature = 800
    datasets = {}
    for m_id in ("M2", "M3"):
        path = find_trajectory(m_id, temperature)
        if path is None:
            continue
        atoms = read(path, index=":")
        symbols = np.array(atoms[0].get_symbols())
        start = atoms[0].get_positions()[symbols == "Na"]
        end = atoms[-1].get_positions()[symbols == "Na"]
        datasets[m_id] = (start, end)
    if not datasets:
        print("[skip] Vector map: no local trajectories")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, m_id in zip(axes, ("M2", "M3")):
        if m_id not in datasets:
            ax.axis("off")
            continue
        start, end = datasets[m_id]
        vectors = end - start
        ax.scatter(start[:, 0], start[:, 1], s=8, color=PAL["gray"], alpha=0.5)
        ax.quiver(start[:, 0], start[:, 1], vectors[:, 0], vectors[:, 1],
                  angles="xy", scale_units="xy", scale=1, color=PAL["orange"],
                  width=0.003, alpha=0.8)
        ax.set_title(m_id, fontsize=8)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
    fig.tight_layout()
    save(fig, "MLIP_Vector_Map")


def plot_mechanism():
    temperature = 1000
    datasets = {}
    for m_id in ("M6", "M8"):
        path = find_trajectory(m_id, temperature)
        if path is None:
            continue
        atoms = read(path, index=":")
        symbols = np.array(atoms[0].get_symbols())
        start = atoms[0].get_positions()[symbols == "Na"]
        end = atoms[-1].get_positions()[symbols == "Na"]
        datasets[m_id] = (start, end)
    if not datasets:
        print("[skip] Mechanism: no local trajectories")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, m_id in zip(axes, ("M6", "M8")):
        if m_id not in datasets:
            ax.axis("off")
            continue
        start, end = datasets[m_id]
        ax.scatter(start[:, 0], start[:, 1], s=8, color=PAL["gray"], alpha=0.5)
        ax.scatter(end[:, 0], end[:, 1], s=8, color=PAL["purple"], alpha=0.7)
        ax.set_title(m_id, fontsize=8)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
    fig.tight_layout()
    save(fig, "MLIP_Mechanism")


def minmax(values, reverse=False):
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
    return 1.0 - scaled if reverse else scaled


def plot_radar(summary, stability):
    frame = summary.merge(stability[["MLIP_ID", "E_rel_meV", "Vol_change_pct"]], on="MLIP_ID")
    metrics = [
        minmax(frame["MD_log10S_RT"]),
        minmax(frame["Ea_eV"], reverse=True),
        minmax(frame["Vol_change_pct"], reverse=True),
        minmax(frame["R2"]),
        minmax(frame["E_rel_meV"], reverse=True),
    ]
    labels = ["Conductivity", "Activation", "Volume", "Fit quality", "Stability"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.4), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(frame)))
    for index, (_, row) in enumerate(frame.iterrows()):
        values = [metric[index] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, color=colors[index], linewidth=1.0,
                marker="o", markersize=2.5, label=row["MLIP_ID"])
        ax.fill(angles, values, color=colors[index], alpha=0.03)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, "MLIP_Radar")


def main():
    MLIP_FIGURES.mkdir(parents=True, exist_ok=True)
    analysis = MLIP_RESULTS / "analysis"
    summary = pd.read_csv(analysis / "final_summary.csv")
    stability = pd.read_csv(analysis / "stability_summary.csv")
    fits = pd.read_csv(analysis / "arrhenius_fits.csv")

    plot_summary(summary, stability)
    plot_arrhenius(fits)
    plot_msd()
    plot_waterfall(summary)
    plot_density()
    plot_displacement()
    plot_vector_map()
    plot_mechanism()
    plot_radar(summary, stability)
    print(f"Written: {MLIP_FIGURES}")


if __name__ == "__main__":
    main()
