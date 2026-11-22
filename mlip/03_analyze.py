#!/usr/bin/env python3
"""Extract diffusion coefficients and room-temperature conductivity from MD trajectories."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import MLIP_RESULTS, MLIP_TRAJECTORIES, SCREENING_RESULTS

T_ROOM = 300.0
KB_EV = 8.617333e-5
TEMPS = [800, 900, 1000, 1100, 1200]

METHOD_BY_ID = {
    "M1": "lnD",
    "M2": "lnD",
    "M3": "lnD",
    "M4": "lnD",
    "M5": "lnSigma",
    "M6": "lnD",
    "M7": "lnSigma",
    "M8": "lnD",
}


def collect_trajectories(root):
    files = {}
    for path in sorted(root.rglob("M*_T*K_r0.traj")):
        match = re.match(r"(M\d+)_T(\d+)K_r0\.traj$", path.name)
        if not match:
            continue
        m_id, temperature = match.group(1), int(match.group(2))
        if temperature in TEMPS and m_id not in files:
            files[m_id] = files.get(m_id, {})
            files[m_id][temperature] = path
    return files


def analyze_trajectory(path, temperature):
    atoms = read(path, index=":")
    structures = [AseAtomsAdaptor.get_structure(frame) for frame in atoms]
    analyzer = DiffusionAnalyzer.from_structures(
        structures,
        specie="Na",
        temperature=temperature,
        time_step=100,
        step_skip=1,
        avg_nsteps=200,
        min_obs=30,
    )
    summary = analyzer.get_summary_dict()
    return float(summary["D"]), float(summary["S"]) / 1000.0


def fit_arrhenius(m_id, temperatures, diffusivities, conductivities):
    method = METHOD_BY_ID.get(m_id, "lnD")
    temperatures = np.asarray(temperatures, dtype=float)
    diffusivities = np.asarray(diffusivities, dtype=float)
    conductivities = np.asarray(conductivities, dtype=float)

    if method == "lnSigma":
        mask = temperatures <= 1000
        y = np.log(conductivities[mask])
        x = 1.0 / (temperatures[mask] * KB_EV)
        slope, intercept = np.polyfit(x, y, 1)
        ea = -slope
        log10_rt = (intercept + slope / (T_ROOM * KB_EV)) / np.log(10)
        y_fit = slope * x + intercept
        fit_range = "800-1000 K"
    else:
        y = np.log(diffusivities)
        x = 1.0 / (temperatures * KB_EV)
        slope, intercept = np.polyfit(x, y, 1)
        ea = -slope
        log_ratio = -ea / KB_EV * (1.0 / T_ROOM - 1.0 / temperatures)
        sigma_rt = float(np.mean(conductivities * np.exp(log_ratio)))
        log10_rt = np.log10(sigma_rt) if sigma_rt > 0 else np.nan
        y_fit = slope * x + intercept
        fit_range = "800-1200 K"

    residuals = y - y_fit
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fit_rows = pd.DataFrame({
        "MLIP_ID": m_id,
        "method": method,
        "temperature": temperatures,
        "x": x,
        "y": y,
        "y_fit": y_fit,
        "Ea_eV": ea,
        "R2": r2,
        "fit_range": fit_range,
    })
    return ea, r2, log10_rt, fit_rows


def main():
    OUTPUT_DIR = MLIP_RESULTS / "analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(SCREENING_RESULTS / "MLIP_Candidates.csv")
    prediction = dict(zip(candidates["MLIP_ID"], candidates["pred_log10S"]))
    s_x = dict(zip(candidates["MLIP_ID"], candidates["S_X"]))

    trajectory_root = MLIP_TRAJECTORIES
    if not trajectory_root.exists():
        trajectory_root = MLIP_RESULTS / "md"
    trajectory_files = collect_trajectories(trajectory_root)
    if not trajectory_files:
        raise FileNotFoundError(f"No M*_T*K_r0.traj files found under {trajectory_root}")

    summary_rows = []
    fit_frames = []
    for m_id in candidates["MLIP_ID"]:
        temperatures, diffusivities, conductivities = [], [], []
        for temperature in TEMPS:
            path = trajectory_files.get(m_id, {}).get(temperature)
            if path is None:
                continue
            try:
                diffusion, conductivity = analyze_trajectory(path, temperature)
            except Exception as error:
                print(f"{m_id} {temperature} K: {error}")
                continue
            if np.isfinite(diffusion) and np.isfinite(conductivity) and conductivity > 0:
                temperatures.append(temperature)
                diffusivities.append(diffusion)
                conductivities.append(conductivity)

        if len(temperatures) < 2:
            print(f"{m_id}: fewer than two valid temperatures, skipped")
            continue

        ea, r2, log10_rt, fit_rows = fit_arrhenius(
            m_id, temperatures, diffusivities, conductivities
        )
        predicted = prediction.get(m_id, np.nan)
        summary_rows.append({
            "MLIP_ID": m_id,
            "S_X_R": s_x.get(m_id, np.nan) / 8.314,
            "method": METHOD_BY_ID.get(m_id, "lnD"),
            "Ea_eV": ea,
            "R2": r2,
            "pred_log10S": predicted,
            "MD_log10S_RT": log10_rt,
            "Delta": log10_rt - predicted if np.isfinite(log10_rt) else np.nan,
            "fit_range": fit_rows["fit_range"].iloc[0],
        })
        fit_frames.append(fit_rows)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "final_summary.csv", index=False)
    if fit_frames:
        pd.concat(fit_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "arrhenius_fits.csv", index=False
        )

    print(summary.to_string(index=False))
    print(f"Written: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
