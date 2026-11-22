# MLIP validation

This folder contains the MACE-MP-0 relaxation, molecular-dynamics, analysis, and
plotting code for the M1–M8 validation set.

## Inputs

- `screening/results/MLIP_Candidates.csv` from the screening stage.
- `mlip/seeds/Na3Zr2Si2PO12_R3c.cif` for the parent structure (local only; not
  redistributed).
- MLIP trajectories under `mlip/trajectories/` (local only; not redistributed).

The MACE-MP-0 weights are downloaded by `mace_mp`; no weights are stored here.

## Scripts

| Script | Role | Output |
|---|---|---|
| `01_relax.py` | Relax M1–M8 with MACE-MP-0 and ASE. | `results/relaxation/` |
| `02_md.py` | Run the NVT MD temperature series and write trajectories. | `results/md/`, `trajectories/` |
| `03_analyze.py` | Extract D, σ, Ea, R², and the 300 K extrapolation. | `results/analysis/final_summary.csv`, `arrhenius_fits.csv` |
| `04_stability.py` | Compute relative energy and volume change from relaxed CIFs. | `results/analysis/stability_summary.csv` |
| `05_plot_results.py` | Plot stability, parity, entropy–Ea, Arrhenius, MSD, waterfall, density, displacement, vector-map, mechanism, and radar panels. | `figures/` |

`04_stability.py` generates `stability_summary.csv`, which
`05_plot_results.py` uses for the stability panel.

## Analysis protocol

`03_analyze.py` uses an explicit, documented method table:

- M5 and M7: `ln σ` over 800–1000 K.
- M1–M4, M6, and M8: `ln D` over 800–1200 K.

This is the protocol described in the manuscript. The method is selected by a
named table, not by an anonymous branch or by post hoc candidate-specific code.

## Data availability

Trajectories, relaxed structures, MD summaries, and figures are **not**
distributed in this repository. They are large and are part of the curated
dataset. Place local copies under `mlip/trajectories/`, `mlip/seeds/`, and
`mlip/results/`; those directories are git-ignored.

The four transport classes (Baseline, Transitional, Isentropic Recovery, and
Entropic Penalty) are assigned after the MLIP/MD runs and are not used by the
relaxation, MD, or analysis scripts.

`05_plot_results.py` contains every panel family. The trajectory-dependent
panels (MSD, density, displacement, vector map, and mechanism) print a
`[skip]` message and return when the local trajectories are absent; the summary,
waterfall, Arrhenius, and radar panels still run from the analysis tables.

## Run order

```bash
python mlip/01_relax.py
python mlip/02_md.py
python mlip/03_analyze.py
python mlip/04_stability.py
python mlip/05_plot_results.py
```

`02_md.py` is expensive and expects a GPU for practical run times. The analysis
and plotting steps can be rerun once the trajectories and relaxed structures are
available locally.
