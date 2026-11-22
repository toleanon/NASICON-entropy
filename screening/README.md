# High-throughput screening

This folder contains the generation, scoring, stratified selection, and
visualization code for the NASICON screening stage.

## Inputs

- `data/nasicon_cleaned.csv` from the data stage.
- `model/figures/best_model_7d.pkl` and `model/processed/scaler_7d.pkl` from the
  model stage.

Seed files are **not included** because they are part of the curated dataset.
If `screening/seeds/seed_candidates.csv` or `screening/seeds/cif_candidates/`
are absent, `01_screening.py` falls back to the formulas in
`data/nasicon_cleaned.csv`. Place local seed files under `screening/seeds/` to
reproduce the original candidate pool; that folder is git-ignored.

## Scripts

```text
01_screening.py
02_select_mlip_candidates.py
03_plot_results.py
```

| Script | Role |
|---|---|
| `01_screening.py` | Generate and neutralize candidates, score with the 7d model, and write the composite and stratified Top30/20/8 lists. |
| `02_select_mlip_candidates.py` | Select the M1–M8 designed contrast set by composition rules and rank it against the full pool. |
| `03_plot_results.py` | Create the screening-space, score-profile, X-site entropy, and M2 SHAP figures. |

Outputs are written to `screening/results/` and `screening/figures/`; both are
ignored by Git.

## Selection versus classification

M1–M8 are a **designed contrast set for mechanism validation**, not the top
eight candidates by composite score. The selection rules cover an undoped
reference, an isovalent Fe/Cr pair, a V anchor, an Al/Sn versus Ge/In pair, a
valence-mismatch extreme, and a radius-contrast extreme.

The four transport classes (Baseline, Transitional, Isentropic Recovery, and
Entropic Penalty) are assigned **after** the MLIP/MD runs from the simulated
transport metrics. They are not used to select candidates.

## Run order

```bash
python screening/01_screening.py
python screening/02_select_mlip_candidates.py
python screening/03_plot_results.py
```

`03_plot_results.py` also requires the model-stage `best_model_7d.pkl` and
`scaler_7d.pkl` for the SHAP panel.
