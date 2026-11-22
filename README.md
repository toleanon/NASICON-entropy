# NASICON data preparation, feature engineering, model benchmark, analytical model, screening, and MLIP validation

Code-only snapshot for the data-curation, site-resolved entropy, 5d/7d
machine-learning, analytical-model, high-throughput screening, and MLIP validation stages of the
NASICON entropy-guided discovery study.

The curated dataset is not redistributed here. The scripts expect the input
table at `data/nasicon.csv`; see `data/README.md` for the expected columns and
data-availability notes.

## Repository layout

```text
nasicon-data-processing/
├── README.md
├── INSTALL.md
├── LICENSE
├── requirements.txt
├── paths.py
├── .gitignore
├── data/
│   ├── README.md
│   ├── prepare_dataset.py
│   └── compute_entropy.py
├── model/
│   ├── 01_feature_engineering.py
│   ├── 02_model_evaluation.py
│   └── 03_plot_results.py
├── analytical_model/
│   ├── 01_fit_formula.py
│   └── 02_plot_results.py
├── screening/
│   ├── README.md
│   ├── 01_screening.py
│   ├── 02_select_mlip_candidates.py
│   └── 03_plot_results.py
└── mlip/
    ├── README.md
    ├── 01_relax.py
    ├── 02_md.py
    ├── 03_analyze.py
    ├── 04_stability.py
    └── 05_plot_results.py
```

`model/processed/`, `model/figures/`, `screening/results/`,
`screening/figures/`, `mlip/results/`, `mlip/figures/`, `mlip/trajectories/`,
and `mlip/seeds/` are generated or local-only and are not tracked.

## Quick start

See `INSTALL.md` for the Python, CUDA, and environment notes.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Place the curated input table at data/nasicon.csv, then:
python data/prepare_dataset.py
python data/compute_entropy.py
python model/01_feature_engineering.py
python model/02_model_evaluation.py
python model/03_plot_results.py
python analytical_model/01_fit_formula.py
python analytical_model/02_plot_results.py
python screening/01_screening.py
python screening/02_select_mlip_candidates.py
python screening/03_plot_results.py
python mlip/01_relax.py
python mlip/02_md.py
python mlip/03_analyze.py
python mlip/04_stability.py
python mlip/05_plot_results.py
```

## Pipeline

| Script | Input | Output |
|---|---|---|
| `data/prepare_dataset.py` | `data/nasicon.csv` | `data/nasicon_cleaned.csv` |
| `data/compute_entropy.py` | `data/nasicon_cleaned.csv` | `data/entropy/nasicon_with_entropy.csv` |
| `model/01_feature_engineering.py` | entropy table | `model/processed/` |
| `model/02_model_evaluation.py` | `model/processed/` | metrics, models, `plot_data.pkl` in `model/figures/` |
| `model/03_plot_results.py` | `model/figures/plot_data.pkl` | publication figures in `model/figures/` |
| `analytical_model/01_fit_formula.py` | `model/processed/` | analytical fit results and `analytical_plot_data.pkl` in `model/figures/` |
| `analytical_model/02_plot_results.py` | `model/figures/analytical_plot_data.pkl` | `Analytical_Model.png/pdf` in `model/figures/` |
| `screening/01_screening.py` | cleaned table, trained model, optional local seeds | screening results in `screening/results/` |
| `screening/02_select_mlip_candidates.py` | screening results | M1–M8 contrast set and ranking in `screening/results/` |
| `screening/03_plot_results.py` | screening results | screening figures in `screening/figures/` |
| `mlip/01_relax.py` | screening candidates, parent CIF | relaxed structures and summary in `mlip/results/relaxation/` |
| `mlip/02_md.py` | relaxed structures | MD summaries in `mlip/results/md/`, trajectories in `mlip/trajectories/` |
| `mlip/03_analyze.py` | MD trajectories | `final_summary.csv` and `arrhenius_fits.csv` in `mlip/results/analysis/` |
| `mlip/04_stability.py` | relaxed structures | `stability_summary.csv` in `mlip/results/analysis/` |
| `mlip/05_plot_results.py` | MLIP analysis tables | MLIP figures in `mlip/figures/` |

The feature engineering uses the main composition-grouped split only. It writes
the 5d/7d matrices, labels, groups, scalers, and full feature tables to
`model/processed/`.

The model benchmark evaluates RF, XGBoost, LightGBM, GBDT, and SVR on the 5d and
7d sets, selects models by cross-validated stability, runs a strict ablation and
group-aware stacking, repeats the grouped split 50 times, and saves the results
needed for figure generation.

`02_model_evaluation.py` uses `N_JOBS = 1` by default to avoid nested joblib
oversubscription on small workstations. Increase it locally if faster execution
is needed.

The analytical stage fits the 5-, 6-, and 7-parameter forms, repeats the grouped
split 50 times, and saves the coefficients and plot data. `analytical_model/02_plot_results.py`
is a pure plotting script: it reads `analytical_plot_data.pkl` and does not refit
the model.

The screening stage generates candidates, applies the trained 7d model, scores
them, and writes the stratified Top30/20/8 lists. `screening/02_select_mlip_candidates.py`
then selects the M1–M8 contrast set by explicit composition rules and ranks the
set against the full pool.

## Plotting scripts

| Script | Reads | Writes |
|---|---|---|
| `model/03_plot_results.py` | `model/figures/plot_data.pkl` | `Model_Performance_Bars`, `Model_Parity`, `Model_Importance`, `Model_Entropy_Delta`, `Model_Repeated_Splits`, `Model_SHAP_Bar`, `Model_SHAP_Beeswarm` |
| `analytical_model/02_plot_results.py` | `model/figures/analytical_plot_data.pkl` | `Analytical_Model.png/pdf` |
| `screening/03_plot_results.py` | `screening/results/` | screening figures in `screening/figures/` |
| `mlip/05_plot_results.py` | MLIP analysis tables and optional local trajectories | `MLIP_*` and `Arrhenius_Fits` figures in `mlip/figures/` |

All plotting scripts are separated from fitting/training. The trajectory-dependent
MLIP panels print `[skip]` and return when local trajectories are absent.

M1–M8 are a **designed contrast set for mechanism validation**, not the top
eight by composite score. The four transport classes (Baseline, Transitional,
Isentropic Recovery, and Entropic Penalty) are assigned **after** the MLIP/MD
runs and are not selection criteria.

The MLIP stage relaxes M1–M8 with MACE-MP-0, runs the MD temperature series,
extracts D/σ/Ea, builds the stability table, and plots the summary plus
trajectory-dependent Fig. 6/7 panels. The analysis table uses the same documented
protocol as the manuscript: M5 and M7 are fitted with ln σ over 800–1000 K; the
other candidates use ln D over 800–1200 K. Trajectories, relaxed structures, and
MD summaries are not redistributed; they are expected under local, git-ignored
directories.

## Entropy columns

`data/compute_entropy.py` writes four columns:

- `S_M_site` — M-site configurational entropy per site.
- `S_X_site` — X-site configurational entropy per site.
- `S_total_site_sum` — `S_M_site + S_X_site`; the unweighted site-sum used by
  the screening score.
- `Delta_Smix_framework` — `n_M * S_M_site + n_X * S_X_site`; the
  stoichiometric total per formula unit.

The two totals are intentionally distinct:

```text
S_total   = S_M + S_X
Delta_Smix = n_M S_M + n_X S_X
```

## Data availability

The curated dataset is not included in this repository. It is provided in the
manuscript's Supplementary Information (Table S2), or available from the
corresponding author on reasonable request. The scripts intentionally stop with
a clear error if required inputs are missing.

This is a code-only release; it does not include the curated dataset,
trajectories, relaxed structures, trained model files, or generated results.

## License and citation

Code is released under the MIT License; see `LICENSE`.

If you use this code, please cite the associated manuscript:

> Entropy-guided interpretable machine learning for high-throughput candidate
> discovery in NASICON solid electrolytes.
