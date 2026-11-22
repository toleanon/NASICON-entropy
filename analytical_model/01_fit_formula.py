#!/usr/bin/env python3
"""Fit the constrained analytical model and save results for plotting."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import FIGURES_DIR, PROCESSED_DIR

R_GAS = 8.314
BASE_FEATURES = ["Na_ratio", "Δr_MX", "Δχ_XM", "Sinter_intensity"]
N_REPEATS = 50


def base_model(x, w1, w2, w3, w4, c):
    return w1 * x[:, 0] + w2 * x[:, 1] + w3 * x[:, 2] + w4 * x[:, 3] + c


def linear_model(x, w1, w2, w3, w4, e, c):
    return w1 * x[:, 0] + w2 * x[:, 1] + w3 * x[:, 2] + w4 * x[:, 3] + e * x[:, 4] + c


def quadratic_model(x, w1, w2, w3, w4, e, f, c):
    return (
        w1 * x[:, 0] + w2 * x[:, 1] + w3 * x[:, 2] + w4 * x[:, 3]
        + e * x[:, 4] - f * x[:, 4] ** 2 + c
    )


def fit_models(x_train, y_train, x_test, y_test):
    base_train = x_train[BASE_FEATURES].values
    base_test = x_test[BASE_FEATURES].values
    sx_train = x_train["S_X"].values
    sx_test = x_test["S_X"].values

    p_base, _ = curve_fit(base_model, base_train, y_train, p0=[1, 0, 1, 0, -7])
    linear_train = np.column_stack([base_train, sx_train])
    linear_test = np.column_stack([base_test, sx_test])
    p_linear, _ = curve_fit(linear_model, linear_train, y_train, p0=[1, 0, 1, 0, 1, -7])

    quadratic_train = linear_train
    quadratic_test = linear_test
    p_quad, _ = curve_fit(
        quadratic_model, quadratic_train, y_train,
        p0=[1, 0, 1, 0, 0.5, 0.1, -7],
        bounds=([-10, -10, -10, -10, -10, 0, -20],
                [10, 10, 10, 10, 10, 10, 10]),
    )

    return {
        "base": {
            "params": p_base,
            "r2_train": r2_score(y_train, base_model(base_train, *p_base)),
            "r2_test": r2_score(y_test, base_model(base_test, *p_base)),
        },
        "linear": {
            "params": p_linear,
            "r2_train": r2_score(y_train, linear_model(linear_train, *p_linear)),
            "r2_test": r2_score(y_test, linear_model(linear_test, *p_linear)),
        },
        "quadratic": {
            "params": p_quad,
            "r2_train": r2_score(y_train, quadratic_model(quadratic_train, *p_quad)),
            "r2_test": r2_score(y_test, quadratic_model(quadratic_test, *p_quad)),
        },
    }


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the processed 5d/7d split.
    x_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_7d.csv")
    x_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_7d.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    scaler = joblib.load(PROCESSED_DIR / "scaler_7d.pkl")

    x_train = pd.DataFrame(
        scaler.inverse_transform(x_train_scaled.values),
        columns=x_train_scaled.columns,
    )
    x_test = pd.DataFrame(
        scaler.inverse_transform(x_test_scaled.values),
        columns=x_test_scaled.columns,
    )
    for frame in (x_train, x_test):
        frame["struct"] = np.round(frame["struct"]).astype(int)

    # Fit the nested models on the primary grouped split.
    fits = fit_models(x_train, y_train, x_test, y_test)

    coef_rows = []
    coef_names = {
        "base": ["w1", "w2", "w3", "w4", "intercept"],
        "linear": ["w1", "w2", "w3", "w4", "S_X", "intercept"],
        "quadratic": ["w1", "w2", "w3", "w4", "S_X", "S_X2", "intercept"],
    }
    for model_name, fit in fits.items():
        for name, value in zip(coef_names[model_name], fit["params"]):
            coef_rows.append({"model": model_name, "parameter": name, "value": value})
    coefficients = pd.DataFrame(coef_rows)
    coefficients.to_csv(FIGURES_DIR / "analytical_coefficients.csv", index=False)

    comparison = pd.DataFrame([
        {
            "model": name,
            "parameters": len(fit["params"]),
            "train_r2": fit["r2_train"],
            "test_r2": fit["r2_test"],
        }
        for name, fit in fits.items()
    ])
    comparison.to_csv(FIGURES_DIR / "analytical_model_comparison.csv", index=False)

    # Repeat the grouped split for uncertainty estimates.
    full = pd.concat([
        pd.read_csv(PROCESSED_DIR / "full_features_train.csv"),
        pd.read_csv(PROCESSED_DIR / "full_features_test.csv"),
    ], ignore_index=True)
    full_base = full[BASE_FEATURES].values
    full_sx = full["S_X"].values
    full_y = full["log10S"].values
    groups = full["canonical"].values

    split_values = {"base": [], "linear": [], "quadratic": []}
    for seed in range(N_REPEATS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train, test = next(gss.split(full, full_y, groups=groups))
        x_tr = np.column_stack([full_base[train], full_sx[train]])
        x_te = np.column_stack([full_base[test], full_sx[test]])
        y_tr = full_y[train]
        y_te = full_y[test]

        p_base, _ = curve_fit(base_model, full_base[train], y_tr, p0=[1, 0, 1, 0, -7])
        p_linear, _ = curve_fit(linear_model, x_tr, y_tr, p0=[1, 0, 1, 0, 1, -7])
        p_quad, _ = curve_fit(
            quadratic_model, x_tr, y_tr,
            p0=[1, 0, 1, 0, 0.5, 0.1, -7],
            bounds=([-10, -10, -10, -10, -10, 0, -20],
                    [10, 10, 10, 10, 10, 10, 10]),
        )
        split_values["base"].append(r2_score(y_te, base_model(full_base[test], *p_base)))
        split_values["linear"].append(r2_score(y_te, linear_model(x_te, *p_linear)))
        split_values["quadratic"].append(r2_score(y_te, quadratic_model(x_te, *p_quad)))

    split_base = np.array(split_values["base"])
    split_linear = np.array(split_values["linear"])
    split_quad = np.array(split_values["quadratic"])
    repeated = pd.DataFrame({
        "base_r2": split_base,
        "linear_r2": split_linear,
        "quadratic_r2": split_quad,
    })
    repeated["delta_linear"] = split_linear - split_base
    repeated["delta_quadratic"] = split_quad - split_base
    repeated.to_csv(FIGURES_DIR / "analytical_repeated_splits.csv", index=False)

    x_all = pd.concat([x_train, x_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    sx_all = x_all["S_X"].values
    struct_all = x_all["struct"].values
    is_mixed = sx_all > 0.01

    median_base = np.median(full_base, axis=0)
    sx_grid = np.linspace(0, x_train["S_X"].max(), 100)
    grid = np.column_stack([
        np.full(100, median_base[0]),
        np.full(100, median_base[1]),
        np.full(100, median_base[2]),
        np.full(100, median_base[3]),
        sx_grid,
    ])
    quad_curve = quadratic_model(grid, *fits["quadratic"]["params"])
    base_line = np.median(base_model(x_train[BASE_FEATURES].values, *fits["base"]["params"]))
    pred_train = quadratic_model(
        np.column_stack([x_train[BASE_FEATURES].values, x_train["S_X"].values]),
        *fits["quadratic"]["params"],
    )
    pred_test = quadratic_model(
        np.column_stack([x_test[BASE_FEATURES].values, x_test["S_X"].values]),
        *fits["quadratic"]["params"],
    )

    # Save everything needed by the plotting script.
    plot_data = {
        "sx_all": sx_all,
        "y_all": y_all.values,
        "struct_all": struct_all,
        "is_mixed": is_mixed,
        "sx_grid": sx_grid,
        "quad_curve": quad_curve,
        "base_line": base_line,
        "r2_base": fits["base"]["r2_test"],
        "r2_linear": fits["linear"]["r2_test"],
        "r2_quad": fits["quadratic"]["r2_test"],
        "r2_quad_train": fits["quadratic"]["r2_train"],
        "y_train": y_train.values,
        "y_test": y_test.values,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "split_base": split_base,
        "split_linear": split_linear,
        "split_quad": split_quad,
        "p_quadratic": ttest_rel(split_quad, split_base).pvalue,
        "r_gas": R_GAS,
    }
    joblib.dump(plot_data, FIGURES_DIR / "analytical_plot_data.pkl")

    report = [
        "Constrained analytical model",
        "=" * 40,
        comparison.to_string(index=False),
        "",
        "Coefficients:",
        coefficients.to_string(index=False),
        "",
        f"Repeated grouped splits: {N_REPEATS}",
        f"Median Delta R2 (linear - base): {np.median(split_linear - split_base):.4f}",
        f"Median Delta R2 (quadratic - base): {np.median(split_quad - split_base):.4f}",
        f"Paired p-value (quadratic vs base): {ttest_rel(split_quad, split_base).pvalue:.4e}",
    ]
    (FIGURES_DIR / "analytical_report.txt").write_text("\n".join(report), encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"Written: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
