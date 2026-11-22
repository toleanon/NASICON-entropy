#!/usr/bin/env python3
"""Train and evaluate the 5d/7d NASICON models; save metrics and plot data."""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp, ttest_rel
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_predict,
    cross_val_score,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import FIGURES_DIR, PROCESSED_DIR

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_REPEATS = 50
N_JOBS = 1  # Single-process default for portability.

MODEL_CONFIGS = {
    "RF": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [8, 10, 12, 15, None],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.7],
        },
    },
    "XGB": {
        "model": XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0),
        "params": {
            "n_estimators": [150, 250],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.15],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0.5, 1, 2],
        },
    },
    "LGBM": {
        "model": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1),
        "params": {
            "n_estimators": [150, 250],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [10, 20],
            "reg_lambda": [1, 2],
        },
    },
    "GBDT": {
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [150, 250],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "C": [1, 5, 10, 50],
            "gamma": ["scale", 0.01, 0.05, 0.1],
            "epsilon": [0.05, 0.1, 0.2],
        },
    },
}


def load_dataset(suffix):
    x_train = pd.read_csv(PROCESSED_DIR / f"X_train_{suffix}.csv")
    x_test = pd.read_csv(PROCESSED_DIR / f"X_test_{suffix}.csv")
    return x_train, x_test, x_train.columns.tolist()


def train_and_evaluate(x_train, x_test, y_train, y_test, label, groups):
    rows = []
    predictions = {}
    importances = {}
    best_params = {}
    models = {}
    gkf = GroupKFold(n_splits=5)

    for name, config in MODEL_CONFIGS.items():
        grid = GridSearchCV(
            config["model"], config["params"], cv=gkf, scoring="r2", n_jobs=N_JOBS
        )
        grid.fit(x_train, y_train, groups=groups)
        model = grid.best_estimator_
        best_params[name] = grid.best_params_
        models[name] = model

        cv_scores = cross_val_score(
            model, x_train, y_train, cv=gkf, groups=groups, scoring="r2"
        )
        prediction = model.predict(x_test)
        predictions[name] = prediction
        rows.append({
            "Model": name,
            "Best Params": str(grid.best_params_),
            "CV R2 mean": cv_scores.mean(),
            "CV R2 std": cv_scores.std(),
            "Test R2": r2_score(y_test, prediction),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
            "Test MAE": mean_absolute_error(y_test, prediction),
        })
        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_

    results = pd.DataFrame(rows)
    results["CV_stability"] = results["CV R2 mean"] - 0.5 * results["CV R2 std"]
    best_name = results.loc[results["CV_stability"].idxmax(), "Model"]
    return (
        results,
        predictions,
        importances,
        models[best_name],
        best_name,
        best_params,
        models,
    )


def repeated_split_analysis(best_params):
    full_train = pd.read_csv(PROCESSED_DIR / "full_features_train.csv")
    full_test = pd.read_csv(PROCESSED_DIR / "full_features_test.csv")
    full = pd.concat([full_train, full_test], ignore_index=True)

    features_5 = ["Na_ratio", "Δr_MX", "Δχ_XM", "Sinter_intensity", "struct"]
    features_7 = features_5 + ["S_M", "S_X"]
    for column in features_5:
        if full[column].isna().any():
            fill = full[column].mode()[0] if column == "struct" else full[column].median()
            full[column] = full[column].fillna(fill)

    y_all = full["log10S"].values
    groups_all = full["canonical"].values
    results = {name: {"5d": [], "7d": []} for name in MODEL_CONFIGS}

    for seed in range(N_REPEATS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train, test = next(gss.split(full, y_all, groups=groups_all))
        for name in MODEL_CONFIGS:
            params = best_params[name]
            for tag, features in (("5d", features_5), ("7d", features_7)):
                model = clone(MODEL_CONFIGS[name]["model"]).set_params(**params)
                model.fit(full.iloc[train][features], y_all[train])
                score = r2_score(y_all[test], model.predict(full.iloc[test][features]))
                results[name][tag].append(score)

    rows = []
    for name in MODEL_CONFIGS:
        scores_5 = np.array(results[name]["5d"])
        scores_7 = np.array(results[name]["7d"])
        delta = scores_7 - scores_5
        _, p_value = ttest_rel(scores_7, scores_5)
        rows.append({
            "Model": name,
            "5d_Test_R2_mean": scores_5.mean(),
            "5d_Test_R2_std": scores_5.std(),
            "7d_Test_R2_mean": scores_7.mean(),
            "7d_Test_R2_std": scores_7.std(),
            "Delta_R2_mean": delta.mean(),
            "Delta_R2_std": delta.std(),
            "p_value": p_value,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(FIGURES_DIR / "repeated_split_results.csv", index=False)
    return summary


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the processed feature matrices.
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().reset_index(drop=True)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().reset_index(drop=True)
    groups_train = pd.read_csv(PROCESSED_DIR / "groups_train.csv").squeeze().reset_index(drop=True)
    groups_test = pd.read_csv(PROCESSED_DIR / "groups_test.csv").squeeze().reset_index(drop=True)

    x_train_5, x_test_5, features_5 = load_dataset("5d")
    x_train_7, x_test_7, features_7 = load_dataset("7d")

    for label, x_train, x_test in (("5d", x_train_5, x_test_5), ("7d", x_train_7, x_test_7)):
        for column in x_train.columns:
            _, p_value = ks_2samp(x_train[column], x_test[column])
            if p_value < 0.05:
                print(f"Distribution shift: {label} / {column} (p={p_value:.3e})")

    # Train the 5d and 7d model families.
    results_5, pred_5, imp_5, best_5, best_name_5, params_5, models_5 = train_and_evaluate(
        x_train_5, x_test_5, y_train, y_test, "5d", groups_train
    )
    results_7, pred_7, imp_7, best_7, best_name_7, params_7, models_7 = train_and_evaluate(
        x_train_7, x_test_7, y_train, y_test, "7d", groups_train
    )

    results_5["Feature Set"] = "5d"
    results_7["Feature Set"] = "7d"
    combined = pd.concat([results_5, results_7], ignore_index=True)
    combined["CV_stability"] = combined["CV R2 mean"] - 0.5 * combined["CV R2 std"]
    best_overall = combined.loc[combined["CV_stability"].idxmax()]

    delta_rows = []
    for name in MODEL_CONFIGS:
        row_5 = results_5[results_5["Model"] == name].iloc[0]
        row_7 = results_7[results_7["Model"] == name].iloc[0]
        delta_rows.append({
            "Model": name,
            "Delta_CV_R2": row_7["CV R2 mean"] - row_5["CV R2 mean"],
            "Delta_Test_R2": row_7["Test R2"] - row_5["Test R2"],
        })
    delta_df = pd.DataFrame(delta_rows)

    strict_rows = []
    for name in ("RF", "LGBM"):
        model = clone(MODEL_CONFIGS[name]["model"]).set_params(**params_7[name])
        model.fit(x_train_5, y_train)
        strict_score = r2_score(y_test, model.predict(x_test_5))
        score_7 = results_7[results_7["Model"] == name].iloc[0]["Test R2"]
        strict_rows.append({
            "Model": name,
            "5d_R2_with_7d_params": strict_score,
            "7d_R2": score_7,
            "Delta_strict": score_7 - strict_score,
        })
    strict_df = pd.DataFrame(strict_rows)

    feature_importance = pd.DataFrame({
        "Feature": features_7,
        "Importance": models_7["RF"].feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Group-aware stacking on the 7d feature set.
    estimators = []
    for name in MODEL_CONFIGS:
        estimators.append(
            (name, clone(MODEL_CONFIGS[name]["model"]).set_params(**params_7[name]))
        )

    oof = np.zeros((len(x_train_7), len(estimators)))
    fitted_base = []
    gkf = GroupKFold(n_splits=5)
    for index, (name, estimator) in enumerate(estimators):
        oof[:, index] = cross_val_predict(
            estimator, x_train_7, y_train, cv=gkf, groups=groups_train, n_jobs=N_JOBS
        )
        estimator.fit(x_train_7, y_train)
        fitted_base.append((name, estimator))

    meta = Ridge().fit(oof, y_train)
    base_test = np.column_stack([model.predict(x_test_7) for _, model in fitted_base])
    pred_stack = meta.predict(base_test)

    cv_stack = []
    for train, test in gkf.split(x_train_7, y_train, groups=groups_train):
        oof_fold = np.zeros((len(train), len(estimators)))
        for index, (_, estimator) in enumerate(estimators):
            oof_fold[:, index] = cross_val_predict(
                clone(estimator), x_train_7.iloc[train], y_train.iloc[train],
                cv=GroupKFold(n_splits=5), groups=groups_train.iloc[train], n_jobs=N_JOBS
            )
        fold_meta = Ridge().fit(oof_fold, y_train.iloc[train])
        fold_models = []
        for _, estimator in estimators:
            fitted = clone(estimator).fit(x_train_7.iloc[train], y_train.iloc[train])
            fold_models.append(fitted)
        fold_base = np.column_stack([model.predict(x_train_7.iloc[test]) for model in fold_models])
        cv_stack.append(r2_score(y_train.iloc[test], fold_meta.predict(fold_base)))
    cv_stack = np.array(cv_stack)

    stack_row = pd.DataFrame([{
        "Model": "Stacking",
        "Best Params": "Ridge ensemble (group-aware OOF)",
        "CV R2 mean": cv_stack.mean(),
        "CV R2 std": cv_stack.std(),
        "CV_stability": cv_stack.mean() - 0.5 * cv_stack.std(),
        "Test R2": r2_score(y_test, pred_stack),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred_stack)),
        "Test MAE": mean_absolute_error(y_test, pred_stack),
        "Feature Set": "7d",
    }])
    final = pd.concat([combined, stack_row], ignore_index=True)

    final.to_csv(FIGURES_DIR / "model_comparison_5d_vs_7d.csv", index=False)
    delta_df.to_csv(FIGURES_DIR / "delta_entropy_contribution.csv", index=False)
    strict_df.to_csv(FIGURES_DIR / "strict_ablation.csv", index=False)
    feature_importance.to_csv(FIGURES_DIR / "feature_importance.csv", index=False)

    stacking_weights = pd.DataFrame({
        "Base Model": [name for name, _ in fitted_base],
        "Weight": meta.coef_,
    })
    stacking_weights.to_csv(FIGURES_DIR / "stacking_weights.csv", index=False)

    predictions = pd.DataFrame({
        "y_test": y_test,
        "pred_5d": pred_5[best_name_5],
        "pred_7d": pred_7[best_name_7],
        "pred_stacking": pred_stack,
    })
    predictions.to_csv(FIGURES_DIR / "predictions.csv", index=False)

    joblib.dump(best_5, FIGURES_DIR / "best_model_5d.pkl")
    joblib.dump(best_7, FIGURES_DIR / "best_model_7d.pkl")
    joblib.dump({"fitted_base": fitted_base, "meta": meta},
                FIGURES_DIR / "stacking_model_7d.pkl")

    report = [
        "NASICON model performance report",
        "=" * 60,
        f"Selected model: {best_overall['Feature Set']} / {best_overall['Model']}",
        f"CV R2: {best_overall['CV R2 mean']:.4f} +/- {best_overall['CV R2 std']:.4f}",
        f"Test R2: {best_overall['Test R2']:.4f}",
        f"Test RMSE: {best_overall['Test RMSE']:.4f}",
        f"Test MAE: {best_overall['Test MAE']:.4f}",
        "",
        final.to_string(index=False),
        "",
        "Entropy contribution:",
        delta_df.to_string(index=False),
        "",
        "Strict ablation:",
        strict_df.to_string(index=False),
    ]
    (FIGURES_DIR / "performance_report.txt").write_text("\n".join(report), encoding="utf-8")

    # Repeated grouped splits for stability.
    repeated = repeated_split_analysis(params_7)

    # SHAP values for the 7d random forest.
    shap_model = models_7["RF"]
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(x_train_7)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).ravel()[0])

    plot_data = {
        "results": final,
        "delta": delta_df,
        "strict": strict_df,
        "repeated": repeated,
        "y_test": y_test,
        "pred_5d": pred_5[best_name_5],
        "pred_7d": pred_7[best_name_7],
        "pred_stacking": pred_stack,
        "features_5d": features_5,
        "features_7d": features_7,
        "feature_importance": feature_importance,
        "stacking_weights": stacking_weights,
        "shap_values": shap_values,
        "shap_data": x_train_7,
        "shap_expected": expected_value,
    }
    joblib.dump(plot_data, FIGURES_DIR / "plot_data.pkl")

    print(f"Selected: {best_overall['Feature Set']} / {best_overall['Model']}")
    print(f"Test R2: {best_overall['Test R2']:.4f}")
    print(f"Stacking Test R2: {r2_score(y_test, pred_stack):.4f}")
    print(f"Written: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
