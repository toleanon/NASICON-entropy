#!/usr/bin/env python3
"""Build grouped 5d/7d feature matrices for the NASICON benchmark."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR, PROCESSED_DIR

INPUT = DATA_DIR / "entropy" / "nasicon_with_entropy.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MANUAL_MAP = {
    "Na3Zr2(SiO4)2(PO4)": "Na3Zr2Si2PO12",
    "Na3Zr2PSi2O12": "Na3Zr2Si2PO12",
}


def canonical_formula(formula):
    try:
        canonical = Composition(formula).reduced_formula
    except Exception:
        canonical = formula
    return MANUAL_MAP.get(canonical, MANUAL_MAP.get(formula, canonical))


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input table not found: {INPUT}. Run compute_entropy.py first."
        )

    # Load and canonicalize the entropy table.
    df = pd.read_csv(INPUT)
    required = ["Formula", "log10S", "N0", "N1", "N2", "N3", "N4", "N5",
                "D1", "D2", "D3", "D4", "D5", "Neg1", "Neg2", "Neg3", "Neg4", "Neg5",
                "Tems", "t2", "struct", "S_M_site", "S_X_site"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["canonical"] = df["Formula"].apply(canonical_formula)

    # Collapse equivalent records and keep one row per composition/process/label.
    df = df.drop_duplicates(
        subset=["canonical", "Tems", "t2", "struct", "log10S"], keep="first"
    ).reset_index(drop=True)

    # Attach the site-resolved entropy descriptors.
    df["S_M"] = df["S_M_site"]
    df["S_X"] = df["S_X_site"]

    eps = 1e-9
    m_sum = df[["N1", "N2", "N3"]].sum(axis=1)
    x_sum = df[["N4", "N5"]].sum(axis=1)
    r_m = (df["N1"] * df["D1"] + df["N2"] * df["D2"] + df["N3"] * df["D3"]) / (m_sum + eps)
    r_x = (df["N4"] * df["D4"] + df["N5"] * df["D5"]) / (x_sum + eps)
    neg_m = (df["N1"] * df["Neg1"] + df["N2"] * df["Neg2"] + df["N3"] * df["Neg3"]) / (m_sum + eps)
    neg_x = (df["N4"] * df["Neg4"] + df["N5"] * df["Neg5"]) / (x_sum + eps)

    df["Na_ratio"] = df["N0"] / (m_sum + eps)
    df["Δr_MX"] = r_m - r_x
    df["Δχ_XM"] = neg_x - neg_m
    df["Sinter_intensity"] = df["Tems"] * np.log10(df["t2"] + 1.0)

    features_5d = ["Na_ratio", "Δr_MX", "Δχ_XM", "Sinter_intensity", "struct"]
    features_7d = features_5d + ["S_M", "S_X"]

    # Composition-grouped train/test split.
    gss = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        gss.split(df, df["log10S"], groups=df["canonical"])
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, features in {"5d": features_5d, "7d": features_7d}.items():
        x_train = df.loc[train_idx, features].copy()
        x_test = df.loc[test_idx, features].copy()

        for column in features:
            if column == "struct":
                fill_value = x_train[column].mode()[0] if x_train[column].notna().any() else 0
            else:
                fill_value = x_train[column].median() if x_train[column].notna().any() else 0.0
            x_train[column] = x_train[column].fillna(fill_value)
            x_test[column] = x_test[column].fillna(fill_value)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        pd.DataFrame(x_train_scaled, columns=features).to_csv(
            PROCESSED_DIR / f"X_train_{name}.csv", index=False
        )
        pd.DataFrame(x_test_scaled, columns=features).to_csv(
            PROCESSED_DIR / f"X_test_{name}.csv", index=False
        )
        joblib.dump(scaler, PROCESSED_DIR / f"scaler_{name}.pkl")
        joblib.dump(features, PROCESSED_DIR / f"features_{name}.pkl")

    df.loc[train_idx, "log10S"].to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    df.loc[test_idx, "log10S"].to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    df.loc[train_idx, "canonical"].to_csv(PROCESSED_DIR / "groups_train.csv", index=False)
    df.loc[test_idx, "canonical"].to_csv(PROCESSED_DIR / "groups_test.csv", index=False)

    columns = ["Index", "Formula", "canonical"] + features_7d + ["log10S"]
    df.loc[train_idx, columns].to_csv(
        PROCESSED_DIR / "full_features_train.csv", index=False
    )
    df.loc[test_idx, columns].to_csv(
        PROCESSED_DIR / "full_features_test.csv", index=False
    )

    overlap = set(df.loc[train_idx, "canonical"]) & set(df.loc[test_idx, "canonical"])
    print(f"Rows after deduplication: {len(df)}")
    print(f"Training rows: {len(train_idx)}, test rows: {len(test_idx)}")
    print(f"Canonical groups shared by train/test: {len(overlap)}")
    print(f"Written: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
