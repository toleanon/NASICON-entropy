#!/usr/bin/env python3
"""Clean the curated NASICON table and write the modelling table."""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR

INPUT = DATA_DIR / "nasicon.csv"
OUTPUT = DATA_DIR / "nasicon_cleaned.csv"

HOST_ELEMENTS = [
    "Zr", "Hf", "Ge", "Sn", "Ti", "V", "Al", "Sc", "Y", "In",
    "Nb", "Mo", "Fe", "Cr", "Mn", "Mg", "Zn", "Cu", "La", "Ce",
    "Pr", "Eu", "Lu", "Yb", "Ca", "Sr", "Ba",
]


def extract_host_element(formula):
    """Return the first host element found in a chemical formula."""
    if not isinstance(formula, str):
        return "Unknown"
    for element in HOST_ELEMENTS:
        if re.search(element + r"(?=\d|$|[A-Z])", formula):
            return element
    return "Unknown"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input table not found: {INPUT}. See data/README.md for the expected file."
        )

    df = pd.read_csv(INPUT)
    print(f"Input rows: {len(df)}")

    # 1. Drop records missing the target or the formula.
    df = df.dropna(subset=["Formula", "log10S"])

    # 2. Group by host element for within-group imputation.
    df["Main_Element"] = df["Formula"].apply(extract_host_element)

    # 3. Drop collinear or mostly missing columns.
    drop_cols = ["di"]
    if "Vol" in df.columns and df["Vol"].isna().mean() >= 0.4:
        drop_cols.append("Vol")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 4. Impute synthesis and lattice fields by host-element median.
    for col in ["Temm", "t1", "Tems", "t2", "Temt", "th", "a", "b", "c", "Vol"]:
        if col not in df.columns or not df[col].isna().any():
            continue
        df[col] = df.groupby("Main_Element")[col].transform(
            lambda values: values.fillna(values.median())
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 5. Fill composition and electronegativity fields.
    if "N0" in df.columns and df["N0"].isna().any():
        df["N0"] = df["N0"].fillna(df["N0"].median())
    for col in ["N1", "N2", "N3", "N4", "N5", "Neg1", "Neg2", "Neg3", "Neg4", "Neg5"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0)

    df = df.drop(columns=["Main_Element"])

    # 6. Remove exact duplicate descriptor rows.
    feature_cols = [
        col for col in df.columns
        if col not in {"S", "log10S", "Index", "Reference"}
    ]
    df = df.drop_duplicates(subset=feature_cols, keep="first")

    df.to_csv(OUTPUT, index=False)
    print(f"Cleaned rows: {len(df)}")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
