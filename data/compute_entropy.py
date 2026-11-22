#!/usr/bin/env python3
"""Compute site-resolved configurational entropy for NASICON compositions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR

INPUT = DATA_DIR / "nasicon_cleaned.csv"
OUTPUT = DATA_DIR / "entropy" / "nasicon_with_entropy.csv"
R_GAS = 8.314


def entropy_features(row):
    """Return M-site, X-site, and total entropy descriptors for one record."""
    # M site: Zr, M', M''.
    m_counts = [row["N1"], row["N2"], row["N3"]]
    n_m = sum(m_counts)
    s_m = 0.0
    if n_m > 0:
        probs = [count / n_m for count in m_counts if count > 0]
        s_m = -R_GAS * sum(p * np.log(p) for p in probs)

    # X site: Si and P.
    x_counts = [row["N4"], row["N5"]]
    n_x = sum(x_counts)
    s_x = 0.0
    if n_x > 0:
        probs = [count / n_x for count in x_counts if count > 0]
        s_x = -R_GAS * sum(p * np.log(p) for p in probs)

    # Screening site-sum and stoichiometric total are distinct quantities.
    s_total = s_m + s_x
    delta_smix = n_m * s_m + n_x * s_x

    return pd.Series({
        "S_M_site": s_m,
        "S_X_site": s_x,
        "S_total_site_sum": s_total,
        "Delta_Smix_framework": delta_smix,
    })


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input table not found: {INPUT}. Run prepare_dataset.py first."
        )

    df = pd.read_csv(INPUT)
    entropy = df.apply(entropy_features, axis=1)
    df_final = pd.concat([df, entropy], axis=1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT, index=False)

    print(f"Rows: {len(df_final)}")
    print(f"Written: {OUTPUT}")
    print(
        df_final[
            ["S_M_site", "S_X_site", "S_total_site_sum", "Delta_Smix_framework"]
        ].describe()
    )

    sample = df_final[df_final["Formula"] == "Na3Zr2Si2PO12"]
    if not sample.empty:
        row = sample.iloc[0]
        print(
            "Na3Zr2Si2PO12: "
            f"S_M={row['S_M_site']:.2f}, "
            f"S_X={row['S_X_site']:.2f}, "
            f"S_total={row['S_total_site_sum']:.2f}, "
            f"Delta_Smix={row['Delta_Smix_framework']:.2f} J mol-1 K-1"
        )


if __name__ == "__main__":
    main()
