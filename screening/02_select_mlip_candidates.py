#!/usr/bin/env python3
"""Select the M1-M8 contrast set and rank it against the screening pool."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import SCREENING_RESULTS

FULL_PATH = SCREENING_RESULTS / "full_candidates.csv"
TOP30_PATH = SCREENING_RESULTS / "Top30_Stratified.csv"
TOP20_PATH = SCREENING_RESULTS / "Top20_Stratified.csv"
TOP8_PATH = SCREENING_RESULTS / "Top8_Stratified.csv"
OUTPUT_DIR = SCREENING_RESULTS

TARGETS = [
    ("M1", None, 3.746),
    ("M2", "Fe", 7.524),
    ("M3", "Cr", 7.524),
    ("M4", "Mg", 8.593),
    ("M5", "V", 8.806),
    ("M6", ("Al", "Sn"), 9.351),
    ("M7", ("Ge", "In"), 9.351),
    ("M8", "Yb", 10.823),
]

ROLES = {
    "M1": "Undoped reference",
    "M2": "Isovalent Fe/Cr pair",
    "M3": "Isovalent Fe/Cr pair",
    "M4": "Valence-mismatch extreme",
    "M5": "V single-dopant anchor",
    "M6": "Al/Sn vs Ge/In isentropic pair",
    "M7": "Al/Sn vs Ge/In isentropic pair",
    "M8": "Radius-contrast extreme",
}


def matches(frame, selector):
    if selector is None:
        return frame["dopant"].eq("None")
    if isinstance(selector, tuple):
        mask = pd.Series(True, index=frame.index)
        for element in selector:
            mask &= frame["Formula"].str.contains(element, regex=False)
        return mask
    return frame["dopant"].eq(selector)


def main():
    full = pd.read_csv(FULL_PATH)
    full["S_total"] = full["S_M"] + full["S_X"]
    full["dopant"] = full["dopant"].fillna("None")

    top30 = set(pd.read_csv(TOP30_PATH)["Formula"]) if TOP30_PATH.exists() else set()
    top20 = set(pd.read_csv(TOP20_PATH)["Formula"]) if TOP20_PATH.exists() else set()
    top8 = set(pd.read_csv(TOP8_PATH)["Formula"]) if TOP8_PATH.exists() else set()

    rows = []
    used = set()
    for mlip_id, selector, target in TARGETS:
        candidates = full[matches(full, selector)].copy()
        if candidates.empty:
            raise ValueError(f"No candidate found for {mlip_id} ({selector})")
        candidates = candidates[~candidates["Formula"].isin(used)]
        candidates["distance"] = (candidates["S_total"] - target).abs()
        row = candidates.sort_values(
            ["distance", "composite"], ascending=[True, False]
        ).iloc[0]
        used.add(row["Formula"])
        record = row.to_dict()
        record.update({
            "MLIP_ID": mlip_id,
            "design_role": ROLES[mlip_id],
            "target_S_total": target,
            "in_Top30": row["Formula"] in top30,
            "in_Top20": row["Formula"] in top20,
            "in_Top8": row["Formula"] in top8,
            "rationale": (
                f"{ROLES[mlip_id]}; closest screened candidate to the "
                f"design target S_total = {target:.3f} J mol-1 K-1."
            ),
        })
        rows.append(record)

    selected = pd.DataFrame(rows)
    selected.to_csv(OUTPUT_DIR / "MLIP_Candidates.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selection_rationale.csv", index=False)

    full["rank_all"] = full["composite"].rank(
        method="min", ascending=False
    ).astype(int)
    ranking = selected.merge(
        full[["Formula", "rank_all"]], on="Formula", how="left"
    ).sort_values("MLIP_ID")
    ranking.to_csv(OUTPUT_DIR / "M1_M8_Composite_Ranking.csv", index=False)

    print(selected[["MLIP_ID", "Formula", "design_role", "S_total"]].to_string(index=False))
    print(f"Written: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
