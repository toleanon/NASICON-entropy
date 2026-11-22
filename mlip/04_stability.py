#!/usr/bin/env python3
"""Build the structural stability table from relaxed candidate structures."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import MLIP_RESULTS, SCREENING_RESULTS

RELAX_DIR = MLIP_RESULTS / "relaxation"
CANDIDATE_CSV = SCREENING_RESULTS / "MLIP_Candidates.csv"
OUTPUT_DIR = MLIP_RESULTS / "analysis"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATE_CSV)
    s_total = dict(zip(candidates["MLIP_ID"], candidates["S_M"] + candidates["S_X"]))
    s_x = dict(zip(candidates["MLIP_ID"], candidates["S_X"]))
    s_m = dict(zip(candidates["MLIP_ID"], candidates["S_M"]))

    summary = pd.read_csv(RELAX_DIR / "relaxation_summary.csv")
    summary = summary[summary["converged"] == True].copy()

    rows = []
    for _, row in summary.iterrows():
        m_id = row["MLIP_ID"]
        init_file = RELAX_DIR / f"init_{m_id}.cif"
        relaxed_file = RELAX_DIR / f"relaxed_{m_id}.cif"

        volume_change = np.nan
        if init_file.exists() and relaxed_file.exists():
            initial = read(init_file)
            relaxed = read(relaxed_file)
            volume_change = (
                (relaxed.get_volume() - initial.get_volume())
                / initial.get_volume()
                * 100
            )

        rows.append({
            "MLIP_ID": m_id,
            "Formula": row["Formula"],
            "S_total": s_total.get(m_id, np.nan),
            "S_X": s_x.get(m_id, np.nan),
            "S_M": s_m.get(m_id, np.nan),
            "E_per_atom_meV": row["final_energy"] / row["n_atoms"] * 1000,
            "Vol_change_pct": volume_change,
            "Final_Fmax": row["final_fmax"],
        })

    stability = pd.DataFrame(rows)
    stability["E_rel_meV"] = (
        stability["E_per_atom_meV"] - stability["E_per_atom_meV"].min()
    )
    stability.to_csv(OUTPUT_DIR / "stability_summary.csv", index=False)

    print(stability.to_string(index=False))
    print(f"Written: {OUTPUT_DIR / 'stability_summary.csv'}")


if __name__ == "__main__":
    main()
