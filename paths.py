"""Resolve the repository root and common project directories."""

import sys
from pathlib import Path


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _find_project_root(marker_files=("requirements.txt", "README.md")):
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in marker_files):
            return candidate
    raise RuntimeError(
        f"Project root not found above {current}; expected {marker_files}."
    )


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
PROCESSED_DIR = MODEL_DIR / "processed"
FIGURES_DIR = MODEL_DIR / "figures"
SCREENING_DIR = PROJECT_ROOT / "screening"
SCREENING_RESULTS = SCREENING_DIR / "results"
SCREENING_FIGURES = SCREENING_DIR / "figures"
MLIP_DIR = PROJECT_ROOT / "mlip"
MLIP_RESULTS = MLIP_DIR / "results"
MLIP_FIGURES = MLIP_DIR / "figures"
MLIP_TRAJECTORIES = MLIP_DIR / "trajectories"
MLIP_SEEDS = MLIP_DIR / "seeds"
