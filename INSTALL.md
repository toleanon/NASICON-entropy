# Installation

## Requirements

- Python 3.10 or newer (tested with Python 3.12).
- `pip`.
- Optional: NVIDIA GPU with a CUDA-compatible PyTorch build for the MLIP stage.

## Basic setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## PyTorch and CUDA

`requirements.txt` pins `torch==2.5.1`. PyPI may provide a CPU build on some
platforms. For GPU use, install the CUDA wheel that matches the driver, for
example CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.4:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining packages:

```bash
pip install -r requirements.txt
```

Verify the GPU build:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## MLIP notes

- `mace-torch` downloads the MACE-MP-0 weights on first use; no weights are
  stored in this repository.
- `pymatgen-analysis-diffusion` provides the diffusion analyzer used by the MD
  analysis script.
- LightGBM is required by the model benchmark.
- Plotting scripts set the Matplotlib `Agg` backend, so a display is not needed.
- All paths are resolved through `paths.py`; there are no hard-coded absolute
  paths in the repository.

## Data and trajectories

The curated dataset, relaxed structures, MLIP trajectories, trained model files,
and generated results are not distributed. Place local inputs under
`data/`, `mlip/seeds/`, and `mlip/trajectories/` as described in `README.md` and
the module README files.

## Run order

See the Quick start section in `README.md`. The high-throughput screening stage
requires the model stage; the MLIP analysis and plotting stages require local
relaxation/MD outputs.
