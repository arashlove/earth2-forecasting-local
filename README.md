# FCN3 local inference (Hugging Face weights + Earth2Studio)

Run **FourCastNet 3** forecasts locally in Python using **weights from Hugging Face** and the **Earth2Studio** FCN3 implementation (no NIM server). ERA5 initial conditions are fetched from the ARCO mirror.

End-to-end: WSL2 (or Linux), Poetry, CUDA PyTorch, run 24h forecast, verify against ERA5.

## Overview

| Step | What |
|------|------|
| 0 | Install Poetry (if needed) |
| 1 | Create Poetry env (Python 3.11) and install project: `poetry install` |
| 2 | Install CUDA PyTorch in the same env |
| 3 | Install torch-harmonics; optional: HF login, pre-download weights, CUDA extension |
| 4 | Run 24h forecast: `run-fcn3` |
| 5 | Verify forecast vs ERA5: `verify-fcn3` |

All commands below are from the **project root**, in WSL2 or a terminal.

## Project layout

Installable package (src layout). After `poetry install`, `run-fcn3` and `verify-fcn3` are available in the env.

```
earth2-forecasting-local/
├── pyproject.toml          # Poetry config, deps, CLI entry points
├── src/
│   └── earth2_forecasting_local/
│       ├── __init__.py
│       ├── utils.py         # parse_time, rmse
│       ├── run_fcn3.py      # 24h forecast (run-fcn3)
│       └── verify.py         # RMSE vs ERA5 (verify-fcn3)
└── README.md
```

## Requirements

- Python 3.11+
- CUDA GPU (≥60 GB VRAM recommended for FCN3)
- [Poetry](https://python-poetry.org/)

---

## 0) Install Poetry (if needed)

In WSL2 or Linux:

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry --version
```

## 1) Create env and install project

`poetry install` installs all project dependencies (Earth2Studio from GitHub with fcn3 extra, makani, scipy, ruamel.yaml, moviepy, numba/llvmlite, etc.). No separate `poetry add` needed.

```bash
poetry env use 3.11
poetry install

# Confirm
poetry run python -c "import earth2studio; print(earth2studio.__file__)"
poetry run run-fcn3 --help
```

## 2) Install CUDA PyTorch (GPU)

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Check GPU
poetry run python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"
```

(Adjust `cu118` / `cu124` if your driver needs a different CUDA version.)

## 3) Install torch-harmonics + optional extras

FCN3 uses **torch-harmonics**. Install it (GPU works with or without the optional CUDA extension):

```bash
poetry run pip install torch-harmonics
```

**Optional — faster GPU:** build the CUDA extension. In WSL this often fails unless the full CUDA Toolkit is installed and `CUDA_HOME` is set:

```bash
export CUDA_HOME=/usr/local/cuda   # or your CUDA toolkit path
export FORCE_CUDA_EXTENSION=1
poetry run pip install --no-build-isolation torch-harmonics
```

If that fails (e.g. `CUDA_HOME environment variable is not set`), keep the simple install above; the model still runs on GPU, possibly a bit slower.

**Optional — Hugging Face login** (not needed for public `nvidia/fourcastnet3`; only if gated or rate limits):

```bash
poetry run huggingface-cli login
```

**Optional — pre-download weights** (first run will download anyway):

```bash
poetry run python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('nvidia/fourcastnet3')
print('Cached at:', path)
"
```

## 4) Run a 24h forecast

```bash
poetry run run-fcn3 --t0 2023-01-01T00:00:00Z --steps 4 --out fcn3_24h.npy
```

- **`--t0`**: initial condition time (ERA5, UTC).
- **`--steps`**: number of 6h steps (4 = 24h).
- **`--out`**: output path for final state array `(C, H, W)`.
- **`--weights`**: (optional) Different HF ref (e.g. `hf://nvidia/fourcastnet3@main`) or local path with same layout as HF repo.

Example output:

```
saved fcn3_24h.npy  shape=(72, 721, 1440)  t0=2023-01-01T00:00:00+00:00  lead=24h
```

**First run — model assets:** The first time you run `run-fcn3`, Earth2Studio downloads the FCN3 package from Hugging Face (`nvidia/fourcastnet3`) and caches it locally. You’ll see progress for:

| File | Purpose |
|------|---------|
| `config.json` | Model config (variable list, grid, settings). |
| `orography.nc` | Elevation on the model grid (fixed input). |
| `land_mask.nc` | Land/sea mask on the model grid. |
| `global_means.npy` / `global_stds.npy` | Per-variable stats for normalization. |
| `mins.npy` / `maxs.npy` | Per-variable min/max for normalization. |
| `best_ckpt_mp0.tar` | Model weights checkpoint (~2.65 GB). |

After this one-time download, later runs use the cache and skip the download.

**Fetching ARCO (initial condition):** After the model is loaded, the script fetches the **ERA5 initial condition** at `--t0` for all 72 variables from the **ARCO** (Analysis-Ready, Cloud-Optimized) mirror. You’ll see `Fetching ARCO zarr array for variable: ...` and a progress bar (72/72 variables). That state is the starting point the model rolls forward from. ARCO data is streamed on demand; the first run for a given date can take a couple of minutes.

## 5) Verify against ERA5 (+24h)

```bash
poetry run verify-fcn3 --t0 2023-01-01T00:00:00Z --pred fcn3_24h.npy
```

Prints RMSE between the 24h forecast and ERA5 at t0+24h.

---

## One-shot (after `poetry install`)

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
poetry run pip install torch-harmonics
poetry run run-fcn3 --t0 2023-01-01T00:00:00Z --steps 4 --out fcn3_24h.npy
poetry run verify-fcn3 --t0 2023-01-01T00:00:00Z --pred fcn3_24h.npy
```

---

## Changing weights

- **Another HF ref:** `poetry run run-fcn3 --weights "hf://nvidia/fourcastnet3@main" ...`
- **Local directory:** same layout as HF repo (e.g. `config.json` + weight files): `--weights /path/to/weights`.

Architecture and code stay the same (Earth2Studio); only the weights (and optional `config.json`) change.

---

## What the CLIs do

**`run-fcn3`** (`earth2_forecasting_local.run_fcn3`): loads FCN3 from HF via Earth2Studio, gets ERA5 IC from ARCO for `model.variables` at `t0`, builds `(1,1,1,C,H,W)` + coords, rolls out N×6h steps, saves last step as `(C,H,W)` to `--out`.

**`verify-fcn3`** (`earth2_forecasting_local.verify`): loads forecast `.npy` and ERA5 at t0+24h from ARCO, prints RMSE.

---

## Notes

- **Weights**: Hugging Face (`nvidia/fourcastnet3`) by default; **code**: Earth2Studio FCN3. Variables and grid from the model’s `config.json` in the HF package.
- **Missing module after install**: All runtime deps are in `pyproject.toml`. Use a fresh `poetry lock` then `poetry install`; if you still see a missing module, ensure you have the latest `pyproject.toml`.
- **llvmlite/numba build fails**: Delete `poetry.lock`, run `poetry lock` and `poetry install` so the pinned numba/llvmlite/moviepy versions are used.
- **`torch.OutOfMemoryError: CUDA out of memory`**: FCN3 needs a large GPU (≥60 GB VRAM recommended); use a machine or cloud with a bigger GPU.
- **Shape/coords errors**: Paste the traceback; the code can be adjusted for your Earth2Studio version (0.12.x vs 0.13.x).

## WSL GPU setup (if needed)

1. **WSL2** with a distro (e.g. Ubuntu).
2. **NVIDIA driver** on Windows (same as for native CUDA).
3. **WSL CUDA**: PyTorch’s `cu121` wheel is usually enough; optional full CUDA Toolkit in WSL for building torch-harmonics extension.
4. **Poetry** in WSL: `curl -sSL https://install.python-poetry.org | python3 -`.

Then run steps 0–5 in order.
