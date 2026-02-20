"""Run FCN3 ensemble forecast using NCAR_ERA5 + NetCDF (Earth2Studio ensemble workflow)."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from earth2studio.models.px.fcn3 import FCN3
from earth2studio.data import NCAR_ERA5
from earth2studio.io import NetCDF4Backend
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble as run_ensemble


# Default output variables (subset for smaller NetCDF; model has 72)
DEFAULT_OUT_VARS = ["u10m", "v10m", "t2m", "msl", "tcwv"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FCN3 ensemble forecast: NCAR_ERA5 ICs, NetCDF output (Earth2Studio ensemble workflow)."
    )
    ap.add_argument(
        "--time",
        nargs="+",
        default=["2024-09-24"],
        help="Initialization time(s), e.g. 2024-09-24 or 2024-09-24T00:00:00. Default: 2024-09-24",
    )
    ap.add_argument("--nsteps", type=int, default=16, help="Number of 6h forecast steps (default: 16)")
    ap.add_argument("--nensemble", type=int, default=4, help="Number of ensemble members (default: 4)")
    ap.add_argument(
        "--out",
        default="fcn3_ensemble.nc",
        help="Output NetCDF path (default: fcn3_ensemble.nc)",
    )
    ap.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help=f"Output variables (default: {DEFAULT_OUT_VARS}). Use model default if not set.",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Ensemble batch size (default: 1)")
    ap.add_argument("--no-verbose", action="store_true", help="Disable progress output")
    args = ap.parse_args()

    out_vars = args.vars if args.vars is not None else DEFAULT_OUT_VARS

    try:
        # Load FCN3 (default Hugging Face package)
        model = FCN3.load_model(FCN3.load_default_package())

        # Data source: NCAR ERA5 initial conditions
        ds = NCAR_ERA5()
        io = NetCDF4Backend(args.out, backend_kwargs={"mode": "w"})

        # No perturbation (FCN3 hidden Markov formulation)
        perturbation = Zero()

        run_ensemble(
            time=args.time,
            nsteps=args.nsteps,
            nensemble=args.nensemble,
            prognostic=model,
            data=ds,
            io=io,
            perturbation=perturbation,
            batch_size=args.batch_size,
            output_coords={"variable": np.array(out_vars)},
            verbose=not args.no_verbose,
        )
        print(f"Ensemble forecast written to {args.out}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
