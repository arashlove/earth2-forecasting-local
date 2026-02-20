"""Run FCN3 24h (or N-step) forecast from ERA5 initial condition."""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import OrderedDict

import numpy as np
import torch

from .utils import parse_time


def _debug_shapes(debug: bool, phase: str, x: torch.Tensor | None = None, coords: OrderedDict | None = None) -> None:
    if not debug:
        return
    print(f"[debug] {phase}:", file=sys.stderr)
    if x is not None:
        print(f"  x.shape={x.shape} ndim={x.dim()}", file=sys.stderr)
    if coords is not None:
        print(f"  coords type={type(coords).__name__} len(coords)={len(coords)} keys={list(coords.keys())}", file=sys.stderr)
        for k, v in coords.items():
            arr = getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else "?")
            print(f"    coords[{k!r}].shape/len={arr}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="2023-01-01T00:00:00Z")
    ap.add_argument("--steps", type=int, default=4, help="6h steps; 4 = 24h")
    ap.add_argument("--out", default="forecast_24h.npy")
    ap.add_argument(
        "--weights",
        default=None,
        help="Override weights: HF ref (e.g. hf://nvidia/fourcastnet3@main) or local path. Default: official FCN3 package.",
    )
    ap.add_argument("--debug", action="store_true", help="Print tensor/coords shapes and re-raise with phase context on error.")
    args = ap.parse_args()

    phase = "parse_args"
    t0 = parse_time(args.t0)

    try:
        # 1) Load FCN3 from Hugging Face or local package via Earth2Studio
        phase = "load_model"
        from earth2studio.models.px.fcn3 import FCN3
        from earth2studio.models.auto import Package

        if args.weights is None:
            package = FCN3.load_default_package()
        else:
            package = Package(
                args.weights,
                cache_options={
                    "cache_storage": Package.default_cache("fcn3"),
                    "same_names": True,
                },
            )
        model = FCN3.load_model(package)
        model = model.cuda().eval()

        # 2) Fetch ERA5 initial condition (ARCO mirror)
        phase = "fetch_ARCO"
        from earth2studio.data import ARCO

        variables = list(model.variables)
        ds = ARCO()
        da0 = ds(time=t0.replace(tzinfo=None), variable=variables).load()

        # 3) Build input tensor (1, 1, 1, C, H, W) — batch_func requires len(shape)==len(coords)==6
        phase = "build_input"
        x = da0.to_numpy().astype("float32")
        while x.ndim > 3:
            x = x.squeeze(0)
        x = torch.from_numpy(x)[None, None, None, ...]
        x = x.cuda()

        ref = model.input_coords()
        coord_values = {
            "batch": np.array([0]),
            "time": np.array([np.datetime64(t0.replace(tzinfo=None))]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(variables),
            "lat": np.linspace(90.0, -90.0, 721),
            "lon": np.linspace(0.0, 360.0, 1440, endpoint=False),
        }
        coords = OrderedDict((k, coord_values[k]) for k in ref if k in coord_values)

        _debug_shapes(args.debug, "before create_iterator", x=x, coords=coords)

        # 4) Roll out N steps via model's iterator
        phase = "rollout (create_iterator)"
        with torch.inference_mode():
            it = model.create_iterator(x, coords)
            next(it)  # skip initial condition
            for step in range(args.steps):
                phase = f"rollout step {step + 1}/{args.steps}"
                cur, cur_coords = next(it)
                _debug_shapes(args.debug, phase, x=cur, coords=cur_coords)
            final = cur.detach().cpu().numpy().squeeze()

        phase = "save"
        np.save(args.out, final)
        print(f"saved {args.out}  shape={final.shape}  t0={t0.isoformat()}  lead={args.steps*6}h")

    except Exception:
        print(f"Error during: {phase}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print("Run with --debug to see tensor/coords shapes and full traceback.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
