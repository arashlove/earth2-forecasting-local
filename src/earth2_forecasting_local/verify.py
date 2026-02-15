"""Verify FCN3 forecast against ERA5 (RMSE)."""
from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
from earth2studio.data import ARCO
from earth2studio.models.px.fcn3 import FCN3

from .utils import parse_time, rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="2023-01-01T00:00:00Z")
    ap.add_argument("--pred", default="fcn3_24h.npy")
    ap.add_argument("--hours", type=int, default=24)
    args = ap.parse_args()

    t0 = parse_time(args.t0)
    tT = t0 + timedelta(hours=args.hours)

    pred = np.load(args.pred).astype(np.float32)

    model = FCN3.load_model(FCN3.load_default_package())
    variables = list(model.variables)

    ds = ARCO()
    daT = ds(time=tT.replace(tzinfo=None), variable=variables).load()
    truth = daT.to_numpy().astype(np.float32)

    print("RMSE:", rmse(pred, truth))


if __name__ == "__main__":
    main()
