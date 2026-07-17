"""
Inspect/validate CSV uploads before trusting their predictions.

For each CSV it reports: detected format, frames/windows, missing-joint fraction,
an out-of-distribution score vs the model's training data, the spike-count decision,
and (if you pass a known label) whether it was correct.

Use it to sanity-check a new CSV source before relying on the backend's output, since
the model is only reliable on data resembling its Kinect training distribution.

Usage:
  python -m train.check_csv path/to/file.csv
  python -m train.check_csv "C:/folder/*.csv"
  python -m train.check_csv path/to/file.csv --label TD     # ASD|TD to score correctness
"""
import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from services.feature_engineering import prepare_csv_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "xgboost_asd_spatiotemporal_optimized.json")
STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect_feature_stats.npz")

WINDOW_RISK_THRESHOLD = 0.30
MIN_ATYPICAL_FRACTION = 0.10
MIN_ATYPICAL_WINDOWS = 2


def decide(p):
    n = len(p)
    cnt = int(np.sum(p >= WINDOW_RISK_THRESHOLD))
    min_w = max(MIN_ATYPICAL_WINDOWS, int(n * MIN_ATYPICAL_FRACTION))
    flag = (float(np.mean(p)) >= 0.50) if n < MIN_ATYPICAL_WINDOWS else (cnt >= min_w)
    return cnt, n, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--label", choices=["ASD", "TD"], default=None)
    args = ap.parse_args()

    model = xgb.XGBClassifier(); model.load_model(MODEL)
    stats = np.load(STATS) if os.path.exists(STATS) else None
    mu, sd, base = (stats["mu"], stats["sd"], float(stats["baseline"])) if stats is not None else (None, None, None)

    files = glob.glob(args.path) if any(c in args.path for c in "*?[") else [args.path]
    if not files:
        print(f"No files match {args.path}"); return

    correct = 0
    for f in sorted(files):
        try:
            X, ts, kind = prepare_csv_data(pd.read_csv(f))
            p = model.predict_proba(X)[:, 1]
            cnt, n, flag = decide(p)
            ood = float(np.abs((X - mu) / sd).mean()) if mu is not None else float("nan")
            ood_flag = "" if (mu is None or ood <= base * 1.3) else "  ⚠️OUT-OF-DISTRIBUTION(low confidence)"
            verdict = "ATYPICAL" if flag else "typical"
            ok = ""
            if args.label:
                hit = flag == (args.label == "ASD")
                correct += hit
                ok = "  [OK]" if hit else "  [WRONG]"
            print(f"{os.path.basename(f):32s} {kind:11s} win={n:3d} spikes={cnt:3d} "
                  f"mean={p.mean():.3f} OOD={ood:.2f} -> {verdict}{ok}{ood_flag}")
        except Exception as e:
            print(f"{os.path.basename(f):32s} ERROR: {e}")

    if args.label and len(files) > 1:
        print(f"\nAccuracy vs label '{args.label}': {correct}/{len(files)}")


if __name__ == "__main__":
    main()
