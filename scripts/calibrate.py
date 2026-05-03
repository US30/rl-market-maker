"""
Calibrate Hawkes parameters from LOBSTER data and save to JSON.

Usage:
    python scripts/calibrate.py --data data/lobster/ --out rlmm/calibration/params.json
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlmm.calibration.lobster_loader import load_event_stream
from rlmm.calibration.hawkes_fit import fit_hawkes, save_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Directory with LOBSTER CSVs")
    p.add_argument("--out", default="rlmm/calibration/params.json")
    p.add_argument("--restarts", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    msgs = glob.glob(os.path.join(args.data, "*message*"))
    books = glob.glob(os.path.join(args.data, "*orderbook*"))
    if not msgs:
        print(f"No message CSV found in {args.data}")
        sys.exit(1)

    events, _ = load_event_stream(msgs[0], books[0] if books else None)
    T = events[-1].time - events[0].time if events else 0.0
    print(f"Loaded {len(events)} events over {T:.1f} seconds.")

    hawkes = fit_hawkes(events, T=T, n_restarts=args.restarts, verbose=True)
    save_params(hawkes, args.out)


if __name__ == "__main__":
    main()
