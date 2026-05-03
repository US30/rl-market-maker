"""
Notebook 4: LOBSTER Calibration

Fits Hawkes parameters to real LOBSTER data, then:
  1. Plots fitted vs empirical inter-arrival distributions
  2. Compares simulated stylized facts with LOBSTER stylized facts
  3. Saves calibrated params to rlmm/calibration/params.json

Requires LOBSTER data in data/lobster/ directory.
Download samples from: https://lobsterdata.com/info/DataSamples.php

Run: python -m rlmm.notebooks.04_lobster_calib --data data/lobster/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rlmm.calibration.lobster_loader import load_event_stream, compute_stylized_facts
from rlmm.calibration.hawkes_fit import fit_hawkes, save_params
from rlmm.sim.hawkes import MultivariateHawkes
from rlmm.sim.flow import OrderFlowSampler

SAVE_DIR = "results/figs"
CALIB_OUT = "rlmm/calibration/params.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/lobster/", help="Path to LOBSTER data directory")
    p.add_argument("--message", default=None, help="Direct path to message CSV")
    p.add_argument("--book", default=None, help="Direct path to book CSV")
    return p.parse_args()


def find_lobster_files(data_dir: str):
    """Auto-find message and book CSVs in data_dir."""
    import glob
    msgs = glob.glob(os.path.join(data_dir, "*message*"))
    books = glob.glob(os.path.join(data_dir, "*orderbook*"))
    return (msgs[0] if msgs else None), (books[0] if books else None)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    args = parse_args()

    # Find data files
    msg_path = args.message
    book_path = args.book
    if msg_path is None:
        msg_path, book_path = find_lobster_files(args.data)

    if msg_path is None or not os.path.exists(msg_path):
        print(f"LOBSTER message file not found in {args.data}")
        print("Download sample data from https://lobsterdata.com/info/DataSamples.php")
        print("Place in data/lobster/ and rerun.")
        return

    print(f"Loading LOBSTER data from {msg_path}...")
    events, book = load_event_stream(msg_path, book_path)
    T = events[-1].time - events[0].time if events else 0.0
    print(f"Loaded {len(events)} events over {T:.1f} seconds.")

    # Stylized facts from real data
    real_facts = compute_stylized_facts(events, book)
    print("\nReal LOBSTER stylized facts:")
    for k, v in real_facts.items():
        print(f"  {k}: {v:.4f}")

    # Fit Hawkes
    print("\nFitting Hawkes parameters (MLE, 3 restarts)...")
    hawkes = fit_hawkes(events, T=T, n_restarts=3, verbose=True)

    # Save params
    os.makedirs(os.path.dirname(CALIB_OUT), exist_ok=True)
    save_params(hawkes, CALIB_OUT)

    # Simulate with fitted params and compare
    print("\nSimulating with fitted params...")
    sim_events = hawkes.simulate(T_max=T, seed=0)
    sim_facts = compute_stylized_facts(sim_events)
    print("Simulated stylized facts:")
    for k, v in sim_facts.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # Figure: Compare inter-arrival time distributions
    # ------------------------------------------------------------------
    from rlmm.sim.hawkes import MO_BUY, MO_SELL

    real_times = np.array([e.time for e in events])
    real_etypes = np.array([e.etype for e in events])
    sim_times = np.array([e.time for e in sim_events])
    sim_etypes = np.array([e.etype for e in sim_events])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    type_names = ["MO-Buy", "MO-Sell", "LO-Buy", "LO-Sell"]

    for etype in range(4):
        ax = axes[etype // 2, etype % 2]
        r_idx = np.where(real_etypes == etype)[0]
        s_idx = np.where(sim_etypes == etype)[0]
        if len(r_idx) > 1:
            real_iat = np.diff(real_times[r_idx])
            ax.hist(real_iat, bins=50, density=True, alpha=0.6, label="LOBSTER", color="steelblue")
        if len(s_idx) > 1:
            sim_iat = np.diff(sim_times[s_idx])
            ax.hist(sim_iat, bins=50, density=True, alpha=0.6, label="Simulated", color="coral")
        ax.set_title(f"IAT: {type_names[etype]}")
        ax.set_xlabel("Inter-arrival time (sec)")
        ax.legend()

    plt.tight_layout()
    path = f"{SAVE_DIR}/04_hawkes_calibration.png"
    plt.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
