"""
LOBSTER data loader.

LOBSTER format:
  Message file: TICKER_DATE_START_END_message_LEVELS.csv
  Columns: Time, Type, OrderID, Size, Price, Direction
  Types: 1=LO submit, 2=LO cancel (partial), 3=LO cancel (full), 4=exec visible, 5=exec hidden

  Orderbook file: TICKER_DATE_START_END_orderbook_LEVELS.csv
  Columns: AskPrice1, AskSize1, BidPrice1, BidSize1, ... (alternating, n_levels pairs)

We extract:
  - Timestamps of each event type
  - Event classification: MO (type 4/5), LO (type 1), cancel (2/3)
  - Direction: +1 buy, -1 sell

Output: list of (time, etype) matching Hawkes event types:
    0=MO-buy, 1=MO-sell, 2=LO-buy, 3=LO-sell
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class LOBSTEREevnt(NamedTuple):
    time: float
    etype: int   # 0=MO-buy, 1=MO-sell, 2=LO-buy, 3=LO-sell


def load_lobster_messages(message_path: str | Path) -> pd.DataFrame:
    """Load raw LOBSTER message CSV."""
    df = pd.read_csv(
        message_path,
        header=None,
        names=["time", "type", "order_id", "size", "price", "direction"],
    )
    df["price"] = df["price"] / 10000.0   # LOBSTER prices are in 1/10000 USD
    return df


def load_lobster_book(book_path: str | Path, n_levels: int = 10) -> pd.DataFrame:
    """Load LOBSTER orderbook snapshot CSV."""
    cols = []
    for i in range(1, n_levels + 1):
        cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
    df = pd.read_csv(book_path, header=None, names=cols)
    for col in [c for c in cols if "price" in c]:
        df[col] = df[col] / 10000.0
    return df


def messages_to_hawkes_events(msgs: pd.DataFrame) -> list[LOBSTEREevnt]:
    """
    Convert LOBSTER messages to Hawkes event stream.

    LOBSTER types:
        1 = new LO submission
        2 = partial cancel
        3 = full cancel
        4 = execution against visible LO (i.e., incoming MO)
        5 = execution against hidden LO

    Direction: +1 = buy side, -1 = sell side.
    We use execution events (4,5) as MO proxies, submissions (1) as LO proxies.
    """
    events: list[LOBSTEREevnt] = []

    for _, row in msgs.iterrows():
        t = float(row["time"])
        typ = int(row["type"])
        direction = int(row["direction"])

        if typ in (4, 5):
            # Market order: direction=1 → buyer aggresses (MO-buy), direction=-1 → MO-sell
            etype = 0 if direction == 1 else 1
            events.append(LOBSTEREevnt(t, etype))
        elif typ == 1:
            # Limit order submission: direction=1 → LO-buy, direction=-1 → LO-sell
            etype = 2 if direction == 1 else 3
            events.append(LOBSTEREevnt(t, etype))

    return sorted(events, key=lambda e: e.time)


def load_event_stream(
    message_path: str | Path,
    book_path: str | Path | None = None,
    n_levels: int = 10,
) -> tuple[list[LOBSTEREevnt], pd.DataFrame | None]:
    """
    Convenience loader: returns (events, book_df).
    """
    msgs = load_lobster_messages(message_path)
    events = messages_to_hawkes_events(msgs)
    book = load_lobster_book(book_path, n_levels) if book_path else None
    return events, book


def compute_stylized_facts(events: list[LOBSTEREevnt], book: pd.DataFrame | None = None) -> dict:
    """
    Compute stylized facts for comparison with simulated LOB:
      - Inter-arrival times by event type
      - MO sign autocorrelation
      - Trade clustering (Fano factor)
    """
    from scipy.stats import pearsonr

    etypes = np.array([e.etype for e in events])
    times = np.array([e.time for e in events])

    facts = {}

    for etype, name in [(0, "mo_buy"), (1, "mo_sell"), (2, "lo_buy"), (3, "lo_sell")]:
        idx = np.where(etypes == etype)[0]
        if len(idx) > 1:
            iat = np.diff(times[idx])
            facts[f"{name}_mean_iat"] = float(iat.mean())
            facts[f"{name}_std_iat"] = float(iat.std())
            facts[f"{name}_rate"] = float(1.0 / iat.mean())

    # MO sign autocorrelation (lag-1)
    mo_mask = (etypes == 0) | (etypes == 1)
    mo_signs = np.where(etypes[mo_mask] == 0, 1.0, -1.0)
    if len(mo_signs) > 2:
        r, _ = pearsonr(mo_signs[:-1], mo_signs[1:])
        facts["mo_sign_autocorr_lag1"] = float(r)

    return facts
