"""
Event-driven L3 limit-order-book with price-time priority.

Supports: limit orders, market orders, cancellations.
Tracks queue position for the agent's resting orders.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np


@dataclass
class Order:
    order_id: int
    side: int       # 0 = bid, 1 = ask
    price: float
    size: float
    is_agent: bool = False


class Fill(NamedTuple):
    side: int       # aggressor side (0=buy MO, 1=sell MO)
    price: float
    size: float
    is_agent_passive: bool


class LOB:
    """
    Price-time priority LOB.

    Prices are rounded to `tick_size` grid.
    `n_levels` controls how many levels are returned by get_snapshot().
    """

    def __init__(self, tick_size: float = 0.01, n_levels: int = 10):
        self.tick_size = tick_size
        self.n_levels = n_levels

        # price -> deque[Order] (time-ordered within each price level)
        self._bids: dict[float, deque[Order]] = {}
        self._asks: dict[float, deque[Order]] = {}

        self._orders: dict[int, Order] = {}
        self._next_id: int = 0

        # most-recent fills this step (cleared on demand)
        self.fills: list[Fill] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def _book(self, side: int) -> dict[float, deque[Order]]:
        return self._bids if side == 0 else self._asks

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_limit(self, side: int, price: float, size: float, is_agent: bool = False) -> int:
        price = self._tick(price)
        order = Order(self._next_id, side, price, size, is_agent)
        self._next_id += 1
        self._orders[order.order_id] = order
        book = self._book(side)
        if price not in book:
            book[price] = deque()
        book[price].append(order)
        return order.order_id

    def submit_market(self, side: int, size: float) -> list[Fill]:
        """
        side=0: buy MO (aggresses asks ascending).
        side=1: sell MO (aggresses bids descending).
        """
        contra_book = self._asks if side == 0 else self._bids
        price_iter = (
            iter(sorted(contra_book.keys()))
            if side == 0
            else iter(sorted(contra_book.keys(), reverse=True))
        )

        new_fills: list[Fill] = []
        remaining = size

        for price in price_iter:
            if remaining <= 1e-10:
                break
            if price not in contra_book:
                continue
            queue = contra_book[price]
            while queue and remaining > 1e-10:
                order = queue[0]
                fill_size = min(order.size, remaining)
                order.size -= fill_size
                remaining -= fill_size
                f = Fill(side, price, fill_size, order.is_agent)
                new_fills.append(f)
                if order.size <= 1e-10:
                    queue.popleft()
                    self._orders.pop(order.order_id, None)
            if not queue:
                del contra_book[price]

        self.fills.extend(new_fills)
        return new_fills

    def cancel(self, order_id: int) -> bool:
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._book(order.side)
        q = book.get(order.price)
        if q is not None:
            try:
                q.remove(order)
            except ValueError:
                pass
            if not q:
                del book[order.price]
        return True

    def clear_fills(self):
        self.fills.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        return min(self._asks) if self._asks else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb is not None and ba is not None else None

    def queue_position(self, order_id: int) -> Optional[float]:
        """Total size ahead of `order_id` in its queue. None if not found."""
        order = self._orders.get(order_id)
        if order is None:
            return None
        q = self._book(order.side).get(order.price)
        if q is None:
            return None
        ahead = 0.0
        for o in q:
            if o.order_id == order_id:
                break
            ahead += o.size
        return ahead

    def get_snapshot(self, n_levels: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (bid_sizes, ask_sizes) each of length n_levels.
        bid_sizes[0] = size at best bid, ask_sizes[0] = size at best ask.
        """
        n = n_levels or self.n_levels
        bid_prices = sorted(self._bids.keys(), reverse=True)[:n]
        ask_prices = sorted(self._asks.keys())[:n]

        bid_sizes = np.array(
            [sum(o.size for o in self._bids[p]) for p in bid_prices], dtype=np.float32
        )
        ask_sizes = np.array(
            [sum(o.size for o in self._asks[p]) for p in ask_prices], dtype=np.float32
        )

        bid_sizes = np.pad(bid_sizes, (0, n - len(bid_sizes)))
        ask_sizes = np.pad(ask_sizes, (0, n - len(ask_sizes)))
        return bid_sizes, ask_sizes

    def total_bid_size(self) -> float:
        return sum(o.size for q in self._bids.values() for o in q)

    def total_ask_size(self) -> float:
        return sum(o.size for q in self._asks.values() for o in q)

    def order_imbalance(self) -> float:
        """(bid_volume - ask_volume) / (bid_volume + ask_volume) in [-1, 1]."""
        bv = self.total_bid_size()
        av = self.total_ask_size()
        denom = bv + av
        return (bv - av) / denom if denom > 0 else 0.0

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb >= ba

    def reset(self):
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self.fills.clear()
        self._next_id = 0
