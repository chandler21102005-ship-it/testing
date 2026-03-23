"""
polymarket_strike_capture.py
-----------------------------
Standalone strike price capture logic for Polymarket Up/Down markets.

WHAT THIS DOES:
  Connects to Polymarket's real-time Chainlink WebSocket feed and captures
  the exact BTC/ETH/SOL/XRP price at the start of each 5-minute and
  15-minute round boundary. This captured price is the "strike price" —
  the price BTC must be above (for YES/Up) or below (for NO/Down) at
  round end for the market to resolve in that direction.

WHY THIS MATTERS:
  Polymarket resolves all Up/Down markets using the Chainlink oracle price,
  NOT Binance or any other exchange. The strike price is the Chainlink price
  at exactly the round start boundary (e.g. 15:00:00, 15:05:00, 15:10:00).
  If you use Binance prices for your strike, you will get the wrong number
  and your signal calculations will be off.

HOW BOUNDARY DETECTION WORKS:
  Round boundaries are simple multiples of the interval:
    5-min boundaries:  boundary_ts = (unix_ts // 300) * 300
    15-min boundaries: boundary_ts = (unix_ts // 900) * 900

  On every incoming Chainlink tick:
    1. Compute which boundary this tick belongs to.
    2. If the boundary is different from the last seen boundary,
       a new round has started — record this tick's price as the strike.
    3. If same boundary, just update the rolling price buffer (for signals).

  The first tick after the clock crosses a boundary IS the strike.
  No averaging, no approximation, no staleness.

MID-ROUND START PROTECTION:
  When the bot first starts, it arrives mid-round. The first tick for each
  asset initialises the tracker WITHOUT recording a strike (because we
  don't know what the real boundary price was before we connected).
  The next boundary crossing will then correctly capture the strike.
  This means: the first round after bot startup is always skipped.

WEBSOCKET FEED DETAILS:
  URL:      wss://ws-live-data.polymarket.com
  Topic:    crypto_prices_chainlink
  Keep-alive: send text "PING" every 5 seconds (not WS ping frame)
  Reconnect: exponential backoff with 60s cap

PUBLIC INTERFACE (for LLMs integrating this):
  feed = CryptoFeed()
  await feed.start()                                    # run as background task

  # Get the strike price for a round:
  strike = feed.get_boundary_price("BTC", 300, boundary_ts)
  # Returns float if captured, None if bot started mid-round or not yet arrived

  # Get latest live price:
  price = feed.latest_price("BTC")

  # Get recent price history for signal calculation:
  prices = feed.get_prices_last_n_seconds("BTC", 30)   # numpy array

  await feed.stop()                                     # clean shutdown

HOW TO COMPUTE boundary_ts FOR THE CURRENT ROUND:
  import time
  interval_sec  = 300                                   # 5-min round
  boundary_ts   = (int(time.time()) // interval_sec) * interval_sec
  strike        = feed.get_boundary_price("BTC", interval_sec, boundary_ts)

DEPENDENCIES:
  pip install websockets numpy loguru
"""

import asyncio
import json
import time
from collections import deque
from typing import Optional

import numpy as np
import websockets
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Polymarket RTDS (Real-Time Data Stream) — Chainlink price feed
RTDS_URL = "wss://ws-live-data.polymarket.com"

# Chainlink symbol → asset label used throughout the bot
SYMBOL_MAP: dict[str, str] = {
    "btc/usd": "BTC",
    "eth/usd": "ETH",
    "sol/usd": "SOL",
    "xrp/usd": "XRP",
}

ASSETS = list(SYMBOL_MAP.values())   # ["BTC", "ETH", "SOL", "XRP"]

# Only track 5-min and 15-min boundaries — the rounds we actually bet on.
# 1-hour and 4-hour markets exist on Polymarket but are not tracked here.
TRACKED_INTERVALS = [300, 900]       # seconds

# Rolling tick buffer per asset (~1 tick/sec → 3600 = ~1 hour of history)
BUFFER_MAXLEN = 3600

# Polymarket RTDS requires a text "PING" every 5 seconds to stay connected.
# This is NOT a standard WebSocket ping frame — it must be a text message.
PING_INTERVAL_SEC = 5.0

# Reconnect backoff cap
MAX_RECONNECT_DELAY = 60.0

# If no tick arrives for 30s, treat the connection as a zombie and reconnect.
# Silent drops happen on Polymarket's end after several hours of uptime.
WATCHDOG_SEC = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# CryptoFeed
# ─────────────────────────────────────────────────────────────────────────────

class CryptoFeed:
    """
    Real-time Chainlink price feed with automatic strike price capture.

    Maintains:
      1. Rolling tick buffer per asset — for momentum/signal calculation.
      2. Boundary price dict — exact Chainlink price at each round start.

    Runs forever as a background asyncio task via start().
    """

    def __init__(self):
        # Rolling tick buffer: asset → deque of (unix_timestamp, price)
        self._ticks: dict[str, deque] = {
            asset: deque(maxlen=BUFFER_MAXLEN) for asset in ASSETS
        }

        # Strike prices: (asset, interval_sec, boundary_ts) → price
        # Example key: ("BTC", 300, 1772655000) → 67846.61
        # Populated automatically on every boundary crossing.
        self._boundary_prices: dict[tuple, float] = {}

        # Last seen boundary per (asset, interval).
        # None = feed just started, first tick will initialise without capturing.
        self._last_boundary: dict[tuple, Optional[int]] = {
            (asset, interval): None
            for asset in ASSETS
            for interval in TRACKED_INTERVALS
        }

        # When start() was called — used externally to detect mid-round starts.
        # If boundary_ts < started_at, the bot was not running at that boundary.
        self.started_at: float = 0.0

        self._running = False

    # ── Public interface ──────────────────────────────────────────────────────

    def latest_price(self, asset: str) -> Optional[float]:
        """Most recent Chainlink price for the asset, or None if not yet received."""
        buf = self._ticks.get(asset)
        if not buf:
            return None
        return buf[-1][1]

    def get_prices_last_n_seconds(self, asset: str, n_seconds: int) -> np.ndarray:
        """
        Returns a numpy array of prices from the last n_seconds. Oldest first.
        Used for momentum and volatility calculation in the signal engine.
        Returns empty array if no data yet.
        """
        buf = self._ticks.get(asset)
        if not buf:
            return np.array([])
        cutoff = time.time() - n_seconds
        prices = [price for ts, price in buf if ts >= cutoff]
        return np.array(prices, dtype=float)

    def get_boundary_price(
        self,
        asset:        str,
        interval_sec: int,
        boundary_ts:  int,
    ) -> Optional[float]:
        """
        Returns the Chainlink strike price recorded at the start of a round.

        This is the price that Polymarket uses as the strike for all
        Up/Down markets of this asset and interval starting at boundary_ts.

        Returns None if:
          - Bot started after this boundary (mid-round start)
          - This boundary hasn't crossed yet
          - interval_sec is not in TRACKED_INTERVALS

        Usage:
          boundary_ts = (int(time.time()) // 300) * 300
          strike = feed.get_boundary_price("BTC", 300, boundary_ts)
          if strike is None:
              # Skip this round — strike unknown
        """
        return self._boundary_prices.get((asset, interval_sec, boundary_ts))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """
        Start the feed as a background task. Reconnects automatically on error.
        Call this with asyncio.create_task(feed.start()).
        """
        self._running   = True
        self.started_at = time.time()
        logger.info(f"CryptoFeed: starting Chainlink stream (started_at={self.started_at:.3f})...")

        delay = 1.0
        while self._running:
            try:
                await self._run_session()
                delay = 1.0
            except Exception as e:
                logger.warning(f"CryptoFeed: session error: {e}")

            if self._running:
                logger.info(f"CryptoFeed: reconnecting in {delay:.0f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def stop(self):
        """Signal clean shutdown."""
        self._running = False
        logger.info("CryptoFeed: stopped.")

    # ── WebSocket session ─────────────────────────────────────────────────────

    async def _run_session(self):
        """
        One WebSocket connection session.
        Subscribes to Chainlink prices, receives ticks, sends keep-alive PINGs.
        Breaks on connection loss or watchdog timeout — outer loop reconnects.
        """
        async with websockets.connect(
            RTDS_URL,
            ping_interval = None,   # We send manual text "PING", not WS pings
            close_timeout = 5,
        ) as ws:
            logger.info("CryptoFeed: connected to Polymarket RTDS.")

            # Subscribe to Chainlink price feed for all assets
            await ws.send(json.dumps({
                "action": "subscribe",
                "subscriptions": [{
                    "topic":   "crypto_prices_chainlink",
                    "type":    "*",
                    "filters": "",
                }],
            }))
            logger.info("CryptoFeed: subscribed to crypto_prices_chainlink.")

            last_ping_ts = time.time()
            last_tick_ts = time.time()

            while self._running:

                # ── Send PING every 5s to keep connection alive ───────────
                # Polymarket RTDS requires this text message, not a WS ping frame.
                now = time.time()
                if now - last_ping_ts >= PING_INTERVAL_SEC:
                    try:
                        await ws.send("PING")
                        last_ping_ts = now
                    except Exception as e:
                        logger.warning(f"CryptoFeed: PING failed: {e}")
                        break

                # ── Receive next message (2s timeout keeps PING loop alive) ─
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    # No message — check watchdog
                    if time.time() - last_tick_ts > WATCHDOG_SEC:
                        logger.warning(
                            f"CryptoFeed: no tick in {WATCHDOG_SEC:.0f}s — "
                            f"zombie connection, reconnecting..."
                        )
                        break
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"CryptoFeed: connection closed: {e}")
                    break

                last_tick_ts = time.time()

                # Parse and process the message
                try:
                    self._handle_message(json.loads(raw))
                except json.JSONDecodeError:
                    pass

    # ── Core logic: boundary detection and strike capture ────────────────────

    def _handle_message(self, msg: dict):
        """
        Process one incoming Chainlink price tick.

        STEP 1: Extract asset and price from the message.
        STEP 2: Append to rolling price buffer (for signals/momentum).
        STEP 3: Check if a new round boundary has been crossed.
                If yes → record this price as the strike for that round.

        BOUNDARY CROSSING LOGIC (the core of this file):
          Every tick belongs to a boundary:
            boundary_ts = (tick_unix_seconds // interval) * interval

          Three cases:
            A. _last_boundary is None (first tick ever for this asset+interval)
               → Initialise tracker. Do NOT record a strike.
                 We arrived mid-round — we don't know the true boundary price.
                 The NEXT boundary crossing will capture correctly.

            B. boundary_ts == _last_boundary (same round as before)
               → Nothing to do for strike. Just store the tick in the buffer.

            C. boundary_ts != _last_boundary (new round just started!)
               → THIS TICK IS THE FIRST TICK OF A NEW ROUND.
                 Record price as the strike for boundary_ts.
                 Update _last_boundary to this new boundary.

        WHY USE THE PAYLOAD TIMESTAMP (not time.time()):
          The payload contains the timestamp from when Chainlink recorded
          the price. Using this instead of wall clock time means boundary
          detection is accurate even with small network delays.
        """
        if not isinstance(msg, dict):
            return
        if msg.get("topic") != "crypto_prices_chainlink":
            return

        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return

        # Map Chainlink symbol (e.g. "btc/usd") to our asset label ("BTC")
        symbol = payload.get("symbol", "").lower()
        asset  = SYMBOL_MAP.get(symbol)
        if asset is None:
            return   # Asset we don't track (e.g. LINK/USD)

        # Parse price and timestamp
        try:
            price  = float(payload["value"])
            raw_ts = payload.get("timestamp")
            # Polymarket sends timestamps in milliseconds
            ts     = float(raw_ts) / 1000.0 if raw_ts else time.time()
        except (KeyError, TypeError, ValueError):
            return

        if price <= 0:
            return

        # ── Step 1: Store tick in rolling buffer ──────────────────────────
        self._ticks[asset].append((ts, price))

        # ── Step 2: Check every tracked interval for boundary crossing ────
        tick_ts_int = int(ts)

        for interval in TRACKED_INTERVALS:
            # Compute which boundary this tick belongs to
            # e.g. tick at 15:07:23 with interval=300 → boundary_ts=15:05:00
            boundary_ts = (tick_ts_int // interval) * interval
            key         = (asset, interval)
            last        = self._last_boundary[key]

            if last is None:
                # ── Case A: First tick ever — arrived mid-round ───────────
                # Initialise tracker to current boundary WITHOUT recording
                # a strike. We don't know the true price at boundary start.
                self._last_boundary[key] = boundary_ts
                logger.debug(
                    f"CryptoFeed: {asset} {interval // 60}m — "
                    f"first tick mid-round, boundary={boundary_ts}, "
                    f"strike NOT captured (bot started mid-round)."
                )

            elif boundary_ts != last:
                # ── Case C: New boundary crossed — new round started ──────
                # This is the first Chainlink tick of the new round.
                # Record this price as the exact strike for this round.
                self._boundary_prices[(asset, interval, boundary_ts)] = price
                self._last_boundary[key] = boundary_ts
                logger.info(
                    f"CryptoFeed: ✔ strike captured | "
                    f"{asset} {interval // 60}m | "
                    f"boundary_ts={boundary_ts} | "
                    f"strike=${price:.6f}"
                )
            # Case B: boundary_ts == last → same round, nothing to do


# ─────────────────────────────────────────────────────────────────────────────
# Usage example
# ─────────────────────────────────────────────────────────────────────────────

async def example():
    """
    Shows how to use CryptoFeed to capture strike prices.

    In a real bot:
      1. Start feed as a background task at bot startup.
      2. Wait ~8 seconds for the WebSocket to connect and receive first ticks.
      3. At the signal window (e.g. 60s into a round), call get_boundary_price()
         to retrieve the strike for the current round.
      4. Use strike in your signal calculation.
    """
    feed = CryptoFeed()

    # Start as background task — runs forever, reconnects on error
    feed_task = asyncio.create_task(feed.start())

    # Wait for connection and first ticks
    print("Waiting 10s for feed to connect and receive first ticks...")
    await asyncio.sleep(10)

    # How to get boundary_ts for the current 5-min round
    interval_sec  = 300
    boundary_ts   = (int(time.time()) // interval_sec) * interval_sec

    # Get the strike price for the current round
    strike = feed.get_boundary_price("BTC", interval_sec, boundary_ts)

    if strike is not None:
        print(f"BTC strike price for current round: ${strike:.4f}")
        print(f"Current BTC price:                  ${feed.latest_price('BTC'):.4f}")
        print(f"Gap from strike:                    ${feed.latest_price('BTC') - strike:.4f}")
    else:
        print("Strike not captured yet — bot started mid-round.")
        print("Wait for the next 5-min boundary to cross.")

    # Clean shutdown
    await feed.stop()
    feed_task.cancel()


if __name__ == "__main__":
    asyncio.run(example())
