"""
polymarket_sell_logic.py
------------------------
Standalone example of how to SELL (exit) a position on Polymarket mid-market.
This is NOT redemption — redemption is only for winning positions at market close.
This is a real limit SELL order placed on the Polymarket CLOB while the market
is still live, used to cut losses (stop loss) or exit on a directional signal.

CONTEXT: Polymarket uses ERC-1155 conditional tokens. When you BUY a YES or NO
share, you receive an on-chain token. To exit before resolution, you place a SELL
order on the CLOB — someone else buys your token at the current market price.

VERIFIED AGAINST: https://docs.polymarket.com/developers/CLOB/orders/create-order
The official docs confirm SELL orders use side=SELL with OrderArgs, posted via
client.post_order(signed_order, OrderType.GTC). That is exactly what this does.

IMPORTANT GOTCHA — share discrepancy:
  Polymarket deducts its fee from the ERC-1155 tokens themselves, not from USDC.
  So if the fill says you received 1.62 shares, your wallet actually holds ~1.60.
  If you try to sell 1.62 you get a 400 error ("not enough balance").
  Fix: always fetch the actual on-chain balance after buying, use that for the sell.

IMPORTANT GOTCHA — stale credentials:
  The CLOB returns a misleading 400 "Size X lower than minimum: Y" when API
  credentials are expired — NOT because the size is wrong. Always refresh
  credentials before placing a sell order to avoid a wasted failed attempt.

DEPENDENCIES:
  pip install py-clob-client
"""

import asyncio
import math
import time
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    OrderType,
    BalanceAllowanceParams,
    AssetType,
)
from py_clob_client.order_builder.constants import SELL


# ── Configuration — replace with your own values ───────────────────────────────

HOST        = "https://clob.polymarket.com"
CHAIN_ID    = 137                          # Polygon mainnet
PRIVATE_KEY = "0xYOUR_PRIVATE_KEY"
FUNDER      = "0xYOUR_PROXY_WALLET"       # The address you deposit USDC into

# For email/Magic wallet login use signature_type=1.
# For MetaMask/browser wallet use signature_type=0.
SIGNATURE_TYPE = 1

# How many cents below best bid to set the limit sell price.
# e.g. best_bid=0.30, slip=0.02 → sell_price=0.28
# Lower slip = better price but higher risk of not filling.
SLIP_CENTS = 2.0

# How long to wait for the sell order to fill before giving up (seconds).
ORDER_TIMEOUT_SEC = 15


# ── Step 1: Initialize the CLOB client ─────────────────────────────────────────

def make_client() -> ClobClient:
    """
    Create and authenticate a Polymarket CLOB client.
    Must be called before any trading operations.
    signature_type=1 is for email/Magic proxy wallets (most common for retail).
    """
    client = ClobClient(
        host           = HOST,
        chain_id       = CHAIN_ID,
        key            = PRIVATE_KEY,
        signature_type = SIGNATURE_TYPE,
        funder         = FUNDER,
    )
    # Derive L2 API credentials — required for authenticated endpoints (order placement).
    # These expire periodically; call again if you get 401 errors.
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    return client


# ── Step 2: Fetch actual on-chain share balance ─────────────────────────────────

async def fetch_actual_shares(
    client:      ClobClient,
    token_id:    str,
    fill_shares: float,
) -> float:
    """
    Fetch the real ERC-1155 token balance for a given token_id from the CLOB cache.

    WHY THIS IS NECESSARY:
      Polymarket deducts its trading fee from the shares themselves (not USDC).
      Your fill confirmation says e.g. 1.619994 shares, but your wallet only
      holds 1.599594 (raw balance = 1599594 / 1_000_000).
      If you try to sell 1.619994, the CLOB rejects with "not enough balance".
      Always use this function to get the real number before placing a SELL.

    HOW IT WORKS:
      1. Call update_balance_allowance to force the CLOB to sync your on-chain balance.
      2. Call get_balance_allowance to read the cached result.
      3. raw_balance / 1_000_000 = actual shares (Polymarket stores balances in
         micro-units, same as USDC's 6 decimal places).
      4. Retry up to 8 times (1s apart) because the cache can lag by 1-2 seconds
         right after a buy fills.

    Args:
        client:      Authenticated ClobClient.
        token_id:    The ERC-1155 token ID of the shares you hold (YES or NO token).
        fill_shares: The share count reported by the fill confirmation (used as
                     fallback if the cache stays empty after all retries).

    Returns:
        Actual on-chain share count (float). Falls back to fill_shares if cache fails.
    """
    loop = asyncio.get_event_loop()

    for attempt in range(8):
        try:
            # Force the CLOB to re-read your on-chain balance for this token.
            await loop.run_in_executor(
                None,
                lambda: client.update_balance_allowance(
                    BalanceAllowanceParams(
                        asset_type = AssetType.CONDITIONAL,
                        token_id   = token_id,
                    )
                )
            )
            # Read the now-updated balance from the CLOB cache.
            result = await loop.run_in_executor(
                None,
                lambda: client.get_balance_allowance(
                    BalanceAllowanceParams(
                        asset_type = AssetType.CONDITIONAL,
                        token_id   = token_id,
                    )
                )
            )
            raw = float(result.get("balance", 0) or 0)
            if raw > 0:
                actual = raw / 1_000_000
                print(
                    f"[fetch_actual_shares] confirmed {actual:.6f} shares "
                    f"(fill reported {fill_shares:.6f}, "
                    f"discrepancy={fill_shares - actual:+.6f})"
                )
                return actual
        except Exception as e:
            print(f"[fetch_actual_shares] attempt {attempt + 1}/8 failed: {e}")

        await asyncio.sleep(1)

    # Fallback: if cache never shows a non-zero balance, use the fill amount.
    # This risks a 400 error on the SELL if the discrepancy is large.
    print(
        f"[fetch_actual_shares] cache empty after 8 attempts — "
        f"falling back to fill_shares={fill_shares:.6f}"
    )
    return fill_shares


# ── Step 3: Place the SELL order ────────────────────────────────────────────────

async def sell_position(
    client:        ClobClient,
    token_id:      str,
    actual_shares: float,
) -> dict:
    """
    Place a limit SELL order on the Polymarket CLOB to exit a held position.

    This is the core function. It:
      1. Refreshes API credentials proactively (stale creds cause fake 400 errors).
      2. Reads the order book to find the current best bid.
      3. Sets the sell price slightly below best bid (by SLIP_CENTS) to ensure fill.
      4. Floors actual_shares to 2 decimal places (Polymarket minimum tick).
      5. Signs and posts the order as a GTC limit sell.
      6. Polls every 1.5s until filled or timeout.

    Args:
        client:        Authenticated ClobClient.
        token_id:      The ERC-1155 token ID of the shares to sell (YES or NO token).
                       Get this from the market's clobTokenIds[0] (YES) or [1] (NO).
        actual_shares: The REAL on-chain share count from fetch_actual_shares().
                       Do NOT use the fill-reported share count here — it's always
                       slightly higher than what you actually hold.

    Returns:
        dict with keys:
          success    (bool)  — True if the sell filled
          fill_price (float) — price per share the sell executed at
          fill_usdc  (float) — total USDC received
          error      (str)   — error description if success=False
    """
    loop = asyncio.get_event_loop()

    # ── STEP A: Proactive credential refresh ─────────────────────────────────
    #
    # CRITICAL: The CLOB returns a misleading 400 error "Size X lower than
    # minimum: Y" when API credentials are stale — NOT because the size is
    # actually wrong. Proof: the same size succeeds immediately after refreshing.
    # Refresh BEFORE placing the order to avoid wasting a failed attempt.
    try:
        creds = await loop.run_in_executor(
            None, lambda: client.create_or_derive_api_creds()
        )
        client.set_api_creds(creds)
        print("[sell_position] credentials refreshed.")
    except Exception as e:
        # Non-fatal — proceed with existing creds, retry will refresh again.
        print(f"[sell_position] credential refresh failed: {e} — proceeding anyway.")

    # ── STEP B: Read the order book to determine sell price ───────────────────
    #
    # We sell at: best_bid - SLIP_CENTS/100
    # Example: best_bid=0.30, slip=0.02 → sell_price=0.28
    # The slip ensures we undercut the best bid slightly so our order gets
    # matched immediately rather than sitting in the book.
    try:
        ob = await loop.run_in_executor(
            None, lambda: client.get_order_book(token_id)
        )
        bids = ob.bids if hasattr(ob, "bids") else ob.get("bids", [])
        if not bids:
            return {"success": False, "error": "no_bids_in_order_book"}

        def get_price(entry):
            return float(entry.price if hasattr(entry, "price") else entry["price"])

        best_bid   = max(get_price(b) for b in bids)
        sell_price = max(best_bid - (SLIP_CENTS / 100), 0.01)
        print(f"[sell_position] best_bid={best_bid:.4f} → sell_price={sell_price:.4f}")
    except Exception as e:
        return {"success": False, "error": f"order_book_fetch_failed: {e}"}

    # ── STEP C: Calculate shares to sell ─────────────────────────────────────
    #
    # Floor to 2 decimal places. Polymarket's minimum order increment is 0.01
    # shares. Never round UP — you can only sell what you actually hold.
    # e.g. actual_shares=1.599594 → shares_to_sell=1.59
    shares_to_sell = math.floor(actual_shares * 100) / 100
    print(
        f"[sell_position] selling {shares_to_sell} shares "
        f"(actual={actual_shares:.6f}, floored to 2dp)"
    )

    # ── STEP D: Build, sign, and post the SELL order ──────────────────────────
    #
    # OrderArgs:
    #   token_id     = the ERC-1155 token you want to sell
    #   price        = your limit price (you'll get this price or better)
    #   size         = number of shares to sell
    #   side         = SELL (from py_clob_client.order_builder.constants)
    #   fee_rate_bps = 0 (Polymarket currently charges 0% fees)
    #
    # OrderType.GTC = Good-Till-Cancelled — stays open until filled or cancelled.
    # You could also use OrderType.FOK (Fill-Or-Kill) for immediate-or-cancel.
    try:
        order_args   = OrderArgs(
            token_id     = token_id,
            price        = sell_price,
            size         = shares_to_sell,
            side         = SELL,
            fee_rate_bps = 0,
        )
        signed_order = client.create_order(order_args)
        response     = await loop.run_in_executor(
            None,
            lambda: client.post_order(signed_order, OrderType.GTC)
        )
        order_id = (
            response.get("orderID") or
            response.get("order_id") or
            response.get("id") or ""
        )
        if not order_id:
            return {"success": False, "error": f"order_rejected: {response}"}

        print(f"[sell_position] order submitted: {order_id}")
    except Exception as e:
        return {"success": False, "error": f"order_submission_exception: {e}"}

    # ── STEP E: Poll until filled or timeout ──────────────────────────────────
    #
    # The CLOB matches orders asynchronously. Poll get_order() every 1.5s.
    # Order states: OPEN → MATCHED/FILLED (success) or CANCELLED (failure).
    # Timeout after ORDER_TIMEOUT_SEC seconds and cancel the open order.
    deadline = time.time() + ORDER_TIMEOUT_SEC

    while time.time() < deadline:
        await asyncio.sleep(1.5)
        try:
            status       = await loop.run_in_executor(
                None, lambda: client.get_order(order_id)
            )
            state        = status.get("status", "").upper()
            size_matched = float(status.get("size_matched", 0) or 0)
            price        = float(status.get("price", sell_price) or sell_price)

            if state in ("MATCHED", "FILLED") or size_matched > 0:
                fill_usdc = size_matched * price if size_matched > 0 else shares_to_sell * sell_price
                print(
                    f"[sell_position] filled: {size_matched} shares @ {price:.4f} "
                    f"= ${fill_usdc:.4f}"
                )
                return {
                    "success":    True,
                    "fill_price": price,
                    "fill_usdc":  fill_usdc,
                    "error":      "",
                }

            if state in ("CANCELLED", "CANCELED"):
                return {"success": False, "error": "order_cancelled_externally"}

        except Exception as e:
            print(f"[sell_position] poll error: {e}")

    # Timed out — cancel the open order to avoid a dangling resting sell.
    print(f"[sell_position] timed out after {ORDER_TIMEOUT_SEC}s — cancelling order.")
    try:
        await loop.run_in_executor(None, lambda: client.cancel_order(order_id))
    except Exception as e:
        print(f"[sell_position] cancel failed: {e}")

    return {"success": False, "error": "order_timed_out"}


# ── Step 4: Full stop-loss exit flow ────────────────────────────────────────────

async def stop_loss_exit(
    client:       ClobClient,
    token_id:     str,
    fill_shares:  float,
    max_attempts: int = 3,
) -> dict:
    """
    Complete stop-loss exit: fetch real shares, then sell with retries.

    Call this when your stop-loss condition triggers (e.g. share price <= 0.30).

    The retry loop is important: if the first sell attempt fails (e.g. transient
    network error), credentials are refreshed again at the top of sell_position()
    on the next attempt. actual_shares is passed in on every retry — no re-fetch
    needed since the share count doesn't change between attempts.

    Args:
        client:       Authenticated ClobClient.
        token_id:     ERC-1155 token ID of shares to sell.
        fill_shares:  Fill-reported share count (used only as fallback in
                      fetch_actual_shares if the cache never populates).
        max_attempts: How many times to retry the sell before giving up.

    Returns:
        Same dict as sell_position(): {success, fill_price, fill_usdc, error}
    """
    # Fetch actual on-chain shares ONCE — reuse on every retry.
    # This is the real number the wallet holds, accounting for Polymarket's
    # fee deduction from the ERC-1155 tokens.
    actual_shares = await fetch_actual_shares(client, token_id, fill_shares)

    for attempt in range(1, max_attempts + 1):
        print(f"[stop_loss_exit] attempt {attempt}/{max_attempts}")
        result = await sell_position(client, token_id, actual_shares)

        if result["success"]:
            print(
                f"[stop_loss_exit] sold successfully | "
                f"received ${result['fill_usdc']:.4f} USDC"
            )
            return result

        print(f"[stop_loss_exit] attempt {attempt} failed: {result['error']}")
        if attempt < max_attempts:
            await asyncio.sleep(1)

    print(f"[stop_loss_exit] all {max_attempts} attempts failed — position stays open.")
    return {"success": False, "error": f"all_{max_attempts}_attempts_failed"}


# ── Example usage ───────────────────────────────────────────────────────────────

async def example():
    """
    Example: you bought 1.62 YES shares at 0.68 and now the price has dropped
    to 0.30 (your stop-loss threshold). Exit the position.

    Replace token_id and fill_shares with your actual values.
    token_id comes from the market's clobTokenIds[0] for YES, [1] for NO.
    fill_shares comes from the size_matched field of your original buy order.
    """
    client = make_client()

    token_id    = "YOUR_TOKEN_ID_HERE"  # YES or NO token ID from the market
    fill_shares = 1.619994              # what the buy fill said you received

    result = await stop_loss_exit(
        client      = client,
        token_id    = token_id,
        fill_shares = fill_shares,
    )

    if result["success"]:
        print(f"Exit complete. Received: ${result['fill_usdc']:.4f}")
    else:
        print(f"Exit failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(example())
