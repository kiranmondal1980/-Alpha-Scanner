"""
Alpha-Scanner — scanner_engine.py  v1.0
============================================================
Signal & Research Hub — companion module to QuantBengal Pro.

ARCHITECTURAL RULES ENFORCED:
  - NO broker imports (no SmartApi, no pyotp, no place_order()).
  - Data source is yfinance ONLY. No Angel One session anywhere.
  - This file NEVER executes trades. It only reads market data,
    evaluates strategies_alpha.py, writes signals to Supabase, and
    optionally pings Telegram. If you're looking for the execution
    engine, that's QuantBengal Pro's main.py — a separate system.

NAMING CONSISTENCY (matches existing QuantBengal codebase):
  - INDEX_CHOICE convention reused per-asset (each asset is scanned,
    not just one — see ASSET_UNIVERSE below).
  - CAPITAL is read the same way (env var) for context/logging only;
    this engine has no risk manager because it never sizes a trade.
  - IST timezone handling identical to main.py (_now_ist()).
  - Supabase helper naming mirrors main.py's _db_* functions.

RUNS ON: GitHub Actions (scheduled) or a free-tier always-on poller.
         No AWS/Oracle VM required — this whole module is designed
         to survive on ephemeral, ~15-min-interval free compute.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from strategies_alpha import AlphaStrategyFactory, RegimeDetector, ALL_ALPHA_STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AlphaScanner")
IST = pytz.timezone("Asia/Kolkata")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CAPITAL = float(os.environ.get("CAPITAL", "25000"))   # context/logging only

# Asset universe: display name -> yfinance ticker
ASSET_UNIVERSE = {
    "BANKNIFTY":    "^NSEBANK",
    "NIFTY":        "^NSEI",
    "SENSEX":       "^BSESN",
    "CRUDE OIL":    "CL=F",
    "NATURAL GAS":  "NG=F",
    "GOLD":         "GC=F",
    "SILVER":       "SI=F",
}

SCAN_INTERVAL_SECONDS   = 15 * 60     # 15 minutes
YF_INTERVAL             = "15m"
YF_PERIOD               = "5d"        # enough bars for ADX/BB/Z-score warm-up
MIN_ALERT_GAP_SECONDS   = 15 * 60     # anti-spam: min gap between repeat alerts

# ── Supabase (optional — engine runs fine without it, just skips persistence) ─
try:
    from supabase import create_client
    _SB_URL = os.environ.get("SUPABASE_URL", "")
    _SB_KEY = os.environ.get("SUPABASE_KEY", "")
    supabase = create_client(_SB_URL, _SB_KEY) if _SB_URL and _SB_KEY else None
    if supabase:
        logger.info("✅ Supabase client connected.")
    else:
        logger.warning("⚠️ Supabase not configured — signals will not be persisted.")
except Exception as exc:
    supabase = None
    logger.warning(f"Supabase init failed: {exc} — running without DB.")

# ── Telegram (optional) ────────────────────────────────────────────────────────
tg_bot = None
TELEGRAM_CHAT_ID: Optional[int] = None
try:
    from telegram import Bot as TelegramBot

    _raw_token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    _raw_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    _clean_chat_id = "".join(c for i, c in enumerate(_raw_chat_id) if c.isdigit() or (c == "-" and i == 0))

    if _raw_token and _clean_chat_id:
        TELEGRAM_CHAT_ID = int(_clean_chat_id)
        tg_bot = TelegramBot(token=_raw_token)
        logger.info(f"✅ Telegram bot initialised. Target ID: {TELEGRAM_CHAT_ID}")
    else:
        logger.warning("⚠️ Telegram not configured — alerts disabled.")
except Exception as exc:
    tg_bot = None
    logger.warning(f"Telegram setup error: {exc}")


def _now_ist() -> datetime:
    return datetime.now(IST)


async def _send_telegram(message: str):
    if not tg_bot or TELEGRAM_CHAT_ID is None:
        return
    try:
        await tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="HTML")
    except Exception as exc:
        logger.warning(f"Telegram send failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL STATE MEMORY  (Anti-Spam)
# ══════════════════════════════════════════════════════════════════════════════

class SignalStateMemory:
    """
    Tracks the last alerted (signal, timestamp) per asset so the engine only
    pings Telegram / writes an "alerted" row when:
      (a) the consensus signal changed since last time, OR
      (b) more than MIN_ALERT_GAP_SECONDS has elapsed since the last alert
          for that asset (even if the signal is unchanged — a heartbeat).

    In-memory by default (dict). If Supabase is configured, the engine also
    persists every evaluation (regardless of alert-worthiness) to the
    `signals` table — this class only governs *Telegram noise*, not DB writes.
    """

    def __init__(self):
        self._last: dict[str, dict] = {}   # asset -> {"signal": str, "ts": datetime}

    def should_alert(self, asset: str, new_signal: str) -> bool:
        prev = self._last.get(asset)
        now = _now_ist()

        if prev is None:
            self._last[asset] = {"signal": new_signal, "ts": now}
            return new_signal != "HOLD"   # first-ever eval: alert only if actionable

        changed = prev["signal"] != new_signal
        elapsed = (now - prev["ts"]).total_seconds()
        stale   = elapsed > MIN_ALERT_GAP_SECONDS

        if changed or (stale and new_signal != "HOLD"):
            self._last[asset] = {"signal": new_signal, "ts": now}
            return True

        return False

    def snapshot(self) -> dict:
        return {k: {"signal": v["signal"], "ts": v["ts"].isoformat()} for k, v in self._last.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCH — yfinance with tenacity exponential backoff
# ══════════════════════════════════════════════════════════════════════════════

class YFinanceRateLimitError(Exception):
    """Raised when Yahoo Finance appears to be blocking/throttling requests."""


def _looks_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate", "429", "too many requests", "blocked", "timeout", "timed out"))


@retry(
    retry=retry_if_exception_type(YFinanceRateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _download_sync(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Blocking yfinance call — wrapped in tenacity retry and run via
    asyncio.to_thread() by the caller so the event loop never blocks.
    """
    try:
        df = yf.download(
            tickers=ticker, period=period, interval=interval,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            # Not necessarily a rate limit — could be a genuinely closed
            # market / bad ticker. We do NOT retry on empty data, only on
            # exceptions that look like throttling. Caller handles empty df.
            return pd.DataFrame()
        return df
    except Exception as exc:
        if _looks_rate_limited(exc):
            raise YFinanceRateLimitError(str(exc)) from exc
        # Non rate-limit errors bubble up immediately (no point retrying)
        raise


async def fetch_candles(ticker: str) -> pd.DataFrame:
    """Async wrapper — runs the blocking, retry-wrapped download in a thread."""
    try:
        return await asyncio.to_thread(_download_sync, ticker, YF_PERIOD, YF_INTERVAL)
    except Exception as exc:
        logger.error(f"fetch_candles: giving up on {ticker} after retries: {exc}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE PERSISTENCE (fire-and-forget via asyncio.to_thread)
# ══════════════════════════════════════════════════════════════════════════════

def _insert_signal_row_sync(row: dict):
    if not supabase:
        return
    try:
        supabase.table("signals").insert(row).execute()
    except Exception as exc:
        logger.error(f"Supabase insert (signals) failed: {exc}")


async def _log_signal_to_db(asset: str, consensus: dict, per_strategy: dict, alerted: bool):
    """
    Persists one row per scan cycle per asset to the `signals` table.
    Fire-and-forget: scheduled as a task so the scan loop never waits on DB I/O.
    """
    if not supabase:
        return

    row = {
        "symbol":    asset,
        "strategy":  "ALPHA_CONSENSUS",
        "signal":    consensus["signal"],
        "reason":    consensus["reason"],
        "price":     consensus["price"],
        "adx":       consensus.get("adx", 0.0),
        "rsi":       0.0,   # not computed by AlphaStrategyFactory; reserved column
        "regime":    consensus.get("regime", "NEUTRAL"),
        "meta": {
            "bullish_count": consensus["bullish_count"],
            "bearish_count": consensus["bearish_count"],
            "per_strategy": {
                k: {"signal": v["signal"], "reason": v["reason"]}
                for k, v in per_strategy.items()
            },
            "alerted": alerted,
        },
    }
    asyncio.create_task(asyncio.to_thread(_insert_signal_row_sync, row))


# ══════════════════════════════════════════════════════════════════════════════
#  CONSENSUS SCORER
# ══════════════════════════════════════════════════════════════════════════════

def _build_consensus(asset: str, per_strategy: dict) -> dict:
    """
    Combines the 3 Alpha Factory strategy outputs into a single consensus
    verdict for the asset. Majority rule across 3 strategies:
      >=2 BUY_CALL  -> consensus BUY_CALL
      >=2 BUY_PUT   -> consensus BUY_PUT
      otherwise     -> HOLD
    """
    bulls = sum(1 for v in per_strategy.values() if v.get("signal") == "BUY_CALL")
    bears = sum(1 for v in per_strategy.values() if v.get("signal") == "BUY_PUT")

    # Use the ADX/regime/price from whichever strategy ran (they all share
    # the same underlying candle set, so these fields are consistent).
    any_result = next(iter(per_strategy.values()), {})
    price  = any_result.get("price", 0.0)
    adx    = any_result.get("adx", 0.0)
    regime = any_result.get("regime", "NEUTRAL")

    if bulls >= 2:
        signal = "BUY_CALL"
        reason = f"{bulls}/3 Alpha strategies bullish on {asset} (VWAP+OBV / Squeeze / Z-Score consensus)"
    elif bears >= 2:
        signal = "BUY_PUT"
        reason = f"{bears}/3 Alpha strategies bearish on {asset} (VWAP+OBV / Squeeze / Z-Score consensus)"
    else:
        signal = "HOLD"
        reason = f"No 2/3 consensus on {asset} | Bulls:{bulls} Bears:{bears} Neutral:{3 - bulls - bears}"

    return {
        "signal": signal,
        "reason": reason,
        "price": price,
        "adx": adx,
        "regime": regime,
        "bullish_count": bulls,
        "bearish_count": bears,
    }


def _fmt_alert(asset: str, consensus: dict) -> str:
    icon = "🟢" if consensus["signal"] == "BUY_CALL" else "🔴" if consensus["signal"] == "BUY_PUT" else "⚪"
    return (
        f"{icon} <b>ALPHA SIGNAL — {asset}</b>\n"
        f"Signal: {consensus['signal']}\n"
        f"Price: ₹{consensus['price']:,.2f}\n"
        f"Regime: {consensus['regime']} (ADX {consensus['adx']:.1f})\n"
        f"Consensus: {consensus['bullish_count']} bull / {consensus['bearish_count']} bear of 3\n"
        f"{consensus['reason']}\n"
        f"Time: {_now_ist().strftime('%d-%b %H:%M IST')}\n"
        f"<i>Informational only — no orders are placed by this engine.</i>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PER-ASSET EVALUATION  (fully isolated try/except — one bad asset can't
#  crash the scan cycle)
# ══════════════════════════════════════════════════════════════════════════════

async def evaluate_asset(asset: str, ticker: str, state: SignalStateMemory) -> Optional[dict]:
    try:
        df = await fetch_candles(ticker)
        if df is None or df.empty or len(df) < 25:
            logger.warning(f"⚠️ {asset} ({ticker}): no usable candle data this cycle — skipping.")
            return None

        per_strategy = AlphaStrategyFactory.evaluate_all(df)
        consensus = _build_consensus(asset, per_strategy)

        alerted = state.should_alert(asset, consensus["signal"])
        if alerted:
            logger.info(f"🔔 {asset}: {consensus['signal']} — alert-worthy (state changed or heartbeat).")
            await _send_telegram(_fmt_alert(asset, consensus))
        else:
            logger.info(f"…  {asset}: {consensus['signal']} (no change — suppressing duplicate alert).")

        await _log_signal_to_db(asset, consensus, per_strategy, alerted)

        return {"asset": asset, "consensus": consensus, "per_strategy": per_strategy, "alerted": alerted}

    except Exception as exc:
        # Deliberately broad: this asset's failure must never bubble up
        # and stop the rest of the universe from being scanned.
        logger.error(f"❌ evaluate_asset({asset}) failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CORE SCAN CYCLE
# ══════════════════════════════════════════════════════════════════════════════

async def run_scan_cycle(state: SignalStateMemory) -> list:
    """
    Runs one full pass over ASSET_UNIVERSE. Assets are evaluated concurrently
    (asyncio.gather) but each is wrapped in its own try/except inside
    evaluate_asset(), so partial failures never abort the whole cycle.
    """
    logger.info(f"🔍 Alpha-Scanner cycle starting | {len(ASSET_UNIVERSE)} assets | {_now_ist().strftime('%d-%b %H:%M:%S IST')}")

    tasks = [evaluate_asset(asset, ticker, state) for asset, ticker in ASSET_UNIVERSE.items()]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    results = [r for r in results if r is not None]

    logger.info(f"✅ Cycle complete | {len(results)}/{len(ASSET_UNIVERSE)} assets evaluated successfully.")
    return results


async def run_forever():
    """
    Continuous poller (use this if deploying on an always-on free-tier
    worker). If instead you're triggering this via GitHub Actions cron
    every 15 minutes, call run_scan_cycle() once via run_once() below
    and let the workflow's schedule handle the interval.
    """
    state = SignalStateMemory()
    logger.info("=" * 65)
    logger.info("  ALPHA-SCANNER ENGINE v1.0 — yfinance-only, no broker")
    logger.info(f"  Universe: {list(ASSET_UNIVERSE.keys())}")
    logger.info(f"  Interval: {SCAN_INTERVAL_SECONDS}s | Capital context: ₹{CAPITAL:,.0f}")
    logger.info("=" * 65)

    while True:
        try:
            await run_scan_cycle(state)
        except Exception as exc:
            # Belt-and-braces: even a bug in the cycle orchestration itself
            # must not kill the poller.
            logger.error(f"run_scan_cycle crashed unexpectedly: {exc}")
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def run_once():
    """
    Single-pass entry point — ideal for a GitHub Actions cron job
    (e.g. every 15 minutes) where the workflow itself manages scheduling
    and each run is a fresh process (so SignalStateMemory starts empty
    each time; Supabase's `signals` table is the durable source of truth
    for "did this signal already fire recently" across process restarts —
    see NOTE below).
    """
    state = SignalStateMemory()
    return await run_scan_cycle(state)


# ══════════════════════════════════════════════════════════════════════════════
#  NOTE ON STATE ACROSS EPHEMERAL RUNS (GitHub Actions)
# ══════════════════════════════════════════════════════════════════════════════
# SignalStateMemory is in-process memory. On GitHub Actions, each scheduled
# run is a brand-new process/container, so in-memory state resets every time.
# Two ways to handle this depending on how you deploy:
#
#   1. Always-on poller (Streamlit Cloud background thread, a Railway/Render
#      free worker, etc.) — run_forever() keeps state naturally between
#      cycles. This is the simplest option and needs no extra code.
#
#   2. GitHub Actions cron (truly ephemeral) — before calling run_once(),
#      seed SignalStateMemory from the last row per asset in Supabase's
#      `signals` table (order by created_at desc, limit 1 per symbol) so
#      duplicate-alert suppression survives across process restarts. This
#      hook point is intentionally left simple to wire up once you decide
#      final deployment target; ask and I'll add a `_seed_state_from_db()`
#      function here.
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("Alpha-Scanner stopped by user.")
