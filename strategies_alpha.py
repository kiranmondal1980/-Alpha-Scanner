"""
Alpha-Scanner — strategies_alpha.py  v1.0
============================================================
Signal & Research Hub — companion module to QuantBengal Pro.

ARCHITECTURAL RULES ENFORCED:
  - NO broker imports (no SmartApi, no pyotp, no order placement).
  - NO execution methods anywhere in this file. This module only
    ever returns a signal dict. It never calls place_order() and
    never imports broker_api.
  - Data source is yfinance-shaped OHLCV only (columns:
    open, high, low, close, volume) — same convention as
    QuantBengal's app.py / strategy.py so the transition feels native.

NAMING CONSISTENCY (matches existing QuantBengal codebase):
  - INDEX_CHOICE, CAPITAL are read the same way (env vars) where relevant.
  - ADX_MOMENTUM_THRESHOLD = 25.0 (identical to strategy.py RegimeDetector).
  - Regime labels reuse "MOMENTUM" / "NEUTRAL" strings so any downstream
    dashboard code that already branches on these strings keeps working.

STRATEGIES IMPLEMENTED:
  1. VWAP + OBV            — volume-weighted momentum confirmation
  2. Bollinger Squeeze      — volatility breakout (same squeeze logic
                               family as QuantBengal's existing
                               "Bollinger Squeeze Breakout" strategy,
                               reimplemented independently here)
  3. Z-Score Mean Reversion — statistical arbitrage on price deviation
                               from its rolling mean

Each strategy function returns a dict:
    {
        "signal":    "BUY_CALL" | "BUY_PUT" | "HOLD",
        "reason":    human-readable explanation,
        "price":     float,
        "indicator_snapshot": {...}   # for Signal Audit tab / logging
    }

No stop_loss / target / order fields are computed here — this hub is
informational only. (QuantBengal Pro's execution engine remains the
sole system that manages SL/TGT/orders.)
"""

import logging
import numpy as np
import pandas as pd

from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger("AlphaFactory")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS — kept consistent with QuantBengal's strategy.py RegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

ADX_MOMENTUM_THRESHOLD = 25.0   # identical threshold to strategy.py
ADX_PERIOD             = 14

REGIME_MOMENTUM = "MOMENTUM"
REGIME_NEUTRAL  = "NEUTRAL"

ALL_ALPHA_STRATEGIES = [
    "VWAP + OBV",
    "Bollinger Squeeze Breakout",
    "Z-Score Mean Reversion",
]


def _sf(val, default: float = 0.0) -> float:
    """Safe float cast — mirrors _safe_float() in strategy.py."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises a yfinance-style dataframe to lowercase OHLCV columns.
    Safe to call even if columns are already normalised.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    out.rename(columns={"adj close": "close", "adj_close": "close"}, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = out.get("close", np.nan)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out.dropna(subset=["close"], inplace=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME DETECTOR — ADX-based, mirrors strategy.py's RegimeDetector API
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Chooses which Alpha Factory strategy family is "in regime" right now.

    Mapping (informational only — no trades are placed anywhere in this file):
      ADX > 25            → MOMENTUM   → favour VWAP+OBV / Bollinger Squeeze
      ADX <= 25           → NEUTRAL    → favour Z-Score Mean Reversion
    """

    ADX_MOMENTUM_THRESHOLD = ADX_MOMENTUM_THRESHOLD
    ADX_PERIOD             = ADX_PERIOD

    REGIME_MOMENTUM = REGIME_MOMENTUM
    REGIME_NEUTRAL  = REGIME_NEUTRAL

    @classmethod
    def detect(cls, df: pd.DataFrame) -> tuple[str, float]:
        df = _ensure_columns(df)
        if df is None or len(df) < cls.ADX_PERIOD * 2:
            logger.warning("RegimeDetector: insufficient data — defaulting to NEUTRAL.")
            return cls.REGIME_NEUTRAL, 0.0

        try:
            adx_series = ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=cls.ADX_PERIOD
            ).adx()
            adx_value = _sf(adx_series.iloc[-1])
            regime = cls.REGIME_MOMENTUM if adx_value > cls.ADX_MOMENTUM_THRESHOLD else cls.REGIME_NEUTRAL
            return regime, adx_value
        except Exception as exc:
            logger.error(f"RegimeDetector error: {exc} — defaulting to NEUTRAL.")
            return cls.REGIME_NEUTRAL, 0.0

    @classmethod
    def recommended_strategy(cls, regime: str) -> str:
        """Suggests which Alpha Factory strategy fits the current regime."""
        if regime == cls.REGIME_MOMENTUM:
            return "VWAP + OBV"
        return "Z-Score Mean Reversion"


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY FACTORY
# ─────────────────────────────────────────────────────────────────────────────

class AlphaStrategyFactory:
    """
    Stateless calculator for all three Alpha Factory strategies.
    No broker object is accepted or required anywhere in this class —
    by design, it is architecturally incapable of placing an order.
    """

    # ── Shared indicator builder ────────────────────────────────────────────

    @staticmethod
    def build_indicators(candles: pd.DataFrame) -> pd.DataFrame:
        """
        candles: DataFrame with at least [open, high, low, close, volume],
                 (yfinance download shape, or QuantBengal candle list
                 converted to a DataFrame).
        """
        df = _ensure_columns(candles)
        if df.empty or len(df) < 25:
            return df

        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

        # VWAP (rolling/session-cumulative — cumulative is fine for intraday
        # 15m bars; for multi-day daily bars this behaves like a running VWAP)
        typical_price = (h + l + c) / 3
        cum_vol = v.cumsum().replace(0, np.nan)
        df["vwap"] = (typical_price * v).cumsum() / cum_vol
        df["vwap"] = df["vwap"].fillna(method="bfill").fillna(c)

        # OBV
        try:
            df["obv"] = OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
        except Exception:
            df["obv"] = 0.0
        df["obv_ema"] = EMAIndicator(close=df["obv"], window=10).ema_indicator()

        # Bollinger Bands (for squeeze detection)
        bb = BollingerBands(close=c, window=20, window_dev=2)
        df["bb_u"] = bb.bollinger_hband()
        df["bb_l"] = bb.bollinger_lband()
        df["bb_m"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_u"] - df["bb_l"]) / df["bb_m"].replace(0, np.nan)

        # ATR (context / volatility reference)
        try:
            df["atr"] = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
        except Exception:
            df["atr"] = c * 0.005

        # Z-Score of price vs its rolling mean (statistical arbitrage input)
        roll_mean = c.rolling(window=20).mean()
        roll_std  = c.rolling(window=20).std().replace(0, np.nan)
        df["zscore"] = (c - roll_mean) / roll_std
        df["zscore"] = df["zscore"].fillna(0.0)

        # ADX (regime context, logged alongside every signal)
        try:
            df["adx"] = ADXIndicator(high=h, low=l, close=c, window=ADX_PERIOD).adx()
        except Exception:
            df["adx"] = 20.0

        return df

    # ── Strategy 1: VWAP + OBV (Volume-Weighted Momentum) ──────────────────

    @staticmethod
    def signal_vwap_obv(df: pd.DataFrame) -> dict:
        if df is None or df.empty or len(df) < 20:
            return {"signal": "HOLD", "reason": "Insufficient data for VWAP+OBV.",
                    "price": 0.0, "indicator_snapshot": {}}

        lat = df.iloc[-1]
        p1  = df.iloc[-2]
        c        = _sf(lat["close"])
        vwap     = _sf(lat["vwap"])
        obv_now  = _sf(lat["obv"])
        obv_ema  = _sf(lat["obv_ema"])
        obv_prev = _sf(p1["obv"])
        obv_ema_p = _sf(p1["obv_ema"])

        above_vwap = c > vwap
        below_vwap = c < vwap
        obv_rising  = obv_now > obv_ema and obv_prev <= obv_ema_p
        obv_falling = obv_now < obv_ema and obv_prev >= obv_ema_p

        bull = above_vwap and obv_rising
        bear = below_vwap and obv_falling

        snapshot = {"close": round(c, 2), "vwap": round(vwap, 2),
                    "obv": round(obv_now, 0), "obv_ema": round(obv_ema, 0)}

        if bull:
            return {"signal": "BUY_CALL",
                    "reason": f"Price ₹{c:,.1f} above VWAP ₹{vwap:,.1f} | OBV crossed above its EMA (volume confirms up-move)",
                    "price": c, "indicator_snapshot": snapshot}
        if bear:
            return {"signal": "BUY_PUT",
                    "reason": f"Price ₹{c:,.1f} below VWAP ₹{vwap:,.1f} | OBV crossed below its EMA (volume confirms down-move)",
                    "price": c, "indicator_snapshot": snapshot}
        return {"signal": "HOLD",
                "reason": f"No VWAP/OBV confluence | Price {'above' if above_vwap else 'below'} VWAP, OBV not confirming",
                "price": c, "indicator_snapshot": snapshot}

    # ── Strategy 2: Bollinger Squeeze Breakout (Volatility) ─────────────────

    @staticmethod
    def signal_bollinger_squeeze(df: pd.DataFrame, squeeze_threshold: float = 0.04) -> dict:
        if df is None or df.empty or len(df) < 20:
            return {"signal": "HOLD", "reason": "Insufficient data for Bollinger Squeeze.",
                    "price": 0.0, "indicator_snapshot": {}}

        lat = df.iloc[-1]
        c   = _sf(lat["close"])
        bu  = _sf(lat["bb_u"])
        bl  = _sf(lat["bb_l"])
        bm  = _sf(lat["bb_m"])
        bw  = _sf(lat["bb_width"], squeeze_threshold * 2)

        is_squeeze = bw < squeeze_threshold
        breakout_up   = is_squeeze and c > bu
        breakout_down = is_squeeze and c < bl

        snapshot = {"close": round(c, 2), "bb_upper": round(bu, 2),
                    "bb_lower": round(bl, 2), "bb_width": round(bw, 4),
                    "squeeze_active": is_squeeze}

        if breakout_up:
            return {"signal": "BUY_CALL",
                    "reason": f"BB squeeze (width {bw:.3f}) resolved upward — breakout above ₹{bu:,.1f}",
                    "price": c, "indicator_snapshot": snapshot}
        if breakout_down:
            return {"signal": "BUY_PUT",
                    "reason": f"BB squeeze (width {bw:.3f}) resolved downward — breakdown below ₹{bl:,.1f}",
                    "price": c, "indicator_snapshot": snapshot}
        return {"signal": "HOLD",
                "reason": f"{'Coiling inside squeeze' if is_squeeze else 'No squeeze'} | width {bw:.3f} vs threshold {squeeze_threshold}",
                "price": c, "indicator_snapshot": snapshot}

    # ── Strategy 3: Z-Score Mean Reversion (Statistical Arbitrage) ─────────

    @staticmethod
    def signal_zscore_reversion(df: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5) -> dict:
        if df is None or df.empty or len(df) < 20:
            return {"signal": "HOLD", "reason": "Insufficient data for Z-Score Mean Reversion.",
                    "price": 0.0, "indicator_snapshot": {}}

        lat = df.iloc[-1]
        c   = _sf(lat["close"])
        z   = _sf(lat["zscore"])

        # Extreme negative z-score = price far BELOW its mean → expect reversion UP
        bull = z <= -abs(entry_z)
        # Extreme positive z-score = price far ABOVE its mean → expect reversion DOWN
        bear = z >= abs(entry_z)

        snapshot = {"close": round(c, 2), "zscore": round(z, 2), "entry_threshold": entry_z}

        if bull:
            return {"signal": "BUY_CALL",
                    "reason": f"Z-Score {z:.2f} ≤ -{entry_z} — price statistically oversold, expecting mean reversion up",
                    "price": c, "indicator_snapshot": snapshot}
        if bear:
            return {"signal": "BUY_PUT",
                    "reason": f"Z-Score {z:.2f} ≥ +{entry_z} — price statistically overbought, expecting mean reversion down",
                    "price": c, "indicator_snapshot": snapshot}
        return {"signal": "HOLD",
                "reason": f"Z-Score {z:.2f} within ±{entry_z} band — no statistical edge currently",
                "price": c, "indicator_snapshot": snapshot}

    # ── Dispatcher ───────────────────────────────────────────────────────────

    @classmethod
    def evaluate(cls, strategy_name: str, candles: pd.DataFrame) -> dict:
        """
        Single entry point used by scanner_engine.py.
        Builds indicators once, then dispatches to the requested strategy.
        Always returns a dict — never raises, to keep the async scan loop alive
        even if one symbol's data is malformed.
        """
        try:
            df = cls.build_indicators(candles)
            if df.empty:
                return {"signal": "HOLD", "reason": "No usable candle data.",
                        "price": 0.0, "indicator_snapshot": {}, "adx": 0.0, "regime": REGIME_NEUTRAL}

            regime, adx_value = RegimeDetector.detect(df)

            if strategy_name == "VWAP + OBV":
                result = cls.signal_vwap_obv(df)
            elif strategy_name == "Bollinger Squeeze Breakout":
                result = cls.signal_bollinger_squeeze(df)
            elif strategy_name == "Z-Score Mean Reversion":
                result = cls.signal_zscore_reversion(df)
            else:
                logger.warning(f"Unknown alpha strategy '{strategy_name}' — defaulting to VWAP + OBV")
                result = cls.signal_vwap_obv(df)

            result["adx"]    = round(adx_value, 1)
            result["regime"] = regime
            result["strategy"] = strategy_name
            return result

        except Exception as exc:
            logger.error(f"AlphaStrategyFactory.evaluate failed for '{strategy_name}': {exc}")
            return {"signal": "HOLD", "reason": f"Evaluation error: {exc}",
                    "price": 0.0, "indicator_snapshot": {}, "adx": 0.0,
                    "regime": REGIME_NEUTRAL, "strategy": strategy_name}

    @classmethod
    def evaluate_all(cls, candles: pd.DataFrame) -> dict:
        """
        Runs all 3 strategies against the same candle set — used to power
        the "Institutional Sentiment Gauge" equivalent on alpha_dashboard.py.
        Returns {strategy_name: result_dict, ...}
        """
        df = cls.build_indicators(candles)
        out = {}
        for strat in ALL_ALPHA_STRATEGIES:
            out[strat] = cls.evaluate(strat, candles if df.empty else df)
        return out
