"""
Alpha-Scanner — alpha_dashboard.py  v1.0
============================================================
Signal & Research Hub — companion module to QuantBengal Pro.

ARCHITECTURAL RULES ENFORCED:
  - READ-ONLY. This file contains zero broker imports, zero
    place_order() calls, and zero session-state execution controls.
    It only ever SELECTs from Supabase's `signals` table (written by
    scanner_engine.py) and renders it.
  - No "Connect to Angel One" panel exists anywhere in this file —
    unlike app.py, there is nothing here that could accidentally be
    wired up to place a live order.
  - Uses st.cache_data so repeated reruns (auto-refresh, tab switches)
    don't hammer Supabase and the UI never freezes/flattens.

NAMING CONSISTENCY (matches existing QuantBengal codebase):
  - Same IST timezone handling pattern as app.py (_utc_iso_to_ist_str
    helper — this dashboard reuses that exact pattern to avoid the
    5h30m timestamp bug you already fixed once in app.py).
  - Reuses the "Institutional Sentiment Gauge" visual language and the
    "Signal Audit" tab concept from app.py, adapted to a multi-asset,
    multi-strategy consensus view instead of a single-index one.
  - ASSET_UNIVERSE list matches scanner_engine.py exactly, so filters
    in the sidebar line up 1:1 with what the scanner actually writes.
"""

import os
import json
from datetime import datetime, timedelta

import pandas as pd
import pytz
import streamlit as st

# ── Supabase (dashboard is inert without it — no JSON fallback, since ────────
#    this hub has no local execution engine writing a JSON ledger) ───────────
try:
    from supabase import create_client
    _SB_URL = os.environ.get("SUPABASE_URL", "")
    _SB_KEY = os.environ.get("SUPABASE_KEY", "")
    _supabase = create_client(_SB_URL, _SB_KEY) if _SB_URL and _SB_KEY else None
except Exception:
    _supabase = None

def _sb_ok() -> bool:
    return _supabase is not None

IST = pytz.timezone("Asia/Kolkata")

# Must match scanner_engine.py ASSET_UNIVERSE keys exactly
ASSET_UNIVERSE = [
    "BANKNIFTY", "NIFTY", "SENSEX",
    "CRUDE OIL", "NATURAL GAS", "GOLD", "SILVER",
]

ALPHA_STRATEGIES = ["VWAP + OBV", "Bollinger Squeeze Breakout", "Z-Score Mean Reversion"]

REFRESH_TTL_SECONDS = 60   # st.cache_data TTL — balances freshness vs. Supabase load


# ── Timestamp helper — identical pattern to app.py's _utc_iso_to_ist_str ────
def _utc_iso_to_ist_str(utc_iso: str, fmt: str = "%d-%b %H:%M") -> str:
    if not utc_iso:
        return "—"
    try:
        ts = pd.to_datetime(utc_iso, utc=True).tz_convert("Asia/Kolkata")
        return ts.strftime(fmt)
    except Exception:
        return str(utc_iso)[:16].replace("T", " ")


def now_ist() -> datetime:
    return datetime.now(IST)


def sf(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED SUPABASE READS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner=False)
def fetch_latest_signals(limit: int = 500) -> pd.DataFrame:
    """
    Pulls the most recent `signals` rows written by scanner_engine.py.
    Read-only SELECT — no writes ever happen from this dashboard.
    """
    if not _sb_ok():
        return pd.DataFrame()
    try:
        resp = (
            _supabase.table("signals")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["created_at_ist"] = df["created_at"].apply(_utc_iso_to_ist_str)
        return df
    except Exception as exc:
        st.session_state["_last_fetch_error"] = str(exc)
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner=False)
def fetch_latest_per_asset() -> pd.DataFrame:
    """
    Returns the single most recent row per asset (from ALPHA_CONSENSUS
    strategy rows only) — this powers the Sentiment Gauge / overview cards.
    """
    df = fetch_latest_signals(limit=1000)
    if df.empty:
        return df
    consensus_df = df[df.get("strategy", "") == "ALPHA_CONSENSUS"].copy()
    if consensus_df.empty:
        return consensus_df
    consensus_df.sort_values("created_at", ascending=False, inplace=True)
    latest = consensus_df.drop_duplicates(subset=["symbol"], keep="first")
    return latest


def _extract_meta(row) -> dict:
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + LIGHT CSS  (visually consistent with QuantBengal Pro palette)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Alpha-Scanner — Signal & Research Hub",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --royal:#1e3a8a; --royal-l:#e8eeff;
  --emerald:#047857; --emerald-l:#ecfdf5;
  --crimson:#dc2626; --crimson-l:#fff1f1;
  --muted:#64748b; --border:#dde3ee; --bg:#f0f4f8;
  --mono:'JetBrains Mono',monospace; --sans:'DM Sans',sans-serif;
}
.stApp{background:var(--bg)!important;}
.hero{background:linear-gradient(135deg,var(--royal),#1d4ed8 60%,#0f766e);
  border-radius:10px;padding:20px 22px;margin-bottom:16px;}
.hero h2{color:#fff;font-family:var(--mono);font-size:17px;margin:0 0 6px}
.hero p{color:rgba(255,255,255,.8);font-family:var(--sans);font-size:13px;margin:0;line-height:1.6}
.badge-ro{display:inline-block;background:rgba(255,255,255,.15);color:#fff;
  border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:3px 10px;
  font-family:var(--mono);font-size:9px;letter-spacing:1px;margin-top:8px}
.mcard{background:#fff;border:1px solid var(--border);border-radius:10px;
  padding:14px 16px;box-shadow:0 1px 6px rgba(15,23,42,.08);}
.mcard-label{font-family:var(--mono);font-size:9px;color:var(--muted);
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px}
.mcard-value{font-family:var(--mono);font-size:18px;font-weight:700;color:#0f172a}
.v-emerald{color:var(--emerald)!important}.v-crimson{color:var(--crimson)!important}
.gauge-track{position:relative;height:20px;border-radius:6px;background:#e2e8f0;overflow:hidden;}
.gauge-fill-bull{position:absolute;left:0;top:0;height:100%;
  background:linear-gradient(90deg,#047857,#10b981);}
.gauge-fill-bear{position:absolute;right:0;top:0;height:100%;
  background:linear-gradient(90deg,#ef4444,#dc2626);}
.sec-hdr{font-family:var(--mono);font-size:9px;color:var(--muted);
  text-transform:uppercase;letter-spacing:2px;padding:14px 0 8px;border-top:1px solid var(--border);}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
  <h2>🔎 Alpha-Scanner — Signal & Research Hub</h2>
  <p>Multi-asset, multi-strategy consensus scanner. VWAP+OBV · Bollinger Squeeze · Z-Score Mean Reversion.
     Runs on yfinance data only — no broker connection, no order execution.</p>
  <span class="badge-ro">● READ-ONLY DASHBOARD</span>
</div>
""", unsafe_allow_html=True)

if not _sb_ok():
    st.error(
        "Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY environment "
        "variables — this dashboard has no local fallback because there is no "
        "execution engine writing a JSON ledger for this hub."
    )
    st.stop()

# Manual refresh (dashboard is cache-backed, so this is the "force update" button)
col_r1, col_r2 = st.columns([1, 5])
with col_r1:
    if st.button("🔄 Refresh Now", use_container_width=True):
        fetch_latest_signals.clear()
        fetch_latest_per_asset.clear()
        st.rerun()
with col_r2:
    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:var(--muted);padding-top:8px">'
        f'Auto-cached {REFRESH_TTL_SECONDS}s · Last page load: {now_ist().strftime("%d-%b %H:%M:%S IST")}</div>',
        unsafe_allow_html=True,
    )

latest_per_asset = fetch_latest_per_asset()
all_signals = fetch_latest_signals()

if st.session_state.get("_last_fetch_error"):
    st.warning(f"Last Supabase read had an issue: {st.session_state['_last_fetch_error']}")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_gauge, tab_assets, tab_audit, tab_about = st.tabs([
    "📊 SENTIMENT GAUGE", "🗂️ ASSET OVERVIEW", "📡 SIGNAL AUDIT", "ℹ️ ABOUT"
])

# ── TAB 1 — Institutional Sentiment Gauge (multi-asset consensus) ──────────
with tab_gauge:
    if latest_per_asset.empty:
        st.info("No consensus signals logged yet. The scanner writes here every ~15 minutes once running.")
    else:
        for _, row in latest_per_asset.iterrows():
            asset = row.get("symbol", "—")
            signal = row.get("signal", "HOLD")
            meta = _extract_meta(row)
            bulls = meta.get("bullish_count", 0)
            bears = meta.get("bearish_count", 0)
            total = 3
            bull_pct = bulls / total * 100
            bear_pct = bears / total * 100

            verdict_color = "v-emerald" if signal == "BUY_CALL" else "v-crimson" if signal == "BUY_PUT" else ""
            verdict_icon = "🟢" if signal == "BUY_CALL" else "🔴" if signal == "BUY_PUT" else "⚪"

            st.markdown(f"""
            <div class="mcard" style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <div class="mcard-label" style="margin:0">{asset}</div>
                <div style="font-family:var(--mono);font-size:12px;font-weight:700" class="{verdict_color}">
                  {verdict_icon} {signal}
                </div>
              </div>
              <div class="gauge-track">
                <div class="gauge-fill-bull" style="width:{bull_pct}%"></div>
                <div class="gauge-fill-bear" style="width:{bear_pct}%"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:6px;
                          font-family:var(--mono);font-size:10px;color:var(--muted)">
                <span>▲ {bulls}/3 bullish</span>
                <span>Price ₹{sf(row.get('price')):,.2f} · ADX {sf(row.get('adx')):.1f} · {row.get('regime','')}</span>
                <span>{bears}/3 bearish ▼</span>
              </div>
              <div style="font-family:var(--sans);font-size:11px;color:var(--muted);margin-top:6px">
                {row.get('reason','')} &nbsp;·&nbsp; {row.get('created_at_ist','')}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:8px">'
            'Consensus = majority vote across VWAP+OBV, Bollinger Squeeze Breakout, and Z-Score Mean Reversion. '
            'Informational only — no orders are placed by this hub.</div>',
            unsafe_allow_html=True,
        )

# ── TAB 2 — Asset Overview (per-strategy breakdown for a selected asset) ───
with tab_assets:
    if all_signals.empty:
        st.info("No signals logged yet.")
    else:
        available_assets = sorted(all_signals["symbol"].dropna().unique().tolist()) or ASSET_UNIVERSE
        sel_asset = st.selectbox("Select asset", available_assets, key="asset_sel")

        asset_rows = all_signals[all_signals["symbol"] == sel_asset].copy()
        consensus_rows = asset_rows[asset_rows.get("strategy", "") == "ALPHA_CONSENSUS"]

        if consensus_rows.empty:
            st.info(f"No consensus data yet for {sel_asset}.")
        else:
            latest_row = consensus_rows.sort_values("created_at", ascending=False).iloc[0]
            meta = _extract_meta(latest_row)
            per_strategy = meta.get("per_strategy", {})

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="mcard"><div class="mcard-label">PRICE</div>'
                            f'<div class="mcard-value">₹{sf(latest_row.get("price")):,.2f}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="mcard"><div class="mcard-label">REGIME (ADX)</div>'
                            f'<div class="mcard-value">{latest_row.get("regime","")} ({sf(latest_row.get("adx")):.1f})</div></div>',
                            unsafe_allow_html=True)
            with c3:
                sig = latest_row.get("signal", "HOLD")
                sig_cls = "v-emerald" if sig == "BUY_CALL" else "v-crimson" if sig == "BUY_PUT" else ""
                st.markdown(f'<div class="mcard"><div class="mcard-label">CONSENSUS</div>'
                            f'<div class="mcard-value {sig_cls}">{sig}</div></div>',
                            unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="mcard"><div class="mcard-label">LAST UPDATED</div>'
                            f'<div class="mcard-value" style="font-size:13px">{latest_row.get("created_at_ist","")}</div></div>',
                            unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr">PER-STRATEGY BREAKDOWN</div>', unsafe_allow_html=True)
            if per_strategy:
                for strat_name in ALPHA_STRATEGIES:
                    detail = per_strategy.get(strat_name, {})
                    s = detail.get("signal", "HOLD")
                    r = detail.get("reason", "—")
                    s_cls = "v-emerald" if s == "BUY_CALL" else "v-crimson" if s == "BUY_PUT" else ""
                    st.markdown(f"""
                    <div class="mcard" style="margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between">
                        <span style="font-family:var(--mono);font-size:11px;font-weight:700">{strat_name}</span>
                        <span style="font-family:var(--mono);font-size:11px;font-weight:700" class="{s_cls}">{s}</span>
                      </div>
                      <div style="font-family:var(--sans);font-size:12px;color:var(--muted);margin-top:4px">{r}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Per-strategy detail not available for this row (older log format).")

            st.markdown('<div class="sec-hdr">RECENT HISTORY — THIS ASSET</div>', unsafe_allow_html=True)
            hist_cols = ["created_at_ist", "signal", "price", "adx", "regime", "reason"]
            hist_show = [c for c in hist_cols if c in consensus_rows.columns]
            st.dataframe(
                consensus_rows.sort_values("created_at", ascending=False)[hist_show].head(30),
                use_container_width=True, hide_index=True,
            )

# ── TAB 3 — Signal Audit (raw log, all assets, all strategies) ─────────────
with tab_audit:
    st.markdown('<div class="sec-hdr" style="border-top:none;padding-top:0">RAW SIGNAL LOG — LAST 500 ROWS</div>',
                unsafe_allow_html=True)

    if all_signals.empty:
        st.info("No signals logged yet. The Alpha-Scanner engine writes here every ~15 minutes once running.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            asset_filter = st.multiselect("Filter by asset", sorted(all_signals["symbol"].dropna().unique().tolist()))
        with f2:
            signal_filter = st.multiselect("Filter by signal", ["BUY_CALL", "BUY_PUT", "HOLD"])
        with f3:
            strategy_filter = st.multiselect("Filter by strategy", sorted(all_signals["strategy"].dropna().unique().tolist()))

        display_df = all_signals.copy()
        if asset_filter:
            display_df = display_df[display_df["symbol"].isin(asset_filter)]
        if signal_filter:
            display_df = display_df[display_df["signal"].isin(signal_filter)]
        if strategy_filter:
            display_df = display_df[display_df["strategy"].isin(strategy_filter)]

        show_cols = [c for c in ["created_at_ist", "symbol", "strategy", "signal", "price", "adx", "regime", "reason"]
                     if c in display_df.columns]
        st.dataframe(
            display_df.sort_values("created_at", ascending=False)[show_cols],
            use_container_width=True, hide_index=True, height=460,
        )
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:6px">'
            f'{len(display_df)} rows shown (of {len(all_signals)} loaded)</div>',
            unsafe_allow_html=True,
        )

# ── TAB 4 — About ───────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="mcard">
      <div style="font-family:var(--mono);font-size:13px;font-weight:700;color:var(--royal);margin-bottom:10px">
        What Alpha-Scanner is — and isn't
      </div>
      <div style="font-family:var(--sans);font-size:13px;color:#334155;line-height:1.7">
        <p><strong>Alpha-Scanner</strong> is an independent, read-only Signal & Research Hub. It scans
        BankNifty, Nifty, Sensex, Crude Oil, Natural Gas, Gold, and Silver every ~15 minutes using
        <code>yfinance</code> data, evaluates three statistical/technical strategies
        (VWAP+OBV, Bollinger Squeeze Breakout, Z-Score Mean Reversion), and logs a consensus signal
        to Supabase.</p>
        <p><strong>It does not hold broker credentials, does not log in to any exchange session,
        and does not place, modify, or cancel any order.</strong> This dashboard only ever reads
        from Supabase's <code>signals</code> table.</p>
        <p>QuantBengal Pro (the separate execution engine) remains the only system in this
        environment authorised to trade.</p>
      </div>
    </div>
    <div class="mcard" style="margin-top:12px;border-color:rgba(220,38,38,.3);background:var(--crimson-l)">
      <div style="font-family:var(--mono);font-size:10px;font-weight:700;color:var(--crimson);
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">⚠️ Disclaimer</div>
      <div style="font-family:var(--sans);font-size:12px;color:#334155;line-height:1.6">
        Signals shown here are informational and derived from historical/technical indicators only.
        They do not constitute investment advice and are not a guarantee of future performance.
        This tool is not registered as a SEBI Investment Advisor. Consult a SEBI-registered
        financial advisor before making any investment decision.
      </div>
    </div>
    """, unsafe_allow_html=True)
