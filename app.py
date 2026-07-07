import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go

st.set_page_config(page_title="VCP Scanner", layout="wide")
st.title("🌀 VCP (Volatility Contraction Pattern) Scanner")
st.caption("Scans a watchlist, scores each name on VCP criteria, and lets you drill into any result's chart.")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Watchlist")
watchlist_input = st.sidebar.text_area(
    "Tickers (comma or newline separated)",
    value="NVDA, INTC, PFE, TSLA, AAPL, PLTR, MU, SPY, NFLX, AMD, SMCI, AMZN, MSFT, QQQ, NKE, MRVL, WMT, AVGO, IWM, GOOGL, GOOG, SLV, META, TSM, KLAC, XOM, CRM, LRCX, SLB, CVX, FCX, CSX, MRK, AA, ARM, BA, UNH, VOO, HON, HAL, MCD, MCHP, OXY, JPM, GLD, SBUX, DIA, GE, LULU, RTX, LLY, LOW, COST, UNP",
    height=150
)
period = st.sidebar.selectbox("History window", ["6mo", "1y", "2y", "3y"], index=1)

st.sidebar.header("Pivot Detection")
pivot_window = st.sidebar.slider(
    "Pivot lookback (bars each side)", 3, 15, 5,
    help="A bar must be the highest/lowest within this many bars on BOTH sides to count as a swing pivot."
)

st.sidebar.header("VCP Criteria")
num_contractions = st.sidebar.slider("Contractions to evaluate", 2, 5, 3)
max_pullback_pct = st.sidebar.slider("Max allowed pullback %", 10, 60, 35)
shrink_tolerance_pct = st.sidebar.slider(
    "Shrink tolerance %", 0, 30, 10,
    help="A later pullback must be at most (100 - tolerance)% the size of the prior one to count as a real contraction."
)
near_high_pct = st.sidebar.slider("Must be within X% of base high", 5, 40, 15)
vol_lookback = st.sidebar.slider("Volume trend lookback (bars)", 10, 60, 20)

st.sidebar.header("RMV (Relative Measured Volatility)")
rmv_atr_length = st.sidebar.slider("RMV smoothing (1 = raw, DeepVue-style)", 1, 14, 1)
rmv_lookback = st.sidebar.slider("RMV lookback period", 5, 30, 15)
rmv_tight_level = st.sidebar.slider("RMV tight threshold", 5, 40, 20)
rmv_extended_level = st.sidebar.slider("RMV extended threshold", 60, 95, 80)

MAX_SCORE = 6


def parse_watchlist(raw):
    tickers = raw.replace("\n", ",").split(",")
    return [t.strip().upper() for t in tickers if t.strip()]


@st.cache_data(ttl=3600)
def get_data(sym, per):
    df = yf.download(sym, period=per, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])
    return df


def compute_rmv(df, atr_length=1, lookback=15):
    """
    Direct port of the ThinkScript RMV: raw (or lightly smoothed) True Range,
    normalized as a percentile rank against its own rolling highest/lowest
    over `lookback` bars. Self-calibrating 0-100 scale, same as the chart
    version -- 0 = tightest range in the window, 100 = widest.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    if atr_length <= 1:
        atr_val = true_range
    else:
        atr_val = true_range.rolling(atr_length).mean()

    highest_atr = atr_val.rolling(lookback).max()
    lowest_atr = atr_val.rolling(lookback).min()
    atr_range = highest_atr - lowest_atr

    rmv = np.where(atr_range > 0, (atr_val - lowest_atr) / atr_range * 100, 0)
    return pd.Series(rmv, index=df.index)


def dedupe_adjacent(idx_array, min_gap):
    if len(idx_array) == 0:
        return idx_array
    out = [idx_array[0]]
    for i in idx_array[1:]:
        if i - out[-1] >= min_gap:
            out.append(i)
    return np.array(out)


def detect_pivots(df, pivot_window):
    highs = df["High"].values
    lows = df["Low"].values

    high_idx = argrelextrema(highs, np.greater_equal, order=pivot_window)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=pivot_window)[0]

    high_idx = dedupe_adjacent(high_idx, pivot_window)
    low_idx = dedupe_adjacent(low_idx, pivot_window)

    pivots = []
    for i in high_idx:
        pivots.append((df.index[i], df["High"].iloc[i], "H"))
    for i in low_idx:
        pivots.append((df.index[i], df["Low"].iloc[i], "L"))
    pivots.sort(key=lambda x: x[0])

    clean_pivots = []
    for p in pivots:
        if clean_pivots and clean_pivots[-1][2] == p[2]:
            if p[2] == "H" and p[1] > clean_pivots[-1][1]:
                clean_pivots[-1] = p
            elif p[2] == "L" and p[1] < clean_pivots[-1][1]:
                clean_pivots[-1] = p
        else:
            clean_pivots.append(p)
    return clean_pivots


def compute_legs(clean_pivots):
    legs = []
    for i in range(len(clean_pivots) - 1):
        d1, p1, t1 = clean_pivots[i]
        d2, p2, t2 = clean_pivots[i + 1]
        if t1 == "H" and t2 == "L":
            pullback_pct = (p1 - p2) / p1 * 100
            legs.append({
                "high_date": d1, "high_price": p1,
                "low_date": d2, "low_price": p2,
                "pullback_pct": pullback_pct
            })
    return legs


def score_vcp(df, pivot_window, num_contractions, max_pullback_pct,
               shrink_tolerance_pct, near_high_pct, vol_lookback,
               rmv_atr_length, rmv_lookback, rmv_tight_level, rmv_extended_level):
    clean_pivots = detect_pivots(df, pivot_window)
    legs = compute_legs(clean_pivots)
    recent_legs = legs[-num_contractions:] if len(legs) >= num_contractions else legs

    score = 0
    notes = []

    has_enough_legs = len(recent_legs) >= 2
    if has_enough_legs:
        score += 1
        notes.append("✅ Enough swing legs detected to evaluate a base structure")
    else:
        notes.append("❌ Not enough distinct swing legs found — try a smaller pivot lookback")

    all_within_max = all(leg["pullback_pct"] <= max_pullback_pct for leg in recent_legs) if recent_legs else False
    if all_within_max and has_enough_legs:
        score += 1
        notes.append(f"✅ All recent pullbacks are under {max_pullback_pct}%")
    else:
        notes.append(f"❌ At least one pullback exceeds {max_pullback_pct}%")

    is_shrinking = True
    if len(recent_legs) >= 2:
        for i in range(1, len(recent_legs)):
            prior = recent_legs[i - 1]["pullback_pct"]
            current = recent_legs[i]["pullback_pct"]
            allowed_max = prior * (1 - shrink_tolerance_pct / 100)
            if current > allowed_max:
                is_shrinking = False
                break
    else:
        is_shrinking = False

    if is_shrinking and has_enough_legs:
        score += 1
        notes.append("✅ Pullbacks are shrinking in sequence (classic VCP contraction)")
    else:
        notes.append("❌ Pullbacks are not consistently shrinking")

    recent_vol = df["Volume"].tail(vol_lookback // 2).mean()
    baseline_vol = df["Volume"].tail(vol_lookback).mean()
    vol_declining = recent_vol < baseline_vol
    if vol_declining:
        score += 1
        notes.append("✅ Volume is contracting (recent avg below the longer baseline)")
    else:
        notes.append("❌ Volume is not currently contracting")

    base_high = df["High"].tail(vol_lookback * 3).max()
    current_close = df["Close"].iloc[-1]
    pct_off_high = (base_high - current_close) / base_high * 100
    near_high = pct_off_high <= near_high_pct
    if near_high:
        score += 1
        notes.append(f"✅ Within {near_high_pct}% of the base high")
    else:
        notes.append(f"❌ More than {near_high_pct}% off the base high")

    rmv_series = compute_rmv(df, rmv_atr_length, rmv_lookback)
    current_rmv = rmv_series.iloc[-1]
    rmv_is_tight = current_rmv <= rmv_tight_level
    if rmv_is_tight:
        score += 1
        notes.append(f"✅ RMV is tight ({current_rmv:.1f}, below {rmv_tight_level}) — coiled volatility")
    else:
        notes.append(f"❌ RMV is not currently tight ({current_rmv:.1f})")

    return {
        "score": score,
        "notes": notes,
        "clean_pivots": clean_pivots,
        "recent_legs": recent_legs,
        "base_high": base_high,
        "current_close": current_close,
        "pct_off_high": pct_off_high,
        "baseline_vol": baseline_vol,
        "vol_declining": vol_declining,
        "rmv_series": rmv_series,
        "current_rmv": current_rmv,
        "rmv_is_tight": rmv_is_tight,
    }


def render_detail_chart(ticker, df, result, rmv_tight_level, rmv_extended_level):
    clean_pivots = result["clean_pivots"]
    recent_legs = result["recent_legs"]
    base_high = result["base_high"]
    baseline_vol = result["baseline_vol"]
    rmv_series = result["rmv_series"]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))

    h_dates = [p[0] for p in clean_pivots if p[2] == "H"]
    h_vals = [p[1] for p in clean_pivots if p[2] == "H"]
    l_dates = [p[0] for p in clean_pivots if p[2] == "L"]
    l_vals = [p[1] for p in clean_pivots if p[2] == "L"]

    fig.add_trace(go.Scatter(
        x=h_dates, y=h_vals, mode="markers", name="Swing High",
        marker=dict(symbol="triangle-down", size=10, color="red")
    ))
    fig.add_trace(go.Scatter(
        x=l_dates, y=l_vals, mode="markers", name="Swing Low",
        marker=dict(symbol="triangle-up", size=10, color="lime")
    ))

    for leg in recent_legs:
        fig.add_trace(go.Scatter(
            x=[leg["high_date"], leg["low_date"]],
            y=[leg["high_price"], leg["low_price"]],
            mode="lines+text",
            line=dict(color="orange", width=2, dash="dot"),
            text=["", f"-{leg['pullback_pct']:.1f}%"],
            textposition="bottom right",
            textfont=dict(color="orange", size=12),
            showlegend=False
        ))

    fig.add_hline(y=base_high, line_dash="dash", line_color="gray",
                  annotation_text="Base High", annotation_position="top left")

    fig.update_layout(
        height=600,
        title=f"{ticker} — Price with Detected Pivots",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="steelblue"))
    vol_fig.add_hline(y=baseline_vol, line_dash="dash", line_color="orange",
                      annotation_text=f"{vol_lookback}-bar avg")
    vol_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_rangeslider_visible=False)
    st.plotly_chart(vol_fig, use_container_width=True)

    # RMV subplot -- mirrors the ThinkScript pane: line + tight/extended
    # reference bands + shaded tight zone
    rmv_fig = go.Figure()
    rmv_fig.add_trace(go.Scatter(
        x=df.index, y=rmv_series, name="RMV", line=dict(color="cyan", width=2)
    ))
    rmv_fig.add_hline(y=rmv_tight_level, line_dash="dash", line_color="green",
                      annotation_text=f"Tight ({rmv_tight_level})")
    rmv_fig.add_hline(y=rmv_extended_level, line_dash="dash", line_color="red",
                      annotation_text=f"Extended ({rmv_extended_level})")
    rmv_fig.add_hline(y=50, line_dash="dot", line_color="gray")

    # Shade the tight zone (0 to tight threshold) whenever RMV is inside it
    tight_zone = rmv_series.where(rmv_series <= rmv_tight_level, np.nan)
    rmv_fig.add_trace(go.Scatter(
        x=df.index, y=tight_zone, fill="tozeroy",
        fillcolor="rgba(0,100,0,0.4)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))

    current_rmv = result["current_rmv"]
    rmv_fig.update_layout(
        height=220,
        title=f"RMV: {current_rmv:.1f}" + (" (TIGHT)" if result["rmv_is_tight"] else ""),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(rmv_fig, use_container_width=True)


# ============================================================
# MAIN: RUN SCAN
# ============================================================
tickers = parse_watchlist(watchlist_input)

if st.sidebar.button("🔍 Run Scan", type="primary"):
    results = {}
    errors = []
    progress = st.progress(0, text="Scanning...")

    for i, t in enumerate(tickers):
        try:
            df = get_data(t, period)
            if df.empty or len(df) < pivot_window * 4:
                errors.append(f"{t}: insufficient data")
                continue
            result = score_vcp(df, pivot_window, num_contractions, max_pullback_pct,
                                shrink_tolerance_pct, near_high_pct, vol_lookback,
                                rmv_atr_length, rmv_lookback, rmv_tight_level, rmv_extended_level)
            results[t] = {"df": df, "result": result}
        except Exception as e:
            errors.append(f"{t}: {e}")
        progress.progress((i + 1) / len(tickers), text=f"Scanning... {t}")

    progress.empty()
    st.session_state["scan_results"] = results
    if errors:
        st.session_state["scan_errors"] = errors

if "scan_errors" in st.session_state and st.session_state["scan_errors"]:
    with st.expander(f"⚠️ {len(st.session_state['scan_errors'])} ticker(s) skipped"):
        for e in st.session_state["scan_errors"]:
            st.write(e)

# ============================================================
# RESULTS TABLE + SCORE FILTER
# ============================================================
if "scan_results" in st.session_state and st.session_state["scan_results"]:
    results = st.session_state["scan_results"]

    st.markdown("---")
    min_score = st.slider(
        f"Minimum score to display (out of {MAX_SCORE})",
        0, MAX_SCORE, MAX_SCORE - 1
    )

    rows = []
    for t, data in results.items():
        r = data["result"]
        rows.append({
            "Ticker": t,
            "Score": r["score"],
            "% Off High": round(r["pct_off_high"], 1),
            "RMV": round(r["current_rmv"], 1),
            "RMV Tight": "Yes" if r["rmv_is_tight"] else "No",
            "Vol Contracting": "Yes" if r["vol_declining"] else "No",
            "Contractions Found": len(r["recent_legs"]),
        })

    results_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    filtered_df = results_df[results_df["Score"] >= min_score].reset_index(drop=True)

    st.subheader(f"Results: {len(filtered_df)} of {len(results_df)} names scoring ≥ {min_score}")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    if not filtered_df.empty:
        st.markdown("---")
        selected_ticker = st.selectbox("Select a ticker to view its chart:", filtered_df["Ticker"].tolist())

        if selected_ticker:
            data = results[selected_ticker]
            df = data["df"]
            result = data["result"]

            col1, col2 = st.columns([3, 1])
            with col2:
                st.subheader(f"{selected_ticker} Score")
                st.metric("Score", f"{result['score']} / {MAX_SCORE}")
                st.metric("Current RMV", f"{result['current_rmv']:.1f}",
                           delta="TIGHT" if result["rmv_is_tight"] else None)
                st.markdown("**Checklist:**")
                for n in result["notes"]:
                    st.write(n)

                if result["recent_legs"]:
                    st.markdown("**Pullback legs:**")
                    leg_table = pd.DataFrame([{
                        "High Date": leg["high_date"].date(),
                        "High": round(leg["high_price"], 2),
                        "Low Date": leg["low_date"].date(),
                        "Low": round(leg["low_price"], 2),
                        "Pullback %": round(leg["pullback_pct"], 1)
                    } for leg in result["recent_legs"]])
                    st.dataframe(leg_table, use_container_width=True, hide_index=True)

            with col1:
                render_detail_chart(selected_ticker, df, result, rmv_tight_level, rmv_extended_level)
    else:
        st.info("No names meet that score threshold. Try lowering the minimum score.")
else:
    st.info("👈 Enter your watchlist and click **Run Scan** to get started.")
