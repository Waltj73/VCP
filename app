import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go

st.set_page_config(page_title="VCP Scanner", layout="wide")
st.title("🌀 VCP (Volatility Contraction Pattern) Scanner")
st.caption("Detects swing pivots, measures shrinking pullback legs, and flags candidates near their highs with drying-up volume.")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Ticker & Data")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
period = st.sidebar.selectbox("History window", ["6mo", "1y", "2y", "3y"], index=1)

st.sidebar.header("Pivot Detection")
pivot_window = st.sidebar.slider(
    "Pivot lookback (bars each side)", 3, 15, 5,
    help="A bar must be the highest/lowest within this many bars on BOTH sides to count as a swing pivot. Larger = fewer, more significant pivots."
)

st.sidebar.header("VCP Criteria")
num_contractions = st.sidebar.slider("Contractions to evaluate", 2, 5, 3)
max_pullback_pct = st.sidebar.slider("Max allowed pullback %", 10, 60, 35)
shrink_tolerance_pct = st.sidebar.slider(
    "Shrink tolerance %", 0, 30, 10,
    help="How much 'slack' to allow — a later pullback must be at most (100 - tolerance)% the size of the prior one to count as a real contraction."
)
near_high_pct = st.sidebar.slider("Must be within X% of base high", 5, 40, 15)
vol_lookback = st.sidebar.slider("Volume trend lookback (bars)", 10, 60, 20)

# ============================================================
# DATA FETCH
# ============================================================
@st.cache_data(ttl=3600)
def get_data(sym, per):
    df = yf.download(sym, period=per, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])
    return df

try:
    df = get_data(ticker, period)

    if df.empty:
        st.error(f"No data returned for {ticker}. Check the symbol.")
        st.stop()

    # Guard against stale/partial last bar mid-session
    last_date = df.index[-1].date()
    today = pd.Timestamp.now().date()
    if last_date == today:
        st.info(f"Note: today's bar ({today}) may still be forming — high/low/close can still change until the close.")

    # ============================================================
    # PIVOT DETECTION
    # ============================================================
    highs = df["High"].values
    lows = df["Low"].values

    high_idx = argrelextrema(highs, np.greater_equal, order=pivot_window)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=pivot_window)[0]

    # Deduplicate plateaus (argrelextrema can flag flat runs as multiple points)
    def dedupe_adjacent(idx_array, min_gap):
        if len(idx_array) == 0:
            return idx_array
        out = [idx_array[0]]
        for i in idx_array[1:]:
            if i - out[-1] >= min_gap:
                out.append(i)
        return np.array(out)

    high_idx = dedupe_adjacent(high_idx, pivot_window)
    low_idx = dedupe_adjacent(low_idx, pivot_window)

    # Merge into a single alternating pivot sequence (by date order)
    pivots = []
    for i in high_idx:
        pivots.append((df.index[i], df["High"].iloc[i], "H"))
    for i in low_idx:
        pivots.append((df.index[i], df["Low"].iloc[i], "L"))
    pivots.sort(key=lambda x: x[0])

    # Enforce strict alternation: if two same-type pivots appear back to back,
    # keep only the more extreme one (highest high / lowest low)
    clean_pivots = []
    for p in pivots:
        if clean_pivots and clean_pivots[-1][2] == p[2]:
            if p[2] == "H" and p[1] > clean_pivots[-1][1]:
                clean_pivots[-1] = p
            elif p[2] == "L" and p[1] < clean_pivots[-1][1]:
                clean_pivots[-1] = p
        else:
            clean_pivots.append(p)

    # ============================================================
    # LEG / CONTRACTION MEASUREMENT
    # ============================================================
    # A "leg" is a High -> Low swing (a pullback). We want the most recent
    # N legs and check whether they're shrinking in sequence.
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

    recent_legs = legs[-num_contractions:] if len(legs) >= num_contractions else legs

    # ============================================================
    # VCP SCORING
    # ============================================================
    score = 0
    max_score = 4
    notes = []

    # 1. Do we have enough legs at all?
    has_enough_legs = len(recent_legs) >= 2
    if has_enough_legs:
        score += 1
        notes.append("✅ Enough swing legs detected to evaluate a base structure")
    else:
        notes.append("❌ Not enough distinct swing legs found — try a smaller pivot lookback")

    # 2. Are pullbacks within the allowed max size (not a crash, a real base)?
    all_within_max = all(leg["pullback_pct"] <= max_pullback_pct for leg in recent_legs) if recent_legs else False
    if all_within_max and has_enough_legs:
        score += 1
        notes.append(f"✅ All recent pullbacks are under {max_pullback_pct}%")
    else:
        notes.append(f"❌ At least one pullback exceeds {max_pullback_pct}% — may be too volatile for a clean base")

    # 3. Are the legs actually shrinking in sequence?
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

    # 4. Volume drying up recently vs its own longer trend?
    recent_vol = df["Volume"].tail(vol_lookback // 2).mean()
    baseline_vol = df["Volume"].tail(vol_lookback).mean()
    vol_declining = recent_vol < baseline_vol
    if vol_declining:
        score += 1
        notes.append("✅ Volume is contracting (recent avg below the longer baseline)")
    else:
        notes.append("❌ Volume is not currently contracting")

    # 5. Proximity to base high
    base_high = df["High"].tail(vol_lookback * 3).max()
    current_close = df["Close"].iloc[-1]
    pct_off_high = (base_high - current_close) / base_high * 100
    near_high = pct_off_high <= near_high_pct

    # ============================================================
    # LAYOUT
    # ============================================================
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("VCP Score")
        st.metric("Score", f"{score} / {max_score}")
        if score >= 3 and near_high:
            st.success("🌀 Looks like a genuine VCP candidate")
        elif score >= 2:
            st.warning("⚠️ Partial match — some VCP traits present")
        else:
            st.error("Not currently showing VCP structure")

        st.metric("% off base high", f"{pct_off_high:.1f}%",
                   delta=f"{'within' if near_high else 'outside'} {near_high_pct}% threshold")

        st.markdown("---")
        st.markdown("**Checklist:**")
        for n in notes:
            st.write(n)

        if recent_legs:
            st.markdown("---")
            st.markdown("**Detected pullback legs (most recent last):**")
            leg_table = pd.DataFrame([{
                "High Date": leg["high_date"].date(),
                "High": round(leg["high_price"], 2),
                "Low Date": leg["low_date"].date(),
                "Low": round(leg["low_price"], 2),
                "Pullback %": round(leg["pullback_pct"], 1)
            } for leg in recent_legs])
            st.dataframe(leg_table, use_container_width=True, hide_index=True)

    with col1:
        st.subheader(f"{ticker} — Price with Detected Pivots")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        ))

        # Plot all pivots
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

        # Annotate the recent legs being scored with % labels and connecting lines
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

        # Base high reference line
        fig.add_hline(y=base_high, line_dash="dash", line_color="gray",
                      annotation_text="Base High", annotation_position="top left")

        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume subplot
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="steelblue"))
        vol_fig.add_hline(y=baseline_vol, line_dash="dash", line_color="orange",
                          annotation_text=f"{vol_lookback}-bar avg")
        vol_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_rangeslider_visible=False)
        st.plotly_chart(vol_fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
