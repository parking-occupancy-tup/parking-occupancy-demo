"""T12: Streamlit demo of the parking occupancy models (CRISP-DM deployment step).

Replays the test period of the study (16-31 May 2019): the user picks a zone (or the
city average), a date and an hour and sees the forecasts of the saved models next to the
actual occupancy and the same-hour-last-week baseline, plus the whole day as a chart.

Files (written by ``python -m src.demo_data``), next to this script:
``models/<name>[_h24].joblib`` and ``data/demo.parquet``.

Run: ``streamlit run app/app.py`` (main repo) or ``streamlit run app.py`` (demo repo).
"""
from __future__ import annotations

import html
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
TARGET = "occupancy_rate"
BASELINE = "occ_lag_168h"
MODEL_LABELS = {"gradient_boosting": "Gradient Boosting", "linear_regression": "Linear regression"}
HORIZONS = {"Next hour": "", "Next day (24 h ahead)": "_h24"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CITY = "City average (34 zones)"
# AreaName "All" is a real zone in the sensor data (mainly Albert St and Jolimont), not "all zones".
ZONE_LABEL = {"All": "All (Albert St / Jolimont)"}

# Test-set results (reports/evaluation.md), shown in the banner and the explainer.
GB_RMSE, BASELINE_RMSE = {"": 6.14, "_h24": 8.33}, {"": 9.68, "_h24": 9.70}

COLORS = {"Actual": "#111827", "Gradient Boosting": "#2563eb", "Linear regression": "#f59e0b",
          "Same hour last week": "#9ca3af"}
LEVELS = [(50, "Plenty of space", "#16a34a"), (85, "Getting busy", "#f59e0b"), (101, "Nearly full", "#dc2626")]

CSS = """
<style>
.block-container {max-width: 1150px; padding-top: 2.2rem;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.banner {border-left: 4px solid #2563eb; background: #e8effd; padding: 0.75rem 1rem;
         border-radius: 6px; margin: 0.2rem 0 1rem 0; color: #111827;}
.card {background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.9rem 1rem;
       height: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.03);}
.card.best {border: 1.5px solid #2563eb;}
.card .label {color: #4b5563; font-size: 0.85rem;}
.card .value {font-size: 2.1rem; font-weight: 700; color: #111827; line-height: 1.25;}
.card .delta {display: inline-block; background: #f3f4f6; color: #4b5563; border-radius: 999px;
              padding: 0.05rem 0.55rem; font-size: 0.75rem;}
.gauge {position: relative; height: 6px; border-radius: 3px; margin: 0.45rem 0 0.35rem 0;
        background: linear-gradient(90deg, #16a34a 0 50%, #f59e0b 50% 85%, #dc2626 85% 100%);}
.gauge .mark {position: absolute; top: -4px; width: 3px; height: 14px; background: #111827;}
.level {font-size: 0.8rem; font-weight: 600;}
</style>
"""


@st.cache_resource
def load_models(suffix: str) -> dict[str, dict]:
    out = {}
    for name in MODEL_LABELS:
        p = HERE / "models" / f"{name}{suffix}.joblib"
        if p.exists():
            out[name] = joblib.load(p)
    return out


@st.cache_data
def load_rows() -> pd.DataFrame:
    return pd.read_parquet(HERE / "data" / "demo.parquet")


def predict(rows: pd.DataFrame, bundles: dict[str, dict]) -> pd.DataFrame:
    """Rows plus one prediction column per model (NaN where a needed input is missing)."""
    out = rows.copy()
    for name, b in bundles.items():
        x = rows[b["features"]]
        ok = x.notna().all(axis=1)
        out[name] = np.nan
        if ok.any():
            out.loc[ok, name] = np.clip(b["model"].predict(x[ok]), 0, 100)
    return out


def city_average(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean over zones per hour, using only rows where every column is known (fair comparison)."""
    full = pred.dropna(subset=cols)
    return full.groupby("window_start")[cols].mean()


def level(v: float) -> tuple[str, str]:
    return next((name, color) for limit, name, color in LEVELS if v < limit)


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.1f}%"


def card(label: str, value, actual=None, best: bool = False, gauge: bool = False) -> str:
    body = f'<div class="label">{html.escape(label)}</div><div class="value">{fmt(value)}</div>'
    if gauge and not pd.isna(value):
        name, color = level(value)
        body += (f'<div class="gauge"><div class="mark" style="left: calc({value:.1f}% - 1px)"></div></div>'
                 f'<div class="level" style="color:{color}">{name}</div>')
    elif actual is not None and not pd.isna(value) and not pd.isna(actual):
        body += f'<span class="delta">{value - actual:+.1f} pp vs actual</span>'
    return f'<div class="card{" best" if best else ""}">{body}</div>'


def day_chart(day: pd.DataFrame, hour: int) -> alt.Chart:
    long = (day.rename_axis("hour").reset_index()
            .melt("hour", var_name="series", value_name="occupancy"))
    order = list(COLORS)
    scale = alt.Scale(domain=order, range=[COLORS[s] for s in order])
    base = alt.Chart(long).encode(
        x=alt.X("hour:Q", title=None, scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(values=list(range(0, 24, 3)), labelExpr="(datum.value < 10 ? '0' : '') + datum.value + ':00'")),
        y=alt.Y("occupancy:Q", title=None, scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 25, 50, 75, 100], labelExpr="datum.value + '%'")),
        color=alt.Color("series:N", scale=scale, sort=order, title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.condition(alt.datum.series == "Same hour last week", alt.value([5, 4]), alt.value([1, 0])),
        strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(3), alt.value(2)),
        tooltip=[alt.Tooltip("hour:Q", title="Hour"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("occupancy:Q", title="Occupancy (%)", format=".1f")])
    band = alt.Chart(pd.DataFrame({"x": [hour - 0.4], "x2": [hour + 0.4]})).mark_rect(
        color="#2563eb", opacity=0.12).encode(x="x:Q", x2="x2:Q")
    point = base.transform_filter((alt.datum.hour == hour) & (alt.datum.series == "Gradient Boosting")).mark_circle(
        size=90, opacity=1)
    return (band + base.mark_line(interpolate="linear") + point).properties(height=340)


def main() -> None:
    st.set_page_config(page_title="Parking occupancy forecast", page_icon="🅿️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Parking occupancy forecast")
    st.caption("City of Melbourne, 2019. Replay of the study's test period (16–31 May), "
               "which the models never saw during training or tuning.")

    rows = load_rows()
    zones = sorted(rows["AreaName"].unique())
    with st.sidebar:
        st.subheader("Choose a window")
        horizon = st.radio("Forecast horizon", list(HORIZONS), help=(
            "Next hour uses the occupancy of the previous hour. "
            "Next day uses only values from 24 hours or more before."))
        zone = st.selectbox("Zone", [CITY, *zones], format_func=lambda z: ZONE_LABEL.get(z, z))
        dates = sorted(rows["window_start"].dt.date.unique())
        date = st.selectbox("Date", dates, format_func=lambda d: f"{DAYS[d.weekday()][:3]} {d:%d %b %Y}")
        hour = st.slider("Hour", 0, 23, 12, format="%d:00")

    suffix = HORIZONS[horizon]
    bundles = load_models(suffix)
    if not bundles:
        st.error("No models found. Run `python -m src.demo_data` first.")
        st.stop()
    cols = [TARGET, *bundles, BASELINE]
    on_day = rows[rows["window_start"].dt.date == date]
    if zone == CITY:
        day = city_average(predict(on_day, bundles), cols)
    else:
        day = predict(on_day[on_day["AreaName"] == zone], bundles).set_index("window_start")[cols]
    day.index = day.index.hour

    st.subheader(f"{ZONE_LABEL.get(zone, zone)}, {DAYS[date.weekday()]} {date:%d %B %Y}, "
                 f"{hour:02d}:00–{(hour + 1) % 24:02d}:00")
    gain = 1 - GB_RMSE[suffix] / BASELINE_RMSE[suffix]
    st.markdown(f'<div class="banner"><b>Our model is {gain:.0%} more accurate</b> than reusing last week\'s value '
                f'(error {GB_RMSE[suffix]:.2f} vs {BASELINE_RMSE[suffix]:.2f} pp, {horizon.lower()}).</div>',
                unsafe_allow_html=True)

    if hour not in day.index:
        st.warning("No data for this window (sensor coverage too low). Pick another hour.")
    else:
        r = day.loc[hour]
        cards = [card("Actual occupancy", r[TARGET], gauge=True)]
        cards += [card(f"{MODEL_LABELS[n]}{' (best)' if n == 'gradient_boosting' else ''}", r[n], r[TARGET],
                       best=n == "gradient_boosting") for n in bundles]
        cards.append(card("Same hour last week", r[BASELINE], r[TARGET]))
        for c, h in zip(st.columns(len(cards)), cards):
            c.markdown(h, unsafe_allow_html=True)
        if pd.isna(r[TARGET]):
            st.info("The actual rate is missing for this window (low sensor coverage).")

    with st.container(border=True):
        st.markdown("**The whole day**")
        chart = day.rename(columns={TARGET: "Actual", BASELINE: "Same hour last week", **MODEL_LABELS})
        st.altair_chart(day_chart(chart, hour), use_container_width=True)
        if zone == CITY:
            st.caption("Average of the 34 zones, over the zones where every value is known in that hour.")
        else:
            st.caption("Gaps are hours with too little sensor coverage. Small zones (few bays) are harder to predict.")

    with st.expander("How this works"):
        st.markdown(
            "- **Occupancy rate** = share of monitored bay-time in the zone during which a vehicle was present, "
            "per hour, computed from in-ground sensor records.\n"
            "- The models were trained on March–April 2019, tuned on 1–15 May, and are shown here on 16–31 May.\n"
            f"- On this period, Gradient Boosting's error (RMSE) was {GB_RMSE[''] :.2f} percentage points for the next "
            f"hour and {GB_RMSE['_h24']:.2f} for the next day, versus {BASELINE_RMSE[''] :.2f} for the "
            "same-hour-last-week baseline.\n"
            "- Gauge on the actual occupancy: green below 50%, amber 50–85%, red 85% and above.\n"
            "- Data: City of Melbourne Open Data, *On-street Car Parking Sensor Data – 2019*. "
            "Study by Ezekiel Jairus Solitario and Jose Salvador Rentoria, TUP Manila.")


if __name__ == "__main__":
    main()
