from __future__ import annotations

import datetime
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "generated_data" / "model_outputs"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions_by_sku_country.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance_by_sku_country.csv"
GLOBAL_BY_COUNTRY_FILE = OUTPUT_DIR / "global_predictions_by_country.csv"
GLOBAL_BY_SKU_FILE = OUTPUT_DIR / "global_predictions_by_sku.csv"
GLOBAL_ALL_FILE = OUTPUT_DIR / "global_predictions_all.csv"
METRICS_FILE = OUTPUT_DIR / "training_metrics.json"

SPLIT_ORDER = ["train", "validation", "test", "forecast"]

# train <-> forecast colors swapped vs original
SPLIT_COLORS = {
    "train":      "#00CFFF",   # was forecast cyan  → now train
    "validation": "#AC70CF",
    "test":       "#007FCC",
    "forecast":   "#14B588",   # was train teal     → now forecast
}

LINE_COLORS = ["#14B588", "#AC70CF", "#007FCC", "#00CFFF"]

FEATURE_LABEL_MAP = {
    "group_1_website_traffic_lag_1":   "Website traffic 1 day ago",
    "group_1_website_traffic_lag_3":   "Website traffic 3 days ago",
    "group_1_website_traffic_lag_7":   "Website traffic 7 days ago",
    "group_1_website_traffic_mov_3":   "Website traffic avg (3-day)",
    "group_1_website_traffic_mov_7":   "Website traffic avg (7-day)",
    "group_1_website_traffic_mov_14":  "Website traffic avg (14-day)",
    "group_1_website_traffic_mov_28":  "Website traffic avg (28-day)",
    "group_1_website_traffic_emov_3":  "Website traffic trend (3-day)",
    "group_1_website_traffic_emov_7":  "Website traffic trend (7-day)",
    "group_1_website_traffic_emov_14": "Website traffic trend (14-day)",
    "group_1_website_traffic_emov_28": "Website traffic trend (28-day)",
    "group_1_demand_lag_1":   "Demand 1 day ago",
    "group_1_demand_lag_3":   "Demand 3 days ago",
    "group_1_demand_lag_7":   "Demand 7 days ago",
    "group_1_demand_mov_3":   "Demand avg (3-day)",
    "group_1_demand_mov_7":   "Demand avg (7-day)",
    "group_1_demand_mov_14":  "Demand avg (14-day)",
    "group_1_demand_mov_28":  "Demand avg (28-day)",
    "group_1_demand_emov_3":  "Demand trend (3-day)",
    "group_1_demand_emov_7":  "Demand trend (7-day)",
    "group_1_demand_emov_14": "Demand trend (14-day)",
    "group_1_demand_emov_28": "Demand trend (28-day)",
    "group_2_add_to_carts_lag_1":   "Add to carts 1 day ago",
    "group_2_add_to_carts_lag_3":   "Add to carts 3 days ago",
    "group_2_add_to_carts_mov_3":   "Add to carts avg (3-day)",
    "group_2_add_to_carts_mov_7":   "Add to carts avg (7-day)",
    "group_2_add_to_carts_mov_14":  "Add to carts avg (14-day)",
    "group_2_add_to_carts_emov_3":  "Add to carts trend (3-day)",
    "group_2_add_to_carts_emov_7":  "Add to carts trend (7-day)",
    "group_2_add_to_carts_emov_14": "Add to carts trend (14-day)",
    "group_2_conversion_rate_lag_1":   "Conversion rate 1 day ago",
    "group_2_conversion_rate_lag_3":   "Conversion rate 3 days ago",
    "group_2_conversion_rate_mov_3":   "Conversion rate avg (3-day)",
    "group_2_conversion_rate_mov_7":   "Conversion rate avg (7-day)",
    "group_2_conversion_rate_mov_14":  "Conversion rate avg (14-day)",
    "group_2_conversion_rate_emov_3":  "Conversion rate trend (3-day)",
    "group_2_conversion_rate_emov_7":  "Conversion rate trend (7-day)",
    "group_2_conversion_rate_emov_14": "Conversion rate trend (14-day)",
    "day_of_week":      "Day of week",
    "month":            "Month",
    "day_of_month":     "Day of month",
    "is_weekend":       "Is weekend",
    "is_holiday":       "Is holiday",
    "website_traffic":  "Website traffic",
    "add_to_carts":     "Add to carts",
    "conversion_rate":  "Conversion rate",
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Demandly", layout="wide")

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg:           #ffffff;
        --bg-card:      #f8f9fb;
        --bg-card-alt:  #f2f4f8;
        --border:       rgba(0,0,0,0.09);
        --border-mid:   rgba(0,0,0,0.14);
        --text:         #0f1117;
        --text-muted:   #6b7280;
        --text-dim:     #9ca3af;
        --accent-red:   #e63946;
    }

    /* ── Override baseweb design tokens for light dropdowns ── */
    :root {
        --bc-colors-backgroundPrimary: #ffffff !important;
        --bc-colors-backgroundSecondary: #f8f9fb !important;
        --bc-colors-backgroundTertiary: #f2f4f8 !important;
        --bc-colors-contentPrimary: #0f1117 !important;
        --bc-colors-contentSecondary: #6b7280 !important;
        --bc-colors-backgroundOverlay: #ffffff !important;
        --bc-colors-backgroundOverlayArt: #ffffff !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; height: 0; }

    /* Hide the keyboard_double_arrow_left sidebar collapse button */
    button[data-testid="collapsedControl"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display: none !important; }
    section[data-testid="stSidebar"] button span[data-testid="stIconMaterial"] { display: none !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fb !important;
        border-right: 1px solid rgba(0,0,0,0.08) !important;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #f8f9fb !important;
        padding-top: 1.5rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .sidebar-heading {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #9ca3af !important;
        padding: 0 0.25rem;
        margin-bottom: 0.5rem;
        margin-top: 1.2rem;
        font-family: 'Inter', sans-serif;
        display: block;
    }
    .sidebar-divider {
        height: 1px;
        background: rgba(0,0,0,0.07);
        margin: 1.2rem 0 0.8rem;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', system-ui, sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp { background-color: var(--bg) !important; }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }

    p, span, div, label { color: var(--text); }

    /* ── Top navbar ── */
    .navbar {
        display: flex;
        align-items: center;
        gap: 2.5rem;
        padding: 0 2.5rem;
        border-bottom: 1px solid var(--border);
        height: 60px;
        background: var(--bg);
        position: sticky;
        top: 0;
        z-index: 100;
    }

    .navbar-brand {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 300;
        color: #0f1117 !important;
        letter-spacing: -0.01em;
        white-space: nowrap;
        padding-right: 2rem;
        border-right: 1px solid var(--border);
    }

    .navbar-brand strong {
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    /* ── Page content wrapper ── */
    .page-content {
        padding: 2rem 2.5rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* ── Intro card ── */
    .intro-card {
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        background: transparent;
    }

    .intro-card h2 {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem !important;
        letter-spacing: -0.02em;
    }

    .intro-card p {
        font-size: 0.875rem;
        color: var(--text-muted);
        line-height: 1.6;
        margin: 0;
        font-weight: 400;
    }

    /* ── KPI metric cards row ── */
    .kpi-row {
        display: grid;
        gap: 0;
        border: 1px solid var(--border);
        border-radius: 4px;
        margin-bottom: 2rem;
        overflow: hidden;
    }

    .kpi-cell {
        padding: 1.4rem 1.8rem;
        border-right: 1px solid var(--border);
        background: transparent;
    }

    .kpi-cell:last-child { border-right: none; }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f1117;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }

    .kpi-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 400;
        font-family: 'Inter', sans-serif;
    }

    /* ── Chart label ── */
    .chart-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.6rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Inline filter row ── */
    .filter-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.3rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Radio styled as tab bar ── */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        gap: 0 !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 0 !important;
        background: transparent !important;
    }
    div[data-testid="stRadio"] > div > label {
        border-radius: 0 !important;
        padding: 0.75rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -1px !important;
        background: transparent !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] > div > label > div:last-child {
        color: var(--text-muted) !important;
    }
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        border-bottom: 2px solid var(--accent-red) !important;
    }
    div[data-testid="stRadio"] > div > label:has(input:checked) > div:last-child {
        color: #0f1117 !important;
    }
    div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
    div[data-testid="stRadio"] > label { display: none !important; }

    /* ── Inputs, selects — closed state ── */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: var(--border-mid) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
    }

    div[data-baseweb="select"] input { color: var(--text) !important; }

    /* ── Dropdown popover — Streamlit renders this in a body-level portal ── */
    /* Target every layer of the portal wrapper */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] > div > div,
    div[data-baseweb="popover"] > div > div > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div,
    [data-baseweb="menu"] > ul,
    ul[role="listbox"],
    ul[role="listbox"] > div {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.12) !important;
        border-radius: 4px !important;
    }

    /* Every individual option row */
    li[role="option"],
    [role="option"] {
        background-color: #ffffff !important;
        color: #0f1117 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* Hover + selected state */
    li[role="option"]:hover,
    [role="option"]:hover,
    li[role="option"][aria-selected="true"],
    [role="option"][aria-selected="true"] {
        background-color: #f2f4f8 !important;
        color: #0f1117 !important;
    }

    /* All text/span/div children inside options */
    li[role="option"] *,
    [role="option"] * {
        color: #0f1117 !important;
    }

    /* Checkmark SVG in multiselect */
    li[role="option"] svg { fill: #7a8aaa !important; }
    li[role="option"][aria-selected="true"] svg { fill: #00CFFF !important; }

    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-color: var(--border-mid) !important;
        border-radius: 4px !important;
    }

    input[type="text"], input[type="date"] {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    span[data-baseweb="tag"] {
        background-color: rgba(0,0,0,0.06) !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: 3px !important;
    }

    span[data-baseweb="tag"] span {
        color: #0f1117 !important;
        font-weight: 400 !important;
        font-size: 0.8rem !important;
    }

    .stMultiSelect label,
    .stSelectbox label,
    .stSlider label,
    .stDateInput label,
    .stRadio label {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
    }

    /* ── Streamlit native metric override (hide, we use custom KPI) ── */
    div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.02) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        padding: 1.2rem 1.4rem !important;
    }

    div[data-testid="stMetricLabel"] > div {
        color: var(--text-muted) !important;
        font-weight: 400 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: var(--text) !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* ── DataFrames — white interior, dark border ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        overflow: hidden !important;
    }

    div[data-testid="stDataFrame"] ::-webkit-scrollbar { width: 6px; height: 6px; }
    div[data-testid="stDataFrame"] ::-webkit-scrollbar-track { background: #f0f0f0; }
    div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 3px; }

    /* ── Expander — dark header, muted title ── */
    div[data-testid="stExpander"] {
        background: #f8f9fb !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details > summary {
        background: #f2f4f8 !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        padding: 0.7rem 1rem !important;
        border-radius: 4px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #0f1117 !important;
        background: #e9ecf2 !important;
    }

    div[data-testid="stExpander"] details[open] > div,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #f8f9fb !important;
        color: var(--text) !important;
    }

    /* ── st.json viewer — dark background ── */
    div[data-testid="stJson"],
    div[data-testid="stJson"] > div,
    div[data-testid="stJson"] pre,
    div[data-testid="stJson"] code {
        background: #f8f9fb !important;
        background-color: #f8f9fb !important;
        color: #0f1117 !important;
        font-family: 'Inter', monospace !important;
        font-size: 0.85rem !important;
    }

    /* JSON key labels */
    div[data-testid="stJson"] span[style*="color: rgb(0, 0, 128)"],
    div[data-testid="stJson"] span[style*="color: rgb(0, 128, 0)"] {
        color: #7a8aaa !important;
    }

    /* JSON numeric / bool values */
    div[data-testid="stJson"] span[style*="color: rgb(0, 0, 255)"] {
        color: #00CFFF !important;
    }

    /* JSON string values */
    div[data-testid="stJson"] span[style*="color: rgb(163, 21, 21)"] {
        color: #14B588 !important;
    }

    /* ── Radio ── */
    div[data-testid="stRadio"] > label {
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Alert ── */
    div[data-testid="stAlert"] {
        background: rgba(0,0,0,0.02) !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: 4px !important;
        color: var(--text-muted) !important;
    }

    /* ── Feature importance legend ── */
    .fi-legend {
        display: flex;
        gap: 2rem;
        margin-bottom: 1rem;
        margin-top: 0.3rem;
    }

    .fi-legend-item { display: flex; flex-direction: column; gap: 0.2rem; }

    .fi-legend-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .fi-legend-sub {
        font-size: 0.775rem;
        color: var(--text-muted);
        margin-left: 1.3rem;
        font-family: 'Inter', sans-serif;
    }

    .fi-dot-pos { width: 9px; height: 9px; border-radius: 2px; background: #00CFFF; flex-shrink: 0; }
    .fi-dot-neg { width: 9px; height: 9px; border-radius: 2px; background: #AC70CF; flex-shrink: 0; }

    /* ── Section header (title + subtitle above KPI rows) ── */
    .section-header {
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f1117;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }

    .section-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 400;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ── Section note ── */
    .section-note {
        color: var(--text-muted);
        font-size: 0.875rem;
        margin-top: -0.2rem;
        margin-bottom: 1.2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.6;
    }

    /* ── Kill white chart container borders (Altair/Vega iframe wrapper) ── */
    div[data-testid="stVegaLiteChart"] > div,
    div[data-testid="stVegaLiteChart"] canvas,
    div[data-testid="stArrowVegaLiteChart"] > div,
    .vega-embed,
    .vega-embed summary,
    .vega-embed .vega-actions {
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }
    /* Vega canvas background */
    .vega-embed canvas {
        background: #ffffff !important;
    }

    /* ── Date input — force dark background ── */
    div[data-testid="stDateInput"] input,
    div[data-testid="stDateInput"] > div > div,
    div[data-testid="stDateInput"] div[data-baseweb="input"],
    div[data-baseweb="calendar"],
    div[data-baseweb="datepicker"] {
        background-color: rgba(255,255,255,0.04) !important;
        color: var(--text) !important;
        border-color: var(--border-mid) !important;
    }

    /* The date range input text color specifically */
    div[data-testid="stDateInput"] input {
        color: #0f1117 !important;
        caret-color: #0f1117 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart config ─────────────────────────────────────────────────────────────
CHART_BG = "#ffffff"

# Register a custom theme so every chart gets the dark background automatically
def _dark_theme():
    return {
        "config": {
            "background": CHART_BG,
            "view": {"fill": CHART_BG, "stroke": "transparent"},
            "axis": {
                "labelColor": "#374151",
                "titleColor": "#6b7280",
                "gridColor": "rgba(0,0,0,0.07)",
                "domainColor": "rgba(0,0,0,0.15)",
                "tickColor": "rgba(0,0,0,0.15)",
                "labelFont": "Inter, sans-serif",
                "titleFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
            "legend": {
                "labelColor": "#374151",
                "titleColor": "#6b7280",
                "fillColor": "#f8f9fb",
                "strokeColor": "rgba(0,0,0,0.1)",
                "padding": 8,
                "labelFont": "Inter, sans-serif",
                "titleFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 11,
            },
        }
    }

alt.themes.register("dark", _dark_theme)
alt.themes.enable("dark")

CHART_CONFIG = dict(
    background=CHART_BG,
    axis_label_color="#374151",
    axis_title_color="#6b7280",
    grid_color="rgba(0,0,0,0.07)",
    legend_label_color="#374151",
    legend_title_color="#6b7280",
    legend_fill="#f8f9fb",
    legend_stroke="rgba(0,0,0,0.1)",
)


def _axis(title=""):
    return alt.Axis(
        labelColor=CHART_CONFIG["axis_label_color"],
        titleColor=CHART_CONFIG["axis_title_color"],
        gridColor=CHART_CONFIG["grid_color"],
        domainColor="rgba(0,0,0,0.15)",
        tickColor="rgba(0,0,0,0.15)",
        title=title,
        labelFont="Inter, sans-serif",
        titleFont="Inter, sans-serif",
        labelFontSize=11,
        titleFontSize=11,
    )


def _legend_cfg():
    return alt.Legend(
        labelColor=CHART_CONFIG["legend_label_color"],
        titleColor=CHART_CONFIG["legend_title_color"],
        fillColor=CHART_CONFIG["legend_fill"],
        strokeColor=CHART_CONFIG["legend_stroke"],
        padding=8,
        labelFont="Inter, sans-serif",
        titleFont="Inter, sans-serif",
        labelFontSize=11,
        titleFontSize=11,
    )


# ── Data helpers ─────────────────────────────────────────────────────────────
def require_file(path: Path) -> None:
    if not path.exists():
        st.error("A required model output is missing. Run the training pipeline first, then refresh.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_csv(path: Path, date_columns: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    if abs(float(value)) >= 1000:
        return f"{float(value):,.0f}{suffix}"
    return f"{float(value):,.2f}{suffix}"


def split_sort_key(split: str) -> int:
    if split in SPLIT_ORDER:
        return SPLIT_ORDER.index(split)
    return len(SPLIT_ORDER)


def ordered_unique(values: pd.Series) -> list:
    return sorted(values.dropna().unique().tolist(), key=lambda v: (split_sort_key(v), str(v)))


def filter_by_multiselect(df: pd.DataFrame, column: str, selected: list[str]) -> pd.DataFrame:
    if column not in df.columns or not selected:
        return df
    return df[df[column].isin(selected)]


def filter_by_date_range(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if "date" not in df.columns or not date_range:
        return df
    start, end = date_range
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


def calculate_visible_metrics(df: pd.DataFrame) -> dict:
    scored = df.dropna(subset=["actual_demand", "predicted_demand"]).copy()
    total_actual = scored["actual_demand"].sum() if not scored.empty else None
    total_predicted = scored["predicted_demand"].sum() if not scored.empty else None
    if scored.empty:
        return {"rows": len(df), "total_actual": total_actual, "total_predicted": total_predicted, "mae": None, "mape": None}
    abs_err = (scored["actual_demand"] - scored["predicted_demand"]).abs()
    non_zero = scored[scored["actual_demand"] != 0]
    mape = None
    if not non_zero.empty:
        mape = ((non_zero["actual_demand"] - non_zero["predicted_demand"]).abs()
                / non_zero["actual_demand"].abs()).mean() * 100
    return {"rows": len(df), "total_actual": total_actual, "total_predicted": total_predicted,
            "mae": abs_err.mean(), "mape": mape}


def get_split_color_scale():
    splits = list(SPLIT_COLORS.keys())
    colors = [SPLIT_COLORS[s] for s in splits]
    return alt.Scale(domain=splits, range=colors)


def metrics_table(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    for col in ["mae", "rmse", "mape_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_table(df: pd.DataFrame, max_rows: int = 500) -> None:
    """Render a DataFrame as a dark HTML table — fully CSS-controlled, no canvas."""
    display = df.head(max_rows).copy()
    for col in display.select_dtypes(include="float").columns:
        display[col] = display[col].apply(lambda v: f"{v:,.4f}" if pd.notna(v) else "")
    for col in display.select_dtypes(include="datetime").columns:
        display[col] = display[col].dt.strftime("%Y-%m-%d")

    headers = "".join(f"<th>{c}</th>" for c in display.columns)
    rows = "".join(
        f"<tr>{''.join(f'<td>{v}</td>' for v in row)}</tr>"
        for _, row in display.iterrows()
    )

    st.markdown(f"""
    <div style="overflow-x:auto;overflow-y:auto;max-height:420px;border-radius:4px;
                border:1px solid rgba(0,0,0,0.09);background:#ffffff;">
      <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;
                    font-size:0.84rem;color:#0f1117;background:#ffffff;">
        <thead>
          <tr style="background:#f2f4f8;border-bottom:1px solid rgba(0,0,0,0.09);">
            {headers}
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <style>
      table td,table th{{padding:0.55rem 1rem;text-align:left;
        border-bottom:1px solid rgba(0,0,0,0.06);white-space:nowrap;color:#0f1117!important;}}
      table th{{font-weight:600;font-size:0.75rem;color:#6b7280!important;
        text-transform:uppercase;letter-spacing:0.05em;}}
      tbody tr:last-child td{{border-bottom:none;}}
      tbody tr:hover td{{background:#f8f9fb;}}
    </style>
    """, unsafe_allow_html=True)


# ── Custom KPI row (matches screenshot card style) ───────────────────────────
def render_kpi_row(cells: list[tuple[str, str]], cols: int | None = None) -> None:
    """cells = list of (value, label). Rendered as a single bordered grid row."""
    n = cols or len(cells)
    grid_cols = " ".join(["1fr"] * n)
    items_html = ""
    for val, label in cells:
        items_html += f'<div class="kpi-cell"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>'
    st.markdown(
        f'<div class="kpi-row" style="grid-template-columns:{grid_cols};">{items_html}</div>',
        unsafe_allow_html=True,
    )


# ── Chart functions ───────────────────────────────────────────────────────────
def make_actual_vs_predicted_chart(df: pd.DataFrame) -> alt.Chart:
    chart_df = (
        df.groupby(["date", "split"], as_index=False)
        .agg(actual_demand=("actual_demand", "sum"), predicted_demand=("predicted_demand", "sum"))
        .sort_values("date")
    )
    chart_df.loc[chart_df["split"] == "forecast", "actual_demand"] = pd.NA

    pred_df = chart_df[["date", "split", "predicted_demand"]].copy().rename(columns={"predicted_demand": "demand"})
    pred_df["series"] = "Predicted"

    act_df = (chart_df[["date", "split", "actual_demand"]]
              .dropna(subset=["actual_demand"]).copy()
              .rename(columns={"actual_demand": "demand"}))
    act_df["series"] = "Actual"

    color_scale = get_split_color_scale()

    base_enc = dict(
        x=alt.X("date:T", axis=_axis("Date")),
        y=alt.Y("demand:Q", axis=_axis("Demand")),
        color=alt.Color("split:N", scale=color_scale, title="Split", legend=_legend_cfg()),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("split:N", title="Split"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("demand:Q", title="Demand", format=",.1f"),
        ],
    )

    predicted_chart = alt.Chart(pred_df).mark_line(strokeWidth=1.25).encode(**base_enc)
    actual_chart    = alt.Chart(act_df).mark_line(strokeWidth=1, strokeDash=[5, 3]).encode(**base_enc)

    return (
        (predicted_chart + actual_chart)
        .properties(height=400)
        .configure_view(fill=CHART_BG, stroke="transparent")
        .configure_legend(
            labelColor=CHART_CONFIG["legend_label_color"],
            titleColor=CHART_CONFIG["legend_title_color"],
            fillColor=CHART_CONFIG["legend_fill"],
            strokeColor=CHART_CONFIG["legend_stroke"],
            padding=8,
            labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif",
        )
        .interactive()
    )


def make_split_chart(df: pd.DataFrame) -> alt.Chart:
    split_df = df["split"].value_counts().rename_axis("split").reset_index(name="rows")
    split_df["order"] = split_df["split"].map(split_sort_key)
    return (
        alt.Chart(split_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("split:N", title="Split", sort=alt.SortField("order"),
                    axis=alt.Axis(labelColor=CHART_CONFIG["axis_label_color"],
                                  titleColor=CHART_CONFIG["axis_title_color"],
                                  labelAngle=0,
                                  gridColor=CHART_CONFIG["grid_color"],
                                  labelFont="Inter, sans-serif",
                                  titleFont="Inter, sans-serif")),
            y=alt.Y("rows:Q", title="Rows", axis=_axis("Rows")),
            color=alt.Color("split:N", scale=get_split_color_scale(), legend=None),
            tooltip=[alt.Tooltip("split:N", title="Split"), alt.Tooltip("rows:Q", title="Rows", format=",")],
        )
        .properties(height=240)
        .configure_view(fill=CHART_BG, stroke="transparent")
    )


def make_feature_importance_chart(df: pd.DataFrame) -> alt.Chart:
    chart_df = df.copy()
    chart_df["feature_label"] = chart_df["feature"].map(FEATURE_LABEL_MAP).fillna(chart_df["feature"])
    chart_df["direction"] = chart_df["coefficient"].apply(
        lambda v: "Positive coefficient" if v >= 0 else "Negative coefficient"
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("feature_label:N", sort="-x", title=None,
                    axis=alt.Axis(
                        labelColor=CHART_CONFIG["axis_label_color"],
                        titleColor=CHART_CONFIG["axis_title_color"],
                        labelFont="Inter, sans-serif",
                        titleFont="Inter, sans-serif",
                        labelFontSize=11,
                        labelLimit=260,
                    )),
            x=alt.X("importance:Q", title="Importance", axis=_axis("Importance")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Positive coefficient", "Negative coefficient"],
                                range=["#00CFFF", "#AC70CF"]),
                title="Direction",
                legend=_legend_cfg(),
            ),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
                alt.Tooltip("coefficient:Q", title="Coefficient", format=".4f"),
            ],
        )
        .properties(height=max(280, min(680, len(chart_df) * 28)))
        .configure_view(fill=CHART_BG, stroke="transparent")
        .configure_legend(
            labelColor=CHART_CONFIG["legend_label_color"],
            titleColor=CHART_CONFIG["legend_title_color"],
            fillColor=CHART_CONFIG["legend_fill"],
            strokeColor=CHART_CONFIG["legend_stroke"],
            padding=8,
            labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif",
        )
    )


def make_global_chart(df: pd.DataFrame, actual_col: str, predicted_col: str, group_col: str | None) -> alt.Chart:
    grouping = ["date", "split"]
    if group_col:
        grouping.append(group_col)
    chart_df = (
        df.groupby(grouping, as_index=False)
        .agg(actual=(actual_col, "sum"), predicted=(predicted_col, "sum"))
        .sort_values("date")
    )
    chart_df.loc[chart_df["split"] == "forecast", "actual"] = pd.NA
    long_df = chart_df.melt(
        id_vars=grouping, value_vars=["actual", "predicted"],
        var_name="series", value_name="demand",
    ).dropna(subset=["demand"])
    long_df["series"] = long_df["series"].map({"actual": "Actual", "predicted": "Predicted"})

    color_field  = "series:N" if not group_col else f"{group_col}:N"
    stroke_field = "series:N" if group_col else "split:N"
    color_scale  = alt.Scale(range=LINE_COLORS)

    return (
        alt.Chart(long_df)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", axis=_axis("Date")),
            y=alt.Y("demand:Q", axis=_axis("Demand")),
            color=alt.Color(color_field, scale=color_scale,
                            title=group_col.title() if group_col else "Series",
                            legend=_legend_cfg()),
            strokeDash=alt.StrokeDash(stroke_field, title="Line"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("split:N", title="Split"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("demand:Q", title="Demand", format=",.1f"),
            ],
        )
        .properties(height=400)
        .configure_view(fill=CHART_BG, stroke="transparent")
        .configure_legend(
            labelColor=CHART_CONFIG["legend_label_color"],
            titleColor=CHART_CONFIG["legend_title_color"],
            fillColor=CHART_CONFIG["legend_fill"],
            strokeColor=CHART_CONFIG["legend_stroke"],
            padding=8,
            labelFont="Inter, sans-serif",
            titleFont="Inter, sans-serif",
        )
        .interactive()
    )


# ── Tab renderers ─────────────────────────────────────────────────────────────
def render_overview(predictions: pd.DataFrame, metrics: dict) -> None:
    summary = metrics.get("summary", {})
    countries = predictions["country"].nunique()
    skus      = predictions["sku"].nunique()
    date_from = predictions["date"].min().date()
    date_to   = predictions["date"].max().date()

    st.markdown(
        """
        <div class="intro-card">
            <h2>Demand forecasting studio</h2>
            <p>Explore forecast accuracy, future demand, model drivers, and portfolio-level performance through a compact dashboard built for the project outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_row([
        (str(countries),       "Countries"),
        (str(skus),            "Products"),
        (str(date_from),       "Data from"),
        (str(date_to),         "Data to"),
    ])

    # Overall model quality
    overall = metrics.get("global_metrics_overall", {})
    horizon = summary.get("forecast_horizon", "—")
    n_models = summary.get("n_models_country_sku", "—")

    st.markdown("#### Model quality at a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall MAPE",      format_number(overall.get("mape_pct"), "%"))
    col2.metric("Overall MAE",       format_number(overall.get("mae")))
    col3.metric("Forecast horizon",  format_number(horizon))
    col4.metric("Models trained",    format_number(n_models))


def render_predictions(predictions: pd.DataFrame,
                       selected_countries, selected_skus,
                       selected_splits, selected_dates) -> None:
    # ── Filter data ──────────────────────────────────────────────────────────
    filtered = predictions.copy()
    filtered = filter_by_multiselect(filtered, "country", selected_countries)
    filtered = filter_by_multiselect(filtered, "sku", selected_skus)
    filtered = filter_by_multiselect(filtered, "split", selected_splits)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filtered = filter_by_date_range(filtered, selected_dates)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Predictions by SKU and Country</div>
            <div class="section-sub">Compare actual demand, predicted demand, and the future forecast horizon by country and SKU.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    vm = calculate_visible_metrics(filtered)
    render_kpi_row([
        (format_number(vm["total_predicted"]), "Total forecasted demand"),
        (format_number(vm["total_actual"]),    "Total actual demand"),
        (format_number(vm["mae"]),             "MAE"),
        (format_number(vm["mape"], "%"),       "MAPE"),
        (format_number(vm["rows"]),            "Rows shown"),
    ])

    # ── Chart label ──────────────────────────────────────────────────────────
    sku_label  = (", ".join(selected_skus[:3]) + ("…" if len(selected_skus) > 3 else "")) or "No SKUs"
    ctry_label = (", ".join(selected_countries[:3]) + ("…" if len(selected_countries) > 3 else "")) or "No countries"
    st.markdown(
        f'<div class="chart-label">{sku_label} · {ctry_label} — Actual vs Forecasted</div>',
        unsafe_allow_html=True,
    )

    st.altair_chart(make_actual_vs_predicted_chart(filtered), use_container_width=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander("Raw data"):
            render_table(filtered.sort_values(["date", "country", "sku"]))
    with exp2:
        with st.expander("Split distribution"):
            st.altair_chart(make_split_chart(filtered), use_container_width=True)


def render_insights(feature_importance: pd.DataFrame, country: str, sku: str, top_n: int) -> None:

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Feature Importance</div>
            <div class="section-sub">See which engineered features contribute most to each selected SKU-country model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = (
        feature_importance[
            (feature_importance["country"] == country) & (feature_importance["sku"] == sku)
        ]
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    if selected.empty:
        st.warning("No feature importance rows match the selected filters.")
        return

    st.markdown(
        """
        <div class="fi-legend">
            <div class="fi-legend-item">
                <div class="fi-legend-label"><span class="fi-dot-pos"></span>Positive coefficient</div>
                <div class="fi-legend-sub">Increases predicted demand</div>
            </div>
            <div class="fi-legend-item">
                <div class="fi-legend-label"><span class="fi-dot-neg"></span>Negative coefficient</div>
                <div class="fi-legend-sub">Decreases predicted demand</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.altair_chart(make_feature_importance_chart(selected), use_container_width=True)

    with st.expander("Raw feature data"):
        display_df = selected.copy()
        display_df["feature"] = display_df["feature"].map(FEATURE_LABEL_MAP).fillna(display_df["feature"])
        render_table(display_df)


def render_portfolio(table_choice: str, selected_groups, selected_splits_pv, selected_dates) -> None:
    if table_choice == "By country":
        df = load_csv(GLOBAL_BY_COUNTRY_FILE)
        actual_col, predicted_col, group_col = "actual_demand_sum", "predicted_demand_sum", "country"
    elif table_choice == "By SKU":
        df = load_csv(GLOBAL_BY_SKU_FILE)
        actual_col, predicted_col, group_col = "actual_demand", "predicted_demand", "sku"
    else:
        df = load_csv(GLOBAL_ALL_FILE)
        actual_col, predicted_col, group_col = "actual_demand_sum", "predicted_demand_sum", None

    filtered = df.copy()
    filtered = filter_by_multiselect(filtered, "split", selected_splits_pv)
    if group_col and selected_groups is not None:
        filtered = filter_by_multiselect(filtered, group_col, selected_groups)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filtered = filter_by_date_range(filtered, selected_dates)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Global Tables</div>
            <div class="section-sub">Switch between total demand, country-level demand, and SKU-level demand views.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.altair_chart(make_global_chart(filtered, actual_col, predicted_col, group_col), use_container_width=True)

    with st.expander("Raw data"):
        render_table(filtered.sort_values("date"))


def render_training(metrics: dict, selected_countries_tq: list, selected_splits_tq: list) -> None:
    summary = metrics.get("summary", {})
    overall = metrics.get("global_metrics_overall", {})

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Training Results</div>
            <div class="section-sub">Review forecast quality, split performance, and the training setup behind the model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_row([
        (format_number(overall.get("mape_pct"), "%"), "Overall MAPE"),
        (format_number(overall.get("mae")),           "Overall MAE"),
        (format_number(overall.get("rmse")),          "Overall RMSE"),
        (format_number(summary.get("n_models_country_sku")), "Models"),
    ])

    render_kpi_row([
        (format_number(summary.get("n_rows_evaluated")),  "Rows evaluated"),
        (format_number(summary.get("forecast_horizon")),  "Forecast horizon"),
        (format_number(summary.get("min_history_length")),"Min history"),
        (format_number(summary.get("ridge_alpha")),       "Ridge alpha"),
    ])

    per_model = metrics_table(metrics.get("per_model_metrics_by_split", []))

    by_split = metrics_table(metrics.get("global_metrics_by_split", []))
    if not by_split.empty:
        st.markdown("#### Metrics by split")
        split_chart = (
            alt.Chart(by_split)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("split:N", title="Split", sort=SPLIT_ORDER,
                        axis=alt.Axis(labelColor=CHART_CONFIG["axis_label_color"],
                                      titleColor=CHART_CONFIG["axis_title_color"],
                                      labelAngle=0,
                                      gridColor=CHART_CONFIG["grid_color"],
                                      labelFont="Inter, sans-serif",
                                      titleFont="Inter, sans-serif")),
                y=alt.Y("mape_pct:Q", title="MAPE (%)", axis=_axis("MAPE (%)")),
                color=alt.Color("split:N", scale=get_split_color_scale(), legend=None),
                tooltip=[
                    alt.Tooltip("split:N",    title="Split"),
                    alt.Tooltip("mae:Q",      title="MAE",          format=".2f"),
                    alt.Tooltip("rmse:Q",     title="RMSE",         format=".2f"),
                    alt.Tooltip("mape_pct:Q", title="MAPE (%)",     format=".2f"),
                    alt.Tooltip("n_obs:Q",    title="Observations", format=","),
                ],
            )
            .properties(height=280)
            .configure_view(fill=CHART_BG, stroke="transparent")
        )
        st.altair_chart(split_chart, use_container_width=True)

        with st.expander("Metrics table"):
            disp = by_split.rename(columns={"n_obs": "Observations", "mae": "MAE",
                                             "rmse": "RMSE", "mape_pct": "MAPE (%)"})
            for c in ["MAE", "RMSE", "MAPE (%)"]:
                if c in disp.columns:
                    disp[c] = disp[c].round(2)
            render_table(disp)

    if not per_model.empty:
        st.markdown("#### SKU-country metrics")
        show_count = st.slider("Rows to display", min_value=10, max_value=100, value=25, step=5)
        mf = per_model.copy()
        if selected_countries_tq:
            mf = filter_by_multiselect(mf, "country", selected_countries_tq)
        if selected_splits_tq:
            mf = filter_by_multiselect(mf, "split", selected_splits_tq)
        mf = mf.sort_values("mape_pct", ascending=False).head(show_count)
        mf = mf.rename(columns={"n_obs": "Observations", "mae": "MAE", "rmse": "RMSE", "mape_pct": "MAPE (%)"})
        for c in ["MAE", "RMSE", "MAPE (%)"]:
            if c in mf.columns:
                mf[c] = mf[c].round(2)
        with st.expander("SKU-country metrics table"):
            render_table(mf)

    with st.expander("Training configuration"):
        st.json(summary)


# ── Date preset helper ───────────────────────────────────────────────────────
DATE_PRESETS = ["Custom", "Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "Past 2 Years"]

def apply_date_preset(preset: str, min_date, max_date):
    today = datetime.date.today()
    if preset == "Past Week":
        return (max(min_date, today - datetime.timedelta(weeks=1)), min(max_date, today))
    elif preset == "Past Month":
        return (max(min_date, today - datetime.timedelta(days=30)), min(max_date, today))
    elif preset == "Past 3 Months":
        return (max(min_date, today - datetime.timedelta(days=90)), min(max_date, today))
    elif preset == "Past 6 Months":
        return (max(min_date, today - datetime.timedelta(days=180)), min(max_date, today))
    elif preset == "Past Year":
        return (max(min_date, today - datetime.timedelta(days=365)), min(max_date, today))
    elif preset == "Past 2 Years":
        return (max(min_date, today - datetime.timedelta(days=730)), min(max_date, today))
    return (min_date, max_date)


# ── Top navbar ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="navbar">
        <span class="navbar-brand"><strong>Demandly</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
predictions        = load_csv(PREDICTIONS_FILE)
feature_importance = load_csv(FEATURE_IMPORTANCE_FILE, date_columns=())
metrics            = load_metrics(METRICS_FILE)
per_model_df       = metrics_table(metrics.get("per_model_metrics_by_split", []))

# ── Tab navigation ────────────────────────────────────────────────────────────
TABS = ["Overview", "Forecast view", "Model drivers", "Portfolio", "Training quality"]
active = st.radio("nav", TABS, horizontal=True, label_visibility="collapsed", key="active_tab")

# ── Sidebar — always present, content switches per tab ────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding: 0 0.25rem 1rem; border-bottom: 1px solid rgba(0,0,0,0.08); margin-bottom:1rem;'>"
        "<span style='font-family:Inter,sans-serif; font-size:1.1rem; font-weight:800;"
        " letter-spacing:-0.03em; color:#0f1117;'>Demandly</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    # Dropdown portal dark mode fix
    st.markdown("""<style>
    body > div[data-baseweb="popover"], body > div[data-baseweb="popover"] *,
    body > div > div[data-baseweb="popover"], body > div > div[data-baseweb="popover"] *,
    body ul[role="listbox"], body ul[role="listbox"] *,
    body [data-baseweb="menu"], body [data-baseweb="menu"] * {
        background:#ffffff!important;background-color:#ffffff!important;color:#0f1117!important;
    }
    body li[role="option"]{background:#ffffff!important;color:#0f1117!important;}
    body li[role="option"]:hover{background:#f2f4f8!important;}
    </style>""", unsafe_allow_html=True)

    if active == "Overview":
        st.markdown("""
        <div style='padding:0.2rem 0.25rem;'>
            <p style='font-size:0.82rem; color:#374151; line-height:1.6; margin:0 0 1.1rem; font-weight:400;'>
                Purpose-built demand forecasting. One model per SKU and country, trained on sales signals and fully explainable.
            </p>
            <div style='height:1px; background:rgba(0,0,0,0.08); margin-bottom:1.1rem;'></div>
            <div style='font-size:0.62rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:#9ca3af; margin-bottom:0.75rem;'>What it does</div>
            <div style='display:flex; flex-direction:column; gap:0.6rem; margin-bottom:1.2rem;'>
                <div style='display:flex; gap:0.7rem; align-items:flex-start;'>
                    <div style='width:4px; height:4px; border-radius:50%; background:#00CFFF; margin-top:6px; flex-shrink:0;'></div>
                    <span style='font-size:0.8rem; color:#6b7280; line-height:1.5;'>Trains a Ridge regression model per SKU and country pair</span>
                </div>
                <div style='display:flex; gap:0.7rem; align-items:flex-start;'>
                    <div style='width:4px; height:4px; border-radius:50%; background:#00CFFF; margin-top:6px; flex-shrink:0;'></div>
                    <span style='font-size:0.8rem; color:#6b7280; line-height:1.5;'>Engineered features: traffic, add-to-carts, conversion rate, demand lags</span>
                </div>
                <div style='display:flex; gap:0.7rem; align-items:flex-start;'>
                    <div style='width:4px; height:4px; border-radius:50%; background:#00CFFF; margin-top:6px; flex-shrink:0;'></div>
                    <span style='font-size:0.8rem; color:#6b7280; line-height:1.5;'>30-day forecast horizon across train, validation, test, and forecast splits</span>
                </div>
                <div style='display:flex; gap:0.7rem; align-items:flex-start;'>
                    <div style='width:4px; height:4px; border-radius:50%; background:#00CFFF; margin-top:6px; flex-shrink:0;'></div>
                    <span style='font-size:0.8rem; color:#6b7280; line-height:1.5;'>Every forecast traceable to its feature drivers via importance scores</span>
                </div>
            </div>
            <div style='height:1px; background:rgba(0,0,0,0.08); margin-bottom:1.1rem;'></div>
            <div style='font-size:0.62rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:#9ca3af; margin-bottom:0.75rem;'>Navigate</div>
            <div style='display:flex; flex-direction:column; gap:0.4rem;'>
                <div style='display:flex; gap:0.6rem; align-items:center; padding:0.45rem 0.6rem; border-radius:4px; background:rgba(0,0,0,0.03);'>
                    <div style='width:5px; height:5px; border-radius:1px; background:#e63946; flex-shrink:0;'></div>
                    <div><div style='font-size:0.78rem; color:#0f1117; font-weight:500;'>Forecast view</div><div style='font-size:0.7rem; color:#9ca3af;'>Predicted vs actual by SKU</div></div>
                </div>
                <div style='display:flex; gap:0.6rem; align-items:center; padding:0.45rem 0.6rem; border-radius:4px; background:rgba(0,0,0,0.03);'>
                    <div style='width:5px; height:5px; border-radius:1px; background:#AC70CF; flex-shrink:0;'></div>
                    <div><div style='font-size:0.78rem; color:#0f1117; font-weight:500;'>Model drivers</div><div style='font-size:0.7rem; color:#9ca3af;'>Feature importance per model</div></div>
                </div>
                <div style='display:flex; gap:0.6rem; align-items:center; padding:0.45rem 0.6rem; border-radius:4px; background:rgba(0,0,0,0.03);'>
                    <div style='width:5px; height:5px; border-radius:1px; background:#007FCC; flex-shrink:0;'></div>
                    <div><div style='font-size:0.78rem; color:#0f1117; font-weight:500;'>Portfolio</div><div style='font-size:0.7rem; color:#9ca3af;'>Demand totals by country or SKU</div></div>
                </div>
                <div style='display:flex; gap:0.6rem; align-items:center; padding:0.45rem 0.6rem; border-radius:4px; background:rgba(0,0,0,0.03);'>
                    <div style='width:5px; height:5px; border-radius:1px; background:#14B588; flex-shrink:0;'></div>
                    <div><div style='font-size:0.78rem; color:#0f1117; font-weight:500;'>Training quality</div><div style='font-size:0.7rem; color:#9ca3af;'>MAPE, MAE, RMSE per split</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif active == "Forecast view":
        st.markdown('<span class="sidebar-heading">Forecast view</span>', unsafe_allow_html=True)
        selected_countries = st.multiselect("Country",
            options=sorted(predictions["country"].dropna().unique()),
            default=sorted(predictions["country"].dropna().unique()),
            key="fv_country")
        selected_skus = st.multiselect("SKU",
            options=sorted(predictions["sku"].dropna().unique()),
            default=sorted(predictions["sku"].dropna().unique())[:5],
            key="fv_sku")
        selected_splits = st.multiselect("Split",
            options=ordered_unique(predictions["split"]),
            default=ordered_unique(predictions["split"]),
            key="fv_split")
        min_date = predictions["date"].min().date()
        max_date = predictions["date"].max().date()
        fv_preset = st.selectbox("Quick range", DATE_PRESETS, index=0, key="fv_preset")
        fv_default = apply_date_preset(fv_preset, min_date, max_date)
        selected_dates = st.date_input("Date range",
            value=fv_default, min_value=min_date, max_value=max_date, key="fv_dates")

    elif active == "Model drivers":
        st.markdown('<span class="sidebar-heading">Model drivers</span>', unsafe_allow_html=True)
        fi_country = st.selectbox("Country",
            options=sorted(feature_importance["country"].dropna().unique()), key="fi_country")
        fi_skus = sorted(feature_importance.loc[
            feature_importance["country"] == fi_country, "sku"].dropna().unique())
        fi_sku = st.selectbox("SKU", options=fi_skus, key="fi_sku")
        fi_top_n = st.slider("Top features", min_value=5, max_value=30, value=15, step=5, key="fi_topn")

    elif active == "Portfolio":
        st.markdown('<span class="sidebar-heading">Portfolio</span>', unsafe_allow_html=True)
        pv_choice = st.radio("View", options=["By country", "By SKU", "All demand"], key="pv_choice")
        if pv_choice == "By country":
            pv_df = load_csv(GLOBAL_BY_COUNTRY_FILE)
            pv_groups = st.multiselect("Country",
                options=sorted(pv_df["country"].dropna().unique()),
                default=sorted(pv_df["country"].dropna().unique()), key="pv_groups")
        elif pv_choice == "By SKU":
            pv_df = load_csv(GLOBAL_BY_SKU_FILE)
            pv_groups = st.multiselect("SKU",
                options=sorted(pv_df["sku"].dropna().unique()),
                default=sorted(pv_df["sku"].dropna().unique())[:8], key="pv_groups")
        else:
            pv_df = load_csv(GLOBAL_ALL_FILE)
            pv_groups = None
        pv_splits = st.multiselect("Split",
            options=ordered_unique(pv_df["split"]),
            default=ordered_unique(pv_df["split"]), key="pv_splits")
        pv_min = pv_df["date"].min().date()
        pv_max = pv_df["date"].max().date()
        pv_dates = st.date_input("Date range",
            value=(pv_min, pv_max), min_value=pv_min, max_value=pv_max, key="pv_dates")

    elif active == "Training quality":
        st.markdown('<span class="sidebar-heading">Training quality</span>', unsafe_allow_html=True)
        tq_countries = st.multiselect("Country",
            options=sorted(per_model_df["country"].dropna().unique()) if not per_model_df.empty else [],
            default=sorted(per_model_df["country"].dropna().unique()) if not per_model_df.empty else [],
            key="tq_country")
        tq_splits = st.multiselect("Split",
            options=ordered_unique(per_model_df["split"]) if not per_model_df.empty else [],
            default=ordered_unique(per_model_df["split"]) if not per_model_df.empty else [],
            key="tq_split")

# ── Page content ──────────────────────────────────────────────────────────────
st.markdown('<div class="page-content">', unsafe_allow_html=True)

if active == "Overview":
    render_overview(predictions, metrics)

elif active == "Forecast view":
    render_predictions(predictions, selected_countries, selected_skus, selected_splits, selected_dates)

elif active == "Model drivers":
    render_insights(feature_importance, fi_country, fi_sku, fi_top_n)

elif active == "Portfolio":
    render_portfolio(pv_choice, pv_groups, pv_splits, pv_dates)

elif active == "Training quality":
    render_training(metrics, tq_countries, tq_splits)

st.markdown('</div>', unsafe_allow_html=True)
