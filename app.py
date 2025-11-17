import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SOPL 2025 - Partnership Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# GOOGLE SHEETS CONFIG (Option B — RECOMMENDED)
# ============================================================
GOOGLE_SHEET_CSV_URL = st.secrets["gsheet_url"]  # <- YOU ALREADY ADDED THIS TO SECRETS


# ============================================================
# CSS FIX: LIGHT THEME + FIX FULLSCREEN ICON + READABLE TEXT
# ============================================================
st.markdown("""
<style>

html, body, .stApp { background: #ffffff !important; }

[data-testid="stSidebar"] { background: #f6f7fb !important; }

/* Fix ALL text color to dark */
.stApp, .stApp * { color: #0f172a !important; }

/* Make Altair/Vega menu icons LIGHT instead of black */
.vega-embed summary svg {
    stroke: #0f172a !important;    /* icon color */
    fill: #0f172a !important;
}
.vega-embed .vega-actions {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}
.vega-embed .vega-actions a {
    color: #0f172a !important;
    font-weight: 500 !important;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 24px;
}

.main-header {
    font-size: 2rem;
    font-weight: 800;
}

.sub-header {
    color: #64748b !important;
    font-size: .9rem;
}

.chart-caption {
    color: #64748b !important;
    font-size: .8rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ALTAIR LIGHT THEME
# ============================================================
def atlas_light_theme():
    return {
        "config": {
            "background": "#ffffff",
            "view": {"stroke": "transparent"},
            "range": {
                "category": ["#3b308f", "#4f46e5", "#1d4ed8", "#0f766e", "#0891b2"]
            },
            "axis": {
                "labelColor": "#334155",
                "titleColor": "#0f172a",
                "gridColor": "#e2e8f0",
                "domainColor": "#cbd5e1"
            },
            "legend": {
                "labelColor": "#334155",
                "titleColor": "#0f172a"
            },
            "title": {
                "color": "#0f172a",
                "fontSize": 16,
                "fontWeight": 700
            }
        }
    }

alt.themes.register("atlas_light", atlas_light_theme)
alt.themes.enable("atlas_light")
alt.renderers.set_embed_options(actions={"export": True, "source": False})


# ============================================================
# LOAD DATA FROM GOOGLE SHEETS
# ============================================================
@st.cache_data(show_spinner=True)
def load_data():
    encodings = ["utf-8", "utf-8-sig", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(GOOGLE_SHEET_CSV_URL, encoding=enc)
        except:
            continue
    st.error("❌ Could not load Google Sheet. Check your secrets.toml URL.")
    return pd.DataFrame()


# ============================================================
# UTILITIES
# ============================================================
def value_counts_pct(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["category", "percent"])
    pct = (s.value_counts() / len(s)) * 100
    df = pct.reset_index()
    df.columns = ["category", "percent"]
    return df


def donut_chart(df_pct, cat_field, val_field, title):
    if df_pct.empty:
        st.info("No data for this chart.")
        return

    df = df_pct.copy()
    df[val_field] = df[val_field].round(1)

    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{val_field}:Q"),
        color=alt.Color(f"{cat_field}:N", legend=alt.Legend(title=None))
    )

    donut = base.mark_arc(innerRadius=70)
    text = base.mark_text(
        radius=110,
        size=14,
        color="#0f172a"
    ).encode(text=alt.Text(f"{val_field}:Q", format=".1f"))

    chart = (donut + text).properties(width=380, height=380, title=title)
    st.altair_chart(chart, use_container_width=True)


def bar_pct(df_pct, cat_field, val_field, title):
    if df_pct.empty:
        st.info("No data for this chart.")
        return

    df = df_pct.copy()
    df[val_field] = df[val_field].round(1)

    base = alt.Chart(df).encode(
        x=alt.X(f"{val_field}:Q", title="Percent (%)", axis=alt.Axis(format=".0f")),
        y=alt.Y(f"{cat_field}:N", sort="-x", title=None)
    )

    bars = base.mark_bar(color="#3b308f")
    labels = base.mark_text(align="left", baseline="middle", dx=4).encode(
        text=alt.Text(f"{val_field}:Q", format=".1f")
    )

    chart = (bars + labels).properties(
        height=max(300, len(df) * 30),
        title=title
    )
    st.altair_chart(chart, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-header">SOPL 2025 Insights Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Partnership analytics and strategic insights</div>', unsafe_allow_html=True)
    st.write("")

    df = load_data()
    if df.empty:
        st.stop()

    # Column references
    COL_REGION = "Please select the region where your company is headquartered."
    COL_REVENUE = "What is your company’s estimated annual revenue?"
    COL_EMP = "What is your company’s total number of employees?"
    COL_INDUSTRY = "What industry sector does your company operate in?"
    COL_DEAL = "How does your average deal size involving partners compare to direct or non-partner deals?"
    COL_CAC = "How does your customer acquisition cost (CAC) from partners compared to direct sales and marketing?"
    COL_SCYCLE = "How does your partner-led sales cycle compare to your direct sales cycle?"
    COL_WIN = "What’s your win rate for deals where partners are involved?"

    # Normalize region labels
    def norm_region(x):
        if pd.isna(x): return None
        s = str(x)
        if "North America" in s: return "North America"
        if "Latin America" in s: return "Latin America"
        if "Asia" in s or "APAC" in s: return "Asia Pacific"
        if "Europe" in s or "EMEA" in s: return "Europe"
        return s

    df["RegionStd"] = df[COL_REGION].map(norm_region)

    # ---------------- Filters ----------------
    st.sidebar.header("Filters")

    regions = sorted(df["RegionStd"].dropna().unique())
    sel_regions = st.sidebar.multiselect("Region", regions, default=regions)

    revenues = sorted(df[COL_REVENUE].dropna().unique())
    sel_revenues = st.sidebar.multiselect("Revenue", revenues, default=revenues)

    employees = sorted(df[COL_EMP].dropna().unique())
    sel_emp = st.sidebar.multiselect("Company Size", employees, default=employees)

    flt = df.copy()
    flt = flt[flt["RegionStd"].isin(sel_regions)]
    flt = flt[flt[COL_REVENUE].isin(sel_revenues)]
    flt = flt[flt[COL_EMP].isin(sel_emp)]

    st.caption(f"Responses in view: {len(flt)}")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Performance", "Geography", "Partner & Impact", "Data"]
    )

    # ============================================================
    # TAB 1 — OVERVIEW
    # ============================================================
    with tab1:
        st.markdown('<div class="section-header">Company Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            donut_chart(
                value_counts_pct(flt["RegionStd"]),
                "category", "percent",
                "Regional Distribution (%)"
            )

        with c2:
            bar_pct(
                value_counts_pct(flt[COL_REVENUE]),
                "category", "percent",
                "Revenue Bands (%)"
            )

        c3, c4 = st.columns(2)

        with c3:
            bar_pct(
                value_counts_pct(flt[COL_EMP]),
                "category", "percent",
                "Company Size (%)"
            )

        with c4:
            bar_pct(
                value_counts_pct(flt[COL_INDUSTRY]),
                "category", "percent",
                "Industry Distribution (%)"
            )

    # ============================================================
    # TAB 2 — PERFORMANCE
    # ============================================================
    with tab2:
        st.markdown('<div class="section-header">Performance vs Direct</div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)

        with p1:
            bar_pct(value_counts_pct(flt[COL_DEAL]), "category", "percent",
                    "Deal Size vs Direct (%)")

        with p2:
            bar_pct(value_counts_pct(flt[COL_CAC]), "category", "percent",
                    "CAC vs Direct (%)")

        st.markdown('<div class="section-header">Sales Cycle & Win Rate</div>', unsafe_allow_html=True)

        p3, p4 = st.columns(2)

        with p3:
            bar_pct(value_counts_pct(flt[COL_SCYCLE]), "category", "percent",
                    "Sales Cycle vs Direct (%)")

        with p4:
            # Win-rate binned chart (10% bands)
            win = flt[COL_WIN].dropna()
            if len(win) > 0:
                bins = list(range(0, 101, 10))
                labels = [f"{b}-{b+10}%" for b in bins[:-1]]
                binned = pd.cut(win, bins=bins, labels=labels, include_lowest=True)
                bar_pct(value_counts_pct(binned), "category", "percent",
                        "Win Rate Distribution (%)")

    # ============================================================
    # TAB 3 — GEOGRAPHY
    # ============================================================
    with tab3:
        st.markdown('<div class="section-header">Geographic Mix</div>', unsafe_allow_html=True)

        region_pct = value_counts_pct(flt["RegionStd"])

        g1, g2 = st.columns(2)

        with g1:
            donut_chart(region_pct, "category", "percent", "Region (%)")

        with g2:
            bar_pct(region_pct, "category", "percent", "Region (%)")

        st.markdown('<div class="section-header">World Map (%)</div>', unsafe_allow_html=True)

        coords = {
            "North America": (40, -100),
            "Latin America": (-15, -60),
            "Europe": (50, 10),
            "Asia Pacific": (15, 100)
        }

        map_df = region_pct.copy()
        map_df["lat"] = map_df["category"].map(lambda x: coords.get(x, (None, None))[0])
        map_df["lon"] = map_df["category"].map(lambda x: coords.get(x, (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"])

        if len(map_df) > 0:
            layer = pdk.Layer(
                "ScatterplotLayer",
                map_df,
                get_position=["lon", "lat"],
                get_fill_color=[59, 48, 143, 180],
                get_radius="percent * 120000",
                pickable=True,
            )

            view = pdk.ViewState(latitude=20, longitude=0, zoom=1)
            deck = pdk.Deck(layers=[layer], initial_view_state=view,
                            tooltip={"text": "{category}: {percent}%"} )
            st.pydeck_chart(deck)

    # ============================================================
    # TAB 4 — MULTI-SELECT QUESTIONS
    # ============================================================
    with tab4:
        st.markdown('<div class="section-header">Partner & Impact</div>', unsafe_allow_html=True)
        st.info("Send me your exact column names for multi-select questions and I will plug them in cleanly.")

    # ============================================================
    # TAB 5 — DATA
    # ============================================================
    with tab5:
        st.dataframe(flt, use_container_width=True)


    # ============================================================
    # PICKAXE ASSISTANT
    # ============================================================
    st.markdown("<h2>Assistant (SOPL Q&A)</h2>", unsafe_allow_html=True)

    components.html("""
    <div id="deployment-5870ff7d-8fcf-4395-976b-9e9fdefbb0ff"></div>
    <script src="https://studio.pickaxe.co/api/embed/bundle.js" defer></script>
    """, height=600)


if __name__ == "__main__":
    main()
