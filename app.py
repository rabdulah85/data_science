import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, urllib.request, os
import numpy as np
from scipy import stats

GRID_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "idn_grid")

st.set_page_config(
    page_title="GDP Indonesia 514 Districts",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

HOVER = dict(bgcolor="white", font_color="#1a2b3c", font_size=12, bordercolor="#e2e8f0")
FONT  = dict(color="#1a2b3c")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f7f9fc !important;
        color: #1a2b3c !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #1a2b3c !important;
        background-color: transparent !important;
    }
    [data-baseweb="tag"] {
        background-color: #dbeafe !important;
        color: #1a3a5c !important;
        border: 1px solid #93c5fd !important;
    }
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stDownloadButton > button {
        background-color: #f0f4f8 !important;
        color: #1a3a5c !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }
    .stDownloadButton > button:hover {
        background-color: #e2e8f0 !important;
        color: #0f6b8a !important;
    }
    .stToggle label { color: #4a5568 !important; }
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f6b8a 100%);
        padding: 1.8rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .main-header h1, .main-header p { color: white !important; background-color: transparent !important; }
    .main-header h1 { font-family: 'DM Serif Display', serif; font-size: 1.9rem; margin: 0 0 0.3rem 0; }
    .main-header p  { font-size: 0.85rem; opacity: 0.9; margin: 0; }
    .kpi-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1.1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center;
    }
    .kpi-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #6b7c93; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1a3a5c; line-height: 1.1; }
    .kpi-sub   { font-size: 0.75rem; color: #2ca86e; margin-top: 0.2rem; }
    .section-title {
        font-size: 0.95rem; font-weight: 600; color: #1a3a5c;
        border-left: 3px solid #0f6b8a; padding-left: 0.75rem; margin: 1.5rem 0 0.8rem 0;
    }
    [data-testid="stAlert"] {
        background-color: #eff6ff !important; color: #1a2b3c !important;
        border: 1px solid #bfdbfe !important;
    }
    [data-testid="stAlert"] * { color: #1a2b3c !important; }
    .js-plotly-plot .plotly text { fill: #1a2b3c !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url_csv = "https://raw.githubusercontent.com/quarcs-lab/indonesia514/main/gdp/gdp.csv"
    df = pd.read_csv(url_csv)
    df["districtID"] = df["districtID"].astype(str)
    gdp_cols = [c for c in df.columns if c.startswith("gdp_")]
    df_long = df.melt(
        id_vars=["districtID","district_bahasa","district_en",
                 "province_bahasa","province_en","island_bahasa","island_en"],
        value_vars=gdp_cols, var_name="tahun", value_name="gdp"
    )
    df_long["tahun"] = df_long["tahun"].str.replace("gdp_","").astype(int)
    return df, df_long

@st.cache_data
def load_geojson():
    url_geo = "https://raw.githubusercontent.com/quarcs-lab/indonesia514/main/maps/mapIndonesia514_new.geojson"
    import time
    for attempt in range(3):
        try:
            req = urllib.request.Request(url_geo, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                geo = json.loads(r.read())
            return geo
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise e

@st.cache_data
def load_grid_data():
    grid = pd.read_csv(os.path.join(GRID_DATA_DIR, "grid_cells.csv"))
    adm1 = pd.read_csv(os.path.join(GRID_DATA_DIR, "adm1.csv"))
    adm0 = pd.read_csv(os.path.join(GRID_DATA_DIR, "adm0.csv"))
    return grid, adm1, adm0

df, df_long = load_data()
geojson   = load_geojson()
gdp_years = sorted([int(c.replace("gdp_","")) for c in df.columns if c.startswith("gdp_")])
grid_df, adm1_df, adm0_df = load_grid_data()
grid_years = sorted(grid_df["year"].unique().tolist())

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Navigation")
    page = st.radio("", ["📖 Overview","🏠 Home","🗺️ Map","📈 Time Series","🏆 Ranking","📊 Descriptive","📉 Convergence","🔲 Grid GDP","📋 Data"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Filter")

    island_list = ["All"] + sorted(df["island_en"].dropna().unique().tolist())
    island_sel = st.selectbox("Island", island_list)

    if island_sel != "All":
        prov_options = sorted(df[df["island_en"] == island_sel]["province_en"].unique().tolist())
    else:
        prov_options = sorted(df["province_en"].unique().tolist())
    prov_sel = st.multiselect("Province", prov_options, default=prov_options)

    if prov_sel:
        dist_options = sorted(df[df["province_en"].isin(prov_sel)]["district_en"].unique().tolist())
    else:
        dist_options = sorted(df["district_en"].unique().tolist())
    dist_sel = st.multiselect("District", dist_options, default=[])

    tahun_sel = st.select_slider("Year", options=gdp_years, value=2024)
    st.markdown("---")
    st.markdown("<small style='color:#6b7c93'>Source: quarcs-lab/indonesia514<br>514 Districts · 2010–2025</small>",
                unsafe_allow_html=True)

# ── FILTER ────────────────────────────────────
col_gdp = f"gdp_{tahun_sel}"

if dist_sel:
    df_fil = df[df["district_en"].isin(dist_sel)].copy()
elif prov_sel:
    df_fil = df[df["province_en"].isin(prov_sel)].copy()
elif island_sel != "All":
    df_fil = df[df["island_en"] == island_sel].copy()
else:
    df_fil = df.copy()

if dist_sel:
    df_long_f = df_long[df_long["district_en"].isin(dist_sel)]
elif prov_sel:
    df_long_f = df_long[df_long["province_en"].isin(prov_sel)]
elif island_sel != "All":
    df_long_f = df_long[df_long["island_en"] == island_sel]
else:
    df_long_f = df_long

# ── HOME ──────────────────────────────────────
if "Overview" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇩 GDP Indonesia — 514 Districts Dashboard</h1>
        <p>Regional Economic Analysis · 2010–2025 · Institute for Development of Economics and Finance (INDEF)</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">About This Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard visualizes **Gross Domestic Product (GDP) data** for all **514 districts and cities** 
    across Indonesia, covering the period **2010 to 2025**. It is designed to support 
    regional economic research, policy analysis, and data-driven decision making.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">📦 Data</p>', unsafe_allow_html=True)
        st.markdown("""
        - **Source:** [quarcs-lab/indonesia514](https://github.com/quarcs-lab/indonesia514) (GitHub)
        - **Coverage:** 514 kabupaten/kota across 38 provinces and 7 islands
        - **Period:** 2010–2025 (16 years)
        - **GDP measure:** Real GDP at **2010 constant prices** (base year 2010)
        - **Unit:** Billion Rupiah (Miliar Rp)
        - **Update:** Data sourced directly from BPS via the indonesia514 repository
        """)

        st.markdown('<p class="section-title">📐 Methodology</p>', unsafe_allow_html=True)
        st.markdown("""
        - All GDP values are expressed in **real terms** using 2010 as the base year, 
          removing the effect of price changes (inflation) to allow meaningful comparisons over time.
        - **Log transformation** is applied in convergence analysis to reduce skewness 
          caused by large disparities between districts.
        - **Growth rates** are computed as percentage change from 2010 to 2025.
        - **OLS regression** is used to test β-convergence (slope of initial GDP vs growth rate).
        """)

    with col2:
        st.markdown('<p class="section-title">🗂️ Dashboard Pages</p>', unsafe_allow_html=True)
        st.markdown("""
        | Page | Description |
        |------|-------------|
        | 🏠 **Home** | KPI indicators, GDP by province and island share |
        | 🗺️ **Map** | Choropleth map of GDP across 514 districts |
        | 📈 **Time Series** | GDP trends and growth rates over time |
        | 🏆 **Ranking** | Top 20 districts and GDP vs growth scatter |
        | 📊 **Descriptive** | Summary stats, distribution, heatmap, transition dynamics |
        | 📉 **Convergence** | σ-convergence, β-convergence, distribution dynamics |
        | 🔲 **Grid GDP** | 0.25° gridded GDP, PWT-rescaled, 2012–2022 |
        | 📋 **Data** | Data source information & official links |
        """)

        st.markdown('<p class="section-title">🔍 Key Findings</p>', unsafe_allow_html=True)

        # Compute quick stats
        gdp_2010_total = df["gdp_2010"].sum()
        gdp_2025_total = df["gdp_2025"].sum()
        growth_total   = (gdp_2025_total - gdp_2010_total) / gdp_2010_total * 100
        top_dist       = df.loc[df["gdp_2025"].idxmax(), "district_en"]
        bot_dist       = df.loc[df["gdp_2025"].idxmin(), "district_en"]
        ratio          = df["gdp_2025"].max() / df["gdp_2025"].min()

        st.markdown(f"""
        - Total real GDP grew from **{gdp_2010_total/1e6:,.1f}T** (2010) to **{gdp_2025_total/1e6:,.1f}T** (2025) 
          — an increase of **{growth_total:.1f}%** over 15 years
        - Highest GDP district (2025): **{top_dist}**
        - Lowest GDP district (2025): **{bot_dist}**
        - Max/Min ratio: **{ratio:,.0f}x** — indicating extreme regional disparity
        - **β-divergence** detected: districts with higher initial GDP tend to grow faster,
          suggesting regional inequality is widening
        """)

    st.markdown("---")
    st.markdown('<p class="section-title">⚙️ How to Use</p>', unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("""
        **🔽 Filter (Sidebar)**  
        Select by Island → Province → District to zoom into specific regions.
        Use the Year slider to change the reference year for all charts.
        """)
    with u2:
        st.markdown("""
        **📥 Download Charts**  
        Each chart has a Download button to export as interactive HTML — 
        ready to embed in WordPress or share via email.
        """)
    with u3:
        st.markdown("""
        **📉 Convergence Analysis**  
        Use Log GDP toggle for standard convergence analysis.
        OLS results show whether poorer districts are catching up (β-convergence)
        or falling further behind (β-divergence).
        """)

    st.markdown("---")
    st.markdown("---")
    st.markdown("""
    <small>
    📊 <b>Primary Data Source:</b> 
    <a href="https://www.bps.go.id/id/statistics-table/2/MjE5NCMy/-seri-2010--pdrb-atas-dasar-harga-konstan--2010-100--menurut-pengeluaran-kabupaten-kota--milyar-rupiah-.html" target="_blank">
    [SERI 2010] PDRB Atas Dasar Harga Konstan (2010=100) Menurut Pengeluaran Kabupaten/Kota — Badan Pusat Statistik Indonesia
    </a><br>
    🗂️ <b>Dataset Repository:</b> <a href="https://github.com/quarcs-lab/indonesia514" target="_blank">quarcs-lab/indonesia514</a> · 
    Built with Streamlit & Plotly · INDEF 2025
    </small>
    """, unsafe_allow_html=True)


elif "Home" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇩 GDP Indonesia — 514 Districts</h1>
        <p>District-level GDP data · 2010–2025 · Source: quarcs-lab/indonesia514</p>
    </div>""", unsafe_allow_html=True)

    total_gdp = df_fil[col_gdp].sum()
    rata_gdp  = df_fil[col_gdp].mean()
    tertinggi = df_fil.loc[df_fil[col_gdp].idxmax(), "district_en"]
    n_kab     = len(df_fil)

    st.markdown('<p class="section-title">Key Indicators</p>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total GDP {tahun_sel}</div>
            <div class="kpi-value">{total_gdp/1e6:,.1f}T</div>
            <div class="kpi-sub">Trillion Rupiah</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Average GDP</div>
            <div class="kpi-value">{rata_gdp:,.0f}</div>
            <div class="kpi-sub">Billion Rp / District</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Highest GDP</div>
            <div class="kpi-value" style="font-size:1rem">{tertinggi}</div>
            <div class="kpi-sub">{df_fil[col_gdp].max():,.0f} M Rp</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Number of Districts</div>
            <div class="kpi-value">{n_kab}</div>
            <div class="kpi-sub">out of 514 total</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    c1,c2 = st.columns([3,2])
    with c1:
        view_opt = st.radio("View by", ["Province", "Top 20 Districts"], horizontal=True)
        if view_opt == "Province":
            st.markdown('<p class="section-title">GDP per Province</p>', unsafe_allow_html=True)
            df_prov = df_fil.groupby("province_en")[col_gdp].sum().reset_index().sort_values(col_gdp, ascending=True)
            fig = px.bar(df_prov, x=col_gdp, y="province_en", orientation="h",
                         color=col_gdp, color_continuous_scale="Blues",
                         labels={col_gdp:"GDP (M Rp)","province_en":""}, template="plotly_white")
            fig.update_layout(height=max(400, len(df_prov)*22),
                              margin=dict(l=5,r=5,t=10,b=10),
                              coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white",
                              font=FONT, hoverlabel=HOVER)
        else:
            st.markdown('<p class="section-title">Top 20 Districts by GDP</p>', unsafe_allow_html=True)
            df_top = df_fil.nlargest(20, col_gdp)[["district_en","province_en",col_gdp]].sort_values(col_gdp, ascending=True)
            fig = px.bar(df_top, x=col_gdp, y="district_en", orientation="h",
                         color="province_en",
                         labels={col_gdp:"GDP (M Rp)","district_en":"","province_en":"Province"},
                         template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=500, margin=dict(l=5,r=5,t=10,b=10),
                              paper_bgcolor="white", plot_bgcolor="white",
                              font=FONT, hoverlabel=HOVER)
        buf = fig.to_html(full_html=True, include_plotlyjs='cdn')
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇️ Download Bar Chart (HTML)", buf, f"gdp_{view_opt.lower().replace(' ','_')}_{tahun_sel}.html", "text/html")
    with c2:
        st.markdown('<p class="section-title">GDP Share per Island</p>', unsafe_allow_html=True)
        df_pulau = df_fil.groupby("island_en")[col_gdp].sum().reset_index()
        fig2 = px.pie(df_pulau, names="island_en", values=col_gdp, hole=0.45,
                      template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(height=380, margin=dict(l=5,r=5,t=10,b=10),
                           legend_title_text="Island", paper_bgcolor="white",
                           font=FONT, hoverlabel=HOVER)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        buf2 = fig2.to_html(full_html=True, include_plotlyjs='cdn')
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("⬇️ Download Pie Chart (HTML)", buf2, f"gdp_island_{tahun_sel}.html", "text/html")

# ── MAP ───────────────────────────────────────
elif "Map" in page:
    st.markdown(f'<p class="section-title">🗺️ Choropleth Map of GDP by District — {tahun_sel}</p>',
                unsafe_allow_html=True)
    col_opt, col_log = st.columns([3,1])
    with col_opt:
        skala_warna = st.selectbox("Color Scale", ["Blues","YlOrRd","Viridis","RdYlGn","Plasma"], index=0)
    with col_log:
        log_scale = st.checkbox("Log Scale", value=True)

    df_map = df[["districtID","district_en","province_en","island_en",col_gdp]].copy()
    df_map["gdp_plot"] = df_map[col_gdp].clip(lower=1)
    if log_scale:
        df_map["gdp_display"] = np.log10(df_map["gdp_plot"])
        label_color = f"Log10 GDP {tahun_sel}"
    else:
        df_map["gdp_display"] = df_map["gdp_plot"]
        label_color = f"GDP {tahun_sel} (M Rp)"

    fig_map = px.choropleth_mapbox(
        df_map, geojson=geojson,
        locations="districtID", featureidkey="properties.districtID",
        color="gdp_display", color_continuous_scale=skala_warna,
        mapbox_style="carto-positron",
        zoom=3.8, center={"lat": -2.5, "lon": 118}, opacity=0.75,
        hover_name="district_en",
        hover_data={"province_en":True,"island_en":True,
                    col_gdp:":,.0f","gdp_display":False,"districtID":False},
        labels={"gdp_display":label_color,"province_en":"Province",
                "island_en":"Island",col_gdp:f"GDP {tahun_sel} (M Rp)"}
    )
    fig_map.update_layout(height=560, margin=dict(l=0,r=0,t=0,b=0),
                          coloraxis_colorbar=dict(title=label_color, thickness=14, len=0.6),
                          paper_bgcolor="white", font=FONT, hoverlabel=HOVER)
    st.plotly_chart(fig_map, use_container_width=True)
    buf_map = fig_map.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("⬇️ Download Map (HTML)", buf_map, f"gdp_map_{tahun_sel}.html", "text/html")

    st.markdown('<p class="section-title">Summary Statistics</p>', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.metric("Highest GDP", f"{df_map[col_gdp].max():,.0f} M Rp",
                  df_map.loc[df_map[col_gdp].idxmax(),"district_en"])
    with s2:
        st.metric("Lowest GDP", f"{df_map[col_gdp].min():,.0f} M Rp",
                  df_map.loc[df_map[col_gdp].idxmin(),"district_en"])
    with s3:
        st.metric("Median GDP", f"{df_map[col_gdp].median():,.0f} M Rp")
    with s4:
        ratio = df_map[col_gdp].max() / df_map[col_gdp].min()
        st.metric("Max/Min Ratio", f"{ratio:,.0f}x", "Disparity Measure")

# ── TIME SERIES ───────────────────────────────
elif "Time" in page:
    st.markdown('<p class="section-title">📈 GDP Trend per Province (2010–2025)</p>', unsafe_allow_html=True)
    if df_long_f.empty:
        st.warning("No data — adjust filters in the sidebar.")
    else:
        df_trend = df_long_f.groupby(["tahun","province_en"])["gdp"].sum().reset_index()
        fig = px.line(df_trend, x="tahun", y="gdp", color="province_en", markers=True,
                      labels={"gdp":"GDP (M Rp)","tahun":"Year","province_en":"Province"},
                      template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=460, hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white",
                          font=FONT, hoverlabel=HOVER,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(fig, use_container_width=True)
        buf_ts = fig.to_html(full_html=True, include_plotlyjs='cdn')
        st.download_button("⬇️ Download Time Series (HTML)", buf_ts, "gdp_timeseries.html", "text/html")

    st.markdown('<p class="section-title">GDP Growth 2010 → 2025 (Top 20 Districts)</p>', unsafe_allow_html=True)
    if not df_fil.empty:
        df_growth = df_fil[["district_en","province_en","gdp_2010","gdp_2025"]].copy()
        df_growth["growth_pct"] = ((df_growth["gdp_2025"]-df_growth["gdp_2010"])/df_growth["gdp_2010"]*100).round(1)
        df_growth = df_growth.sort_values("growth_pct", ascending=False).head(20)
        fig3 = px.bar(df_growth, x="district_en", y="growth_pct", color="province_en",
                      template="plotly_white",
                      labels={"growth_pct":"Growth (%)","district_en":"District","province_en":"Province"},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=380, xaxis_tickangle=-35, paper_bgcolor="white", plot_bgcolor="white",
                           font=FONT, hoverlabel=HOVER,
                           legend_title_text="Province", margin=dict(b=80))
        st.plotly_chart(fig3, use_container_width=True)

# ── RANKING ───────────────────────────────────
elif "Ranking" in page:
    st.markdown(f'<p class="section-title">🏆 Top 20 Districts — GDP {tahun_sel}</p>', unsafe_allow_html=True)
    top20 = df_fil.nlargest(20, col_gdp)[["district_en","province_en","island_en",col_gdp]].copy()
    fig4 = px.bar(top20.sort_values(col_gdp), x=col_gdp, y="district_en",
                  color="island_en", orientation="h", template="plotly_white",
                  labels={col_gdp:f"GDP {tahun_sel} (M Rp)","district_en":"","island_en":"Island"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(height=520, margin=dict(l=5,r=5,t=10,b=10),
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=FONT, hoverlabel=HOVER)
    st.plotly_chart(fig4, use_container_width=True)
    buf4 = fig4.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("⬇️ Download Ranking Chart (HTML)", buf4, f"gdp_ranking_{tahun_sel}.html", "text/html")

    st.markdown('<p class="section-title">Scatter: GDP vs Growth 2010–2025</p>', unsafe_allow_html=True)
    df_sc = df_fil[["district_en","province_en","island_en","gdp_2010",col_gdp]].copy()
    df_sc["growth"] = ((df_sc[col_gdp]-df_sc["gdp_2010"])/df_sc["gdp_2010"]*100).round(1)
    fig5 = px.scatter(df_sc, x=col_gdp, y="growth", color="island_en",
                      hover_name="district_en", hover_data={"province_en":True},
                      labels={col_gdp:f"GDP {tahun_sel} (M Rp)","growth":"Growth 2010–2025 (%)"},
                      template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.75)
    fig5.update_layout(height=420, legend_title_text="Island",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=FONT, hoverlabel=HOVER)
    st.plotly_chart(fig5, use_container_width=True)

# ── CONVERGENCE ───────────────────────────────
elif "Convergence" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📉 Convergence Analysis</h1>
        <p>GDP convergence across 514 Indonesian districts · 2010–2025</p>
    </div>""", unsafe_allow_html=True)

    use_log = st.toggle("Use Log GDP", value=True,
                        help="Log transformation reduces skewness — standard in convergence analysis")

    # ① SIGMA CONVERGENCE
    st.markdown('<p class="section-title">① σ-Convergence — Is GDP disparity narrowing over time?</p>',
                unsafe_allow_html=True)
    st.caption("σ-convergence occurs when the standard deviation of GDP across districts **decreases** over time.")

    sigma_data = []
    for y in gdp_years:
        vals = df_fil[f"gdp_{y}"].replace(0, np.nan).dropna()
        if use_log:
            vals = np.log(vals)
        sigma_data.append({"Year": y, "Std Dev": vals.std(), "CV": vals.std()/vals.mean()})
    df_sigma = pd.DataFrame(sigma_data)

    c1, c2 = st.columns(2)
    with c1:
        fig_sigma = px.line(df_sigma, x="Year", y="Std Dev", markers=True,
                            title=f"Standard Deviation of {'Log ' if use_log else ''}GDP",
                            template="plotly_white", color_discrete_sequence=["#0f6b8a"])
        fig_sigma.update_traces(line_width=2.5, marker_size=7)
        fig_sigma.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="white",
                                font=FONT, hoverlabel=HOVER, title_font_size=13)
        trend = "↑ Diverging" if df_sigma["Std Dev"].iloc[-1] > df_sigma["Std Dev"].iloc[0] else "↓ Converging"
        color = "red" if "Diverging" in trend else "green"
        fig_sigma.add_annotation(x=gdp_years[-1], y=df_sigma["Std Dev"].iloc[-1],
                                 text=trend, showarrow=True, arrowhead=2,
                                 font_size=11, font_color=color)
        st.plotly_chart(fig_sigma, use_container_width=True)

    with c2:
        fig_cv = px.line(df_sigma, x="Year", y="CV", markers=True,
                         title=f"Coefficient of Variation of {'Log ' if use_log else ''}GDP",
                         template="plotly_white", color_discrete_sequence=["#e05c5c"])
        fig_cv.update_traces(line_width=2.5, marker_size=7)
        fig_cv.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="white",
                             font=FONT, hoverlabel=HOVER, title_font_size=13)
        st.plotly_chart(fig_cv, use_container_width=True)

    delta = df_sigma["Std Dev"].iloc[-1] - df_sigma["Std Dev"].iloc[0]
    direction = "increased" if delta > 0 else "decreased"
    verdict = "**diverging** (σ-divergence)" if delta > 0 else "**converging** (σ-convergence)"
    st.info(f"📊 Std Dev of {'log ' if use_log else ''}GDP has {direction} from "
            f"**{df_sigma['Std Dev'].iloc[0]:.3f}** (2010) to **{df_sigma['Std Dev'].iloc[-1]:.3f}** (2025) → {verdict}.")

    st.markdown("---")

    # ② BETA CONVERGENCE
    st.markdown('<p class="section-title">② β-Convergence — Do poorer districts grow faster?</p>',
                unsafe_allow_html=True)
    st.caption("β-convergence: districts with **lower initial GDP grow faster** — negative slope = convergence.")

    df_beta = df_fil[["districtID","district_en","province_en","island_en","gdp_2010","gdp_2025"]].copy()
    df_beta = df_beta.replace(0, np.nan).dropna()
    df_beta["growth_rate"] = ((df_beta["gdp_2025"] - df_beta["gdp_2010"]) / df_beta["gdp_2010"] * 100)
    df_beta["initial_gdp"] = np.log(df_beta["gdp_2010"]) if use_log else df_beta["gdp_2010"]
    x_label = "Log GDP 2010 (Initial Level)" if use_log else "GDP 2010 (Initial Level, M Rp)"

    fig_beta = px.scatter(df_beta, x="initial_gdp", y="growth_rate",
                          color="island_en", hover_name="district_en",
                          hover_data={"province_en": True, "initial_gdp": ":.2f", "growth_rate": ":.1f"},
                          trendline="ols",
                          labels={"initial_gdp": x_label, "growth_rate": "GDP Growth Rate 2010–2025 (%)",
                                  "island_en": "Island"},
                          template="plotly_white",
                          color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.65)
    fig_beta.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white",
                           font=FONT, hoverlabel=HOVER)
    st.plotly_chart(fig_beta, use_container_width=True)

    slope, intercept, r, p, se = stats.linregress(df_beta["initial_gdp"], df_beta["growth_rate"])
    beta_verdict = "β-convergence ✅" if slope < 0 else "β-divergence ⚠️"
    st.info(f"📊 OLS slope = **{slope:.3f}** (p = {p:.4f}, R² = {r**2:.3f}) → **{beta_verdict}**. "
            f"{'Districts with lower initial GDP tend to grow faster.' if slope < 0 else 'Districts with higher initial GDP tend to grow faster.'}")

    buf_conv = fig_beta.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("⬇️ Download β-Convergence Chart (HTML)", buf_conv, "beta_convergence.html", "text/html")

    st.markdown("---")

    # ③ DISTRIBUTION DYNAMICS
    st.markdown('<p class="section-title">③ Distribution Dynamics — How has the GDP distribution shifted?</p>',
                unsafe_allow_html=True)
    st.caption("Tracks how the full distribution of GDP across districts has evolved from 2010 to 2025.")

    years_compare = st.multiselect("Select years to compare", gdp_years, default=[2010, 2015, 2020, 2025])
    if years_compare:
        df_dist = []
        for y in years_compare:
            vals = df_fil[f"gdp_{y}"].replace(0, np.nan).dropna()
            if use_log:
                vals = np.log(vals)
            df_dist.append(pd.DataFrame({"GDP": vals, "Year": str(y)}))
        df_dist = pd.concat(df_dist)

        fig_dist = px.histogram(df_dist, x="GDP", color="Year", barmode="overlay",
                                nbins=40, opacity=0.6,
                                labels={"GDP": f"{'Log ' if use_log else ''}GDP", "Year": "Year"},
                                template="plotly_white",
                                color_discrete_sequence=px.colors.qualitative.Set1)
        fig_dist.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white",
                               font=FONT, hoverlabel=HOVER)
        st.plotly_chart(fig_dist, use_container_width=True)

# ── DESCRIPTIVE STATISTICS ────────────────────
elif "Descriptive" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Descriptive Statistics</h1>
        <p>Exploratory analysis of district GDP · distribution, correlation & transition dynamics</p>
    </div>""", unsafe_allow_html=True)

    if df_fil.empty:
        st.warning("No data — adjust the filters in the sidebar.")
    else:
        use_log = st.toggle("Use Log GDP", value=True,
                            help="Log transformation reduces skewness — recommended for GDP distributions")
        gdp_cols = [f"gdp_{y}" for y in gdp_years]

        # ── ① SUMMARY STATISTICS ──────────────────
        st.markdown('<p class="section-title">① Summary Statistics — GDP by year (M Rp)</p>',
                    unsafe_allow_html=True)
        st.caption("Distribution of GDP across the selected districts, computed for every year.")

        summ = df_fil[gdp_cols].describe().T
        summ["skew"] = df_fil[gdp_cols].skew().values
        summ["kurtosis"] = df_fil[gdp_cols].kurt().values
        summ.index = [i.replace("gdp_", "") for i in summ.index]
        summ.index.name = "Year"
        summ = summ.rename(columns={"count": "N", "mean": "Mean", "std": "Std Dev", "min": "Min",
                                    "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max",
                                    "skew": "Skewness", "kurtosis": "Kurtosis"})
        st.dataframe(summ.style.format({c: "{:,.1f}" for c in summ.columns}),
                     use_container_width=True, height=400)

        vals_sel = df_fil[col_gdp].replace(0, np.nan).dropna()
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Mean GDP {tahun_sel}</div>
                <div class="kpi-value">{vals_sel.mean():,.0f}</div>
                <div class="kpi-sub">Billion Rp</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Median GDP {tahun_sel}</div>
                <div class="kpi-value">{vals_sel.median():,.0f}</div>
                <div class="kpi-sub">Billion Rp</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Std Dev</div>
                <div class="kpi-value">{vals_sel.std():,.0f}</div>
                <div class="kpi-sub">Billion Rp</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Skewness</div>
                <div class="kpi-value">{vals_sel.skew():.2f}</div>
                <div class="kpi-sub">{'Right-skewed' if vals_sel.skew() > 0 else 'Left-skewed'}</div></div>""",
                unsafe_allow_html=True)

        st.markdown("---")

        # ── ② DISTRIBUTION ────────────────────────
        st.markdown(f'<p class="section-title">② Distribution — GDP {tahun_sel}</p>', unsafe_allow_html=True)
        st.caption("Histogram of the overall distribution and violin plots comparing islands.")

        vals = df_fil[col_gdp].replace(0, np.nan).dropna()
        plot_vals = np.log(vals) if use_log else vals
        x_label = f"{'Log ' if use_log else ''}GDP {tahun_sel}"
        df_d = pd.DataFrame({x_label: plot_vals.values,
                             "Island": df_fil.loc[vals.index, "island_en"].values})

        d1, d2 = st.columns(2)
        with d1:
            fig_hist = px.histogram(df_d, x=x_label, nbins=40, marginal="box",
                                    template="plotly_white", color_discrete_sequence=["#0f6b8a"],
                                    opacity=0.8)
            fig_hist.update_layout(height=400, paper_bgcolor="white", plot_bgcolor="white",
                                   font=FONT, hoverlabel=HOVER, bargap=0.05,
                                   title_text="Histogram + Box", title_font_size=13)
            st.plotly_chart(fig_hist, use_container_width=True)
            buf_h = fig_hist.to_html(full_html=True, include_plotlyjs='cdn')
            st.download_button("⬇️ Download Histogram (HTML)", buf_h, f"gdp_histogram_{tahun_sel}.html", "text/html")
        with d2:
            fig_vio = px.violin(df_d, x="Island", y=x_label, color="Island", box=True, points="outliers",
                                template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_vio.update_layout(height=400, paper_bgcolor="white", plot_bgcolor="white",
                                  font=FONT, hoverlabel=HOVER, showlegend=False,
                                  xaxis_tickangle=-25, title_text="Violin by Island", title_font_size=13)
            st.plotly_chart(fig_vio, use_container_width=True)
            buf_v = fig_vio.to_html(full_html=True, include_plotlyjs='cdn')
            st.download_button("⬇️ Download Violin (HTML)", buf_v, f"gdp_violin_{tahun_sel}.html", "text/html")

        st.markdown("---")

        # ── ③ HEATMAP — Island × Year ─────────────
        st.markdown('<p class="section-title">③ Heatmap — Mean GDP by Island × Year</p>',
                    unsafe_allow_html=True)
        st.caption("Each cell shows the average district GDP within an island, tracing growth over time.")

        heat = df_fil.groupby("island_en")[gdp_cols].mean()
        if use_log:
            heat = np.log(heat.replace(0, np.nan))
        heat.columns = [c.replace("gdp_", "") for c in heat.columns]
        heat = heat.sort_index()
        fig_heat = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                             text_auto=".1f",
                             labels=dict(x="Year", y="Island",
                                         color=f"{'Log ' if use_log else ''}Mean GDP"))
        fig_heat.update_layout(height=max(320, len(heat) * 45), paper_bgcolor="white",
                               font=FONT, hoverlabel=HOVER, margin=dict(l=5, r=5, t=10, b=10))
        fig_heat.update_xaxes(side="bottom")
        st.plotly_chart(fig_heat, use_container_width=True)
        buf_heat = fig_heat.to_html(full_html=True, include_plotlyjs='cdn')
        st.download_button("⬇️ Download Heatmap (HTML)", buf_heat, "gdp_heatmap_island_year.html", "text/html")

        st.markdown("---")

        # ── ④ TRANSITION DYNAMICS ─────────────────
        first_y, last_y = gdp_years[0], gdp_years[-1]
        st.markdown(f'<p class="section-title">④ Transition Dynamics — GDP mobility {first_y} → {last_y}</p>',
                    unsafe_allow_html=True)
        st.caption("Districts are split into 5 quintiles. Each row shows where districts that started in a "
                   "quintile ended up — the diagonal means they stayed in place (low mobility).")

        df_t = df_fil[[f"gdp_{first_y}", f"gdp_{last_y}"]].replace(0, np.nan).dropna()
        if len(df_t) >= 10:
            q_labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"]
            try:
                q_start = pd.qcut(df_t[f"gdp_{first_y}"], 5, labels=q_labels)
                q_end = pd.qcut(df_t[f"gdp_{last_y}"], 5, labels=q_labels)
                trans = pd.crosstab(q_start, q_end, normalize="index") * 100
                trans = trans.reindex(index=q_labels, columns=q_labels)
                fig_tr = px.imshow(trans, text_auto=".0f", color_continuous_scale="Blues",
                                   labels=dict(x=f"Quintile in {last_y}", y=f"Quintile in {first_y}",
                                               color="% of districts"))
                fig_tr.update_layout(height=420, paper_bgcolor="white", font=FONT, hoverlabel=HOVER,
                                     margin=dict(l=5, r=5, t=10, b=10))
                st.plotly_chart(fig_tr, use_container_width=True)
                diag = np.diag(trans.fillna(0).values).mean()
                st.info(f"📊 On average **{diag:.0f}%** of districts remained in their original quintile "
                        f"between {first_y} and {last_y} — "
                        f"{'high persistence (low mobility)' if diag >= 50 else 'moderate mobility'}.")
                buf_tr = fig_tr.to_html(full_html=True, include_plotlyjs='cdn')
                st.download_button("⬇️ Download Transition Matrix (HTML)", buf_tr,
                                   "gdp_transition_matrix.html", "text/html")
            except ValueError:
                st.warning("Not enough distinct GDP values to build quintiles for the current filter.")
        else:
            st.warning("Select at least 10 districts to compute the transition matrix.")

        st.markdown("---")

        # ── ⑤ EXTREME VALUES ──────────────────────
        st.markdown(f'<p class="section-title">⑤ Extreme Values — GDP {tahun_sel}</p>', unsafe_allow_html=True)
        st.caption("The 10 highest and 10 lowest districts by GDP in the selected year.")
        ext = df_fil[["district_en", "province_en", "island_en", col_gdp]].copy()
        ext.columns = ["District", "Province", "Island", f"GDP {tahun_sel}"]
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**🔝 Top 10**")
            st.dataframe(ext.nlargest(10, f"GDP {tahun_sel}").reset_index(drop=True)
                         .style.format({f"GDP {tahun_sel}": "{:,.0f}"}), use_container_width=True, height=390)
        with e2:
            st.markdown("**🔻 Bottom 10**")
            st.dataframe(ext.nsmallest(10, f"GDP {tahun_sel}").reset_index(drop=True)
                         .style.format({f"GDP {tahun_sel}": "{:,.0f}"}), use_container_width=True, height=390)

# ── GRID GDP (0.25°) ──────────────────────────
elif "Grid" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🔲 Grid GDP — 0.25° Cells</h1>
        <p>Gridded local GDP for Indonesia · 2012–2022 · PWT 11.0-rescaled, GADM 4.10 geography</p>
    </div>""", unsafe_allow_html=True)

    g1, g2, g3 = st.columns([2,2,1])
    with g1:
        grid_year = st.select_slider("Year", options=grid_years, value=grid_years[-1], key="grid_year")
    with g2:
        metric_label = st.selectbox(
            "Metric",
            ["GDP per capita (US$ PPP)", "Log GDP per capita", "Total Cell GDP (mil. US$ PPP)", "Population (persons)"],
            key="grid_metric"
        )
    with g3:
        grid_scale = st.selectbox("Color Scale", ["Viridis","Plasma","YlOrRd","Blues","RdYlGn"], index=0, key="grid_scale")

    metric_map = {
        "GDP per capita (US$ PPP)": "gdppc",
        "Log GDP per capita": "ln_gdppc",
        "Total Cell GDP (mil. US$ PPP)": "gdp_pwt",
        "Population (persons)": "pop_pwt",
    }
    metric_col = metric_map[metric_label]

    gy = grid_df[grid_df["year"] == grid_year].copy()
    ay = adm0_df[adm0_df["year"] == grid_year].iloc[0]

    st.markdown('<p class="section-title">Key Indicators</p>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">National GDP {grid_year}</div>
            <div class="kpi-value">{ay['gdp_pwt']/1e6:,.2f}T</div>
            <div class="kpi-sub">Trillion 2021 US$ PPP</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Population</div>
            <div class="kpi-value">{ay['pop_pwt']:,.1f}M</div>
            <div class="kpi-sub">Million persons</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">GDP per Capita</div>
            <div class="kpi-value">${ay['gdppc']:,.0f}</div>
            <div class="kpi-sub">2021 US$ PPP</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Grid Cells</div>
            <div class="kpi-value">{len(gy):,}</div>
            <div class="kpi-sub">0.25° resolution</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f'<p class="section-title">Grid Map — {metric_label} ({grid_year})</p>', unsafe_allow_html=True)
    st.caption("Each point is a 0.25° × 0.25° grid cell centroid. Censored (near-zero population) cells are excluded. "
               "Color scale is clipped to the 2nd–98th percentile so a few extreme low-population cells don't wash out the map.")

    gy_plot = gy[gy[metric_col].notna()].copy()
    if metric_col in ("gdppc", "ln_gdppc"):
        gy_plot = gy_plot[gy_plot["gdppc"] > 0]

    vmin, vmax = gy_plot[metric_col].quantile([0.02, 0.98])

    fig_grid = px.scatter_mapbox(
        gy_plot, lat="latitude", lon="longitude", color=metric_col,
        color_continuous_scale=grid_scale, range_color=[vmin, vmax],
        mapbox_style="carto-positron", zoom=3.6, center={"lat": -2.2, "lon": 118},
        hover_name="name_2",
        hover_data={"name_1": True, metric_col: ":,.1f", "latitude": False, "longitude": False},
        labels={metric_col: metric_label, "name_1": "Province", "name_2": "Regency/City"},
        opacity=0.9,
    )
    fig_grid.update_traces(marker=dict(size=9))
    fig_grid.update_layout(height=560, margin=dict(l=0,r=0,t=0,b=0),
                           coloraxis_colorbar=dict(title=metric_label, thickness=14, len=0.6),
                           paper_bgcolor="white", font=FONT, hoverlabel=HOVER)
    st.plotly_chart(fig_grid, use_container_width=True)
    buf_grid = fig_grid.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("⬇️ Download Grid Map (HTML)", buf_grid, f"grid_gdp_{grid_year}.html", "text/html")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">National GDP per Capita Trend (2012–2022)</p>', unsafe_allow_html=True)
        fig_trend = px.line(adm0_df.sort_values("year"), x="year", y="gdppc", markers=True,
                            labels={"gdppc": "GDP per Capita (US$ PPP)", "year": "Year"},
                            template="plotly_white", color_discrete_sequence=["#0f6b8a"])
        fig_trend.update_traces(line_width=2.5, marker_size=7)
        fig_trend.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white",
                                font=FONT, hoverlabel=HOVER)
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.markdown(f'<p class="section-title">Top 15 Provinces by GDP per Capita ({grid_year})</p>', unsafe_allow_html=True)
        top_prov = adm1_df[adm1_df["year"] == grid_year].nlargest(15, "gdppc")[["name","gdppc"]].sort_values("gdppc")
        fig_prov = px.bar(top_prov, x="gdppc", y="name", orientation="h",
                          color="gdppc", color_continuous_scale="Blues",
                          labels={"gdppc": "GDP per Capita (US$ PPP)", "name": ""},
                          template="plotly_white")
        fig_prov.update_layout(height=380, margin=dict(l=5,r=5,t=10,b=10),
                               coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white",
                               font=FONT, hoverlabel=HOVER)
        st.plotly_chart(fig_prov, use_container_width=True)

    st.markdown("---")

    st.markdown(f'<p class="section-title">Distribution of Cell GDP per Capita — {grid_year}</p>', unsafe_allow_html=True)
    st.caption("Log GDP per capita across all 0.25° grid cells (censored/zero-population cells excluded).")
    dist_vals = gy[gy["gdppc"] > 0]["ln_gdppc"].dropna()
    fig_gdist = px.histogram(dist_vals, x="ln_gdppc", nbins=50, marginal="box",
                             template="plotly_white", color_discrete_sequence=["#0f6b8a"], opacity=0.8)
    fig_gdist.update_layout(height=360, paper_bgcolor="white", plot_bgcolor="white",
                            font=FONT, hoverlabel=HOVER, bargap=0.05,
                            xaxis_title="Log GDP per Capita", showlegend=False)
    st.plotly_chart(fig_gdist, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#6b7c93'>
    📊 <b>Data:</b> Gridded local GDP rescaled to <b>Penn World Table 11.0</b> national accounts
    (0.25° cells, constant 2021 PPP), with <b>GADM 4.10</b> administrative geography. Source collection:
    <i>Local GDP Estimates Around the World</i> (Rossi-Hansberg &amp; Zhang) · PWT 11.0 (Feenstra, Inklaar &amp; Timmer) · GADM 4.10.
    </small>
    """, unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────
elif "Data" in page:
    st.markdown('<p class="section-title">📋 Data Source</p>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard does not provide raw data downloads. All GDP figures displayed are derived 
    from official statistics published by **Badan Pusat Statistik (BPS) Indonesia**.

    For the complete and authoritative dataset, please refer to the original source below.
    """)

    st.markdown("""
    <div class="kpi-card" style="text-align:left; padding:1.5rem;">
        <div class="kpi-label">📊 Original Data Source</div>
        <div style="margin-top:0.5rem; font-size:1rem; font-weight:600; color:#1a3a5c;">
            [SERIES 2010] GRDP at Constant Prices (2010=100)<br>by Expenditure, Regency/City
        </div>
        <div style="margin-top:0.3rem; color:#6b7c93; font-size:0.85rem;">
            Statistics Indonesia (BPS — Badan Pusat Statistik)
        </div>
        <div style="margin-top:0.8rem;">
            <a href="https://www.bps.go.id/id/statistics-table/2/MjE5NCMy/-seri-2010--pdrb-atas-dasar-harga-konstan--2010-100--menurut-pengeluaran-kabupaten-kota--milyar-rupiah-.html" target="_blank">
            🔗 Visit BPS Official Website
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#6b7c93'>
    Compiled dataset (2010–2025, 514 districts) used in this dashboard: 
    <a href="https://github.com/quarcs-lab/indonesia514" target="_blank">quarcs-lab/indonesia514</a>
    </small>
    """, unsafe_allow_html=True)
