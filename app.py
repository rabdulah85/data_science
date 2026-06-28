import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, urllib.request
import numpy as np

st.set_page_config(
    page_title="GDP Indonesia 514 Districts",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
    }
    /* Fix download buttons — light style */
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
    /* Fix invisible text in toggles, captions, info boxes */
    .stToggle label, .stCaption, [data-testid="stCaptionContainer"] {
        color: #4a5568 !important;
    }
    /* Fix header text — keep white inside main-header */
    .main-header h1, .main-header p {
        color: white !important;
        background-color: transparent !important;
    }
    /* Fix chart text — axis labels, ticks, titles */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legendtext,
    .js-plotly-plot .plotly text {
        fill: #1a2b3c !important;
        color: #1a2b3c !important;
    }
    /* Info/warning boxes */
    [data-testid="stAlert"] {
        background-color: #eff6ff !important;
        color: #1a2b3c !important;
        border: 1px solid #bfdbfe !important;
    }
    [data-testid="stAlert"] * {
        color: #1a2b3c !important;
    }
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f6b8a 100%);
        padding: 1.8rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .main-header h1 { font-family: 'DM Serif Display', serif; font-size: 1.9rem; margin: 0 0 0.3rem 0; color: white; }
    .main-header p  { font-size: 0.85rem; opacity: 0.8; margin: 0; color: #cde8f0; }
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
    with urllib.request.urlopen(url_geo) as r:
        geo = json.loads(r.read())
    return geo

df, df_long = load_data()
geojson   = load_geojson()
gdp_years = sorted([int(c.replace("gdp_","")) for c in df.columns if c.startswith("gdp_")])

with st.sidebar:
    st.markdown("### 🗂️ Navigation")
    page = st.radio("", ["🏠 Home","🗺️ Map","📈 Time Series","🏆 Ranking","📋 Data","📉 Convergence"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Filter")

    island_list = ["All"] + sorted(df["island_en"].dropna().unique().tolist())
    island_sel = st.selectbox("Island", island_list)

    if island_sel != "All":
        prov_options = sorted(df[df["island_en"] == island_sel]["province_en"].unique().tolist())
    else:
        prov_options = sorted(df["province_en"].unique().tolist())
    prov_sel = st.multiselect("Province", prov_options, default=prov_options[:5])

    if prov_sel:
        dist_options = sorted(df[df["province_en"].isin(prov_sel)]["district_en"].unique().tolist())
    else:
        dist_options = sorted(df["district_en"].unique().tolist())
    dist_sel = st.multiselect("District", dist_options, default=[])

    tahun_sel = st.select_slider("Year", options=gdp_years, value=2024)

    st.markdown("---")
    st.markdown("<small style='color:#6b7c93'>Source: quarcs-lab/indonesia514<br>514 Districts · 2010–2025</small>",
                unsafe_allow_html=True)

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

if "Home" in page:
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
        st.markdown('<p class="section-title">GDP per Province</p>', unsafe_allow_html=True)
        df_prov = df_fil.groupby("province_en")[col_gdp].sum().reset_index().sort_values(col_gdp)
        fig = px.bar(df_prov, x=col_gdp, y="province_en", orientation="h",
                     color=col_gdp, color_continuous_scale="Blues",
                     labels={col_gdp:"GDP (M Rp)","province_en":""}, template="plotly_white")
        fig.update_layout(height=420, margin=dict(l=5,r=5,t=10,b=10),
                          coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1a2b3c"))
        buf = fig.to_html(full_html=True, include_plotlyjs='cdn')
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇️ Download Bar Chart (HTML)", buf,
                           f"gdp_province_{tahun_sel}.html", "text/html")
    with c2:
        st.markdown('<p class="section-title">GDP Share per Island</p>', unsafe_allow_html=True)
        df_pulau = df_fil.groupby("island_en")[col_gdp].sum().reset_index()
        fig2 = px.pie(df_pulau, names="island_en", values=col_gdp, hole=0.45,
                      template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(height=380, margin=dict(l=5,r=5,t=10,b=10),
                           legend_title_text="Island", paper_bgcolor="white")
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        buf2 = fig2.to_html(full_html=True, include_plotlyjs='cdn')
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("⬇️ Download Pie Chart (HTML)", buf2,
                           f"gdp_island_{tahun_sel}.html", "text/html")

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
                          paper_bgcolor="white")
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
                          font=dict(color="#1a2b3c"),
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
                           font=dict(color="#1a2b3c"),
                           legend_title_text="Province", margin=dict(b=80))
        st.plotly_chart(fig3, use_container_width=True)

elif "Ranking" in page:
    st.markdown(f'<p class="section-title">🏆 Top 20 Districts — GDP {tahun_sel}</p>', unsafe_allow_html=True)
    top20 = df_fil.nlargest(20, col_gdp)[["district_en","province_en","island_en",col_gdp]].copy()
    fig4 = px.bar(top20.sort_values(col_gdp), x=col_gdp, y="district_en",
                  color="island_en", orientation="h", template="plotly_white",
                  labels={col_gdp:f"GDP {tahun_sel} (M Rp)","district_en":"","island_en":"Island"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(height=520, margin=dict(l=5,r=5,t=10,b=10), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1a2b3c"))
    st.plotly_chart(fig4, use_container_width=True)
    buf4 = fig4.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("⬇️ Download Ranking Chart (HTML)", buf4, f"gdp_ranking_{tahun_sel}.html", "text/html")

    st.markdown('<p class="section-title">Scatter: GDP vs Growth 2010–2025</p>', unsafe_allow_html=True)
    df_sc = df_fil[["district_en","province_en","island_en","gdp_2010",col_gdp]].copy()
    df_sc["growth"] = ((df_sc[col_gdp]-df_sc["gdp_2010"])/df_sc["gdp_2010"]*100).round(1)
    fig5 = px.scatter(df_sc, x=col_gdp, y="growth", color="island_en",
                      hover_name="district_en", hover_data={"province_en":True},
                      labels={col_gdp:f"GDP {tahun_sel} (M Rp)","growth":"Growth 2010–2025 (%)"},
                      template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.75)
    fig5.update_layout(height=420, legend_title_text="Island", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1a2b3c"))
    st.plotly_chart(fig5, use_container_width=True)


elif "Convergence" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📉 Convergence Analysis</h1>
        <p>GDP convergence across 514 Indonesian districts · 2010–2025</p>
    </div>""", unsafe_allow_html=True)

    # Log toggle
    use_log = st.toggle("Use Log GDP", value=True,
                        help="Log transformation reduces skewness and is standard in convergence analysis")

    st.markdown('<p class="section-title">① σ-Convergence — Is GDP disparity narrowing over time?</p>',
                unsafe_allow_html=True)
    st.caption("σ-convergence occurs when the standard deviation of GDP across districts **decreases** over time.")

    # Compute sigma convergence
    sigma_data = []
    for y in gdp_years:
        col = f"gdp_{y}"
        vals = df_fil[col].replace(0, np.nan).dropna()
        if use_log:
            vals = np.log(vals)
        sigma_data.append({"Year": y, "Std Dev": vals.std(), "CV": vals.std()/vals.mean()})
    df_sigma = pd.DataFrame(sigma_data)

    c1, c2 = st.columns(2)
    with c1:
        fig_sigma = px.line(df_sigma, x="Year", y="Std Dev", markers=True,
                            title=f"Standard Deviation of {'Log ' if use_log else ''}GDP",
                            template="plotly_white",
                            color_discrete_sequence=["#0f6b8a"])
        fig_sigma.update_traces(line_width=2.5, marker_size=7)
        fig_sigma.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(color="#1a2b3c"), title_font_size=13,
                        hoverlabel=dict(bgcolor="white", font_color="#1a2b3c",
                                       font_size=12, bordercolor="#e2e8f0"))
        fig_sigma.add_annotation(
            x=gdp_years[-1], y=df_sigma["Std Dev"].iloc[-1],
            text="↑ Diverging" if df_sigma["Std Dev"].iloc[-1] > df_sigma["Std Dev"].iloc[0] else "↓ Converging",
            showarrow=True, arrowhead=2, font_size=11,
            font_color="red" if df_sigma["Std Dev"].iloc[-1] > df_sigma["Std Dev"].iloc[0] else "green"
        )
        st.plotly_chart(fig_sigma, use_container_width=True)

    with c2:
        fig_cv = px.line(df_sigma, x="Year", y="CV", markers=True,
                         title=f"Coefficient of Variation of {'Log ' if use_log else ''}GDP",
                         template="plotly_white",
                         color_discrete_sequence=["#e05c5c"])
        fig_cv.update_traces(line_width=2.5, marker_size=7)
        fig_cv.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="white",
                             title_font_size=13)
        st.plotly_chart(fig_cv, use_container_width=True)

    # Sigma interpretation
    delta = df_sigma["Std Dev"].iloc[-1] - df_sigma["Std Dev"].iloc[0]
    direction = "increased" if delta > 0 else "decreased"
    verdict = "**diverging** (σ-divergence)" if delta > 0 else "**converging** (σ-convergence)"
    st.info(f"📊 The standard deviation of {'log ' if use_log else ''}GDP has {direction} from "
            f"**{df_sigma['Std Dev'].iloc[0]:.3f}** (2010) to **{df_sigma['Std Dev'].iloc[-1]:.3f}** (2025), "
            f"suggesting districts are {verdict}.")

    st.markdown("---")
    st.markdown('<p class="section-title">② β-Convergence — Do poorer districts grow faster?</p>',
                unsafe_allow_html=True)
    st.caption("β-convergence occurs when districts with **lower initial GDP grow faster** — a negative slope indicates convergence.")

    # Compute beta convergence
    df_beta = df_fil[["districtID","district_en","province_en","island_en","gdp_2010","gdp_2025"]].copy()
    df_beta = df_beta.replace(0, np.nan).dropna()
    df_beta["growth_rate"] = ((df_beta["gdp_2025"] - df_beta["gdp_2010"]) / df_beta["gdp_2010"] * 100)

    if use_log:
        df_beta["initial_gdp"] = np.log(df_beta["gdp_2010"])
        x_label = "Log GDP 2010 (Initial Level)"
    else:
        df_beta["initial_gdp"] = df_beta["gdp_2010"]
        x_label = "GDP 2010 (Initial Level, M Rp)"

    # OLS trendline
    fig_beta = px.scatter(df_beta, x="initial_gdp", y="growth_rate",
                          color="island_en", hover_name="district_en",
                          hover_data={"province_en": True, "initial_gdp": ":.2f", "growth_rate": ":.1f"},
                          trendline="ols",
                          labels={"initial_gdp": x_label,
                                  "growth_rate": "GDP Growth Rate 2010–2025 (%)",
                                  "island_en": "Island"},
                          template="plotly_white",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          opacity=0.65)
    fig_beta.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(color="#1a2b3c"),
                       hoverlabel=dict(
                           bgcolor="white",
                           font_color="#1a2b3c",
                           font_size=12,
                           bordercolor="#e2e8f0"
                       ))
    st.plotly_chart(fig_beta, use_container_width=True)

    # OLS result
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(df_beta["initial_gdp"], df_beta["growth_rate"])
    beta_verdict = "β-convergence ✅" if slope < 0 else "β-divergence ⚠️"
    st.info(f"📊 OLS slope = **{slope:.3f}** (p = {p:.4f}, R² = {r**2:.3f}) → **{beta_verdict}**. "
            f"{'Districts with lower initial GDP tend to grow faster.' if slope < 0 else 'Districts with higher initial GDP tend to grow faster.'}")

    st.markdown("---")
    st.markdown('<p class="section-title">③ Distribution Dynamics — How has the GDP distribution shifted?</p>',
                unsafe_allow_html=True)
    st.caption("Tracks how the full distribution of GDP across districts has evolved from 2010 to 2025.")

    # Select years to compare
    years_compare = st.multiselect("Select years to compare",
                                   gdp_years, default=[2010, 2015, 2020, 2025])
    if years_compare:
        df_dist = []
        for y in years_compare:
            vals = df_fil[f"gdp_{y}"].replace(0, np.nan).dropna()
            if use_log:
                vals = np.log(vals)
            tmp = pd.DataFrame({"GDP": vals, "Year": str(y)})
            df_dist.append(tmp)
        df_dist = pd.concat(df_dist)

        fig_dist = px.histogram(df_dist, x="GDP", color="Year", barmode="overlay",
                                nbins=40, opacity=0.6,
                                labels={"GDP": f"{'Log ' if use_log else ''}GDP",
                                        "Year": "Year"},
                                template="plotly_white",
                                color_discrete_sequence=px.colors.qualitative.Set1)
        fig_dist.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(color="#1a2b3c"),
                       hoverlabel=dict(bgcolor="white", font_color="#1a2b3c",
                                      font_size=12, bordercolor="#e2e8f0"))
        st.plotly_chart(fig_dist, use_container_width=True)

        # Download
        buf_conv = fig_beta.to_html(full_html=True, include_plotlyjs='cdn')
        st.download_button("⬇️ Download β-Convergence Chart (HTML)", buf_conv,
                           "beta_convergence.html", "text/html")


elif "Data" in page:
    st.markdown('<p class="section-title">📋 Complete Data Table</p>', unsafe_allow_html=True)
    search = st.text_input("🔍 Search district/city...", "")
    cols_show = ["district_en","province_en","island_en"] + [f"gdp_{y}" for y in gdp_years]
    df_show = df_fil[cols_show].copy()
    if search:
        df_show = df_show[df_show["district_en"].str.contains(search, case=False)]
    df_show.columns = ["District","Province","Island"] + [str(y) for y in gdp_years]
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=450)
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"gdp_indonesia_{tahun_sel}.csv", "text/csv")

# ── CONVERGENCE ───────────────────────────────
