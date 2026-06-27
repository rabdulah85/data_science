import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, urllib.request

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GDP Indonesia 514 Kabupaten",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f6b8a 100%);
        padding: 1.8rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { font-family: 'DM Serif Display', serif; font-size: 1.9rem; margin: 0 0 0.3rem 0; color: white; }
    .main-header p  { font-size: 0.85rem; opacity: 0.8; margin: 0; color: #cde8f0; }
    .kpi-card {
        background: white; border: 1px solid #e8edf2;
        border-radius: 10px; padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04); text-align: center;
    }
    .kpi-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #6b7c93; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1a3a5c; line-height: 1.1; }
    .kpi-sub   { font-size: 0.75rem; color: #2ca86e; margin-top: 0.2rem; }
    .section-title {
        font-size: 0.95rem; font-weight: 600; color: #1a3a5c;
        border-left: 3px solid #0f6b8a; padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
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
geojson    = load_geojson()
gdp_years  = sorted([int(c.replace("gdp_","")) for c in df.columns if c.startswith("gdp_")])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Navigasi")
    page = st.radio("", ["🏠 Beranda","🗺️ Peta","📈 Tren Waktu","🏆 Ranking","📋 Data"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Filter")

    pulau_list = ["Semua"] + sorted(df["island_bahasa"].dropna().unique().tolist())
    pulau = st.selectbox("Pulau", pulau_list)

    if pulau != "Semua":
        prov_list = sorted(df[df["island_bahasa"]==pulau]["province_bahasa"].unique().tolist())
    else:
        prov_list = sorted(df["province_bahasa"].unique().tolist())

    prov_sel = st.multiselect("Provinsi", prov_list, default=prov_list[:5])
    tahun_sel = st.select_slider("Tahun", options=gdp_years, value=2024)

    st.markdown("---")
    st.markdown("<small style='color:#6b7c93'>Sumber: quarcs-lab/indonesia514<br>514 Kabupaten/Kota · 2010–2025</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
col_gdp   = f"gdp_{tahun_sel}"
df_fil    = df[df["province_bahasa"].isin(prov_sel)].copy() if prov_sel else df.copy()
df_long_f = df_long[df_long["province_bahasa"].isin(prov_sel)] if prov_sel else df_long

# ─────────────────────────────────────────────
# PAGE: BERANDA
# ─────────────────────────────────────────────
if "Beranda" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇩 GDP Indonesia — 514 Kabupaten/Kota</h1>
        <p>Data PDRB tingkat kabupaten/kota · 2010–2025 · Sumber: quarcs-lab/indonesia514</p>
    </div>""", unsafe_allow_html=True)

    total_gdp = df_fil[col_gdp].sum()
    rata_gdp  = df_fil[col_gdp].mean()
    tertinggi = df_fil.loc[df_fil[col_gdp].idxmax(), "district_bahasa"]
    n_kab     = len(df_fil)

    st.markdown('<p class="section-title">Indikator Utama</p>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total GDP {tahun_sel}</div>
            <div class="kpi-value">{total_gdp/1e6:,.1f}T</div>
            <div class="kpi-sub">Triliun Rupiah</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Rata-rata GDP</div>
            <div class="kpi-value">{rata_gdp:,.0f}</div>
            <div class="kpi-sub">Miliar Rp / Kabupaten</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">GDP Tertinggi</div>
            <div class="kpi-value" style="font-size:1rem">{tertinggi}</div>
            <div class="kpi-sub">{df_fil[col_gdp].max():,.0f} M Rp</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Jumlah Kabupaten</div>
            <div class="kpi-value">{n_kab}</div>
            <div class="kpi-sub">dari 514 total</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<p class="section-title">GDP per Provinsi</p>', unsafe_allow_html=True)
        df_prov = df_fil.groupby("province_bahasa")[col_gdp].sum().reset_index().sort_values(col_gdp)
        fig = px.bar(df_prov, x=col_gdp, y="province_bahasa", orientation="h",
                     color=col_gdp, color_continuous_scale="Blues",
                     labels={col_gdp:"GDP (M Rp)","province_bahasa":""}, template="plotly_white")
        fig.update_layout(height=400, margin=dict(l=5,r=5,t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="section-title">Share GDP per Pulau</p>', unsafe_allow_html=True)
        df_pulau = df.groupby("island_bahasa")[col_gdp].sum().reset_index()
        fig2 = px.pie(df_pulau, names="island_bahasa", values=col_gdp, hole=0.45,
                      template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(height=400, margin=dict(l=5,r=5,t=10,b=10), legend_title_text="Pulau")
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PETA
# ─────────────────────────────────────────────
elif "Peta" in page:
    st.markdown(f'<p class="section-title">🗺️ Peta Choropleth GDP Kabupaten/Kota — {tahun_sel}</p>',
                unsafe_allow_html=True)

    # Color scale selector
    col_opt, col_log = st.columns([3,1])
    with col_opt:
        skala_warna = st.selectbox("Skala Warna", ["Blues","YlOrRd","Viridis","RdYlGn","Plasma"], index=0)
    with col_log:
        log_scale = st.checkbox("Log Scale", value=True,
                                help="Log scale membantu visualisasi karena disparitas GDP sangat besar")

    # Prepare data — use full df for map (all 514), highlight filtered
    df_map = df[["districtID","district_bahasa","province_bahasa","island_bahasa", col_gdp]].copy()
    df_map["gdp_plot"] = df_map[col_gdp].clip(lower=1)

    import numpy as np
    if log_scale:
        df_map["gdp_display"] = np.log10(df_map["gdp_plot"])
        label_color = f"Log10 GDP {tahun_sel}"
    else:
        df_map["gdp_display"] = df_map["gdp_plot"]
        label_color = f"GDP {tahun_sel} (M Rp)"

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson=geojson,
        locations="districtID",
        featureidkey="properties.districtID",
        color="gdp_display",
        color_continuous_scale=skala_warna,
        mapbox_style="carto-positron",
        zoom=3.8,
        center={"lat": -2.5, "lon": 118},
        opacity=0.75,
        hover_name="district_bahasa",
        hover_data={
            "province_bahasa": True,
            "island_bahasa": True,
            col_gdp: ":,.0f",
            "gdp_display": False,
            "districtID": False
        },
        labels={
            "gdp_display": label_color,
            "province_bahasa": "Provinsi",
            "island_bahasa": "Pulau",
            col_gdp: f"GDP {tahun_sel} (M Rp)"
        }
    )
    fig_map.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=label_color,
            thickness=14,
            len=0.6
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Summary stats below map
    st.markdown('<p class="section-title">Statistik Ringkas</p>', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.metric("GDP Tertinggi", f"{df_map[col_gdp].max():,.0f} M Rp",
                  df_map.loc[df_map[col_gdp].idxmax(),"district_bahasa"])
    with s2:
        st.metric("GDP Terendah", f"{df_map[col_gdp].min():,.0f} M Rp",
                  df_map.loc[df_map[col_gdp].idxmin(),"district_bahasa"])
    with s3:
        st.metric("Median GDP", f"{df_map[col_gdp].median():,.0f} M Rp")
    with s4:
        ratio = df_map[col_gdp].max() / df_map[col_gdp].min()
        st.metric("Rasio Maks/Min", f"{ratio:,.0f}x", "Ukuran disparitas")

# ─────────────────────────────────────────────
# PAGE: TREN WAKTU
# ─────────────────────────────────────────────
elif "Tren" in page:
    st.markdown('<p class="section-title">📈 Tren GDP per Provinsi (2010–2025)</p>', unsafe_allow_html=True)
    if not prov_sel:
        st.warning("Pilih minimal 1 provinsi di sidebar.")
    else:
        df_trend = df_long_f.groupby(["tahun","province_bahasa"])["gdp"].sum().reset_index()
        fig = px.line(df_trend, x="tahun", y="gdp", color="province_bahasa", markers=True,
                      labels={"gdp":"GDP (M Rp)","tahun":"Tahun","province_bahasa":"Provinsi"},
                      template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=460, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Pertumbuhan GDP 2010 → 2025 (Top 20 Kabupaten)</p>', unsafe_allow_html=True)
    df_growth = df_fil[["district_bahasa","province_bahasa","gdp_2010","gdp_2025"]].copy()
    df_growth["growth_pct"] = ((df_growth["gdp_2025"]-df_growth["gdp_2010"])/df_growth["gdp_2010"]*100).round(1)
    df_growth = df_growth.sort_values("growth_pct", ascending=False).head(20)
    fig3 = px.bar(df_growth, x="district_bahasa", y="growth_pct", color="province_bahasa",
                  template="plotly_white",
                  labels={"growth_pct":"Pertumbuhan (%)","district_bahasa":"Kabupaten"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(height=380, xaxis_tickangle=-35, legend_title_text="Provinsi", margin=dict(b=80))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: RANKING
# ─────────────────────────────────────────────
elif "Ranking" in page:
    st.markdown(f'<p class="section-title">🏆 Top 20 Kabupaten/Kota — GDP {tahun_sel}</p>', unsafe_allow_html=True)
    top20 = df_fil.nlargest(20, col_gdp)[["district_bahasa","province_bahasa","island_bahasa",col_gdp]].copy()
    fig4 = px.bar(top20.sort_values(col_gdp), x=col_gdp, y="district_bahasa",
                  color="island_bahasa", orientation="h", template="plotly_white",
                  labels={col_gdp:f"GDP {tahun_sel} (M Rp)","district_bahasa":"","island_bahasa":"Pulau"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(height=520, margin=dict(l=5,r=5,t=10,b=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-title">Scatter: GDP vs Pertumbuhan 2010–2025</p>', unsafe_allow_html=True)
    df_sc = df_fil[["district_bahasa","province_bahasa","island_bahasa","gdp_2010",col_gdp]].copy()
    df_sc["growth"] = ((df_sc[col_gdp]-df_sc["gdp_2010"])/df_sc["gdp_2010"]*100).round(1)
    fig5 = px.scatter(df_sc, x=col_gdp, y="growth", color="island_bahasa",
                      hover_name="district_bahasa", hover_data={"province_bahasa":True},
                      labels={col_gdp:f"GDP {tahun_sel} (M Rp)","growth":"Pertumbuhan 2010–2025 (%)"},
                      template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.75)
    fig5.update_layout(height=420, legend_title_text="Pulau")
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DATA
# ─────────────────────────────────────────────
elif "Data" in page:
    st.markdown('<p class="section-title">📋 Tabel Data Lengkap</p>', unsafe_allow_html=True)
    cari = st.text_input("🔍 Cari kabupaten/kota...", "")
    cols_show = ["district_bahasa","province_bahasa","island_bahasa"] + [f"gdp_{y}" for y in gdp_years]
    df_show = df_fil[cols_show].copy()
    if cari:
        df_show = df_show[df_show["district_bahasa"].str.contains(cari, case=False)]
    df_show.columns = ["Kabupaten/Kota","Provinsi","Pulau"] + [str(y) for y in gdp_years]
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=450)
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"gdp_indonesia_{tahun_sel}.csv", "text/csv")
