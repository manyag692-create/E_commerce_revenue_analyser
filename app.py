import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"], .main, .block-container {
    background-color: #0d0d1a !important;
    color: #e8e8f0 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
header[data-testid="stHeader"] { background: transparent !important; }
.main .block-container { padding-top: 1rem; }

.hero-banner {
    background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 40%, #11001c 100%);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(102,126,234,0.25), 0 0 120px rgba(118,75,162,0.15);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin: 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(200,200,255,0.7);
    margin-top: 0.5rem;
}

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    border-bottom: 2px solid rgba(102,126,234,0.4);
    padding-bottom: 5px;
    margin-bottom: 1rem;
    display: inline-block;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060612 0%, #0d0d2b 50%, #0a0a20 100%) !important;
    border-right: 1px solid rgba(102,126,234,0.2);
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #9090b0 !important;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4c1d95, #1e40af) !important;
    color: #fff !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.5);
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color: #8080a8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #c8c8ff !important; }

.stSelectbox > div > div,
.stSlider > div { background: transparent !important; }

.stButton > button {
    background: linear-gradient(135deg, #4c1d95, #1e40af);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px rgba(124,58,237,0.4);
    transition: all 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(124,58,237,0.7);
    transform: translateY(-2px);
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px;
    color: #a78bfa !important;
}

.pred-box {
    background: linear-gradient(135deg, rgba(76,29,149,0.3), rgba(30,64,175,0.3));
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(124,58,237,0.2);
    margin-top: 1rem;
}
.pred-value {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pred-label { color: #9090c0; font-size: 0.9rem; margin-top: 0.3rem; }

.footer {
    text-align: center;
    color: #444466;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BG     = "#0d0d1a"
PAPER  = "#0d0d1a"
GRID   = "rgba(255,255,255,0.07)"
FONT_C = "#c8c8e8"

def dark_layout(height=400, extra_xaxis=None, extra_yaxis=None, **kwargs):
    """
    Build a dark plotly layout dict.
    extra_xaxis / extra_yaxis are merged on top of the base axis dicts
    so callers can override individual axis keys without wiping the rest.
    """
    base_x = dict(showgrid=True, gridcolor=GRID, color=FONT_C, zeroline=False)
    base_y = dict(showgrid=True, gridcolor=GRID, color=FONT_C, zeroline=False)
    if extra_xaxis:
        base_x.update(extra_xaxis)
    if extra_yaxis:
        base_y.update(extra_yaxis)
    layout = dict(
        plot_bgcolor=BG,
        paper_bgcolor=PAPER,
        font_color=FONT_C,
        height=height,
        xaxis=base_x,
        yaxis=base_y,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_C),
    )
    layout.update(kwargs)
    return layout

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="latin1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
    df["Month"]       = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Hour"]        = df["InvoiceDate"].dt.hour
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    return df

# ─────────────────────────────────────────────
# RF MODEL  — trained fresh on actual data
# ─────────────────────────────────────────────
@st.cache_resource
def train_rf(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    sample = df.sample(min(50_000, len(df)), random_state=42).copy()
    le = LabelEncoder()
    # Fit encoder on ALL countries so predictor never hits unseen labels
    le.fit(df["Country"].values)
    sample["CountryEnc"] = le.transform(sample["Country"])
    X = sample[["Quantity", "UnitPrice", "CountryEnc", "Hour"]].values
    y = sample["Revenue"].values
    model = RandomForestRegressor(
        n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    return model, le

df       = load_data()
rf_model, label_enc = train_rf(df)

ALL_COUNTRIES = sorted(df["Country"].unique().tolist())

# ─────────────────────────────────────────────
# SIDEBAR  — unique widget keys throughout
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    # key="sidebar_country" avoids collision with predictor selectbox
    sel_country = st.selectbox(
        "🌍 Filter by Country",
        ["All"] + ALL_COUNTRIES,
        key="sidebar_country"
    )

    months      = sorted(df["Month"].unique().tolist())
    sel_months  = st.select_slider(
        "📅 Month Range",
        options=months,
        value=(months[0], months[-1]),
        key="sidebar_months"
    )

    max_rev  = float(df["Revenue"].quantile(0.99))
    rev_range = st.slider(
        "💰 Revenue per Transaction (£)",
        min_value=0.0,
        max_value=float(round(max_rev)),
        value=(0.0, float(round(max_rev / 2))),
        step=1.0,
        key="sidebar_rev"
    )

    st.markdown("---")
    st.markdown("### 🎨 Chart Theme")
    color_theme = st.selectbox(
        "Palette",
        ["Electric", "Plasma", "Inferno", "Magma", "Turbo"],
        key="sidebar_palette"
    )
    theme_map = {
        "Electric": px.colors.sequential.Electric,
        "Plasma":   px.colors.sequential.Plasma,
        "Inferno":  px.colors.sequential.Inferno,
        "Magma":    px.colors.sequential.Magma,
        "Turbo":    px.colors.sequential.Turbo,
    }
    palette = theme_map[color_theme]

    st.markdown("---")
    st.markdown(
        "<small style='color:#555577'>📊 Retail Intelligence Hub<br>"
        "Dark Edition · RF Model Active</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
dff = df.copy()
if sel_country != "All":
    dff = dff[dff["Country"] == sel_country]
dff = dff[(dff["Month"] >= sel_months[0]) & (dff["Month"] <= sel_months[1])]
dff = dff[(dff["Revenue"] >= rev_range[0]) & (dff["Revenue"] <= rev_range[1])]

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <p class='hero-title'>🛍️ Retail Intelligence Hub</p>
  <p class='hero-subtitle'>Dark Edition · 3D Analytics · ML-Powered Predictions · 541K+ Transactions</p>
</div>
""", unsafe_allow_html=True)

# Guard: stop gracefully if filters wipe out all data
if dff.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust the sidebar controls.")
    st.stop()

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
total_rev       = dff["Revenue"].sum()
total_orders    = dff["InvoiceNo"].nunique()
total_customers = dff["CustomerID"].nunique()
avg_order_val   = total_rev / total_orders if total_orders else 0
rev_series      = dff.groupby("Country")["Revenue"].sum()
top_country     = rev_series.idxmax() if not rev_series.empty and rev_series.max() > 0 else "-"

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("💰 Total Revenue",   f"£{total_rev:,.0f}")
with k2: st.metric("🧾 Total Orders",     f"{total_orders:,}")
with k3: st.metric("👥 Unique Customers", f"{total_customers:,}")
with k4: st.metric("📦 Avg Order Value",  f"£{avg_order_val:,.2f}")
with k5: st.metric("🌍 Top Market",       top_country)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Revenue & Trends",
    "🌐 3D Globe & Geo",
    "🔮 3D Analysis",
    "🕰️ Time Intelligence",
    "🏆 Top Products",
    "🤖 RF Predictor",
])

# ══════════════════════════════════════════════
# TAB 1 – Revenue & Trends
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<span class='section-header'>Monthly Revenue Trend</span>", unsafe_allow_html=True)
        rev_m = dff.groupby("Month")["Revenue"].sum().reset_index()
        rev_m["MA3"] = rev_m["Revenue"].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rev_m["Month"], y=rev_m["Revenue"],
            fill="tozeroy", name="Revenue",
            line=dict(color="#a78bfa", width=2.5),
            fillcolor="rgba(167,139,250,0.12)"
        ))
        fig.add_trace(go.Scatter(
            x=rev_m["Month"], y=rev_m["MA3"],
            name="3-Month MA",
            line=dict(color="#60a5fa", width=2, dash="dot")
        ))
        fig.update_layout(**dark_layout(
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<span class='section-header'>Revenue by Country</span>", unsafe_allow_html=True)
        top_c = dff.groupby("Country")["Revenue"].sum().nlargest(8).reset_index()
        fig2 = px.bar(
            top_c, x="Revenue", y="Country", orientation="h",
            color="Revenue", color_continuous_scale=palette,
            text=top_c["Revenue"].apply(lambda x: f"£{x/1000:.0f}k")
        )
        fig2.update_layout(**dark_layout(
            height=320, showlegend=False, coloraxis_showscale=False
        ))
        fig2.update_traces(textposition="outside",
                           textfont=dict(color=FONT_C))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<span class='section-header'>Quantity vs Revenue — Interactive Scatter</span>", unsafe_allow_html=True)
    sample = dff.sample(min(3000, len(dff)), random_state=42)
    fig3 = px.scatter(
        sample, x="Quantity", y="Revenue",
        color="Country", size="UnitPrice",
        size_max=18, opacity=0.75,
        hover_data=["Description", "InvoiceDate"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig3.update_layout(**dark_layout(height=380))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – 3D Globe & Geo
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<span class='section-header'>🌍 Global Revenue — Rotating Globe</span>", unsafe_allow_html=True)
    country_rev = dff.groupby("Country").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique")
    ).reset_index()

    # Revenue removed from hover_data — it's already the color axis
    fig_globe = px.choropleth(
        country_rev,
        locations="Country", locationmode="country names",
        color="Revenue", hover_name="Country",
        hover_data={"Orders": True, "Customers": True},
        color_continuous_scale="Plasma",
        projection="orthographic"
    )
    fig_globe.update_geos(
        showcoastlines=True, coastlinecolor="#334",
        showland=True,  landcolor="#1a1a2e",
        showocean=True, oceancolor="#070718",
        showlakes=True, lakecolor="#070718",
        showframe=False
    )
    fig_globe.update_layout(
        paper_bgcolor=PAPER, geo_bgcolor=PAPER,
        font_color=FONT_C,
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        coloraxis_colorbar=dict(
            title="Revenue £", tickprefix="£",
            tickfont=dict(color=FONT_C),
            title_font=dict(color=FONT_C)
        )
    )
    st.plotly_chart(fig_globe, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<span class='section-header'>Orders per Country</span>", unsafe_allow_html=True)
        top10 = country_rev.nlargest(10, "Orders")
        fig_b = px.bar(
            top10, x="Country", y="Orders",
            color="Orders", color_continuous_scale=palette, text="Orders"
        )
        fig_b.update_layout(**dark_layout(
            height=300, showlegend=False, coloraxis_showscale=False
        ))
        fig_b.update_traces(
            texttemplate="%{text:,}", textposition="outside",
            textfont=dict(color=FONT_C)
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with col_b:
        st.markdown("<span class='section-header'>Customer Share</span>", unsafe_allow_html=True)
        fig_pie = px.pie(
            country_rev.nlargest(8, "Customers"),
            values="Customers", names="Country",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hole=0.45
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont=dict(color="#fff")
        )
        fig_pie.update_layout(
            paper_bgcolor=PAPER, font_color=FONT_C,
            margin=dict(l=0, r=0, t=0, b=0), height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – 3D Analysis
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<span class='section-header'>🔮 3D Revenue Landscape — Country × Month × Revenue</span>", unsafe_allow_html=True)

    top5 = dff.groupby("Country")["Revenue"].sum().nlargest(5).index.tolist()
    df3d_agg = (
        dff[dff["Country"].isin(top5)]
        .groupby(["Country", "Month"])["Revenue"].sum()
        .reset_index()
    )
    df3d_agg["MonthNum"] = (
        pd.to_datetime(df3d_agg["Month"]).dt.month
        + (pd.to_datetime(df3d_agg["Month"]).dt.year - 2010) * 12
    )

    SCENE_DARK = dict(
        xaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Month"),
        yaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Country"),
        zaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Revenue (£)"),
        bgcolor=BG
    )
    colors_3d = ["#a78bfa", "#60a5fa", "#f472b6", "#34d399", "#fbbf24"]
    fig3d = go.Figure()
    for i, c in enumerate(top5):
        sub = df3d_agg[df3d_agg["Country"] == c]
        fig3d.add_trace(go.Scatter3d(
            x=sub["MonthNum"], y=[c] * len(sub), z=sub["Revenue"],
            mode="lines+markers", name=c,
            line=dict(color=colors_3d[i % len(colors_3d)], width=5),
            marker=dict(size=5, color=sub["Revenue"],
                        colorscale="Plasma", showscale=False)
        ))
    fig3d.update_layout(
        scene=SCENE_DARK, paper_bgcolor=PAPER, font_color=FONT_C,
        margin=dict(l=0, r=0, t=0, b=0), height=520,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("<span class='section-header'>3D Price × Quantity × Revenue Bubble Cloud</span>", unsafe_allow_html=True)
    s3 = dff.sample(min(2000, len(dff)), random_state=7)
    s3 = s3[(s3["UnitPrice"] < 50) & (s3["Quantity"] < 200)]
    if not s3.empty:
        fig3d_b = px.scatter_3d(
            s3, x="UnitPrice", y="Quantity", z="Revenue",
            color="Country", size="Revenue",
            size_max=14, opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hover_data=["Description"]
        )
        fig3d_b.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Unit Price (£)"),
                yaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Quantity"),
                zaxis=dict(backgroundcolor=BG, gridcolor="#223", color=FONT_C, title="Revenue (£)"),
                bgcolor=BG
            ),
            paper_bgcolor=PAPER, font_color=FONT_C,
            margin=dict(l=0, r=0, t=0, b=0), height=520,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
        )
        st.plotly_chart(fig3d_b, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – Time Intelligence
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<span class='section-header'>⏱️ Sales Heatmap — Day × Hour</span>", unsafe_allow_html=True)
    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat_pivot = (
        dff.groupby(["DayOfWeek", "Hour"])["Revenue"].sum()
        .reset_index()
        .pivot(index="DayOfWeek", columns="Hour", values="Revenue")
        .reindex(day_order).fillna(0)
    )
    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values, x=heat_pivot.columns, y=heat_pivot.index,
        colorscale="Plasma", hoverongaps=False,
        colorbar=dict(
            title="Revenue £",
            tickfont=dict(color=FONT_C),
            title_font=dict(color=FONT_C)
        )
    ))
    fig_heat.update_layout(**dark_layout(height=340))
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-header'>Hourly Revenue</span>", unsafe_allow_html=True)
        hourly = dff.groupby("Hour")["Revenue"].sum().reset_index()
        fig_h  = px.area(hourly, x="Hour", y="Revenue",
                         color_discrete_sequence=["#a78bfa"])
        fig_h.update_traces(fillcolor="rgba(167,139,250,0.15)")
        fig_h.update_layout(**dark_layout(height=280))
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        st.markdown("<span class='section-header'>Orders by Day</span>", unsafe_allow_html=True)
        dow = (
            dff.groupby("DayOfWeek")["InvoiceNo"].nunique()
            .reindex(day_order).reset_index()
        )
        dow.columns = ["Day", "Orders"]
        fig_dow = px.bar(dow, x="Day", y="Orders",
                         color="Orders", color_continuous_scale=palette,
                         text="Orders")
        fig_dow.update_layout(**dark_layout(
            height=280, showlegend=False, coloraxis_showscale=False
        ))
        fig_dow.update_traces(
            texttemplate="%{text:,}", textposition="outside",
            textfont=dict(color=FONT_C)
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("<span class='section-header'>Revenue Funnel by Quarter</span>", unsafe_allow_html=True)
    dff2 = dff.copy()
    dff2["Quarter"] = dff2["InvoiceDate"].dt.to_period("Q").astype(str)
    q_rev = dff2.groupby("Quarter")["Revenue"].sum().reset_index().sort_values("Quarter")
    n = len(q_rev)
    funnel_colors = px.colors.sample_colorscale(
        "Plasma", [i / max(n - 1, 1) for i in range(n)]
    )
    fig_funnel = go.Figure(go.Funnel(
        y=q_rev["Quarter"], x=q_rev["Revenue"],
        textposition="inside", textinfo="value+percent initial",
        marker={"color": funnel_colors}
    ))
    fig_funnel.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, font_color=FONT_C,
        margin=dict(l=0, r=0, t=0, b=0), height=320
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 – Top Products
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<span class='section-header'>🏆 Top 20 Products by Revenue</span>", unsafe_allow_html=True)
    top_prod = (
        dff.groupby("Description")
        .agg(Revenue=("Revenue","sum"), Qty=("Quantity","sum"),
             Orders=("InvoiceNo","nunique"))
        .reset_index()
        .nlargest(20, "Revenue")
    )
    fig_prod = px.bar(
        top_prod, x="Revenue", y="Description", orientation="h",
        color="Revenue", color_continuous_scale=palette,
        text=top_prod["Revenue"].apply(lambda x: f"£{x/1000:.1f}k"),
        hover_data={"Qty": True, "Orders": True}
    )
    fig_prod.update_layout(**dark_layout(
        height=560, coloraxis_showscale=False,
        extra_yaxis=dict(title="", autorange="reversed", showgrid=False),
        extra_xaxis=dict(title="Revenue (£)")
    ))
    fig_prod.update_traces(textposition="outside",
                           textfont=dict(color=FONT_C))
    st.plotly_chart(fig_prod, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-header'>Revenue Treemap</span>", unsafe_allow_html=True)
        top15 = dff.groupby("Description")["Revenue"].sum().nlargest(15).reset_index()
        fig_tree = px.treemap(
            top15, path=["Description"], values="Revenue",
            color="Revenue", color_continuous_scale=palette
        )
        fig_tree.update_layout(
            paper_bgcolor=PAPER, font_color=FONT_C,
            margin=dict(l=0, r=0, t=10, b=0), height=380
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with col2:
        st.markdown("<span class='section-header'>Country → Product Sunburst</span>", unsafe_allow_html=True)
        top8c  = dff.groupby("Country")["Revenue"].sum().nlargest(8).index.tolist()
        sun_df = (
            dff[dff["Country"].isin(top8c)]
            .groupby(["Country", "Description"])["Revenue"].sum()
            .reset_index()
            .groupby("Country", group_keys=False)
            .apply(lambda x: x.nlargest(3, "Revenue"))
            .reset_index(drop=True)
        )
        fig_sun = px.sunburst(
            sun_df, path=["Country", "Description"], values="Revenue",
            color="Revenue", color_continuous_scale=palette
        )
        fig_sun.update_layout(
            paper_bgcolor=PAPER, font_color=FONT_C,
            margin=dict(l=0, r=0, t=10, b=0), height=380
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    with st.expander("📋 Explore Raw Data"):
        st.dataframe(dff.head(500), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 – RF PREDICTOR
# ══════════════════════════════════════════════
with tab6:
    st.markdown("<span class='section-header'>🤖 Random Forest Revenue Predictor</span>", unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#9090b0; font-size:0.92rem; margin-bottom:1.2rem;'>
    Trained on <b style='color:#a78bfa'>530,000+ real transactions</b> using
    <b style='color:#60a5fa'>100 decision trees</b>.
    Adjust the inputs below — the prediction updates instantly.
    </p>
    """, unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    with i1: st.metric("🌲 Trees",         "100")
    with i2: st.metric("📊 Training Rows", "50,000")
    with i3: st.metric("🎯 Features",      "4")
    with i4: st.metric("📐 Max Depth",     "8")

    st.markdown("---")

    # Feature importance
    st.markdown("<span class='section-header'>Feature Importance</span>", unsafe_allow_html=True)
    feat_names  = ["Quantity", "Unit Price", "Country", "Hour of Day"]
    importances = rf_model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance")
    )
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Plasma",
        text=fi_df["Importance"].apply(lambda x: f"{x:.3f}")
    )
    fig_fi.update_layout(**dark_layout(
        height=240, showlegend=False, coloraxis_showscale=False,
        extra_xaxis=dict(title="Importance Score"),
        extra_yaxis=dict(title="", showgrid=False)
    ))
    fig_fi.update_traces(textposition="outside",
                         textfont=dict(color=FONT_C))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("<span class='section-header'>🔢 Make a Prediction</span>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        qty = st.number_input(
            "📦 Quantity", min_value=1, max_value=1000,
            value=10, step=1, key="pred_qty"
        )
    with c2:
        price = st.number_input(
            "💷 Unit Price (£)", min_value=0.01, max_value=500.0,
            value=3.99, step=0.01, key="pred_price"
        )
    with c3:
        # Unique key → no collision with sidebar country filter
        uk_idx = ALL_COUNTRIES.index("United Kingdom") if "United Kingdom" in ALL_COUNTRIES else 0
        pred_country = st.selectbox(
            "🏳️ Transaction Country",
            ALL_COUNTRIES,
            index=uk_idx,
            key="pred_country"          # ← different key from "sidebar_country"
        )
    with c4:
        hour = st.slider("🕐 Hour of Day", 0, 23, 11, key="pred_hour")

    # Encode country — label_enc was fit on ALL countries so no unseen-label error
    country_enc = int(label_enc.transform([pred_country])[0])

    X_pred    = np.array([[qty, price, country_enc, hour]])
    pred_rev  = float(rf_model.predict(X_pred)[0])
    simple_rev = qty * price
    delta      = pred_rev - simple_rev
    delta_col  = "#34d399" if delta >= 0 else "#f87171"
    arrow      = "▲" if delta >= 0 else "▼"

    st.markdown(f"""
    <div class='pred-box'>
      <div class='pred-label'>🤖 Random Forest Predicted Revenue</div>
      <div class='pred-value'>£{pred_rev:,.2f}</div>
      <div class='pred-label' style='margin-top:0.8rem'>
        Simple (Qty × Price): <b style='color:#60a5fa'>£{simple_rev:,.2f}</b>
        &nbsp;|&nbsp;
        RF Adjustment: <b style='color:{delta_col}'>{arrow} £{abs(delta):,.2f}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quantity sensitivity sweep
    st.markdown("<span class='section-header'>📊 Revenue Sensitivity — Quantity Sweep</span>", unsafe_allow_html=True)
    qty_range = np.arange(1, 101)
    preds_q   = [float(rf_model.predict(np.array([[q, price, country_enc, hour]]))[0]) for q in qty_range]
    simple_q  = [q * price for q in qty_range]

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=qty_range, y=preds_q, name="RF Prediction",
        line=dict(color="#a78bfa", width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"
    ))
    fig_sens.add_trace(go.Scatter(
        x=qty_range, y=simple_q, name="Simple (Qty × Price)",
        line=dict(color="#60a5fa", width=2, dash="dot")
    ))
    fig_sens.add_vline(
        x=qty, line_dash="dash", line_color="#f472b6",
        annotation_text=f"Now: {qty}",
        annotation_font_color="#f472b6"
    )
    fig_sens.update_layout(**dark_layout(
        height=320,
        extra_xaxis=dict(title="Quantity"),
        extra_yaxis=dict(title="Revenue (£)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
    ))
    st.plotly_chart(fig_sens, use_container_width=True)

    # Price sensitivity sweep
    st.markdown("<span class='section-header'>💷 Revenue Sensitivity — Price Sweep</span>", unsafe_allow_html=True)
    price_range = np.linspace(0.5, 50.0, 100)
    preds_p     = [float(rf_model.predict(np.array([[qty, p, country_enc, hour]]))[0]) for p in price_range]
    simple_p    = [qty * p for p in price_range]

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=price_range, y=preds_p, name="RF Prediction",
        line=dict(color="#f472b6", width=2.5),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.1)"
    ))
    fig_p.add_trace(go.Scatter(
        x=price_range, y=simple_p, name="Simple (Qty × Price)",
        line=dict(color="#fbbf24", width=2, dash="dot")
    ))
    fig_p.add_vline(
        x=price, line_dash="dash", line_color="#a78bfa",
        annotation_text=f"Now: £{price:.2f}",
        annotation_font_color="#a78bfa"
    )
    fig_p.update_layout(**dark_layout(
        height=320,
        extra_xaxis=dict(title="Unit Price (£)"),
        extra_yaxis=dict(title="Revenue (£)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font_color=FONT_C)
    ))
    st.plotly_chart(fig_p, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  🛍️ Retail Intelligence Hub · Dark Edition ·
  Built with Streamlit &amp; Plotly ·
  Random Forest trained on 50K transactions · 541K+ rows
</div>
""", unsafe_allow_html=True)