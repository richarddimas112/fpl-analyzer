import streamlit as st
import pandas as pd
import numpy as np

from src.constants import (
    BOOTSTRAP_URL, FIXTURES_URL, ELEMENT_SUMMARY_URL,
    POSITION_MAP, STATUS_MAP, POS_MODEL_CONFIGS
)
from src.api import (
    fetch_fpl_data, fetch_fixtures_data, fetch_player_history,
    fetch_player_element_summary, fetch_player_history_raw
)
from src.processors import (
    get_current_gw, load_historical_training_data, format_setpiece_order,
    calculate_team_fdrs, calculate_team_strength_analysis,
    compute_all_l5m_avg_mins, process_players
)
from src.models import (
    perform_classical_assumption_tests, check_setpiece_taker,
    train_option_b_models, train_xpoints_model, build_option_c_model_and_view
)
from src.views.tab_player_stats import render_tab_player_stats
from src.views.tab_visualizations import render_tab_visualizations
from src.views.tab_team_strength import render_tab_team_strength
from src.views.tab_fixtures import render_tab_fixtures
from src.views.tab_option_b import render_tab_option_b
from src.views.tab_option_c import render_tab_option_c
from src.views.tab_radar import render_player_comparison_radar_tab
from src.views.tab_hidden_gem import render_tab_hidden_gem

# Page Configuration
st.set_page_config(
    page_title="FPL Analytics & xPoin Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0f172a;
}

.main {
    background-color: #f8fafc;
}

/* Header & Brand Container */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #0f172a;
    font-weight: 700;
}

/* Sidebar Inputs and Selects */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    border-radius: 8px !important;
    border-color: #cbd5e1 !important;
}

/* Brand Banner Card */
.fpl-brand-header {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
    border-radius: 14px;
    padding: 24px 32px;
    color: #ffffff;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.fpl-brand-title {
    display: flex;
    align-items: center;
    gap: 18px;
}

.fpl-logo-badge {
    background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
    width: 52px;
    height: 52px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 4px 12px rgba(56, 239, 125, 0.35);
}

.fpl-brand-name {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin: 0;
    color: #ffffff;
}

.fpl-brand-subtitle {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 4px;
    margin-bottom: 0;
}

.fpl-status-badge {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #38ef7d;
    display: flex;
    align-items: center;
    gap: 6px;
}

.fpl-status-dot {
    width: 8px;
    height: 8px;
    background-color: #38ef7d;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #38ef7d;
}

/* Metric Cards Refinement */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

div[data-testid="stMetricValue"] {
    color: #0f172a;
    font-weight: 800;
    font-size: 1.65rem;
    letter-spacing: -0.02em;
}

/* Pearson Correlation Insight Card */
.corr-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
}

.corr-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
}

.corr-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #2563eb;
}

.corr-desc {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 4px;
}

/* Tabs Design */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f1f5f9;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 6px;
    font-weight: 600;
    color: #64748b;
    border: none !important;
    background-color: transparent;
    padding: 0 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
}

/* Table Container Styling */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid #cbd5e1;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #94a3b8;
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Brand Top Banner
    st.markdown("""
    <div class="fpl-brand-header">
        <div class="fpl-brand-title">
            <div class="fpl-logo-badge">⚽</div>
            <div>
                <h1 class="fpl-brand-name">FPL Scout Analytics & xPoin Predictor</h1>
                <p class="fpl-brand-subtitle">Platform Analisis Data Resmi Premier League & Model Prediksi Performa Pemain</p>
            </div>
        </div>
        <div class="fpl-status-badge">
            <span class="fpl-status-dot"></span> Live API Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch FPL raw data
    fpl_data = fetch_fpl_data()
    fixtures_data = fetch_fixtures_data()
    
    if not fpl_data:
        st.error("Data FPL tidak tersedia. Silakan muat ulang halaman.")
        return

    teams_dict = {t['id']: t['name'] for t in fpl_data.get('teams', [])}
    
    # Calculate FDRs
    fdr_summary = calculate_team_fdrs(fixtures_data, teams_dict)
    
    # Deteksi GW saat ini dan Load Histori Musim Lalu
    current_gw = get_current_gw(fpl_data)
    df_historical = load_historical_training_data()

    # Train 4 Positional Regression Models for xPoin (Option A)
    models_dict = train_xpoints_model(
        fpl_data.get('elements', []), fdr_summary, current_gw, df_historical
    )

    # Train Option B Match xG & xA Models (Option B)
    opt_b_model_xg, opt_b_model_xa, stats_xg, stats_xa = train_option_b_models(
        fpl_data.get('elements', []), fdr_summary, current_gw, df_historical
    )

    # Process Player Dataset
    players_df, team_dict = process_players(
        fpl_data, fdr_summary, models_dict, _opt_b_models=(opt_b_model_xg, opt_b_model_xa)
    )

    if players_df.empty:
        st.warning("Data pemain tidak ditemukan.")
        return

    # -------------------------------------------------------------------------
    # SIDEBAR FILTERS
    # -------------------------------------------------------------------------
    st.sidebar.header("🔍 Filter Pemain FPL")

    # Search Bar
    search_query = st.sidebar.text_input("Cari Nama Pemain", "", placeholder="Misal: Haaland, Palmer...")

    # Position Filter
    all_positions = ["GK", "DEF", "MID", "FWD"]
    selected_positions = st.sidebar.multiselect("Posisi", options=all_positions, default=all_positions)

    # Club Filter
    all_clubs = sorted(list(team_dict.values()))
    selected_clubs = st.sidebar.multiselect("Klub", options=all_clubs, default=all_clubs)

    # Price Slider
    min_price = float(players_df['Harga (£m)'].min())
    max_price = float(players_df['Harga (£m)'].max())
    price_range = st.sidebar.slider(
        "Rentang Harga (£m)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=0.5
    )

    # Minutes Slider
    min_mins_played = st.sidebar.slider(
        "Minimal Menit Bermain",
        min_value=0,
        max_value=int(players_df['Menit Bermain'].max()),
        value=0,
        step=90
    )

    # Ownership Slider (% Ownership)
    min_own = float(players_df['% Ownership'].min()) if not players_df.empty else 0.0
    max_own = float(players_df['% Ownership'].max()) if not players_df.empty else 100.0
    own_range = st.sidebar.slider(
        "Rentang % Ownership",
        min_value=0.0,
        max_value=max_own if max_own > 0 else 100.0,
        value=(0.0, max_own if max_own > 0 else 100.0),
        step=0.5,
        format="%.1f%%"
    )

    # Total Points Slider
    min_poin = int(players_df['Total Poin'].min()) if not players_df.empty else 0
    max_poin = int(players_df['Total Poin'].max()) if not players_df.empty else 300
    poin_range = st.sidebar.slider(
        "Rentang Total Poin",
        min_value=min_poin,
        max_value=max_poin if max_poin > min_poin else min_poin + 1,
        value=(min_poin, max_poin if max_poin > min_poin else min_poin + 1),
        step=1
    )

    # Avg Mins (L5M) Slider
    max_l5m = float(players_df['Avg Mins (L5M)'].max()) if not players_df.empty else 90.0
    min_l5m_mins = st.sidebar.slider(
        "Minimal Avg Mins (L5M)",
        min_value=0.0,
        max_value=max_l5m if max_l5m > 0 else 90.0,
        value=0.0,
        step=5.0,
        format="%.1f"
    )

    # Apply Filters
    filtered_players = players_df[
        (players_df['Posisi'].isin(selected_positions)) &
        (players_df['Klub'].isin(selected_clubs)) &
        (players_df['Harga (£m)'] >= price_range[0]) &
        (players_df['Harga (£m)'] <= price_range[1]) &
        (players_df['Menit Bermain'] >= min_mins_played) &
        (players_df['% Ownership'] >= own_range[0]) &
        (players_df['% Ownership'] <= own_range[1]) &
        (players_df['Total Poin'] >= poin_range[0]) &
        (players_df['Total Poin'] <= poin_range[1]) &
        (players_df['Avg Mins (L5M)'] >= min_l5m_mins)
    ]

    if search_query:
        filtered_players = filtered_players[
            filtered_players['Nama Pemain'].str.contains(search_query, case=False, na=False) |
            filtered_players['Nama Lengkap'].str.contains(search_query, case=False, na=False)
        ]

    # Tabs Layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Player Stats & xPoin", 
        "📈 Visualisasi Data & Radar Pemain", 
        "🛡️ Team Strength Analysis",
        "📅 Fixtures & FDR", 
        "🧮 Option B: Component Model xPoin",
        "🔮 Option C: Current Season Model",
        "💎 Hidden Gem & Haul Predictor"
    ])

    with tab1:
        render_tab_player_stats(filtered_players, players_df, models_dict, fpl_data, teams_dict)

    with tab2:
        render_tab_visualizations(filtered_players, players_df, fpl_data, teams_dict)

    with tab3:
        render_tab_team_strength(fpl_data, players_df, fdr_summary, fixtures_data, teams_dict)

    with tab4:
        render_tab_fixtures(fixtures_data, teams_dict, fdr_summary)

    with tab5:
        render_tab_option_b(filtered_players, stats_xg, stats_xa)

    with tab6:
        render_tab_option_c(fpl_data, fdr_summary, current_gw)

    with tab7:
        render_tab_hidden_gem(fpl_data, fdr_summary, current_gw)

if __name__ == "__main__":
    main()
