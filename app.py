import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.auth import login_user, register_user, logout_user

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
    calculate_team_fdrs, calculate_team_strength_analysis, compute_team_differentials_asof,
    process_players, apply_team_differentials_and_recalc_option_b
)
from src.models import (
    perform_classical_assumption_tests, check_setpiece_taker,
    train_option_b_models, train_xpoints_model, build_option_c_model_and_view
)
from src.views.tab_player_stats import render_tab_player_stats
from src.views.tab_squad_planner import render_tab_squad_planner
from src.views.tab_visualizations import render_tab_visualizations
from src.views.tab_team_strength import render_tab_team_strength
from src.views.tab_fixtures import render_tab_fixtures
from src.views.tab_option_b import render_tab_option_b
from src.views.tab_option_c import render_tab_option_c
from src.views.tab_radar import render_player_comparison_radar_tab
from src.views.tab_hidden_gem import render_tab_hidden_gem
from src.views.navigation import (
    render_navigation_bar
)

# Page Configuration
st.set_page_config(
    page_title="FPL Analytics & xPoin Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Sports Broadcast Pro Styling (High Contrast Dark Sporty Theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

.main {
    background-color: #0b0f19;
}

/* Container Spacing & Max Width for Professional Look */
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 3rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    max-width: 1440px !important;
}

/* Header & Brand Container */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Typography Hierarchy - Sports Broadcast Pro */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.03em !important;
    color: #ffffff !important;
}

h1 { font-weight: 800 !important; }
h2 { font-weight: 800 !important; }
h3 { font-weight: 700 !important; }
h4 { font-weight: 700 !important; }

/* Sidebar Custom Styling - Dark Broadcast Tone */
section[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Keep filter labels readable against the dark sidebar, including when the
   Streamlit theme configured for the deployment is light. */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    color: #e2e8f0 !important;
}

/* Sidebar Inputs and Selects */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    border-radius: 8px !important;
    background-color: #161f33 !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.15s ease-in-out !important;
}

div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
    border-color: #06b6d4 !important;
}

/* Sports Broadcast Live Matchday Header */
.fpl-broadcast-header {
    background: linear-gradient(135deg, #090e1a 0%, #111a2e 50%, #0d233a 100%);
    border-radius: 12px;
    padding: 18px 24px;
    color: #ffffff;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.fpl-broadcast-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981 0%, #06b6d4 50%, #f59e0b 100%);
}

.fpl-brand-title {
    display: flex;
    align-items: center;
    gap: 16px;
}

.fpl-logo-badge {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    width: 48px;
    height: 48px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
}

.fpl-brand-name {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin: 0;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.fpl-brand-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 2px;
    margin-bottom: 0;
    font-family: 'JetBrains Mono', monospace;
}

.fpl-live-pulse-badge {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #10b981;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

.fpl-live-pulse-badge {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #10b981;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* Metric Cards Refinement - Broadcast Studio Aesthetic */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #131b2e 0%, #0f1626 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.15s ease, border-color 0.15s ease !important;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(6, 182, 212, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Streamlit/preview builds wrap metric labels differently; cover nested text. */
[data-testid="stMetric"] p,
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid*="Label"],
[data-testid="stMetric"] [data-testid*="Label"] * {
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
    font-weight: 650 !important;
    letter-spacing: 0 !important;
    opacity: 1 !important;
}

[data-testid="stMetric"] [data-testid*="Delta"],
[data-testid="stMetric"] [data-testid*="Delta"] * {
    color: #86efac !important;
    opacity: 1 !important;
}

[data-testid="stMetric"] [data-testid*="Delta"] svg {
    fill: #86efac !important;
    color: #86efac !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.55rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.03em !important;
}

/* Quick Filter Preset Bar */
.quick-preset-bar {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

/* Pearson Correlation Insight Card */
.corr-card {
    background: #111a2e;
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
}

.corr-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.corr-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #06b6d4;
    font-family: 'JetBrains Mono', monospace;
}

.corr-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Tabs Design - Auto-wrapping High-Contrast Dark */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #0d1322 !important;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    height: auto !important;
}

.stTabs [data-baseweb="tab"] {
    height: auto !important;
    min-height: 36px;
    border-radius: 6px;
    font-weight: 600;
    color: #94a3b8 !important;
    background-color: transparent !important;
    border: none !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    border: 1px solid rgba(6, 182, 212, 0.3) !important;
}

/* Dataframe & Tables styling */
div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden !important;
}

/* Clear keyboard focus and keep controls comfortable on touch screens. */
button:focus-visible, [role="button"]:focus-visible, input:focus-visible {
    outline: 2px solid #22d3ee !important;
    outline-offset: 2px !important;
}
div[data-testid="stButton"] button {
    min-height: 40px;
    border-radius: 8px;
    font-weight: 650;
    transition: border-color .15s ease, background-color .15s ease, transform .15s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: rgba(34, 211, 238, .55);
    transform: translateY(-1px);
}

@media (max-width: 768px) {
    .block-container {
        padding: .75rem 1rem 2rem !important;
    }
    .fpl-broadcast-header {
        align-items: flex-start;
        flex-direction: column;
        gap: 14px;
        padding: 16px;
    }
    .fpl-brand-title { gap: 11px; }
    .fpl-logo-badge { width: 40px; height: 40px; font-size: 20px; flex: 0 0 auto; }
    .fpl-brand-name { font-size: 1.12rem; flex-wrap: wrap; gap: 6px; }
    .fpl-brand-subtitle { font-size: .7rem; line-height: 1.5; }
    .fpl-live-pulse-badge { font-size: .68rem; padding: 5px 9px; }
    div[data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)

def main():
    # Fetch FPL raw data early for dynamic matchday ticker
    fpl_data = fetch_fpl_data()
    fixtures_data = fetch_fixtures_data()
    
    if not fpl_data:
        st.error("Data FPL tidak tersedia. Silakan muat ulang halaman.")
        return

    # Broadcast Live Matchday Header
    events = fpl_data.get('events', [])
    current_event = next((ev for ev in events if ev.get('is_current')), None)
    next_event = next((ev for ev in events if ev.get('is_next')), None)

    active_gw_id = (current_event or next_event or {}).get('id', get_current_gw(fpl_data))
    target_dl_event = next_event or current_event or {}
    raw_deadline = target_dl_event.get('deadline_time')
    dl_gw_id = target_dl_event.get('id', active_gw_id)

    def format_broadcast_deadline(dl_str):
        if not dl_str or str(dl_str).strip() in ['TBD', '', 'None']:
            return 'TBD'
        try:
            s = str(dl_str).replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            wib = timezone(timedelta(hours=7))
            dt_wib = dt.astimezone(wib)
            bulan = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
            return f"{dt_wib.day} {bulan[dt_wib.month]} {dt_wib.year}, {dt_wib.strftime('%H:%M')} WIB"
        except Exception:
            return str(dl_str).replace('T', ' ')[:16] + " UTC"

    deadline_fmt = format_broadcast_deadline(raw_deadline)
    deadline_text = f"DEADLINE GW {dl_gw_id}: {deadline_fmt}" if dl_gw_id != active_gw_id else f"DEADLINE: {deadline_fmt}"

    st.markdown(f"""
    <div class="fpl-broadcast-header">
        <div class="fpl-brand-title">
            <div class="fpl-logo-badge">⚡</div>
            <div>
                <h1 class="fpl-brand-name">
                    FPL SCOUT PRO <span style="font-size: 0.82rem; font-weight: 700; color: #06b6d4; font-family: 'JetBrains Mono', monospace; background: rgba(6,182,212,0.12); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(6,182,212,0.3);">BROADCAST SUITE</span>
                </h1>
                <p class="fpl-brand-subtitle">
                    MATCHDAY ENGINE &bull; GAMEWEEK {active_gw_id} &bull; {deadline_text}
                </p>
            </div>
        </div>
        <div class="fpl-live-pulse-badge">
            MATCHDAY LIVE &bull; OFFICIAL FPL FEED
        </div>
    </div>
    """, unsafe_allow_html=True)

    teams_dict = {t['id']: t['name'] for t in fpl_data.get('teams', [])}
    teams_short_dict = {t['id']: t.get('short_name', t['name'][:3].upper()) for t in fpl_data.get('teams', [])}
    
    # Calculate FDRs with official team short names
    fdr_summary = calculate_team_fdrs(fixtures_data, teams_dict, teams_short_dict=teams_short_dict)
    
    # Deteksi GW saat ini dan Load Histori Musim Lalu
    current_gw = get_current_gw(fpl_data)
    df_historical = load_historical_training_data()

    # Train 4 Positional Regression Models for xPoin (Option A) with accurate FDR extraction
    models_dict = train_xpoints_model(
        fpl_data.get('elements', []), fdr_summary, current_gw, df_historical, fixtures_data=fixtures_data
    )

    # Process Player Dataset (Without Option B models first to calculate team strength)
    players_df, team_dict = process_players(
        fpl_data, fdr_summary, models_dict, _opt_b_models=None
    )

    if players_df.empty:
        st.warning("Data pemain tidak ditemukan.")
        return

    # Calculate comprehensive team strength analysis once for shared use
    df_teams = calculate_team_strength_analysis(fpl_data, players_df, fdr_summary, fixtures_data)

    # Train Option B xG models by position and one pooled xA/90 model with position as a feature.
    opt_b_model_xg, opt_b_model_xa, stats_xg, stats_xa = train_option_b_models(
        fpl_data.get('elements', []), fdr_summary, current_gw, df_historical, _df_teams=df_teams, fixtures_data=fixtures_data
    )

    # -------------------------------------------------------------------------
    # GAMEWEEK DIFFERENTIAL FROM COMPLETED MATCH HISTORY
    # -------------------------------------------------------------------------
    current_gw_num = int(current_gw.get('id', 4) if isinstance(current_gw, dict) else current_gw)
    
    # Gunakan snapshot yang dihitung dari pertandingan selesai sebelum GW target.
    # Dokumen Firebase lama dapat berisi skor tim terkini yang dihitung ulang secara retrospektif.
    target_gw_by_team = {
        int(team_id): int(info['upcoming_all'][0]['gw'])
        for team_id, info in fdr_summary.items()
        if info.get('upcoming_all') and info['upcoming_all'][0].get('gw') is not None
    }
    historical_diffs = compute_team_differentials_asof(
        fpl_data.get('elements', []), fixtures_data,
        target_gws=sorted({int(f['event']) for f in fixtures_data if f.get('event') is not None})
    )
    team_diffs_predict = {}
    for team_id, target_gw in target_gw_by_team.items():
        info = historical_diffs.get((target_gw, team_id))
        if info:
            team_name = teams_dict.get(team_id, f"Team {team_id}")
            opponent_name = teams_dict.get(info['opponent_id'], f"Team {info['opponent_id']}")
            team_diff = {**info, 'opponent': opponent_name}
            team_diffs_predict[team_name] = team_diff
            team_diffs_predict[team_id] = team_diff

    # Enrich players_df with Diff Attack Team & Diff Defense Team and calibrate Option B for the NEXT match
    players_df = apply_team_differentials_and_recalc_option_b(
        players_df, team_diffs_predict, _opt_b_models=(opt_b_model_xg, opt_b_model_xa)
    )
    
    # -------------------------------------------------------------------------
    # SIDEBAR QUICK NAVIGATION & FILTERS
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

    # Pre-fetch Option C data for multi-option comparison (cached)
    df_opt_c, _ = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw, fixtures_data=fixtures_data)

    def render_content_by_id(mod_id):
        if mod_id == "player_stats":
            render_tab_player_stats(filtered_players, players_df, models_dict, fpl_data, teams_dict, fdr_summary=fdr_summary)
        elif mod_id == "squad_planner":
            render_tab_squad_planner(players_df, fpl_data, fdr_summary, current_gw, df_option_c=df_opt_c)
        elif mod_id == "visualizations":
            render_tab_visualizations(filtered_players, players_df, fpl_data, teams_dict, fdr_summary=fdr_summary)
        elif mod_id == "hidden_gem":
            filtered_player_ids = set(filtered_players['id'].tolist()) if 'id' in filtered_players.columns else None
            render_tab_hidden_gem(fpl_data, fdr_summary, current_gw, filtered_player_ids=filtered_player_ids, teams_dict=teams_dict)
        elif mod_id == "option_b":
            render_tab_option_b(
                filtered_players, stats_xg, stats_xa,
                df_teams=df_teams, fixtures=fixtures_data, teams_dict=teams_dict, current_gw=current_gw_num
            )
        elif mod_id == "option_c":
            render_tab_option_c(fpl_data, fdr_summary, current_gw, filtered_players=filtered_players, price_range=price_range, df_option_c=df_opt_c, fixtures_data=fixtures_data)
        elif mod_id == "team_strength":
            render_tab_team_strength(fpl_data, players_df, fdr_summary, fixtures_data, teams_dict)
        elif mod_id == "fixtures":
            render_tab_fixtures(fixtures_data, teams_dict, fdr_summary, fpl_data=fpl_data, df_teams=df_teams, players_df=players_df)

    # Categorized navigation keeps module selection predictable and compact.
    active_nav_id = render_navigation_bar()
    render_content_by_id(active_nav_id)

if __name__ == "__main__":
    main()
