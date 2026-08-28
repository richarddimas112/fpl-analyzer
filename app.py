import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, shapiro, poisson, percentileofscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING (Clean Minimalism Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FPL Analytics & xPoin Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Clean Minimalism" design theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }

    /* Hide default Streamlit top decoration bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Sidebar Styling - Crisp White Minimalist */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #37003c !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Sidebar Inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }

    /* Top Brand Header */
    .fpl-brand-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .fpl-brand-title {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .fpl-logo-badge {
        width: 40px;
        height: 40px;
        background-color: #37003c;
        color: #00ff85;
        font-weight: 900;
        font-size: 1.3rem;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .fpl-brand-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: #37003c;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .fpl-brand-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin: 2px 0 0 0;
        font-weight: 500;
    }
    .fpl-status-badge {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #37003c;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .fpl-status-dot {
        width: 8px;
        height: 8px;
        background-color: #00e676;
        border-radius: 50%;
        display: inline-block;
    }

    /* Metric Cards - Clean Minimalism */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        font-weight: 800 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #37003c !important;
    }

    /* Correlation Callout Box */
    .corr-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #37003c;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .corr-title {
        font-size: 0.8rem;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .corr-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #37003c;
        margin: 4px 0;
    }
    .corr-desc {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
    }

    /* Streamlit Tabs - Clean Border Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #64748b !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0 12px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #37003c !important;
        border-bottom: 2px solid #37003c !important;
    }

    /* DataFrame Table Container */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 4px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }

    /* Radar & Comparison Custom Styles */
    .compare-card-p1 {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0284c7;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .compare-card-p2 {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #e11d48;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .compare-tag-p1 {
        display: inline-block;
        background: #e0f2fe;
        color: #0284c7;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .compare-tag-p2 {
        display: inline-block;
        background: #ffe4e6;
        color: #e11d48;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .vs-circle {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background-color: #37003c;
        color: #00ff85;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1rem;
        letter-spacing: -0.02em;
        margin: 0 auto;
        box-shadow: 0 2px 6px rgba(55, 0, 60, 0.25);
    }
    .radar-summary-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #00ff85;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & MAPS
# -----------------------------------------------------------------------------
import os

# Fungsi untuk mendeteksi GW yang sedang berjalan/akan datang
def get_current_gw(fpl_data):
    events = fpl_data.get('events', [])
    for ev in events:
        if ev.get('is_current'):
            return ev.get('id')
    # Jika pre-season (belum ada yang current), cari GW berikutnya
    for ev in events:
        if ev.get('is_next'):
            return ev.get('id')
    return 1

# Fungsi untuk memuat data historis
@st.cache_data(ttl=86400)
def load_historical_training_data():
    """Load data histori musim lalu untuk stabilisasi model di GW1-GW10."""
    hist_file = "data/historical_train_data.csv"
    if os.path.exists(hist_file):
        try:
            return pd.read_csv(hist_file)
        except Exception as e:
            st.warning(f"Gagal membaca data historis: {e}")
    return pd.DataFrame()

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

STATUS_MAP = {
    'a': '✅ Available',
    'd': '⚠️ Doubtful',
    'i': '🚑 Injured',
    's': '🛑 Suspended',
    'u': '❌ Unavailable'
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def format_setpiece_order(val):
    if pd.isnull(val) or val is None or str(val).strip() == "":
        return "-"
    try:
        v_int = int(float(val))
        return str(v_int) if v_int > 0 else "-"
    except Exception:
        return str(val)

# -----------------------------------------------------------------------------
# DATA CACHING & FETCHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_fpl_data():
    """Fetch main bootstrap data from FPL API."""
    try:
        response = requests.get(BOOTSTRAP_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_fixtures_data():
    """Fetch fixtures schedule from FPL API."""
    try:
        response = requests.get(FIXTURES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []

def fetch_player_history_raw(player_id):
    """Fetch per-match summary history for a given player directly."""
    try:
        url = ELEMENT_SUMMARY_URL.format(player_id)
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            return res.json().get('history', [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=86400)
def fetch_player_history(player_id):
    """Fetch cached per-match summary history for a given player."""
    return fetch_player_history_raw(player_id)

@st.cache_data(ttl=86400)
def fetch_player_element_summary(player_id):
    """Fetch complete element-summary (history, history_past, fixtures) for a given player."""
    try:
        url = ELEMENT_SUMMARY_URL.format(player_id)
        res = requests.get(url, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {'history': [], 'history_past': [], 'fixtures': []}

@st.cache_data(ttl=86400)
def calculate_team_strength_analysis(_fpl_data, players_df, fdr_summary):
    """
    Aggregate comprehensive Premier League team data from player statistics and FPL APIs:
    - Average FPL points of players (and active players)
    - Total squad points and squad market value
    - Offensive metrics: Total Goals, Total Assists, Total xG, Total xA, Total xGI, Top Scorer
    - Defensive metrics: Clean Sheets, Total xGC, Total Saves, Defensive Contribution
    - Fixture Difficulty Ratings: FDR1, FDR3, FDR5, Next Opponent (Home/Away), Schedule Ease
    - Normalized Composite Team Strength Index (0 - 100) and Strength Tiers
    """
    teams = _fpl_data.get('teams', []) if _fpl_data else []
    records = []
    
    for t in teams:
        t_id = t['id']
        t_name = t['name']
        t_short = t.get('short_name', '')
        
        # Filter players for this team
        if 'team' in players_df.columns:
            tp = players_df[players_df['team'] == t_id]
        else:
            tp = players_df[players_df['Klub'] == t_name]
            
        total_players = len(tp)
        if total_players > 0:
            # Pastikan hanya menghitung pemain yang sudah bermain (Menit Bermain > 0)
            active = tp[tp['Menit Bermain'] > 0]
            total_points = int(tp['Total Poin'].sum())
            # Rata-rata poin pemain dihitung HANYA untuk pemain yang sudah bermain (Menit Bermain > 0)
            avg_points_played = float(active['Total Poin'].mean()) if not active.empty else 0.0
            avg_points_all = float(tp['Total Poin'].mean())
            
            total_goals = int(tp['Gol'].sum())
            total_assists = int(tp['Asis'].sum())
            total_xg = float(tp['xG'].sum())
            total_xa = float(tp['xA'].sum())
            total_xgi = float(tp['xGI'].sum())
            
            def_gk = tp[tp['Posisi'].isin(['DEF', 'GK'])]
            clean_sheets = int(def_gk['Clean Sheet'].max()) if not def_gk.empty else 0
            
            if 'expected_goals_conceded' in def_gk.columns:
                total_xgc = float(pd.to_numeric(def_gk['expected_goals_conceded'], errors='coerce').fillna(0.0).sum())
            elif 'xGC' in def_gk.columns:
                total_xgc = float(pd.to_numeric(def_gk['xGC'], errors='coerce').fillna(0.0).sum())
            else:
                total_xgc = 0.0
                
            total_saves = int(tp['Saves'].sum()) if 'Saves' in tp.columns else 0
            avg_form = float(active['Form'].mean()) if not active.empty else 0.0
            total_squad_value = float(tp['Harga (£m)'].sum()) if 'Harga (£m)' in tp.columns else 0.0
            total_bps = int(tp['BPS'].sum()) if 'BPS' in tp.columns else 0
            
            # Top Scorer & Top Creator
            sorted_by_goals = tp.sort_values(by=['Gol', 'xG'], ascending=False)
            top_scorer_row = sorted_by_goals.iloc[0] if not sorted_by_goals.empty else None
            top_scorer_name = f"{top_scorer_row['Nama Pemain']} ({int(top_scorer_row['Gol'])}G)" if top_scorer_row is not None and top_scorer_row['Gol'] > 0 else (top_scorer_row['Nama Pemain'] if top_scorer_row is not None else "-")
            
            sorted_by_assists = tp.sort_values(by=['Asis', 'xA'], ascending=False)
            top_creator_row = sorted_by_assists.iloc[0] if not sorted_by_assists.empty else None
            top_creator_name = f"{top_creator_row['Nama Pemain']} ({int(top_creator_row['Asis'])}A)" if top_creator_row is not None and top_creator_row['Asis'] > 0 else (top_creator_row['Nama Pemain'] if top_creator_row is not None else "-")
            
            sorted_by_pts = tp.sort_values(by=['Total Poin', 'xPoin'], ascending=False)
            top_asset_row = sorted_by_pts.iloc[0] if not sorted_by_pts.empty else None
            top_asset_name = f"{top_asset_row['Nama Pemain']} ({int(top_asset_row['Total Poin'])} pts)" if top_asset_row is not None else "-"
        else:
            total_points = 0
            avg_points_played = 0.0
            avg_points_all = 0.0
            total_goals = 0
            total_assists = 0
            total_xg = 0.0
            total_xa = 0.0
            total_xgi = 0.0
            clean_sheets = 0
            total_xgc = 0.0
            total_saves = 0
            avg_form = 0.0
            total_squad_value = 0.0
            total_bps = 0
            top_scorer_name = "-"
            top_creator_name = "-"
            top_asset_name = "-"
            
        f_info = fdr_summary.get(t_id, {})
        fdr1 = float(f_info.get('FDR1', 3.0))
        fdr3 = float(f_info.get('FDR3', 3.0))
        fdr5 = float(f_info.get('FDR5', 3.0))
        next_opp = f_info.get('Next_Opponent_Fmt', '-')
        
        str_ovr_h = t.get('strength_overall_home', 3) or 3
        str_ovr_a = t.get('strength_overall_away', 3) or 3
        str_ovr = round((str_ovr_h + str_ovr_a) / 2.0, 1)
        
        records.append({
            'team_id': t_id,
            'Klub': t_name,
            'Kode': t_short,
            'Total Pemain': total_players,
            'Pemain Aktif': len(active) if total_players > 0 else 0,
            'Total Poin Skuad': total_points,
            'Rata-rata Poin Pemain': round(avg_points_played, 2),
            'Rata-rata Poin Skuad': round(avg_points_all, 2),
            'Total Gol': total_goals,
            'Total Asis': total_assists,
            'Total xG': round(total_xg, 2),
            'Total xA': round(total_xa, 2),
            'Total xGI': round(total_xgi, 2),
            'Clean Sheet': clean_sheets,
            'Total xGC': round(total_xgc, 2),
            'Total Saves': total_saves,
            'Form Rata-rata': round(avg_form, 2),
            'Nilai Skuad (£m)': round(total_squad_value, 1),
            'Total BPS': total_bps,
            'Top Scorer': top_scorer_name,
            'Top Creator': top_creator_name,
            'Top Aset FPL': top_asset_name,
            'FDR1': round(fdr1, 1),
            'FDR3': round(fdr3, 2),
            'FDR5': round(fdr5, 2),
            'Lawan Berikutnya': next_opp,
            'Official_Strength': str_ovr
        })
        
    df_teams = pd.DataFrame(records)
    if df_teams.empty:
        return df_teams
        
    def min_max_scale(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50.0, index=series.index)
        scaled = (series - mn) / (mx - mn) * 100.0
        return (100.0 - scaled) if invert else scaled

    att_score = 0.6 * min_max_scale(df_teams['Total xG']) + 0.4 * min_max_scale(df_teams['Total Gol'])
    def_score = 0.5 * min_max_scale(df_teams['Clean Sheet']) + 0.5 * min_max_scale(df_teams['Total xGC'], invert=True)
    pts_score = 0.6 * min_max_scale(df_teams['Rata-rata Poin Pemain']) + 0.4 * min_max_scale(df_teams['Total Poin Skuad'])
    fdr_score = min_max_scale(df_teams['FDR3'], invert=True)
    base_score = min_max_scale(df_teams['Official_Strength'])
    
    # 20% Metrik Serangan, 20% Soliditas Pertahanan, 15% Efisiensi Poin Pemain Aktif, 30% Kekuatan Resmi Premier League, 15% Kemudahan Jadwal FDR
    composite = (0.20 * att_score + 0.20 * def_score + 0.15 * pts_score + 0.30 * base_score + 0.15 * fdr_score).round(1)
    
    df_teams['Indeks Kekuatan'] = composite
    df_teams['Skor Serangan'] = att_score.round(1)
    df_teams['Skor Pertahanan'] = def_score.round(1)
    df_teams['Kemudahan Jadwal (%)'] = fdr_score.round(1)
    
    def assign_tier(score):
        if score >= 75:
            return "🏆 Elite Contender"
        elif score >= 60:
            return "🌟 Top Tier Challenger"
        elif score >= 48:
            return "⚖️ Mid-Table Stable"
        else:
            return "⚠️ Underdogs / Rebuilding"
            
    df_teams['Kategori Tim'] = df_teams['Indeks Kekuatan'].apply(assign_tier)
    return df_teams

# -----------------------------------------------------------------------------
# FIXTURE DIFFICULTY RATING (FDR1, FDR3, FDR5) CALCULATIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def calculate_team_fdrs(fixtures, teams_dict):
    """Calculate FDR1, FDR3, FDR5 and next match home status & opponent info for every team."""
    team_upcoming = {t_id: [] for t_id in teams_dict.keys()}
    
    for f in fixtures:
        if not f.get('finished'):
            h_id = f.get('team_h')
            a_id = f.get('team_a')
            h_diff = f.get('team_h_difficulty', 3)
            a_diff = f.get('team_a_difficulty', 3)
            
            if h_id in team_upcoming:
                team_upcoming[h_id].append({'fdr': h_diff, 'is_home': 1, 'gw': f.get('event'), 'opp_id': a_id})
            if a_id in team_upcoming:
                team_upcoming[a_id].append({'fdr': a_diff, 'is_home': 0, 'gw': f.get('event'), 'opp_id': h_id})
                
    fdr_summary = {}
    for t_id, fxs in team_upcoming.items():
        if fxs:
            f1 = float(fxs[0]['fdr'])
            next_is_home = fxs[0]['is_home']
            next_opp_id = fxs[0].get('opp_id')
            f3 = float(np.mean([x['fdr'] for x in fxs[:3]])) if len(fxs) >= 3 else f1
            f5 = float(np.mean([x['fdr'] for x in fxs[:5]])) if len(fxs) >= 5 else f3
        else:
            f1 = 3.0
            f3 = 3.0
            f5 = 3.0
            next_is_home = 1
            next_opp_id = None

        opp_name = teams_dict.get(next_opp_id, 'TBD') if next_opp_id else 'TBD'
        opp_fmt = f"{opp_name} ({'🏠' if next_is_home == 1 else '✈️'})" if next_opp_id else "-"
            
        fdr_summary[t_id] = {
            'FDR1': round(f1, 2),
            'FDR3': round(f3, 2),
            'FDR5': round(f5, 2),
            'Next_Is_Home': next_is_home,
            'Next_Opponent_ID': next_opp_id,
            'Next_Opponent_Name': opp_name,
            'Next_Opponent_Fmt': opp_fmt
        }
        
    return fdr_summary

# -----------------------------------------------------------------------------
# CLASSICAL ASSUMPTION DIAGNOSTICS (REGRESSION ASSUMPTION TESTS)
# -----------------------------------------------------------------------------
def perform_classical_assumption_tests(X, y, feature_labels):
    """
    Menghitung 4 Uji Asumsi Klasik Regresi Linear:
    1. Uji Multikolinearitas (VIF & Tolerance)
    2. Uji Linearitas (linear_rainbow)
    3. Uji Normalitas Residual (scipy.stats.shapiro)
    4. Uji Homoskedastisitas (het_breuschpagan)
    """
    try:
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        residuals = ols_model.resid

        # 1. Uji Multikolinearitas (VIF & Tolerance)
        vif_rows = []
        all_vif_ok = True
        for i, col in enumerate(X.columns):
            label = feature_labels[i] if i < len(feature_labels) else col
            vif_val = float(variance_inflation_factor(X_const.values, i + 1))
            tol_val = 1.0 / vif_val if vif_val != 0 else 0.0
            
            is_ok = (vif_val < 10.0) and (tol_val > 0.10)
            if not is_ok:
                all_vif_ok = False
                
            status_text = "Bebas Multikolinearitas" if is_ok else "Terindikasi Multikolinearitas"
            vif_rows.append({
                'Variabel': label,
                'VIF': round(vif_val, 4),
                'Tolerance': round(tol_val, 4),
                'Status': status_text
            })
        vif_df = pd.DataFrame(vif_rows)

        # 2. Uji Linearitas (linear_rainbow)
        rb_stat, rb_p = linear_rainbow(ols_model)
        rb_passed = float(rb_p) > 0.05
        rb_status = "Linear" if rb_passed else "Non-Linear"

        # 3. Uji Normalitas Residual (Shapiro-Wilk)
        sh_stat, sh_p = shapiro(residuals)
        sh_passed = float(sh_p) > 0.05
        sh_status = "Residual Normal" if sh_passed else "Tidak Normal"

        # 4. Uji Homoskedastisitas (Breusch-Pagan)
        lm_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, X_const)
        bp_passed = float(bp_p) > 0.05
        bp_status = "Homoskedastisitas (Varian Konstan)" if bp_passed else "Heteroskedastisitas"

        return {
            'vif_df': vif_df,
            'all_vif_ok': all_vif_ok,
            'linearity': {
                'stat': round(float(rb_stat), 4),
                'p_value': round(float(rb_p), 4),
                'status': rb_status,
                'passed': rb_passed
            },
            'normality': {
                'stat': round(float(sh_stat), 4),
                'p_value': round(float(sh_p), 4),
                'status': sh_status,
                'passed': sh_passed
            },
            'homoscedasticity': {
                'stat': round(float(lm_stat), 4),
                'p_value': round(float(bp_p), 4),
                'status': bp_status,
                'passed': bp_passed
            }
        }
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# MACHINE LEARNING REGRESSION MODELS (4 POSITIONAL MODELS)
# -----------------------------------------------------------------------------
POS_MODEL_CONFIGS = {
    'FWD': {
        'element_type': 4,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'ICT Index'
        ]
    },
    'MID': {
        'element_type': 3,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'Defensive_Contribution_per_90', 'xGC_per_90', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'Kontribusi Defensif per 90',
            'xGC per 90 (Expected Goals Conceded)',
            'ICT Index'
        ]
    },
    'DEF': {
        'element_type': 2,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'Defensive_Contribution_per_90', 'xGC_per_90', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'Kontribusi Defensif per 90',
            'xGC per 90 (Expected Goals Conceded)',
            'ICT Index'
        ]
    },
    'GK': {
        'element_type': 1,
        'feature_cols': ['bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'xGC_per_90', 'Saves_per_90'],
        'feature_labels': [
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'xGC per 90 (Expected Goals Conceded)',
            'Saves per 90'
        ]
    }
}

def check_setpiece_taker(corner_ord, fk_ord):
    """Return 1 if corner_order <= 2 or freekick_order <= 2, else 0."""
    try:
        if corner_ord is not None and str(corner_ord).strip() not in ["", "None", "-"] and int(corner_ord) <= 2:
            return 1
    except Exception:
        pass
    try:
        if fk_ord is not None and str(fk_ord).strip() not in ["", "None", "-"] and int(fk_ord) <= 2:
            return 1
    except Exception:
        pass
    return 0

@st.cache_data(ttl=86400)
def train_option_b_models(players_list, fdr_summary, current_gw, df_historical):
    """
    Train LinearRegression models for upcoming match xG and xA prediction (Option B)
    separately for positions 'FWD', 'MID', 'DEF'. Excludes 'GK'.
    """
    pos_el_map = {
        'FWD': 4,
        'MID': 3,
        'DEF': 2
    }

    opt_b_models_xg = {}
    opt_b_models_xa = {}
    stats_xg = {}
    stats_xa = {}

    for pos_key, el_type in pos_el_map.items():
        pos_players = [p for p in players_list if p.get('element_type') == el_type]
        top_players = sorted(pos_players, key=lambda p: (p.get('total_points', 0), p.get('minutes', 0)), reverse=True)[:35]

        history_xg_rows = []
        history_xa_rows = []

        for p in top_players:
            p_form = float(p.get('form', 0.0))
            c_ord = p.get('corners_and_indirect_freekicks_order')
            fk_ord = p.get('direct_freekicks_order')
            is_sp = check_setpiece_taker(c_ord, fk_ord)

            p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: m.get('round', m.get('event', 0)))
                for i, m in enumerate(sorted_hist):
                    mins = int(m.get('minutes', 0))
                    if mins > 0:
                        prev_5 = sorted_hist[max(0, i-4):i+1]
                        s_mins = sum(int(x.get('minutes', 0)) for x in prev_5)
                        xg_l5m = (sum(float(x.get('expected_goals', 0.0)) for x in prev_5) / max(1, s_mins)) * 90.0
                        xa_l5m = (sum(float(x.get('expected_assists', 0.0)) for x in prev_5) / max(1, s_mins)) * 90.0
                        ict_90 = (float(m.get('ict_index', 0.0)) / mins) * 90.0
                        
                        was_home = 1 if m.get('was_home') else 0
                        fdr = int(m.get('opponent_team', 3)) if isinstance(m.get('opponent_team'), (int, float)) else 3
                        opp_xgc = float(m.get('expected_goals_conceded', 1.25))

                        xg_match = float(m.get('expected_goals', 0.0))
                        xa_match = float(m.get('expected_assists', 0.0))

                        history_xg_rows.append({
                            'xG_per_90_L5M': xg_l5m,
                            'ict_index_per_90': ict_90,
                            'form': p_form,
                            'was_home': was_home,
                            'FDR': fdr,
                            'Opponent_xGC_per_90': opp_xgc,
                            'xG_match': xg_match
                        })

                        history_xa_rows.append({
                            'xA_per_90_L5M': xa_l5m,
                            'ict_index_per_90': ict_90,
                            'is_setpiece_taker': is_sp,
                            'form': p_form,
                            'was_home': was_home,
                            'FDR': fdr,
                            'Opponent_xGC_per_90': opp_xgc,
                            'xA_match': xa_match
                        })

        if len(history_xg_rows) >= 15:
            df_xg = pd.DataFrame(history_xg_rows)
            df_xa = pd.DataFrame(history_xa_rows)
        else:
            # Position-adjusted realistic synthetic fallback
            np.random.seed(42 + el_type)
            N = 250
            if pos_key == 'FWD':
                xg_l5m = np.random.uniform(0.15, 0.85, N)
                xa_l5m = np.random.uniform(0.05, 0.45, N)
                mult_xg, mult_xa = 0.45, 0.25
            elif pos_key == 'MID':
                xg_l5m = np.random.uniform(0.05, 0.60, N)
                xa_l5m = np.random.uniform(0.05, 0.55, N)
                mult_xg, mult_xa = 0.35, 0.35
            else:  # DEF - substantially lower offensive output
                xg_l5m = np.random.uniform(0.00, 0.12, N)
                xa_l5m = np.random.uniform(0.00, 0.20, N)
                mult_xg, mult_xa = 0.10, 0.12

            ict_90 = np.random.uniform(1.0, 12.0, N)
            sp_taker = np.random.choice([0, 1], N)
            form = np.random.uniform(0.5, 8.5, N)
            home = np.random.choice([0, 1], N)
            fdr = np.random.choice([1, 2, 3, 4, 5], N)
            opp_xgc = np.random.uniform(0.5, 2.5, N)

            y_xg = np.maximum(0, mult_xg*xg_l5m + 0.008*ict_90 + 0.008*form + 0.02*home - 0.01*fdr + 0.02*opp_xgc + np.random.normal(0, 0.015, N))
            y_xa = np.maximum(0, mult_xa*xa_l5m + 0.008*ict_90 + 0.04*sp_taker + 0.008*form + 0.02*home - 0.01*fdr + 0.02*opp_xgc + np.random.normal(0, 0.015, N))

            df_xg = pd.DataFrame({
                'xG_per_90_L5M': xg_l5m, 'ict_index_per_90': ict_90, 'form': form,
                'was_home': home, 'FDR': fdr, 'Opponent_xGC_per_90': opp_xgc, 'xG_match': y_xg
            })
            df_xa = pd.DataFrame({
                'xA_per_90_L5M': xa_l5m, 'ict_index_per_90': ict_90, 'is_setpiece_taker': sp_taker,
                'form': form, 'was_home': home, 'FDR': fdr, 'Opponent_xGC_per_90': opp_xgc, 'xA_match': y_xa
            })

        # --- INCREMENTAL TRAINING LOGIC (OPTION B) ---
        if current_gw <= 10 and not df_historical.empty:
            hist_pos = df_historical[df_historical['element_type'] == el_type]
            if not hist_pos.empty:
                # Inject histori ke model xG
                cols_xg = ['xG_per_90_L5M', 'ict_index_per_90', 'form', 'was_home', 'FDR', 'Opponent_xGC_per_90', 'xG_match']
                if all(c in hist_pos.columns for c in cols_xg):
                    df_xg = pd.concat([df_xg, hist_pos[cols_xg]], ignore_index=True)
                
                # Inject histori ke model xA
                cols_xa = ['xA_per_90_L5M', 'ict_index_per_90', 'is_setpiece_taker', 'form', 'was_home', 'FDR', 'Opponent_xGC_per_90', 'xA_match']
                if all(c in hist_pos.columns for c in cols_xa):
                    df_xa = pd.concat([df_xa, hist_pos[cols_xa]], ignore_index=True)
        
        # Train Model xG
        X_xg = df_xg[['xG_per_90_L5M', 'ict_index_per_90', 'form', 'was_home', 'FDR', 'Opponent_xGC_per_90']]
        y_xg = df_xg['xG_match']
        model_xg = LinearRegression()
        model_xg.fit(X_xg, y_xg)
        opt_b_models_xg[pos_key] = model_xg

        # Train Model xA
        X_xa = df_xa[['xA_per_90_L5M', 'ict_index_per_90', 'is_setpiece_taker', 'form', 'was_home', 'FDR', 'Opponent_xGC_per_90']]
        y_xa = df_xa['xA_match']
        model_xa = LinearRegression()
        model_xa.fit(X_xa, y_xa)
        opt_b_models_xa[pos_key] = model_xa

        # Calculate model statistics & evaluation metrics
        y_pred_xg = model_xg.predict(X_xg)
        r2_xg = round(float(r2_score(y_xg, y_pred_xg)), 4)
        mae_xg = round(float(mean_absolute_error(y_xg, y_pred_xg)), 4)
        eval_df_xg = pd.DataFrame({
            'y_actual': np.round(y_xg, 4),
            'y_pred': np.round(y_pred_xg, 4),
            'Error': np.round(np.abs(y_xg - y_pred_xg), 4)
        }).sort_values(by='y_actual', ascending=False).head(10)

        y_pred_xa = model_xa.predict(X_xa)
        r2_xa = round(float(r2_score(y_xa, y_pred_xa)), 4)
        mae_xa = round(float(mean_absolute_error(y_xa, y_pred_xa)), 4)
        eval_df_xa = pd.DataFrame({
            'y_actual': np.round(y_xa, 4),
            'y_pred': np.round(y_pred_xa, 4),
            'Error': np.round(np.abs(y_xa - y_pred_xa), 4)
        }).sort_values(by='y_actual', ascending=False).head(10)

        stats_xg[pos_key] = {
            'r2': r2_xg,
            'mae': mae_xg,
            'intercept': round(float(model_xg.intercept_), 4),
            'coef_df': pd.DataFrame({
                'Variabel Fitur': list(X_xg.columns),
                'Koefisien (β)': np.round(model_xg.coef_, 4)
            }),
            'eval_df': eval_df_xg
        }

        stats_xa[pos_key] = {
            'r2': r2_xa,
            'mae': mae_xa,
            'intercept': round(float(model_xa.intercept_), 4),
            'coef_df': pd.DataFrame({
                'Variabel Fitur': list(X_xa.columns),
                'Koefisien (β)': np.round(model_xa.coef_, 4)
            }),
            'eval_df': eval_df_xa
        }

    # Flat fallback keys for single-access compatibility
    if 'FWD' in stats_xg:
        stats_xg['r2'] = stats_xg['FWD']['r2']
        stats_xg['mae'] = stats_xg['FWD']['mae']
        stats_xg['intercept'] = stats_xg['FWD']['intercept']
        stats_xg['coef_df'] = stats_xg['FWD']['coef_df']
        stats_xg['eval_df'] = stats_xg['FWD']['eval_df']

    if 'FWD' in stats_xa:
        stats_xa['r2'] = stats_xa['FWD']['r2']
        stats_xa['mae'] = stats_xa['FWD']['mae']
        stats_xa['intercept'] = stats_xa['FWD']['intercept']
        stats_xa['coef_df'] = stats_xa['FWD']['coef_df']
        stats_xa['eval_df'] = stats_xa['FWD']['eval_df']

    return opt_b_models_xg, opt_b_models_xa, stats_xg, stats_xa

@st.cache_data(ttl=86400)
def train_xpoints_model(players_list, fdr_summary, current_gw, df_historical):
    models_dict = {}

    for pos_key, cfg in POS_MODEL_CONFIGS.items():
        pos_el_type = cfg['element_type']
        feature_cols = cfg['feature_cols']
        feature_labels = cfg['feature_labels']

        # Select top active players for this position
        pos_players = [p for p in players_list if p.get('element_type') == pos_el_type]
        top_pos_players = sorted(pos_players, key=lambda p: (p.get('total_points', 0), p.get('minutes', 0)), reverse=True)[:25]

        history_rows = []
        for p in top_pos_players:
            p_form = float(p.get('form', 0.0))
            p_def_contrib = float(p.get('defensive_contribution_per_90', 0.0))
            p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: m.get('round', m.get('event', 0)))
                for i, m in enumerate(sorted_hist):
                    mins = int(m.get('minutes', 0))
                    if mins > 0:
                        prev_5 = sorted_hist[max(0, i-4):i+1]
                        avg_mins_l5m = sum(int(x.get('minutes', 0)) for x in prev_5) / float(len(prev_5))
                        xg90 = (float(m.get('expected_goals', 0.0)) / mins) * 90.0
                        xa90 = (float(m.get('expected_assists', 0.0)) / mins) * 90.0
                        xgc90 = (float(m.get('expected_goals_conceded', 0.0)) / mins) * 90.0
                        saves90 = (float(m.get('saves', 0)) / mins) * 90.0
                        bps90 = (float(m.get('bps', 0)) / mins) * 90.0
                        ict90 = (float(m.get('ict_index', 0.0)) / mins) * 90.0
                        
                        tackles = float(m.get('tackles', 0))
                        interceptions = float(m.get('interceptions', 0))
                        clearances = float(m.get('clearances_blocks_interceptions', m.get('clearances', 0)))
                        recoveries = float(m.get('recoveries', 0))
                        tot_def_actions = tackles + interceptions + clearances + recoveries
                        def_contrib_90 = (tot_def_actions / mins) * 90.0 if tot_def_actions > 0 else p_def_contrib

                        history_rows.append({
                            'xG_per_90': xg90,
                            'xA_per_90': xa90,
                            'bps_per_90': bps90,
                            'form': p_form,
                            'was_home': 1 if m.get('was_home') else 0,
                            'FDR': int(m.get('opponent_team', 3)),
                            'last_minutes_5_match': avg_mins_l5m,
                            'ict_index': ict90,
                            'Defensive_Contribution_per_90': def_contrib_90,
                            'xGC_per_90': xgc90,
                            'Saves_per_90': saves90,
                            'total_points': int(m.get('total_points', 0))
                        })

        if len(history_rows) >= 15:
            df_train = pd.DataFrame(history_rows)
            is_real_history = True
        else:
            # Generate realistic synthetic training dataset specific to position scoring
            np.random.seed(42 + pos_el_type)
            N = 300
            l5m = np.random.uniform(20.0, 90.0, size=N)
            xg = np.random.exponential(0.25 if pos_key in ['FWD', 'MID'] else 0.08, size=N)
            xa = np.random.exponential(0.20 if pos_key in ['FWD', 'MID'] else 0.10, size=N)
            bps = np.random.uniform(8.0, 45.0, size=N)
            form = np.random.uniform(0.5, 8.5, size=N)
            home = np.random.choice([0, 1], size=N)
            fdr = np.random.choice([1, 2, 3, 4, 5], size=N)
            ict = np.random.uniform(1.0, 16.0, size=N)
            def_contrib = np.random.uniform(2.0, 14.0, size=N)
            xgc = np.random.uniform(0.3, 2.5, size=N)
            saves = np.random.uniform(0.5, 6.0, size=N)

            if pos_key == 'FWD':
                y = 2.0*(l5m/90.0) + 4.2*xg + 3.0*xa + 0.05*bps + 0.2*form + 0.35*home - 0.2*fdr + 0.1*ict + np.random.normal(0, 0.2, size=N)
            elif pos_key == 'MID':
                y = 2.0*(l5m/90.0) + 5.0*xg + 3.0*xa + 0.8*(2.5 - xgc) + 0.05*bps + 0.03*def_contrib + 0.2*form + 0.35*home - 0.2*fdr + 0.1*ict + np.random.normal(0, 0.2, size=N)
            elif pos_key == 'DEF':
                y = 2.0*(l5m/90.0) + 6.0*xg + 3.0*xa + 2.0*(2.5 - xgc) + 0.05*bps + 0.05*def_contrib + 0.18*form + 0.4*home - 0.25*fdr + 0.08*ict + np.random.normal(0, 0.2, size=N)
            else: # GK
                y = 2.0*(l5m/90.0) + 2.0*(2.5 - xgc) + 0.33*saves + 0.06*bps + 0.2*form + 0.3*home - 0.2*fdr + np.random.normal(0, 0.2, size=N)

            y = np.clip(y, 0, 24)

            df_train = pd.DataFrame({
                'xG_per_90': xg, 'xA_per_90': xa, 'bps_per_90': bps, 'form': form,
                'was_home': home, 'FDR': fdr, 'last_minutes_5_match': l5m,
                'ict_index': ict, 'Defensive_Contribution_per_90': def_contrib,
                'xGC_per_90': xgc, 'Saves_per_90': saves, 'total_points': y
            })
            is_real_history = False

        # --- INCREMENTAL TRAINING LOGIC (OPTION A) ---
        if current_gw <= 10 and not df_historical.empty:
            hist_pos = df_historical[df_historical['element_type'] == pos_el_type]
            if not hist_pos.empty:
                # Ambil hanya kolom yang dibutuhkan untuk mencegah error concat
                valid_cols = [c for c in feature_cols + ['total_points'] if c in hist_pos.columns]
                df_train = pd.concat([df_train, hist_pos[valid_cols]], ignore_index=True)
                is_real_history = True # Timpa status menjadi riil karena menggunakan data musim lalu
        
        X = df_train[feature_cols]
        y = df_train['total_points']

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        coef_df = pd.DataFrame({
            'Variabel Fitur': feature_labels,
            'Koefisien (β)': [round(c, 4) for c in model.coef_]
        })

        diag_results = perform_classical_assumption_tests(X, y, feature_labels)

        models_dict[pos_key] = {
            'model': model,
            'r2': round(r2, 4),
            'coef_df': coef_df,
            'intercept': round(float(model.intercept_), 4),
            'is_real_history': is_real_history,
            'diag_results': diag_results,
            'feature_cols': feature_cols,
            'feature_labels': feature_labels
        }

    return models_dict

# -----------------------------------------------------------------------------
# DATA PROCESSING HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def compute_all_l5m_avg_mins(elements):
    """
    Hitung avg_minutes_per_match dari 5 pertandingan terakhir (element-summary API).
    Hitung secara paralel untuk performa optimal.
    """
    results = {}
    
    def fetch_one(el):
        p_id = el['id']
        tot_mins = int(el.get('minutes', 0))
        chance = el.get('chance_of_playing_next_round')
        chance_val = 100 if chance is None or pd.isnull(chance) else int(chance)
        
        if tot_mins <= 0:
            return p_id, 0.0
            
        hist = fetch_player_history_raw(p_id)
        if hist:
            sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
            last_5 = sorted_hist[-5:]
            n_m = len(last_5)
            if n_m > 0:
                s_mins = sum(int(m.get('minutes', 0)) for m in last_5)
                return p_id, float(np.clip(s_mins / float(n_m), 0.0, 90.0))
                
        # Fallback for empty history (e.g. pre-season):
        fallback = min(90.0, (float(tot_mins) / 38.0) * (chance_val / 100.0))
        return p_id, float(np.clip(fallback, 0.0, 90.0))

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = executor.map(fetch_one, elements)
        for p_id, val in futures:
            results[p_id] = round(val, 1)
            
    return results

@st.cache_data(ttl=86400)
def process_players(fpl_data, fdr_summary, _models_dict, _opt_b_models=None):
    """Clean & format player dataset with extensive FPL metrics and positional xPoin predictions."""
    teams_list = fpl_data.get('teams', [])
    team_dict = {t['id']: t['name'] for t in teams_list}
    
    elements = fpl_data.get('elements', [])
    df = pd.DataFrame(elements)
    
    if df.empty:
        return pd.DataFrame(), team_dict
    
    # 1. Info Dasar
    df['Nama Pemain'] = df['web_name']
    df['Nama Lengkap'] = df['first_name'] + " " + df['second_name']
    df['Klub'] = df['team'].map(team_dict)
    df['Posisi'] = df['element_type'].map(POSITION_MAP)
    df['Harga (£m)'] = df['now_cost'] / 10.0
    df['Total Poin'] = df['total_points'].astype(int)
    df['Menit Bermain'] = df['minutes'].astype(int)
    df['Gol'] = df['goals_scored'].astype(int)
    df['Asis'] = df['assists'].astype(int)
    df['Clean Sheet'] = df['clean_sheets'].astype(int)

    # 1b. Compute L5M Average Minutes
    l5m_mins_dict = compute_all_l5m_avg_mins(elements)
    df['Avg Mins (L5M)'] = df['id'].map(l5m_mins_dict).fillna(0.0).round(1)
    
    # 2. Expected Metrics & Per 90
    df['xG'] = pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0.0).round(2)
    df['xA'] = pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0.0).round(2)
    df['xGI'] = pd.to_numeric(df['expected_goal_involvements'], errors='coerce').fillna(0.0).round(2)
    df['xG per 90'] = pd.to_numeric(df['expected_goals_per_90'], errors='coerce').fillna(0.0).round(2)
    df['xA per 90'] = pd.to_numeric(df['expected_assists_per_90'], errors='coerce').fillna(0.0).round(2)
    df['xGI per 90'] = pd.to_numeric(df['expected_goal_involvements_per_90'], errors='coerce').fillna(0.0).round(2)
    df['xGC per 90'] = pd.to_numeric(df['expected_goals_conceded_per_90'], errors='coerce').fillna(0.0).round(2)
    df['Saves per 90'] = pd.to_numeric(df['saves_per_90'], errors='coerce').fillna(0.0).round(2)
    df['Defensive Contribution per 90'] = pd.to_numeric(df['defensive_contribution_per_90'], errors='coerce').fillna(0.0).round(2)
    df['ICT Index'] = pd.to_numeric(df['ict_index'], errors='coerce').fillna(0.0).round(1)

    # 3. Tren & Pasar (Form dari bootstrap-static)
    df['Form'] = pd.to_numeric(df['form'], errors='coerce').fillna(0.0).round(2)
    df['% Ownership'] = pd.to_numeric(df['selected_by_percent'], errors='coerce').fillna(0.0).round(1)
    df['Transfers In GW'] = df['transfers_in_event'].fillna(0).astype(int)
    df['Transfers Out GW'] = df['transfers_out_event'].fillna(0).astype(int)
    df['Net Transfers GW'] = df['Transfers In GW'] - df['Transfers Out GW']

    # 4. Set-Piece Takers
    df['Penalti Order'] = df['penalties_order'].apply(format_setpiece_order)
    df['Free Kick Order'] = df['direct_freekicks_order'].apply(format_setpiece_order)
    df['Corner Order'] = df['corners_and_indirect_freekicks_order'].apply(format_setpiece_order)

    # 5. Bonus & Kedisiplinan
    df['BPS'] = df['bps'].fillna(0).astype(int)
    df['Bonus Poin'] = df['bonus'].fillna(0).astype(int)
    df['Kartu Kuning'] = df['yellow_cards'].fillna(0).astype(int)
    df['Kartu Merah'] = df['red_cards'].fillna(0).astype(int)
    df['Saves'] = df['saves'].fillna(0).astype(int)

    # 6. Info Kebugaran
    df['Status'] = df['status'].map(STATUS_MAP).fillna('❓ Unknown')
    df['Peluang Main GW (%)'] = df['chance_of_playing_next_round'].apply(
        lambda x: 100 if pd.isnull(x) or x is None else int(x)
    )
    df['Berita Cedera'] = df['news'].apply(
        lambda x: str(x).strip() if pd.notnull(x) and str(x).strip() != "" else "-"
    )

    # 7. FDR Attachment & Next Opponent
    df['FDR1'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR1', 3.0))
    df['FDR3'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR3', 3.0))
    df['FDR5'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR5', 3.0))
    df['Next_Is_Home'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('Next_Is_Home', 1))
    df['Next_Opponent_ID'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('Next_Opponent_ID'))
    df['Next_Opponent_Name'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('Next_Opponent_Name', 'TBD'))
    df['Lawan GW Berikutnya'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('Next_Opponent_Fmt', '-'))

    # 8. Predict xPoin using 4 Positional LinearRegression Models (Option A)
    mins_played = np.maximum(df['Menit Bermain'], 1.0)

    # Convert all raw/total metrics to per-90 scale for inference
    df['xG_per_90_calc'] = (df['xG'] / mins_played) * 90.0
    df['xA_per_90_calc'] = (df['xA'] / mins_played) * 90.0

    total_ict = pd.to_numeric(df['ict_index'], errors='coerce').fillna(0.0)
    df['ict_index_per_90_calc'] = (total_ict / mins_played) * 90.0

    total_xgc = pd.to_numeric(df['expected_goals_conceded'], errors='coerce').fillna(0.0)
    df['xGC_per_90_calc'] = (total_xgc / mins_played) * 90.0

    df['Defensive_Contribution_per_90_calc'] = pd.to_numeric(df['defensive_contribution_per_90'], errors='coerce').fillna(0.0)
    df['Saves_per_90_calc'] = (df['Saves'] / mins_played) * 90.0
    df['bps_per_90_calc'] = (df['BPS'] / mins_played) * 90.0

    df['xPoin_raw'] = 0.0

    pos_feature_mapping = {
        'FWD': {
            'xG_per_90': df['xG_per_90_calc'],
            'xA_per_90': df['xA_per_90_calc'],
            'bps_per_90': df['bps_per_90_calc'],
            'form': df['Form'],
            'was_home': df['Next_Is_Home'],
            'FDR': df['FDR1'],
            'last_minutes_5_match': df['Avg Mins (L5M)'],
            'ict_index': df['ict_index_per_90_calc']
        },
        'MID': {
            'xG_per_90': df['xG_per_90_calc'],
            'xA_per_90': df['xA_per_90_calc'],
            'bps_per_90': df['bps_per_90_calc'],
            'form': df['Form'],
            'was_home': df['Next_Is_Home'],
            'FDR': df['FDR1'],
            'last_minutes_5_match': df['Avg Mins (L5M)'],
            'Defensive_Contribution_per_90': df['Defensive_Contribution_per_90_calc'],
            'xGC_per_90': df['xGC_per_90_calc'],
            'ict_index': df['ict_index_per_90_calc']
        },
        'DEF': {
            'xG_per_90': df['xG_per_90_calc'],
            'xA_per_90': df['xA_per_90_calc'],
            'bps_per_90': df['bps_per_90_calc'],
            'form': df['Form'],
            'was_home': df['Next_Is_Home'],
            'FDR': df['FDR1'],
            'last_minutes_5_match': df['Avg Mins (L5M)'],
            'Defensive_Contribution_per_90': df['Defensive_Contribution_per_90_calc'],
            'xGC_per_90': df['xGC_per_90_calc'],
            'ict_index': df['ict_index_per_90_calc']
        },
        'GK': {
            'bps_per_90': df['bps_per_90_calc'],
            'form': df['Form'],
            'was_home': df['Next_Is_Home'],
            'FDR': df['FDR1'],
            'last_minutes_5_match': df['Avg Mins (L5M)'],
            'xGC_per_90': df['xGC_per_90_calc'],
            'Saves_per_90': df['Saves_per_90_calc']
        }
    }

    for pos_key, feat_map in pos_feature_mapping.items():
        pos_mask = (df['Posisi'] == pos_key)
        if pos_mask.any() and pos_key in _models_dict:
            model_info = _models_dict[pos_key]
            cols_needed = model_info['feature_cols']
            X_pos = pd.DataFrame({col: feat_map[col][pos_mask] for col in cols_needed})
            preds = model_info['model'].predict(X_pos)
            df.loc[pos_mask, 'xPoin_raw'] = np.clip(preds, 0.0, None)

    # Assign xPoin strictly >= 0.0
    df['xPoin'] = df['xPoin_raw'].round(2)

    # -------------------------------------------------------------------------
    # 9. OPTION B: BOTTOM-UP COMPONENT MODEL CALCULATIONS
    # -------------------------------------------------------------------------
    team_def_xgc = df.groupby('team')['xGC per 90'].mean().to_dict()
    team_att_xg = df.groupby('Klub').apply(lambda x: x.nlargest(11, 'Menit Bermain')['xG per 90'].sum()).to_dict()

    df['Opponent_xGC_per_90'] = df['Next_Opponent_ID'].map(lambda oid: team_def_xgc.get(oid, 1.25)).fillna(1.25)
    df['Opponent_xG_per_90_attack'] = df['Next_Opponent_Name'].map(lambda name: team_att_xg.get(name, 1.5)).fillna(1.5)

    df['is_setpiece_taker'] = df.apply(
        lambda r: check_setpiece_taker(r.get('corners_and_indirect_freekicks_order'), r.get('direct_freekicks_order')), axis=1
    )

    raw_xg_match = np.zeros(len(df))
    raw_xa_match = np.zeros(len(df))

    if _opt_b_models is not None:
        models_xg_dict = _opt_b_models[0]
        models_xa_dict = _opt_b_models[1]
        for pos in ['FWD', 'MID', 'DEF']:
            pos_mask = (df['Posisi'] == pos)
            if pos_mask.any():
                df_pos = df[pos_mask]
                X_xg_pos = df_pos[['xG_per_90_calc', 'ict_index_per_90_calc', 'Form', 'Next_Is_Home', 'FDR1', 'Opponent_xGC_per_90']].rename(
                    columns={'xG_per_90_calc': 'xG_per_90_L5M', 'ict_index_per_90_calc': 'ict_index_per_90', 'Form': 'form', 'Next_Is_Home': 'was_home', 'FDR1': 'FDR'}
                )
                X_xa_pos = df_pos[['xA_per_90_calc', 'ict_index_per_90_calc', 'is_setpiece_taker', 'Form', 'Next_Is_Home', 'FDR1', 'Opponent_xGC_per_90']].rename(
                    columns={'xA_per_90_calc': 'xA_per_90_L5M', 'ict_index_per_90_calc': 'ict_index_per_90', 'Form': 'form', 'Next_Is_Home': 'was_home', 'FDR1': 'FDR'}
                )

                if isinstance(models_xg_dict, dict) and pos in models_xg_dict:
                    raw_xg_match[pos_mask] = models_xg_dict[pos].predict(X_xg_pos)
                elif hasattr(models_xg_dict, 'predict'):
                    raw_xg_match[pos_mask] = models_xg_dict.predict(X_xg_pos)
                else:
                    raw_xg_match[pos_mask] = df_pos['xG_per_90_calc'] * 0.5

                if isinstance(models_xa_dict, dict) and pos in models_xa_dict:
                    raw_xa_match[pos_mask] = models_xa_dict[pos].predict(X_xa_pos)
                elif hasattr(models_xa_dict, 'predict'):
                    raw_xa_match[pos_mask] = models_xa_dict.predict(X_xa_pos)
                else:
                    raw_xa_match[pos_mask] = df_pos['xA_per_90_calc'] * 0.5
    else:
        gk_mask = (df['Posisi'] == 'GK')
        raw_xg_match = np.where(gk_mask, 0.0, df['xG_per_90_calc'] * 0.5)
        raw_xa_match = np.where(gk_mask, 0.0, df['xA_per_90_calc'] * 0.5)

    # KHUSUS UNTUK POSISI 'GK' (Kiper): Paksa (hardcode) nilai raw_xg_match = 0.0 dan raw_xa_match = 0.0
    gk_mask = (df['Posisi'] == 'GK')
    raw_xg_match[gk_mask] = 0.0
    raw_xa_match[gk_mask] = 0.0

    mins_ratio = df['Avg Mins (L5M)'] / 90.0
    df['xG Pred (Match)'] = (np.maximum(0.0, raw_xg_match) * mins_ratio).round(2)
    df['xA Pred (Match)'] = (np.maximum(0.0, raw_xa_match) * mins_ratio).round(2)

    # Component Calculations:
    # a. xMins_Pts
    chance_factor = df['Peluang Main GW (%)'] / 100.0
    avg_mins_l5m = df['Avg Mins (L5M)']
    df['xMins Pts'] = np.where(
        avg_mins_l5m >= 60.0, 2.0 * chance_factor,
        np.where(avg_mins_l5m > 0.0, 1.0 * chance_factor, 0.0)
    ).round(2)

    # b. xG_Pts (GK=10, DEF=6, MID=5, FWD=4)
    poin_gol_map = {'GK': 10.0, 'DEF': 6.0, 'MID': 5.0, 'FWD': 4.0}
    poin_gol = df['Posisi'].map(poin_gol_map).fillna(4.0)
    df['xG Pts'] = (df['xG Pred (Match)'] * poin_gol).round(2)

    # c. xA_Pts (All = 3)
    df['xA Pts'] = (df['xA Pred (Match)'] * 3.0).round(2)

    # d. xSaves_Pts (GK only: expected saves / 3.0)
    exp_saves = df['Saves per 90'] * mins_ratio
    df['xSaves Pts'] = np.where(df['Posisi'] == 'GK', exp_saves / 3.0, 0.0).round(2)

    # e. xDC_Pts (Defensive Contribution: DEF=10, MID=12, FWD=12, GK=0)
    dc_thresh_map = {'DEF': 10, 'MID': 12, 'FWD': 12, 'GK': 0}
    thresholds = df['Posisi'].map(dc_thresh_map).fillna(0)
    dc_pts = []
    for mu, t_val in zip(df['Defensive Contribution per 90'] * mins_ratio, thresholds):
        if t_val > 0 and mu > 0:
            prob = 1.0 - float(poisson.cdf(t_val - 1, mu))
            val = float(np.clip(2.0 * prob, 0.0, 2.0))
        else:
            val = 0.0
        dc_pts.append(val)
    df['xDC Pts'] = np.array(dc_pts).round(2)

    # f. xCS_Pts (Clean Sheet: GK=4, DEF=4, MID=1, FWD=0)
    # PENTING: Jika Avg Mins (L5M) < 60, set xCS_Pts = 0.0 (aturan FPL minimal 60 menit)
    poin_cs_map = {'GK': 4.0, 'DEF': 4.0, 'MID': 1.0, 'FWD': 0.0}
    poin_cs = df['Posisi'].map(poin_cs_map).fillna(0.0)
    prob_cs = np.exp(-df['Opponent_xG_per_90_attack'] * mins_ratio)
    raw_xcs = prob_cs * poin_cs
    df['xCS Pts'] = np.where(df['Avg Mins (L5M)'] >= 60.0, raw_xcs, 0.0).round(2)

    # g. xBP (Bonus Points)
    # FIX: Konversi bps_per_90_calc menjadi ekspektasi match riil dengan mins_ratio.
    # Ini menetralisir inflasi statistik pemain yang hanya bermain menit kecil.
    exp_bps_match = df['bps_per_90_calc'] * mins_ratio
    raw_xbp = (exp_bps_match * 0.02) + ((df['xG Pred (Match)'] + df['xA Pred (Match)']) * 0.5)
    
    # Batasi xBP maksimal 3.0 poin (tanpa pembatasan minimal menit bermain)
    df['xBP'] = np.clip(raw_xbp, 0.0, 3.0).round(2)

    # h. Total Option B xPoin
    df['xPoin (Option B)'] = (
        df['xMins Pts'] + df['xG Pts'] + df['xA Pts'] + 
        df['xSaves Pts'] + df['xDC Pts'] + df['xCS Pts'] + df['xBP']
    ).round(2)

    cols = [
        'id', 'team',
        'Nama Pemain', 'Klub', 'Lawan GW Berikutnya', 'Posisi', 'Harga (£m)', 'xPoin', 'xPoin (Option B)',
        'xG Pred (Match)', 'xA Pred (Match)', 'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
        'Avg Mins (L5M)', 'Total Poin', 'FDR1', 'FDR3', 'FDR5', 'Form', '% Ownership', 'Net Transfers GW',
        'xG', 'xA', 'xGI', 'xG per 90', 'xA per 90', 'xGI per 90',
        'xGC per 90', 'Saves per 90', 'Defensive Contribution per 90',
        'ICT Index', 'BPS', 'Bonus Poin', 'Kartu Kuning', 'Kartu Merah', 'Saves',
        'Penalti Order', 'Free Kick Order', 'Corner Order', 'Status',
        'Peluang Main GW (%)', 'Berita Cedera', 'Menit Bermain', 'Gol', 'Asis',
        'Clean Sheet', 'Nama Lengkap'
    ]
    
    return df[cols], team_dict

# -----------------------------------------------------------------------------
# MAIN APP ENTRY POINT
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# OPTION C: CURRENT SEASON MODEL (TAB 5)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def build_option_c_model_and_view(fpl_data, fdr_summary, current_gw):
    # Model diperbarui setiap 5 gameweek: GW5, GW10, GW15, dst.
    # Cache key includes (current_gw // 5) to force remodeling.
    remodel_cycle = current_gw // 5 
    
    elements = fpl_data.get('elements', [])
    teams = fpl_data.get('teams', [])
    team_strength = {t['id']: t['strength'] for t in teams}
    team_dict = {t['id']: t['name'] for t in teams}
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Calculate Team Current Aggregates for opp_xGC_per90 etc.
    team_xG_total = {}
    team_xGC_total = {}
    
    # Simple proxy: Sum of player xG = team xG. 
    for el in elements:
        t_id = el['team']
        xg = float(el.get('expected_goals', 0))
        xgc = float(el.get('expected_goals_conceded', 0))
        # For xGC, only count GK to avoid duplicating 11x
        if el['element_type'] == 1:
            team_xGC_total[t_id] = team_xGC_total.get(t_id, 0) + xgc
        team_xG_total[t_id] = team_xG_total.get(t_id, 0) + xg
        
    team_matches = current_gw if current_gw > 0 else 1
    
    from concurrent.futures import ThreadPoolExecutor
    all_histories = []
    
    def fetch_hist_c(el):
        hist = fetch_player_history_raw(el['id'])
        return el['id'], el['element_type'], el['team'], hist
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = executor.map(fetch_hist_c, elements)
        for p_id, pos, team_id, hist in futures:
            if hist:
                sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
                pts_history = []
                mins_history = []
                
                for m in sorted_hist:
                    m_dict = dict(m)
                    m_dict['player_id'] = p_id
                    m_dict['posisi'] = pos_map.get(pos, 'Unknown')
                    m_dict['team_id'] = team_id
                    
                    mins = int(m_dict.get('minutes', 0))
                    pts = int(m_dict.get('total_points', 0))
                    pts_history.append(pts)
                    mins_history.append(mins)
                    
                    m_dict['form_calc'] = np.mean(pts_history[-4:]) if pts_history else 0.0
                    m_dict['avg_mins_L5M'] = np.mean(mins_history[-5:]) if mins_history else 0.0
                    
                    for stat in ['expected_goals', 'expected_assists', 'expected_goals_conceded']:
                        val = float(m_dict.get(stat, 0.0))
                        m_dict[f"{stat}_per_90"] = (val / mins * 90) if mins > 0 else 0.0
                    
                    opp_team = m_dict.get('opponent_team')
                    m_dict['FDR_now'] = team_strength.get(opp_team, 3)
                    
                    # Proxy opp stats
                    m_dict['opp_xGC_per90'] = team_xGC_total.get(opp_team, 0) / team_matches
                    m_dict['opp_xG_per90'] = team_xG_total.get(opp_team, 0) / team_matches
                    m_dict['opp_l5m_xGC_per90'] = m_dict['opp_xGC_per90'] # proxy
                    m_dict['opp_l5m_xG_per90'] = m_dict['opp_xG_per90'] # proxy
                    
                    m_dict['laga kandang/tandang'] = 1 if m_dict.get('was_home') else 0
                    
                    all_histories.append(m_dict)
                    
    df_train = pd.DataFrame(all_histories)
    if df_train.empty:
        return None, None, None
        
    numeric_cols = ['minutes', 'influence', 'creativity', 'threat', 'expected_goals', 'expected_assists', 
                    'clearances_blocks_interceptions', 'expected_goals_conceded', 'saves', 'clean_sheets',
                    'FDR_now']
    for c in numeric_cols:
        if c in df_train.columns:
            df_train[c] = pd.to_numeric(df_train[c], errors='coerce').fillna(0.0)

    df_train.rename(columns={
        'expected_goals': 'xG',
        'expected_assists': 'xA',
        'form_calc': 'form',
        'FDR_now': 'FDR',
        'clearances_blocks_interceptions': 'CBIT',
        'expected_goals_conceded': 'xGC'
    }, inplace=True)
    
    features_map = {
        'FWD': ['minutes', 'influence', 'creativity', 'threat', 'xG', 'xA', 'form', 'laga kandang/tandang', 'FDR', 'avg_mins_L5M'],
        'MID': ['minutes', 'influence', 'creativity', 'threat', 'xG', 'xA', 'form', 'laga kandang/tandang', 'FDR', 'avg_mins_L5M', 'CBIT', 'xGC'],
        'DEF': ['minutes', 'influence', 'creativity', 'threat', 'xG', 'xA', 'form', 'laga kandang/tandang', 'FDR', 'avg_mins_L5M', 'CBIT', 'xGC'],
        'GK':  ['minutes', 'form', 'laga kandang/tandang', 'FDR', 'avg_mins_L5M', 'CBIT', 'xGC', 'saves', 'clean_sheets']
    }
    
    models_c = {}
    
    for pos, feats in features_map.items():
        # Using all data to train Option C
        pos_df = df_train[df_train['posisi'] == pos].dropna(subset=feats + ['total_points'])
        if len(pos_df) > 10:
            X = pos_df[feats]
            y = pos_df['total_points']
            model = LinearRegression()
            model.fit(X, y)
            models_c[pos] = {'model': model, 'features': feats}
            
    summary_data = []
    for el in elements:
        p_id = el['id']
        pos = pos_map.get(el['element_type'])
        if pos not in models_c:
            continue
            
        team_id = el['team']
        f_info = fdr_summary.get(team_id, {})
        fdr_next = f_info.get('FDR1', 3.0)
        fdr3 = f_info.get('FDR3', 3.0)
        fdr5 = f_info.get('FDR5', 3.0)
        is_home_next = 1 if f_info.get('Next_Is_Home') == 1 else 0
        
        hist = fetch_player_history_raw(p_id)
        if hist:
            sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
            last_5 = sorted_hist[-5:]
            avg_mins = np.mean([int(m.get('minutes', 0)) for m in last_5]) if last_5 else 0.0
            form_val = np.mean([int(m.get('total_points', 0)) for m in last_5[-4:]]) if last_5 else 0.0
        else:
            avg_mins = 0.0
            form_val = 0.0
            
        tot_mins = float(el.get('minutes', 0))
        def _get_avg(col):
            val = float(el.get(col, 0.0))
            return val / (tot_mins/90) if tot_mins > 0 else 0.0
            
        curr_x = {
            'minutes': avg_mins,
            'influence': float(el.get('influence', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'creativity': float(el.get('creativity', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'threat': float(el.get('threat', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'xG': float(el.get('expected_goals', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'xA': float(el.get('expected_assists', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'form': form_val,
            'laga kandang/tandang': is_home_next,
            'FDR': fdr_next,
            'avg_mins_L5M': avg_mins,
            'CBIT': float(el.get('clearances_blocks_interceptions', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'xGC': float(el.get('expected_goals_conceded', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'saves': float(el.get('saves', 0)) / (tot_mins/90) if tot_mins>0 else 0,
            'clean_sheets': float(el.get('clean_sheets', 0)) / (tot_mins/90) if tot_mins>0 else 0
        }
        
        feats = models_c[pos]['features']
        X_pred = pd.DataFrame([curr_x])[feats]
        
        xpoin_c = models_c[pos]['model'].predict(X_pred)[0]
        
        status = el.get('status', 'a')
        status_map = {'a': 'Available', 'd': 'Doubtful', 'i': 'Injured', 's': 'Suspended', 'u': 'Unavailable'}
        
        summary_data.append({
            'Nama Pemain': el['web_name'],
            'Klub': team_dict.get(team_id, ''),
            'Posisi': pos,
            'Total Poin': el.get('total_points', 0),
            'xPoin (Option C)': max(0.0, float(xpoin_c)),
            'Total xG': float(el.get('expected_goals', 0)),
            'Total xA': float(el.get('expected_assists', 0)),
            'Avg Mins (L5M)': avg_mins,
            'FDR1': fdr_next,
            'FDR3': fdr3,
            'FDR5': fdr5,
            'Status': status_map.get(status, status)
        })
        
    df_view = pd.DataFrame(summary_data)
    if not df_view.empty:
        df_view = df_view.sort_values('xPoin (Option C)', ascending=False)
        
    return df_train, df_view, models_c

# -----------------------------------------------------------------------------
# PLAYER COMPARISON & RADAR CHART MODULE
# -----------------------------------------------------------------------------
def render_player_comparison_radar_tab(players_df, fpl_data, teams_dict):
    st.subheader("⚔️ Komparasi Head-to-Head 2 Pemain & Grafik Radar")
    st.write("Bandingkan profil performa, potensi serangan, efisiensi poin, dan statistik 360° antara dua pemain Fantasy Premier League menggunakan visualisasi grafik radar interaktif.")

    if players_df.empty:
        st.warning("Data pemain tidak tersedia untuk komparasi.")
        return

    # Siapkan dataframe komparasi dengan metrik turunan
    df = players_df.copy()
    if 'Poin per £m' not in df.columns:
        df['Poin per £m'] = np.where(df['Harga (£m)'] > 0, (df['Total Poin'] / df['Harga (£m)']).round(2), 0.0)
    if 'xPoin per £m' not in df.columns:
        df['xPoin per £m'] = np.where(df['Harga (£m)'] > 0, (df['xPoin'] / df['Harga (£m)']).round(2), 0.0)
    if 'Kemudahan Jadwal' not in df.columns:
        df['Kemudahan Jadwal'] = (6.0 - df['FDR1']).round(1)
    if 'Kemudahan Jadwal (3 Laga)' not in df.columns:
        df['Kemudahan Jadwal (3 Laga)'] = (6.0 - df['FDR3']).round(2)

    # Filter hanya pemain yang aktif dan memiliki menit bermain di Premier League
    active_pool = df[df['Menit Bermain'] > 0].copy()
    if active_pool.empty:
        active_pool = df.copy()
        
    active_sorted = active_pool.sort_values(by=['Total Poin', 'xPoin'], ascending=False)
    
    top_p1_default = active_sorted.iloc[0]['Nama Pemain'] if not active_sorted.empty else df.iloc[0]['Nama Pemain']
    top_p2_default = active_sorted.iloc[1]['Nama Pemain'] if len(active_sorted) > 1 else (df.iloc[1]['Nama Pemain'] if len(df) > 1 else df.iloc[0]['Nama Pemain'])

    # Validasi session_state agar tidak menyimpan nama basi / pemain yang sudah tidak main di EPL
    current_p1 = st.session_state.get('comp_p1')
    if not current_p1 or current_p1 not in df['Nama Pemain'].values or current_p1 == 'M.Salah':
        st.session_state['comp_p1'] = top_p1_default

    current_p2 = st.session_state.get('comp_p2')
    if not current_p2 or current_p2 not in df['Nama Pemain'].values or current_p2 == 'M.Salah':
        st.session_state['comp_p2'] = top_p2_default
        
    if st.session_state['comp_p1'] == st.session_state['comp_p2'] and len(active_sorted) > 1:
        st.session_state['comp_p2'] = top_p2_default

    # 1. Rekomendasi Duel Real-Time (Dihitung Dinamis dari Pemain yang Aktif Bermain di EPL)
    realtime_matchups = []
    
    # a. Top 2 Total Poin Keseluruhan
    if len(active_sorted) >= 2:
        r1, r2 = active_sorted.iloc[0], active_sorted.iloc[1]
        realtime_matchups.append((f"👑 {r1['Nama Pemain']} vs {r2['Nama Pemain']} (Top Poin Liga)", r1['Nama Pemain'], r2['Nama Pemain']))
        
    # b. Top 2 Gelandang Aktif (MID)
    mids_act = active_pool[active_pool['Posisi'] == 'MID'].sort_values(by=['Total Poin', 'xPoin'], ascending=False)
    if len(mids_act) >= 2:
        m1, m2 = mids_act.iloc[0], mids_act.iloc[1]
        realtime_matchups.append((f"⚡ {m1['Nama Pemain']} vs {m2['Nama Pemain']} (Top MID)", m1['Nama Pemain'], m2['Nama Pemain']))

    # c. Top 2 Penyerang Aktif (FWD)
    fwds_act = active_pool[active_pool['Posisi'] == 'FWD'].sort_values(by=['Total Poin', 'xPoin'], ascending=False)
    if len(fwds_act) >= 2:
        f1, f2 = fwds_act.iloc[0], fwds_act.iloc[1]
        realtime_matchups.append((f"🎯 {f1['Nama Pemain']} vs {f2['Nama Pemain']} (Top FWD)", f1['Nama Pemain'], f2['Nama Pemain']))

    # d. Top 2 Bek Aktif (DEF)
    defs_act = active_pool[active_pool['Posisi'] == 'DEF'].sort_values(by=['Total Poin', 'xPoin'], ascending=False)
    if len(defs_act) >= 2:
        d1, d2 = defs_act.iloc[0], defs_act.iloc[1]
        realtime_matchups.append((f"🛡️ {d1['Nama Pemain']} vs {d2['Nama Pemain']} (Top DEF)", d1['Nama Pemain'], d2['Nama Pemain']))

    # e. Top 2 Kiper Aktif (GK)
    gks_act = active_pool[active_pool['Posisi'] == 'GK'].sort_values(by=['Total Poin', 'xPoin'], ascending=False)
    if len(gks_act) >= 2:
        g1, g2 = gks_act.iloc[0], gks_act.iloc[1]
        realtime_matchups.append((f"🧤 {g1['Nama Pemain']} vs {g2['Nama Pemain']} (Top GK)", g1['Nama Pemain'], g2['Nama Pemain']))

    with st.expander("⚡ Rekomendasi Duel Real-Time (Top Performer Aktif EPL)", expanded=True):
        st.caption("Pintasan duel di bawah ini dihitung otomatis secara *real-time* dari pemain Premier League yang sedang aktif bermain musim ini (Menit Bermain > 0).")
        if realtime_matchups:
            rt_cols = st.columns(len(realtime_matchups))
            for col, (btn_lbl, target1, target2) in zip(rt_cols, realtime_matchups):
                with col:
                    if st.button(btn_lbl, use_container_width=True, key=f"rt_btn_{target1}_{target2}"):
                        st.session_state['comp_p1'] = target1
                        st.session_state['comp_p2'] = target2
                        st.rerun()

    def format_player_picker(p):
        if not p or not isinstance(p, dict):
            return ""
        name = p.get('Nama Pemain', '')
        klub = p.get('Klub', '')
        pos = p.get('Posisi', '')
        pts = p.get('Total Poin', 0)
        mins = p.get('Menit Bermain', 0)
        try:
            price = f"£{float(p.get('Harga (£m)', 0.0)):.1f}m"
        except (ValueError, TypeError):
            price = "-"
        try:
            xpts = f"{float(p.get('xPoin', 0.0)):.2f}"
        except (ValueError, TypeError):
            xpts = "-"
        return f"{name} ({klub} · {pos}) · {price} · Total: {pts} pts ({mins} min) · xPoin: {xpts}"

    col_filter_hdr1, col_filter_hdr2 = st.columns([1, 1])
    with col_filter_hdr1:
        st.markdown("##### 👥 Pemilihan Pemain Head-to-Head")
    with col_filter_hdr2:
        filter_played_only = st.checkbox(
            "Hanya tampilkan pemain yang sudah bermain di EPL (Menit Bermain > 0)",
            value=True,
            help="Saring daftar agar hanya menampilkan pemain yang telah mencatatkan menit bermain di Premier League musim ini.",
            key="radar_filter_played_only"
        )

    col_sel1, col_swap, col_sel2 = st.columns([10, 2, 10])

    pos_options = ["Semua Posisi", "FWD", "MID", "DEF", "GK"]
    club_options = ["Semua Klub"] + sorted(list(df['Klub'].unique()))

    with col_sel1:
        st.markdown("#### 🔵 Pemain 1 (Biru)")
        f_p1_c1, f_p1_c2 = st.columns(2)
        with f_p1_c1:
            p1_pos_filter = st.selectbox("Filter Posisi (P1):", pos_options, key="p1_pos_filt")
        with f_p1_c2:
            p1_club_filter = st.selectbox("Filter Klub (P1):", club_options, key="p1_club_filt")
        
        p1_pool = df.copy()
        if filter_played_only:
            p1_pool = p1_pool[p1_pool['Menit Bermain'] > 0]
        if p1_pos_filter != "Semua Posisi":
            p1_pool = p1_pool[p1_pool['Posisi'] == p1_pos_filter]
        if p1_club_filter != "Semua Klub":
            p1_pool = p1_pool[p1_pool['Klub'] == p1_club_filter]
        
        # Jika pool kosong karena filter, fallback ke active_pool
        if p1_pool.empty:
            p1_pool = active_pool.copy()
            
        p1_pool = p1_pool.sort_values(by="Total Poin", ascending=False)
        p1_records = p1_pool.to_dict('records')
        
        cur_p1_name = st.session_state.get('comp_p1')
        p1_names = [r.get('Nama Pemain') for r in p1_records]
        if cur_p1_name not in p1_names and p1_records:
            cur_p1_name = p1_records[0].get('Nama Pemain')
            st.session_state['comp_p1'] = cur_p1_name

        p1_idx = 0
        for i, rec in enumerate(p1_records):
            if rec.get('Nama Pemain') == cur_p1_name:
                p1_idx = i
                break
        
        p1_selected_record = st.selectbox(
            "Pilih Pemain 1:",
            options=p1_records,
            index=p1_idx if 0 <= p1_idx < len(p1_records) else 0,
            format_func=format_player_picker,
            key=f"p1_sel_box_{cur_p1_name}"
        )
        if p1_selected_record:
            st.session_state['comp_p1'] = p1_selected_record.get('Nama Pemain')

    with col_swap:
        st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Tukar", help="Tukar posisi Pemain 1 dan Pemain 2", use_container_width=True, key="btn_swap_p1_p2"):
            t = st.session_state.get('comp_p1')
            st.session_state['comp_p1'] = st.session_state.get('comp_p2')
            st.session_state['comp_p2'] = t
            st.rerun()

    with col_sel2:
        st.markdown("#### 🔴 Pemain 2 (Merah)")
        f_p2_c1, f_p2_c2 = st.columns(2)
        with f_p2_c1:
            p2_pos_filter = st.selectbox("Filter Posisi (P2):", pos_options, key="p2_pos_filt")
        with f_p2_c2:
            p2_club_filter = st.selectbox("Filter Klub (P2):", club_options, key="p2_club_filt")
        
        p2_pool = df.copy()
        if filter_played_only:
            p2_pool = p2_pool[p2_pool['Menit Bermain'] > 0]
        if p2_pos_filter != "Semua Posisi":
            p2_pool = p2_pool[p2_pool['Posisi'] == p2_pos_filter]
        if p2_club_filter != "Semua Klub":
            p2_pool = p2_pool[p2_pool['Klub'] == p2_club_filter]
            
        if p2_pool.empty:
            p2_pool = active_pool.copy()

        p2_pool = p2_pool.sort_values(by="Total Poin", ascending=False)
        p2_records = p2_pool.to_dict('records')
        
        cur_p2_name = st.session_state.get('comp_p2')
        p2_names = [r.get('Nama Pemain') for r in p2_records]
        if cur_p2_name not in p2_names and p2_records:
            cur_p2_name = p2_records[0].get('Nama Pemain')
            st.session_state['comp_p2'] = cur_p2_name

        p2_idx = 0
        for i, rec in enumerate(p2_records):
            if rec.get('Nama Pemain') == cur_p2_name:
                p2_idx = i
                break
                
        p2_selected_record = st.selectbox(
            "Pilih Pemain 2:",
            options=p2_records,
            index=p2_idx if 0 <= p2_idx < len(p2_records) else 0,
            format_func=format_player_picker,
            key=f"p2_sel_box_{cur_p2_name}"
        )
        if p2_selected_record:
            st.session_state['comp_p2'] = p2_selected_record.get('Nama Pemain')

    p1 = p1_selected_record
    p2 = p2_selected_record

    if not p1 or not p2:
        st.warning("Silakan pilih kedua pemain untuk memulai komparasi radar.")
        return

    # Visual Profile Cards Row
    st.divider()
    card_col1, card_mid, card_col2 = st.columns([10, 2, 10])

    with card_col1:
        st.markdown(f"""
        <div class="compare-card-p1">
            <span class="compare-tag-p1">{p1.get('Posisi', '')} · {p1.get('Klub', '')}</span>
            <h3 style="margin: 0 0 6px 0; color: #0284c7; font-weight: 800; font-size: 1.4rem;">{p1.get('Nama Pemain', '')}</h3>
            <p style="margin: 0 0 10px 0; font-size: 0.85rem; color: #64748b;">{p1.get('Nama Lengkap', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Harga", f"£{p1.get('Harga (£m)', 0):.1f}m")
        with m2:
            st.metric("Total Poin", f"{int(p1.get('Total Poin', 0))} pts")
        with m3:
            st.metric("Prediksi xPoin", f"{float(p1.get('xPoin', 0)):.2f} pts")
        with m4:
            st.metric("Form", f"{float(p1.get('Form', 0)):.2f}")

    with card_mid:
        st.markdown("""
        <div style="display: flex; height: 100%; align-items: center; justify-content: center; padding-top: 30px;">
            <div class="vs-circle">VS</div>
        </div>
        """, unsafe_allow_html=True)

    with card_col2:
        st.markdown(f"""
        <div class="compare-card-p2">
            <span class="compare-tag-p2">{p2.get('Posisi', '')} · {p2.get('Klub', '')}</span>
            <h3 style="margin: 0 0 6px 0; color: #e11d48; font-weight: 800; font-size: 1.4rem;">{p2.get('Nama Pemain', '')}</h3>
            <p style="margin: 0 0 10px 0; font-size: 0.85rem; color: #64748b;">{p2.get('Nama Lengkap', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Harga", f"£{p2.get('Harga (£m)', 0):.1f}m")
        with m2:
            st.metric("Total Poin", f"{int(p2.get('Total Poin', 0))} pts")
        with m3:
            st.metric("Prediksi xPoin", f"{float(p2.get('xPoin', 0)):.2f} pts")
        with m4:
            st.metric("Form", f"{float(p2.get('Form', 0)):.2f}")

    # Radar Configuration Controls
    st.markdown("### 🕸️ Grafik Radar Komparasi Statistik")
    
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        radar_preset = st.selectbox(
            "Pilih Preset Metrik Radar:",
            options=[
                "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)",
                "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Avg Mins L5M, BPS, xGI/90, ICT, Poin/£m)",
                "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, xGI/90, BPS, % Ownership)",
                "🛡️ Defensif & Kiper (Clean Sheet, Def Contrib/90, Saves, Avg Mins L5M, BPS, xPoin, Kemudahan Jadwal)",
                "🛠️ Kustom (Pilih Metrik Bebas)"
            ],
            key="radar_preset_choice"
        )
    with ctrl2:
        norm_mode = st.selectbox(
            "Skala Nilai Sumbu Radar:",
            options=[
                "Persentil Liga (0 - 100%)",
                "Min-Max Normalisasi (0 - 100)",
                "Nilai Aktual (Raw Values)"
            ],
            help="Skala Persentil direkomendasikan agar metrik dengan skala berbeda (misal menit, xG, poin, harga) dapat dibandingkan secara proporsional.",
            key="radar_norm_mode"
        )
    with ctrl3:
        cohort_mode = st.selectbox(
            "Basis Komparasi Persentil/Min-Max:",
            options=[
                "Seluruh Pemain Liga (All Players)",
                f"Sesama Posisi {p1.get('Nama Pemain')} ({p1.get('Posisi')})",
                f"Sesama Posisi {p2.get('Nama Pemain')} ({p2.get('Posisi')})"
            ],
            key="radar_cohort_mode"
        )

    preset_metric_map = {
        "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)": [
            'xG', 'xA', 'xGI per 90', 'Gol', 'Asis', 'ICT Index', 'xPoin', 'Form'
        ],
        "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Avg Mins L5M, BPS, xGI/90, ICT, Poin/£m)": [
            'Total Poin', 'xPoin', 'Form', 'Avg Mins (L5M)', 'BPS', 'xGI per 90', 'ICT Index', 'Poin per £m'
        ],
        "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, xGI/90, BPS, % Ownership)": [
            'Poin per £m', 'xPoin per £m', 'Form', 'Avg Mins (L5M)', 'xGI per 90', 'BPS', '% Ownership'
        ],
        "🛡️ Defensif & Kiper (Clean Sheet, Def Contrib/90, Saves, Avg Mins L5M, BPS, xPoin, Kemudahan Jadwal)": [
            'Clean Sheet', 'Defensive Contribution per 90', 'Saves', 'Avg Mins (L5M)', 'BPS', 'xPoin', 'Kemudahan Jadwal'
        ]
    }

    all_available_metrics = [
        'Total Poin', 'xPoin', 'xPoin (Option B)', 'Form', 'Harga (£m)', 'Avg Mins (L5M)', 
        'Menit Bermain', 'Gol', 'Asis', 'xG', 'xA', 'xGI', 'xG per 90', 'xA per 90', 'xGI per 90', 
        'ICT Index', 'BPS', 'Bonus Poin', 'Clean Sheet', 'Saves', 'Defensive Contribution per 90', 
        '% Ownership', 'Net Transfers GW', 'Poin per £m', 'xPoin per £m', 'Kemudahan Jadwal'
    ]

    if radar_preset == "🛠️ Kustom (Pilih Metrik Bebas)":
        active_metric_keys = st.multiselect(
            "Pilih minimal 3 metrik statistik untuk grafik radar:",
            options=all_available_metrics,
            default=['Total Poin', 'xPoin', 'xG', 'xA', 'Form', 'ICT Index', 'BPS', 'Poin per £m'],
            key="radar_custom_metrics"
        )
    else:
        active_metric_keys = preset_metric_map.get(radar_preset, preset_metric_map["🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)"])

    if len(active_metric_keys) < 3:
        st.warning("Pilih minimal 3 metrik agar poligon radar chart dapat divisualisasikan dengan proporsional.")
        return

    # Tentukan cohort data (persentil dihitung terhadap pemain aktif yang sudah bermain di EPL)
    if f"Sesama Posisi {p1.get('Nama Pemain')}" in cohort_mode:
        cohort_df = df[df['Posisi'] == p1.get('Posisi')]
    elif f"Sesama Posisi {p2.get('Nama Pemain')}" in cohort_mode:
        cohort_df = df[df['Posisi'] == p2.get('Posisi')]
    else:
        cohort_df = df

    active_cohort = cohort_df[cohort_df['Menit Bermain'] > 0]
    if not active_cohort.empty:
        cohort_df = active_cohort
    elif cohort_df.empty:
        cohort_df = df

    METRIC_META = {
        'xG': {'label': 'xG (Exp. Goals)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xA': {'label': 'xA (Exp. Assists)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xGI': {'label': 'xGI (Goal Involv.)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xG per 90': {'label': 'xG / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'xA per 90': {'label': 'xA / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'xGI per 90': {'label': 'xGI / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'Gol': {'label': 'Total Gol', 'higher_better': True, 'fmt': '{:.0f}'},
        'Asis': {'label': 'Total Asis', 'higher_better': True, 'fmt': '{:.0f}'},
        'Total Poin': {'label': 'Total Poin', 'higher_better': True, 'fmt': '{:.0f} pts'},
        'xPoin': {'label': 'Prediksi xPoin GW', 'higher_better': True, 'fmt': '{:.2f} pts'},
        'xPoin (Option B)': {'label': 'xPoin (Opt B)', 'higher_better': True, 'fmt': '{:.2f} pts'},
        'Form': {'label': 'Form Terkini', 'higher_better': True, 'fmt': '{:.2f}'},
        'Harga (£m)': {'label': 'Harga (£m)', 'higher_better': False, 'fmt': '£{:.1f}m'},
        'Avg Mins (L5M)': {'label': 'Avg Mins (L5M)', 'higher_better': True, 'fmt': '{:.1f} m'},
        'Menit Bermain': {'label': 'Menit Bermain', 'higher_better': True, 'fmt': '{:.0f} m'},
        'ICT Index': {'label': 'ICT Index', 'higher_better': True, 'fmt': '{:.1f}'},
        'BPS': {'label': 'BPS Score', 'higher_better': True, 'fmt': '{:.0f}'},
        'Bonus Poin': {'label': 'Bonus Poin', 'higher_better': True, 'fmt': '{:.0f}'},
        'Clean Sheet': {'label': 'Clean Sheet', 'higher_better': True, 'fmt': '{:.0f}'},
        'Saves': {'label': 'Saves', 'higher_better': True, 'fmt': '{:.0f}'},
        'Defensive Contribution per 90': {'label': 'Def Contrib / 90', 'higher_better': True, 'fmt': '{:.2f}'},
        '% Ownership': {'label': '% Ownership', 'higher_better': True, 'fmt': '{:.1f}%'},
        'Net Transfers GW': {'label': 'Net Transfers GW', 'higher_better': True, 'fmt': '{:+.0f}'},
        'Poin per £m': {'label': 'Poin / £m', 'higher_better': True, 'fmt': '{:.2f}'},
        'xPoin per £m': {'label': 'xPoin / £m', 'higher_better': True, 'fmt': '{:.2f}'},
        'Kemudahan Jadwal': {'label': 'Kemudahan Jadwal', 'higher_better': True, 'fmt': '{:.1f}'},
    }

    theta_labels = []
    r_p1 = []
    r_p2 = []
    customdata_p1 = []
    customdata_p2 = []

    h2h_rows = []
    p1_wins = 0
    p2_wins = 0
    ties = 0

    for m_key in active_metric_keys:
        meta = METRIC_META.get(m_key, {'label': m_key, 'higher_better': True, 'fmt': '{:.2f}'})
        lbl = meta['label']
        higher_better = meta['higher_better']
        fmt = meta['fmt']

        val1 = float(p1.get(m_key, 0.0) or 0.0)
        val2 = float(p2.get(m_key, 0.0) or 0.0)

        try:
            val1_str = fmt.format(val1)
        except (ValueError, TypeError):
            val1_str = str(val1)
        try:
            val2_str = fmt.format(val2)
        except (ValueError, TypeError):
            val2_str = str(val2)

        if "Persentil" in norm_mode:
            c_vals = cohort_df[m_key].dropna().astype(float).values if m_key in cohort_df.columns else np.array([val1, val2])
            if len(c_vals) > 0:
                pct1 = float(percentileofscore(c_vals, val1, kind='rank'))
                pct2 = float(percentileofscore(c_vals, val2, kind='rank'))
                if not higher_better:
                    pct1 = 100.0 - pct1
                    pct2 = 100.0 - pct2
            else:
                pct1, pct2 = 50.0, 50.0
            score1 = round(pct1, 1)
            score2 = round(pct2, 1)
        elif "Min-Max" in norm_mode:
            c_vals = cohort_df[m_key].dropna().astype(float).values if m_key in cohort_df.columns else np.array([val1, val2])
            if len(c_vals) > 0:
                c_min, c_max = float(np.min(c_vals)), float(np.max(c_vals))
                if c_max > c_min:
                    s1 = ((val1 - c_min) / (c_max - c_min)) * 100.0
                    s2 = ((val2 - c_min) / (c_max - c_min)) * 100.0
                    if not higher_better:
                        s1 = 100.0 - s1
                        s2 = 100.0 - s2
                else:
                    s1, s2 = 50.0, 50.0
            else:
                s1, s2 = 50.0, 50.0
            score1 = round(float(np.clip(s1, 0.0, 100.0)), 1)
            score2 = round(float(np.clip(s2, 0.0, 100.0)), 1)
        else:
            score1 = val1
            score2 = val2

        theta_labels.append(lbl)
        r_p1.append(score1)
        r_p2.append(score2)
        customdata_p1.append([val1_str, score1])
        customdata_p2.append([val2_str, score2])

        diff = val1 - val2
        diff_str = f"{diff:+.2f}" if abs(diff) < 100 else f"{diff:+.0f}"
        if abs(diff) < 1e-4:
            winner = "⚪ Seimbang"
            ties += 1
        elif (val1 > val2 and higher_better) or (val1 < val2 and not higher_better):
            winner = f"🔵 {p1.get('Nama Pemain')}"
            p1_wins += 1
        else:
            winner = f"🔴 {p2.get('Nama Pemain')}"
            p2_wins += 1

        h2h_rows.append({
            'Metrik': lbl,
            f"{p1.get('Nama Pemain')} (🔵)": val1_str,
            f"{p2.get('Nama Pemain')} (🔴)": val2_str,
            'Selisih (P1 - P2)': diff_str,
            'Keunggulan': winner
        })

    # Tutup poligon polar (ulangi titik pertama)
    theta_closed = theta_labels + [theta_labels[0]]
    r_p1_closed = r_p1 + [r_p1[0]]
    r_p2_closed = r_p2 + [r_p2[0]]
    customdata_p1_closed = customdata_p1 + [customdata_p1[0]]
    customdata_p2_closed = customdata_p2 + [customdata_p2[0]]

    p1_name = p1.get('Nama Pemain', 'Pemain 1')
    p2_name = p2.get('Nama Pemain', 'Pemain 2')
    p1_club = p1.get('Klub', '-')
    p2_club = p2.get('Klub', '-')

    fig = go.Figure()

    # Trace Pemain 1
    fig.add_trace(go.Scatterpolar(
        r=r_p1_closed,
        theta=theta_closed,
        fill='toself',
        name=f"🔵 {p1_name} ({p1_club})",
        line=dict(color='#0284c7', width=3),
        fillcolor='rgba(2, 132, 199, 0.22)',
        marker=dict(size=7, color='#0284c7', symbol='circle'),
        customdata=customdata_p1_closed,
        hovertemplate=(
            "<b>%{data.name}</b><br>"
            "Metrik: <b>%{theta}</b><br>"
            "Nilai Aktual: <b>%{customdata[0]}</b><br>"
            "Skor Radar: <b>%{r:.1f}</b>"
            "<extra></extra>"
        )
    ))

    # Trace Pemain 2
    fig.add_trace(go.Scatterpolar(
        r=r_p2_closed,
        theta=theta_closed,
        fill='toself',
        name=f"🔴 {p2_name} ({p2_club})",
        line=dict(color='#e11d48', width=3),
        fillcolor='rgba(225, 29, 72, 0.22)',
        marker=dict(size=7, color='#e11d48', symbol='diamond'),
        customdata=customdata_p2_closed,
        hovertemplate=(
            "<b>%{data.name}</b><br>"
            "Metrik: <b>%{theta}</b><br>"
            "Nilai Aktual: <b>%{customdata[0]}</b><br>"
            "Skor Radar: <b>%{r:.1f}</b>"
            "<extra></extra>"
        )
    ))

    max_val = max(max(r_p1), max(r_p2)) if len(r_p1) > 0 else 100
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] if "Nilai Aktual" not in norm_mode else [0, max_val * 1.15 if max_val > 0 else 10],
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
                tickfont=dict(size=10, color="#64748b", family="Plus Jakarta Sans"),
                ticksuffix="%" if "Persentil" in norm_mode else ""
            ),
            angularaxis=dict(
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
                tickfont=dict(size=11, color="#0f172a", family="Plus Jakarta Sans", weight="bold")
            ),
            bgcolor="#ffffff"
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans", color="#1e293b"),
        margin=dict(l=50, r=50, t=50, b=50),
        height=620,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=13, weight="bold")
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 *Arahkan kursor ke titik sudut radar untuk melihat nilai riil FPL dan skor persentil masing-masing pemain.*")

    # Head to Head Comparison Table
    st.divider()
    st.markdown("### 📊 Head-to-Head Matriks Statistik & Pemenang Metrik")

    score_c1, score_c2, score_c3 = st.columns(3)
    with score_c1:
        st.metric(f"🔵 Keunggulan {p1_name}", f"{p1_wins} Metrik", f"{(p1_wins / len(active_metric_keys) * 100):.0f}% Dominasi")
    with score_c2:
        st.metric(f"🔴 Keunggulan {p2_name}", f"{p2_wins} Metrik", f"{(p2_wins / len(active_metric_keys) * 100):.0f}% Dominasi")
    with score_c3:
        st.metric("⚪ Imbang / Seimbang", f"{ties} Metrik")

    h2h_df = pd.DataFrame(h2h_rows)
    st.dataframe(h2h_df, use_container_width=True, height=min(600, len(h2h_df) * 38 + 50))

    # Catatan Analisis Scout FPL
    st.markdown("### 🧠 Catatan Analisis Scout FPL (Tactical Verdict)")

    p1_cost = float(p1.get('Harga (£m)', 0.0) or 0.0)
    p2_cost = float(p2.get('Harga (£m)', 0.0) or 0.0)
    p1_xpts = float(p1.get('xPoin', 0.0) or 0.0)
    p2_xpts = float(p2.get('xPoin', 0.0) or 0.0)
    p1_pts = int(p1.get('Total Poin', 0) or 0)
    p2_pts = int(p2.get('Total Poin', 0) or 0)
    p1_ppm = p1_pts / p1_cost if p1_cost > 0 else 0.0
    p2_ppm = p2_pts / p2_cost if p2_cost > 0 else 0.0
    p1_xg = float(p1.get('xG', 0.0) or 0.0)
    p2_xg = float(p2.get('xG', 0.0) or 0.0)
    p1_xa = float(p1.get('xA', 0.0) or 0.0)
    p2_xa = float(p2.get('xA', 0.0) or 0.0)
    p1_fdr = float(p1.get('FDR1', 3.0) or 3.0)
    p2_fdr = float(p2.get('FDR1', 3.0) or 3.0)

    if p1_xpts > p2_xpts + 0.3:
        xpts_insight = f"**{p1_name}** diunggulkan model untuk Gameweek mendatang dengan proyeksi **{p1_xpts:.2f} pts** vs **{p2_name}** ({p2_xpts:.2f} pts), selisih **+{p1_xpts - p2_xpts:.2f} pts**."
    elif p2_xpts > p1_xpts + 0.3:
        xpts_insight = f"**{p2_name}** diunggulkan model untuk Gameweek mendatang dengan proyeksi **{p2_xpts:.2f} pts** vs **{p1_name}** ({p1_xpts:.2f} pts), selisih **+{p2_xpts - p1_xpts:.2f} pts**."
    else:
        xpts_insight = f"Kedua pemain memiliki proyeksi xPoin yang sangat berimbang untuk Gameweek mendatang ({p1_xpts:.2f} pts vs {p2_xpts:.2f} pts)."

    price_diff = abs(p1_cost - p2_cost)
    if p1_cost < p2_cost and p1_ppm >= p2_ppm:
        value_insight = f"**{p1_name}** lebih hemat **£{price_diff:.1f}m** dan memberikan efisiensi poin per juta lebih tinggi (**{p1_ppm:.2f} pts/£m** vs {p2_ppm:.2f} pts/£m)."
    elif p2_cost < p1_cost and p2_ppm >= p1_ppm:
        value_insight = f"**{p2_name}** lebih hemat **£{price_diff:.1f}m** dan memberikan efisiensi poin per juta lebih tinggi (**{p2_ppm:.2f} pts/£m** vs {p1_ppm:.2f} pts/£m)."
    elif p1_ppm > p2_ppm:
        value_insight = f"Meskipun terdapat selisih harga £{price_diff:.1f}m, **{p1_name}** menghasilkan efisiensi akumulasi poin lebih baik ({p1_ppm:.2f} vs {p2_ppm:.2f} pts/£m)."
    else:
        value_insight = f"**{p2_name}** menghasilkan efisiensi akumulasi poin per harga lebih tinggi ({p2_ppm:.2f} vs {p1_ppm:.2f} pts/£m)."

    style1 = "pencetak gol utama (finisher)" if p1_xg >= p1_xa else "kreator peluang (playmaker)"
    style2 = "pencetak gol utama (finisher)" if p2_xg >= p2_xa else "kreator peluang (playmaker)"
    threat_insight = f"Secara profil ancaman, **{p1_name}** bertindak dominan sebagai *{style1}* (xG: {p1_xg:.2f}, xA: {p1_xa:.2f}), sedangkan **{p2_name}** berperan sebagai *{style2}* (xG: {p2_xg:.2f}, xA: {p2_xa:.2f})."

    opp1 = p1.get('Lawan GW Berikutnya', '-')
    opp2 = p2.get('Lawan GW Berikutnya', '-')
    if p1_fdr < p2_fdr:
        fix_insight = f"Jadwal terdekat menguntungkan **{p1_name}** (Lawan: {opp1}, FDR {p1_fdr:.0f}) dibanding **{p2_name}** (Lawan: {opp2}, FDR {p2_fdr:.0f})."
    elif p2_fdr < p1_fdr:
        fix_insight = f"Jadwal terdekat menguntungkan **{p2_name}** (Lawan: {opp2}, FDR {p2_fdr:.0f}) dibanding **{p1_name}** (Lawan: {opp1}, FDR {p1_fdr:.0f})."
    else:
        fix_insight = f"Tingkat kesulitan lawan berikutnya setara (FDR {p1_fdr:.0f}): {p1_name} vs {opp1} dan {p2_name} vs {opp2}."

    st.markdown(f"""
    <div class="radar-summary-box">
        <h4 style="margin: 0 0 10px 0; color: #37003c; font-weight: 800;">📋 Kesimpulan & Rekomendasi Scout:</h4>
        <ul style="margin: 0; padding-left: 20px; color: #334155; line-height: 1.7; font-size: 0.92rem;">
            <li><strong>Prediksi Gameweek Mendatang:</strong> {xpts_insight}</li>
            <li><strong>Efisiensi Anggaran (Value for Money):</strong> {value_insight}</li>
            <li><strong>Karakteristik Serangan:</strong> {threat_insight}</li>
            <li><strong>Faktor Jadwal Terdekat:</strong> {fix_insight}</li>
        </ul>
    </div>
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Player Stats & xPoin", 
        "📈 Visualisasi Data & Radar Pemain", 
        "🛡️ Team Strength Analysis",
        "📅 Fixtures & FDR", 
        "🧮 Option B: Component Model xPoin",
        "🔮 Option C: Current Season Model"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PLAYER STATS & XPOIN PREDICTOR
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📋 Tabel Statistik & Prediksi xPoin Pemain")

        # Summary Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pemain Terfilter", len(filtered_players))
        with col2:
            top_xpoin = filtered_players.sort_values(by="xPoin", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Top xPoin Laga Ini", f"{top_xpoin['Nama Pemain']} ({top_xpoin['xPoin']} pts)" if top_xpoin is not None else "-")
        with col3:
            top_pts = filtered_players.sort_values(by="Total Poin", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Poin Tertinggi", f"{top_pts['Nama Pemain']} ({top_pts['Total Poin']} pts)" if top_pts is not None else "-")
        with col4:
            top_form = filtered_players.sort_values(by="Form", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Form Terbaik", f"{top_form['Nama Pemain']} ({top_form['Form']})" if top_form is not None else "-")

        # Expander for Custom Column Selection
        default_cols = [
            'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin', 'Avg Mins (L5M)', 'Total Poin',
            'FDR1', 'FDR3', 'FDR5', 'Form', '% Ownership', 'xG', 'xA', 'Status', 'Peluang Main GW (%)'
        ]
        
        with st.expander("⚙️ Pilih Kolom yang Ingin Ditampilkan pada Tabel", expanded=False):
            available_table_cols = [c for c in players_df.columns if c not in ['id', 'team']]
            selected_cols = st.multiselect(
                "Centang/Pilih kolom data FPL yang ingin dimunculkan di tabel:",
                options=available_table_cols,
                default=default_cols,
                key="table_cols_picker"
            )
        
        if not selected_cols:
            selected_cols = default_cols

        # Expander for Positional Regression Model Details
        with st.expander("🤖 Detail 4 Model Regression xPoin Berdasarkan Posisi (FWD, MID, DEF, GK)", expanded=False):
            pos_tab1, pos_tab2, pos_tab3, pos_tab4 = st.tabs(["⚽ FWD (Penyerang)", "🎯 MID (Gelandang)", "🛡️ DEF (Bek)", "🧤 GK (Kiper)"])
            
            tab_mapping = [
                (pos_tab1, "FWD"),
                (pos_tab2, "MID"),
                (pos_tab3, "DEF"),
                (pos_tab4, "GK")
            ]
            
            for tab_obj, pos_key in tab_mapping:
                with tab_obj:
                    pos_info = models_dict.get(pos_key, {})
                    if pos_info:
                        st.markdown(f"#### 📐 Model Regresi Linier Posisi **{pos_key}**")
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("R² Score (Akurasi Fitting)", f"{pos_info['r2']:.4f}")
                        with m2:
                            st.metric("Intercept (Konstanta β₀)", f"{pos_info['intercept']:.4f}")
                        with m3:
                            src_label = 'Histori Pertandingan Aktual API' if pos_info['is_real_history'] else 'Simulasi Regresi Performa FPL'
                            st.metric("Sumber Data Latihan", src_label)

                        st.markdown("##### Bobot / Koefisien Variabel Input (β):")
                        st.dataframe(pos_info['coef_df'], use_container_width=True)
                        st.caption(f"💡 *Model {pos_key} menghitung prediksi xPoin laga mendatang menggunakan variabel X spesifik posisi {pos_key}.*")

        # Expander for Classical Assumption Diagnostics (Per Posisi)
        with st.expander("🧪 Diagnostik Uji Asumsi Klasik Model Regresi", expanded=False):
            diag_pos_choice = st.selectbox(
                "Pilih Posisi Pemain untuk Melihat Uji Asumsi Klasik (FWD | MID | DEF | GK):",
                options=["FWD", "MID", "DEF", "GK"],
                key="pos_diag_selector"
            )
            
            diag_pos_info = models_dict.get(diag_pos_choice, {})
            diag_results = diag_pos_info.get('diag_results') if diag_pos_info else None
            
            if diag_results is not None:
                st.markdown(f"##### 📌 Ringkasan Hasil Uji Asumsi Klasik (Model {diag_pos_choice})")
                
                # Metrics Row
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    v_pass = diag_results['all_vif_ok']
                    st.metric("1. Multikolinearitas", "🟢 Lolos" if v_pass else "🔴 Terindikasi", "VIF < 10" if v_pass else "VIF >= 10")
                with d2:
                    l_pass = diag_results['linearity']['passed']
                    st.metric("2. Linearitas", "🟢 Linear" if l_pass else "🔴 Non-Linear", f"p = {diag_results['linearity']['p_value']}")
                with d3:
                    n_pass = diag_results['normality']['passed']
                    st.metric("3. Normalitas Residual", "🟢 Normal" if n_pass else "🔴 Tidak Normal", f"p = {diag_results['normality']['p_value']}")
                with d4:
                    h_pass = diag_results['homoscedasticity']['passed']
                    st.metric("4. Homoskedastisitas", "🟢 Varian Konstan" if h_pass else "🔴 Heteroskedastisitas", f"p = {diag_results['homoscedasticity']['p_value']}")

                st.divider()

                # 1. Uji Multikolinearitas
                st.markdown(f"##### 1. Uji Multikolinearitas (Variance Inflation Factor & Tolerance) - {diag_pos_choice}")
                st.dataframe(diag_results['vif_df'], use_container_width=True)
                st.caption("💡 *Kriteria: Nilai VIF < 10.0 dan Tolerance > 0.10 menandakan variabel independen bebas dari multikolinearitas.*")

                # 2. Uji Linearitas
                st.markdown(f"##### 2. Uji Linearitas (Rainbow Test) - {diag_pos_choice}")
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    st.markdown(f"**Statistic (Rainbow Test):** `{diag_results['linearity']['stat']}`")
                with lc2:
                    st.markdown(f"**p-value:** `{diag_results['linearity']['p_value']}`")
                with lc3:
                    l_status = "🟢 Linear (p > 0.05)" if diag_results['linearity']['passed'] else "🔴 Non-Linear (p <= 0.05)"
                    st.markdown(f"**Status:** {l_status}")
                st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan hubungan antara variabel independen dan dependen bersifat linear.*")

                # 3. Uji Normalitas Residual
                st.markdown(f"##### 3. Uji Normalitas Residual (Shapiro-Wilk Test) - {diag_pos_choice}")
                nc1, nc2, nc3 = st.columns(3)
                with nc1:
                    st.markdown(f"**Statistic (W):** `{diag_results['normality']['stat']}`")
                with nc2:
                    st.markdown(f"**p-value:** `{diag_results['normality']['p_value']}`")
                with nc3:
                    n_status = "🟢 Residual Normal (p > 0.05)" if diag_results['normality']['passed'] else "🔴 Tidak Normal (p <= 0.05)"
                    st.markdown(f"**Status:** {n_status}")
                st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan residual e = (Y_aktual - Y_prediksi) terdistribusi secara normal.*")

                # 4. Uji Homoskedastisitas
                st.markdown(f"##### 4. Uji Homoskedastisitas (Breusch-Pagan Test) - {diag_pos_choice}")
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    st.markdown(f"**LM Statistic:** `{diag_results['homoscedasticity']['stat']}`")
                with hc2:
                    st.markdown(f"**p-value (LM-Test):** `{diag_results['homoscedasticity']['p_value']}`")
                with hc3:
                    h_status = "🟢 Homoskedastisitas (Varian Konstan)" if diag_results['homoscedasticity']['passed'] else "🔴 Heteroskedastisitas (p <= 0.05)"
                    st.markdown(f"**Status:** {h_status}")
                st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan varian dari residual bersifat konstan (homoskedastisitas).*")
            else:
                st.warning(f"Diagnostik Uji Asumsi Klasik untuk model {diag_pos_choice} tidak dapat dihitung.")

        sorted_players = filtered_players.sort_values(by="xPoin", ascending=False)

        st.dataframe(
            sorted_players[selected_cols],
            use_container_width=True,
            height=520,
            column_config={
                "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
                "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins"),
                "Total Poin": st.column_config.NumberColumn(format="%d pts"),
                "FDR1": st.column_config.NumberColumn(format="%.1f"),
                "FDR3": st.column_config.NumberColumn(format="%.2f"),
                "FDR5": st.column_config.NumberColumn(format="%.2f"),
                "Menit Bermain": st.column_config.NumberColumn(format="%d mins"),
                "Form": st.column_config.NumberColumn(format="%.2f"),
                "% Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                "Net Transfers GW": st.column_config.NumberColumn(format="%+d"),
                "Transfers In GW": st.column_config.NumberColumn(format="%d"),
                "Transfers Out GW": st.column_config.NumberColumn(format="%d"),
                "xG": st.column_config.NumberColumn(format="%.2f"),
                "xA": st.column_config.NumberColumn(format="%.2f"),
                "xGI": st.column_config.NumberColumn(format="%.2f"),
                "xG per 90": st.column_config.NumberColumn(format="%.2f"),
                "xA per 90": st.column_config.NumberColumn(format="%.2f"),
                "xGI per 90": st.column_config.NumberColumn(format="%.2f"),
                "ICT Index": st.column_config.NumberColumn(format="%.1f"),
                "BPS": st.column_config.NumberColumn(format="%d"),
                "Bonus Poin": st.column_config.NumberColumn(format="%d"),
                "Kartu Kuning": st.column_config.NumberColumn(format="%d"),
                "Kartu Merah": st.column_config.NumberColumn(format="%d"),
                "Saves": st.column_config.NumberColumn(format="%d"),
                "Peluang Main GW (%)": st.column_config.ProgressColumn(
                    "Peluang Main GW (%)",
                    min_value=0,
                    max_value=100,
                    format="%d%%"
                )
            }
        )

        # ---------------------------------------------------------------------
        # ENHANCED SECTION: PLAYER PERFORMANCE & PROGRESSION OVER TIME
        # ---------------------------------------------------------------------
        st.divider()
        st.markdown("### 📈 Visualisasi Tren & Progresi Performa Pemain per Gameweek")
        st.write("Visualisasikan performa pertandingan demi pertandingan pemain sepanjang musim berjalan (total poin, gol, asis, xG, xA, dan menit bermain) menggunakan data real-time API FPL.")

        # Player Selector
        player_pool = filtered_players if not filtered_players.empty else players_df
        player_choices = player_pool.to_dict('records')
        
        default_idx = 0
        for i, p in enumerate(player_choices):
            if p.get('Nama Pemain') in ['Haaland', 'M.Salah', 'Saka', 'Palmer']:
                default_idx = i
                break

        def format_player_option(p):
            if not p or not isinstance(p, dict):
                return ""
            p_name = p.get('Nama Pemain', '')
            p_klub = p.get('Klub', '')
            p_pos = p.get('Posisi', '')
            p_pts = p.get('Total Poin', 0)
            try:
                price_str = f"£{float(p.get('Harga (£m)', 0.0)):.1f}m"
            except (ValueError, TypeError):
                price_str = "-"
            try:
                xpts_str = f"{float(p.get('xPoin', 0.0)):.2f}"
            except (ValueError, TypeError):
                xpts_str = "-"
            return f"{p_name} ({p_klub} - {p_pos}) · {price_str} · Total Poin: {p_pts} pts · xPoin: {xpts_str}"

        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            selected_player = st.selectbox(
                "Pilih Pemain untuk Melihat Tren Performa Gameweek:",
                options=player_choices,
                index=default_idx if 0 <= default_idx < len(player_choices) else 0,
                format_func=format_player_option,
                key="player_trend_selector"
            )
        with sel_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            view_mode = st.radio(
                "Mode Tampilan Nilai:",
                options=["Per Gameweek (Match)", "Akumulatif (Kumulatif)"],
                horizontal=True,
                key="player_prog_mode"
            )

        if selected_player:
            sel_pid = selected_player.get('id')
            if not sel_pid:
                for el in fpl_data.get('elements', []):
                    if el.get('web_name') == selected_player.get('Nama Pemain'):
                        sel_pid = el.get('id')
                        break
            sel_pname = selected_player.get('Nama Pemain', 'Pemain')
            sel_club = selected_player.get('Klub', '-')
            sel_pos = selected_player.get('Posisi', '-')
            sel_cost = float(selected_player.get('Harga (£m)', 0.0) or 0.0)
            sel_pts = int(selected_player.get('Total Poin', 0) or 0)
            sel_xg = float(selected_player.get('xG', 0.0) or 0.0)
            sel_xa = float(selected_player.get('xA', 0.0) or 0.0)
            sel_goals = int(selected_player.get('Gol', 0) or 0)
            sel_assists = int(selected_player.get('Asis', 0) or 0)
            sel_form = float(selected_player.get('Form', 0.0) or 0.0)
            sel_status = selected_player.get('Status', 'Tersedia')
            sel_chance = selected_player.get('Peluang Main GW (%)', 100)
            sel_next_opp = selected_player.get('Lawan GW Berikutnya', '-')
            sel_fdr = selected_player.get('FDR1', 3.0)

            # Quick Player Bio & KPI Metrics Card Row
            kp1, kp2, kp3, kp4, kp5, kp6 = st.columns(6)
            with kp1:
                st.metric("Total Poin", f"{sel_pts} pts", f"Form: {sel_form}")
            with kp2:
                st.metric("Total Gol / xG", f"{sel_goals} / {sel_xg:.2f}", f"xG: {sel_xg:.2f}")
            with kp3:
                st.metric("Total Asis / xA", f"{sel_assists} / {sel_xa:.2f}", f"xA: {sel_xa:.2f}")
            with kp4:
                st.metric("Harga (£m)", f"£{sel_cost:.1f}m", f"{sel_pos} · {sel_club}")
            with kp5:
                st.metric("Status Kebugaran", sel_status, f"Peluang: {sel_chance}%")
            with kp6:
                st.metric("Lawan Berikutnya", sel_next_opp, f"FDR: {sel_fdr}")

            # Fetch player element summary
            with st.spinner(f"Mengambil data histori pertandingan {sel_pname} dari API FPL..."):
                p_summary = fetch_player_element_summary(sel_pid)

            p_history = p_summary.get('history', [])
            p_past = p_summary.get('history_past', [])
            p_fixtures = p_summary.get('fixtures', [])

            if p_history:
                # Build match-by-match dataframe
                hist_rows = []
                for h in p_history:
                    opp_id = h.get('opponent_team')
                    opp_name = teams_dict.get(opp_id, f"Team {opp_id}")
                    loc = "H" if h.get('was_home') else "A"
                    gw_round = h.get('round', 1)
                    gw_label = f"GW{gw_round} vs {opp_name} ({loc})"
                    
                    hist_rows.append({
                        'Gameweek': f"GW{gw_round}",
                        'Label Pertandingan': gw_label,
                        'GW_Num': gw_round,
                        'Lawan': f"{opp_name} ({loc})",
                        'Total Poin': int(h.get('total_points', 0)),
                        'Gol': int(h.get('goals_scored', 0)),
                        'Asis': int(h.get('assists', 0)),
                        'xG': round(float(h.get('expected_goals', 0.0)), 2),
                        'xA': round(float(h.get('expected_assists', 0.0)), 2),
                        'xGI': round(float(h.get('expected_goal_involvements', 0.0)), 2),
                        'xGC': round(float(h.get('expected_goals_conceded', 0.0)), 2),
                        'Menit Bermain': int(h.get('minutes', 0)),
                        'BPS': int(h.get('bps', 0)),
                        'Bonus Poin': int(h.get('bonus', 0)),
                        'Clean Sheet': int(h.get('clean_sheets', 0)),
                        'Saves': int(h.get('saves', 0)),
                        'Harga (£m)': h.get('value', 0) / 10.0,
                    })

                df_phist = pd.DataFrame(hist_rows).sort_values(by='GW_Num')

                # Calculate cumulative metrics
                df_phist['Kumulatif Total Poin'] = df_phist['Total Poin'].cumsum()
                df_phist['Kumulatif Gol'] = df_phist['Gol'].cumsum()
                df_phist['Kumulatif Asis'] = df_phist['Asis'].cumsum()
                df_phist['Kumulatif xG'] = df_phist['xG'].cumsum().round(2)
                df_phist['Kumulatif xA'] = df_phist['xA'].cumsum().round(2)
                df_phist['Kumulatif xGI'] = df_phist['xGI'].cumsum().round(2)
                df_phist['Kumulatif Menit Bermain'] = df_phist['Menit Bermain'].cumsum()

                # Metric multiselect
                available_metrics = ['Total Poin', 'xG', 'xA', 'Gol', 'Asis', 'xGI', 'Menit Bermain', 'BPS', 'Bonus Poin']
                default_metrics = ['Total Poin', 'xG', 'xA', 'Gol', 'Asis']
                
                m_col1, m_col2 = st.columns([3, 1])
                with m_col1:
                    chosen_metrics = st.multiselect(
                        "Pilih Metrik untuk Ditampilkan pada Grafik:",
                        options=available_metrics,
                        default=default_metrics,
                        key="p_metrics_multiselect"
                    )
                with m_col2:
                    chart_engine = st.selectbox(
                        "Tipe Grafik:",
                        options=["📈 Streamlit Line Chart", "📶 Streamlit Bar Chart", "✨ Dual-Axis Plotly Combo", "🌊 Streamlit Area Chart"],
                        key="p_chart_engine"
                    )

                if not chosen_metrics:
                    chosen_metrics = default_metrics

                is_cum = (view_mode == "Akumulatif (Kumulatif)")
                plot_cols = [f"Kumulatif {m}" if is_cum else m for m in chosen_metrics]
                
                chart_df = df_phist.set_index('Label Pertandingan')[plot_cols].copy()
                if is_cum:
                    chart_df.columns = [f"{m} (Kumulatif)" for m in chosen_metrics]

                # Render Chart using Streamlit Charting Capabilities
                st.markdown(f"##### 📊 Grafik Progresi {sel_pname} ({view_mode})")
                
                if chart_engine == "📈 Streamlit Line Chart":
                    st.line_chart(chart_df, use_container_width=True)
                elif chart_engine == "📶 Streamlit Bar Chart":
                    st.bar_chart(chart_df, use_container_width=True)
                elif chart_engine == "🌊 Streamlit Area Chart":
                    st.area_chart(chart_df, use_container_width=True)
                else:
                    fig_combo = go.Figure()
                    
                    pts_col = 'Kumulatif Total Poin' if is_cum else 'Total Poin'
                    if 'Total Poin' in chosen_metrics:
                        fig_combo.add_trace(go.Bar(
                            x=df_phist['Label Pertandingan'],
                            y=df_phist[pts_col],
                            name=f"{pts_col}",
                            marker_color='#37003c',
                            opacity=0.85,
                            yaxis='y1'
                        ))
                    
                    line_colors = {
                        'xG': '#00ff85',
                        'xA': '#0284c7',
                        'Gol': '#e11d48',
                        'Asis': '#f59e0b',
                        'xGI': '#8b5cf6',
                        'Menit Bermain': '#64748b',
                        'BPS': '#10b981',
                        'Bonus Poin': '#d97706'
                    }
                    
                    for m in chosen_metrics:
                        if m == 'Total Poin':
                            continue
                        col_name = f"Kumulatif {m}" if is_cum else m
                        use_sec = (m in ['xG', 'xA', 'xGI', 'Gol', 'Asis'] and 'Total Poin' in chosen_metrics and not is_cum)
                        
                        fig_combo.add_trace(go.Scatter(
                            x=df_phist['Label Pertandingan'],
                            y=df_phist[col_name],
                            name=col_name,
                            mode='lines+markers',
                            line=dict(width=3, color=line_colors.get(m, '#0284c7')),
                            marker=dict(size=8),
                            yaxis='y2' if use_sec else 'y1'
                        ))
                    
                    fig_combo.update_layout(
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#f8fafc",
                        font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                        margin=dict(l=30, r=30, t=40, b=30),
                        height=440,
                        hovermode="x unified",
                        xaxis=dict(gridcolor="#e2e8f0", title="Gameweek / Laga Pertandingan"),
                        yaxis=dict(gridcolor="#e2e8f0", title="Total Poin / Angka", side='left'),
                        yaxis2=dict(
                            title="Expected Metrics (xG / xA / Gol / Asis)",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ) if any(m in ['xG', 'xA', 'xGI', 'Gol', 'Asis'] for m in chosen_metrics) and 'Total Poin' in chosen_metrics and not is_cum else None,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_combo, use_container_width=True)

                st.caption("💡 *Tip: Anda dapat beralih antara nilai per laga dan progresi kumulatif sepanjang musim untuk menganalisis tren performa pemain.*")

                # Match-by-match breakdown table
                st.markdown(f"##### 📋 Rincian Match-by-Match Gameweek ({sel_pname})")
                display_cols = ['Gameweek', 'Lawan', 'Total Poin', 'Gol', 'Asis', 'xG', 'xA', 'xGI', 'Menit Bermain', 'BPS', 'Bonus Poin', 'Clean Sheet', 'Saves']
                st.dataframe(
                    df_phist[display_cols],
                    use_container_width=True,
                    column_config={
                        "Total Poin": st.column_config.NumberColumn(format="%d pts"),
                        "Gol": st.column_config.NumberColumn(format="%d"),
                        "Asis": st.column_config.NumberColumn(format="%d"),
                        "xG": st.column_config.NumberColumn(format="%.2f"),
                        "xA": st.column_config.NumberColumn(format="%.2f"),
                        "xGI": st.column_config.NumberColumn(format="%.2f"),
                        "Menit Bermain": st.column_config.NumberColumn(format="%d mins"),
                        "BPS": st.column_config.NumberColumn(format="%d"),
                        "Bonus Poin": st.column_config.NumberColumn(format="%d")
                    }
                )
            else:
                st.info(f"ℹ️ {sel_pname} belum mencatatkan menit bermain pada pertandingan Premier League musim ini.")

            # Past Seasons Career Expander
            if p_past:
                with st.expander(f"🏛️ Tren Riwayat Multi-Musim Sebelumnya ({sel_pname})", expanded=False):
                    st.write(f"Progresi performa {sel_pname} pada musim-musim Premier League sebelumnya:")
                    past_rows = []
                    for s in p_past:
                        past_rows.append({
                            'Musim': s.get('season_name'),
                            'Total Poin': int(s.get('total_points', 0)),
                            'Gol': int(s.get('goals_scored', 0)),
                            'Asis': int(s.get('assists', 0)),
                            'xG': round(float(s.get('expected_goals', 0.0)), 2),
                            'xA': round(float(s.get('expected_assists', 0.0)), 2),
                            'Menit Bermain': int(s.get('minutes', 0)),
                            'Clean Sheet': int(s.get('clean_sheets', 0)),
                            'BPS': int(s.get('bps', 0)),
                            'Harga Awal (£m)': s.get('start_cost', 0) / 10.0,
                            'Harga Akhir (£m)': s.get('end_cost', 0) / 10.0,
                        })
                    df_past = pd.DataFrame(past_rows)
                    
                    st.line_chart(df_past.set_index('Musim')[['Total Poin', 'Gol', 'Asis', 'xG', 'xA']], use_container_width=True)
                    st.dataframe(df_past, use_container_width=True)

            # Upcoming Fixtures Expander
            if p_fixtures:
                with st.expander(f"🗓️ Jadwal Pertandingan Mendatang & FDR ({sel_pname})", expanded=False):
                    st.write(f"Daftar laga mendatang {sel_pname} ({sel_club}) beserta tingkat kesulitan:")
                    next_rows = []
                    for f in p_fixtures[:8]:
                        opp_id = f.get('team_a') if f.get('is_home') else f.get('team_h')
                        opp_name = teams_dict.get(opp_id, f"Team {opp_id}")
                        loc = "🏠 Home" if f.get('is_home') else "✈️ Away"
                        diff = f.get('difficulty', 3)
                        diff_label = {
                            1: "🟢 Sangat Mudah (1)",
                            2: "🟢 Mudah (2)",
                            3: "⚪ Netral / Sedang (3)",
                            4: "🔴 Sulit (4)",
                            5: "🔴 Sangat Sulit (5)"
                        }.get(diff, f"Rating {diff}")
                        
                        next_rows.append({
                            'Gameweek': f"GW{f.get('event')}",
                            'Lawan': opp_name,
                            'Lokasi': loc,
                            'FDR': diff,
                            'Tingkat Kesulitan': diff_label,
                            'Waktu Kickoff': f.get('kickoff_time', '-')[:10] if f.get('kickoff_time') else '-'
                        })
                    df_next_fix = pd.DataFrame(next_rows)
                    st.dataframe(
                        df_next_fix,
                        use_container_width=True,
                        column_config={
                            "FDR": st.column_config.NumberColumn(format="%d")
                        }
                    )

    # -------------------------------------------------------------------------
    # TAB 2: VISUALIZATION & PEARSON CORRELATION (PLOTLY)
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("📈 Visualisasi Interaktif & Radar Komparasi Pemain")
        st.write("Analisis hubungan antar variabel statistik pemain, perbandingan agregat klub, serta komparasi head-to-head 2 pemain dengan grafik radar interaktif.")

        chart_subtab1, chart_subtab2, chart_subtab3 = st.tabs([
            "🔵 Scatter Plot & Pearson Correlation", 
            "📊 Bar Chart Agregat Klub",
            "⚔️ Komparasi 2 Pemain (Radar Chart)"
        ])

        # SECTION 1: SCATTER PLOT & PEARSON R
        with chart_subtab1:
            st.markdown("##### 🔵 Scatter Plot Multi-Metrik dengan Garis Tren Regresi")

            num_cols = [
                'xPoin', 'Total Poin', 'Harga (£m)', 'xG', 'xA', 'xGI',
                'Form', 'BPS', '% Ownership', 'ICT Index', 'xG per 90', 'xA per 90',
                'FDR1', 'FDR3', 'FDR5', 'Menit Bermain', 'Gol', 'Asis'
            ]

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                x_var = st.selectbox("Metrik Sumbu X", options=num_cols, index=3, key="px_x") # xG
            with p_col2:
                y_var = st.selectbox("Metrik Sumbu Y", options=num_cols, index=1, key="px_y") # Total Poin
            with p_col3:
                color_var = st.selectbox("Warna Kelompok", options=['Posisi', 'Klub'], index=0, key="px_color")

            if not filtered_players.empty and len(filtered_players) > 2:
                # Calculate Pearson Correlation
                x_vals = filtered_players[x_var].astype(float)
                y_vals = filtered_players[y_var].astype(float)
                
                # Filter out NaNs if any
                valid_mask = ~(x_vals.isna() | y_vals.isna())
                r_coef, p_val = pearsonr(x_vals[valid_mask], y_vals[valid_mask])

                # Determine correlation strength category
                abs_r = abs(r_coef)
                if abs_r >= 0.8:
                    strength = "Sangat Kuat 🚀"
                elif abs_r >= 0.6:
                    strength = "Kuat 💪"
                elif abs_r >= 0.4:
                    strength = "Sedang ⚖️"
                elif abs_r >= 0.2:
                    strength = "Lemah 📉"
                else:
                    strength = "Sangat Lemah / Tidak Ada Korelasi 🔴"

                direction = "Positif (+)" if r_coef > 0 else "Negatif (-)"

                # Display Correlation Card
                st.markdown(f"""
                <div class="corr-card">
                    <div class="corr-title">Hasil Analisis Korelasi Pearson ({x_var} vs {y_var})</div>
                    <div class="corr-value">r = {r_coef:.4f}</div>
                    <div class="corr-desc">
                        Hubungan <strong>{direction}</strong> dengan tingkat korelasi <strong>{strength}</strong> (p-value = {p_val:.4e}).
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Generate Plotly Scatter Chart
                fig = px.scatter(
                    filtered_players,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    hover_name='Nama Pemain',
                    hover_data=['Klub', 'Posisi', 'Harga (£m)', 'xPoin', 'Total Poin', 'xG', 'xA', 'Form'],
                    trendline="ols",
                    trendline_color_override="#1e293b",
                    title=f"Hubungan {x_var} vs {y_var} (Trendline OLS)"
                )

                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=520,
                    xaxis=dict(gridcolor="#e2e8f0", title=x_var),
                    yaxis=dict(gridcolor="#e2e8f0", title=y_var),
                    legend=dict(bordercolor="#e2e8f0", borderwidth=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 *Garis hitam putus-putus menunjukkan garis tren regresi linear (OLS). Arahkan kursor ke titik untuk detail pemain.*")
            else:
                st.warning("Data pemain terlalu sedikit untuk menghitung korelasi dan membuat scatter plot.")

        # SECTION 2: CLUB AGGREGATE BAR CHART
        with chart_subtab2:
            st.markdown("##### 📊 Bar Chart Komparasi Agregat Antar Klub")

            b_ctrl1, b_ctrl2 = st.columns(2)
            with b_ctrl1:
                bar_metric = st.selectbox(
                    "Pilih Metrik Agregat",
                    options=['Total Poin', 'xPoin', 'Gol', 'Asis', 'Clean Sheet', 'Form', 'xG', 'xA', 'xGI', 'ICT Index', 'BPS', 'Harga (£m)'],
                    index=0,
                    key="bar_metric_px"
                )
            with b_ctrl2:
                agg_type = st.radio(
                    "Jenis Agregasi",
                    options=['Total (Sum)', 'Rata-rata Per Pemain (Mean)'],
                    horizontal=True,
                    key="bar_agg_px"
                )

            if not players_df.empty:
                if agg_type == 'Rata-rata Per Pemain (Mean)':
                    club_agg = players_df.groupby('Klub')[bar_metric].mean().reset_index()
                else:
                    club_agg = players_df.groupby('Klub')[bar_metric].sum().reset_index()

                club_agg[bar_metric] = club_agg[bar_metric].round(2)
                club_agg = club_agg.sort_values(by=bar_metric, ascending=False)

                bar_fig = px.bar(
                    club_agg,
                    x='Klub',
                    y=bar_metric,
                    color=bar_metric,
                    color_continuous_scale='Viridis',
                    title=f"Perbandingan {agg_type} {bar_metric} Antar Klub"
                )

                bar_fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Plus Jakarta Sans", color="#1e293b"),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=480,
                    xaxis=dict(gridcolor="#e2e8f0", title="Klub Premier League"),
                    yaxis=dict(gridcolor="#e2e8f0", title=f"{agg_type} {bar_metric}")
                )

                st.plotly_chart(bar_fig, use_container_width=True)

        # SECTION 3: PLAYER COMPARISON RADAR CHART
        with chart_subtab3:
            render_player_comparison_radar_tab(players_df, fpl_data, teams_dict)

    # -------------------------------------------------------------------------
    # TAB 3: TEAM STRENGTH ANALYSIS MODULE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("🛡️ Analisis Komprehensif Kekuatan Tim Premier League (Team Strength Analysis)")
        st.write("Modul agregasi statistik 20 klub Premier League: mengevaluasi rata-rata poin FPL pemain, daya gedor ofensif (Gol, xG, xA), soliditas pertahanan (Clean Sheet, xGC, Saves), serta tingkat kemudahan jadwal pertandingan mendatang (FDR).")

        df_teams = calculate_team_strength_analysis(fpl_data, players_df, fdr_summary)

        if not df_teams.empty:
            # 1. Highlights / Summary KPI Row
            top_strength_team = df_teams.sort_values(by="Indeks Kekuatan", ascending=False).iloc[0]
            top_attack_team = df_teams.sort_values(by="Total xG", ascending=False).iloc[0]
            top_defense_team = df_teams.sort_values(by="Clean Sheet", ascending=False).iloc[0]
            easiest_fdr_team = df_teams.sort_values(by="FDR3", ascending=True).iloc[0]

            tk1, tk2, tk3, tk4 = st.columns(4)
            with tk1:
                st.metric(
                    "👑 Tim Terkuat (Indeks Tertinggi)",
                    f"{top_strength_team['Klub']} ({top_strength_team['Indeks Kekuatan']})",
                    top_strength_team['Kategori Tim']
                )
            with tk2:
                st.metric(
                    "⚔️ Serangan Tertajam",
                    f"{top_attack_team['Klub']} ({top_attack_team['Total xG']:.2f} xG)",
                    f"{top_attack_team['Total Gol']} Gol dicetak"
                )
            with tk3:
                st.metric(
                    "🛡️ Pertahanan Terkokoh",
                    f"{top_defense_team['Klub']} ({top_defense_team['Clean Sheet']} CS)",
                    f"{top_defense_team['Total Saves']} Saves"
                )
            with tk4:
                st.metric(
                    "🗓️ Jadwal Termudah (FDR3)",
                    f"{easiest_fdr_team['Klub']} (FDR: {easiest_fdr_team['FDR3']:.2f})",
                    f"Lawan: {easiest_fdr_team['Lawan Berikutnya']}"
                )

            # 2. Controls: Filter & Sort
            st.markdown("##### 🔍 Filter & Urutkan Data Tim")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                team_search = st.text_input("Cari Nama Klub", "", placeholder="Misal: Arsenal, Liverpool...", key="team_search_input")
            with t_col2:
                tier_filter = st.selectbox(
                    "Filter Kategori Kekuatan:",
                    options=["Semua Kategori", "🏆 Elite Contender", "🌟 Top Tier Challenger", "⚖️ Mid-Table Stable", "⚠️ Underdogs / Rebuilding"],
                    key="team_tier_filter"
                )
            with t_col3:
                sort_col = st.selectbox(
                    "Urutkan Berdasarkan:",
                    options=[
                        "Indeks Kekuatan", "Rata-rata Poin Pemain", "Total Poin Skuad",
                        "Total Gol", "Total xG", "Clean Sheet", "Total xGC", "FDR3", "Kemudahan Jadwal (%)"
                    ],
                    index=0,
                    key="team_sort_col"
                )

            filtered_teams = df_teams.copy()
            if team_search:
                filtered_teams = filtered_teams[filtered_teams['Klub'].str.contains(team_search, case=False, na=False)]
            if tier_filter != "Semua Kategori":
                filtered_teams = filtered_teams[filtered_teams['Kategori Tim'] == tier_filter]

            is_asc = (sort_col in ["FDR3", "Total xGC"])
            filtered_teams = filtered_teams.sort_values(by=sort_col, ascending=is_asc)

            # 3. Comprehensive Sortable Table
            st.markdown("##### 📋 Tabel Agregasi & Pemeringkatan Kekuatan Tim")
            team_display_cols = [
                'Klub', 'Indeks Kekuatan', 'Kategori Tim', 'Rata-rata Poin Pemain', 'Pemain Aktif', 'Total Poin Skuad',
                'Total Gol', 'Total xG', 'Clean Sheet', 'Total xGC', 'Total Saves',
                'Top Scorer', 'Top Creator', 'Top Aset FPL', 'Lawan Berikutnya', 'FDR1', 'FDR3', 'FDR5'
            ]

            st.dataframe(
                filtered_teams[team_display_cols],
                use_container_width=True,
                height=520,
                column_config={
                    "Indeks Kekuatan": st.column_config.ProgressColumn(
                        "Indeks Kekuatan",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    ),
                    "Rata-rata Poin Pemain": st.column_config.NumberColumn(
                        "Rata-rata Poin (Menit > 0)",
                        help="Rata-rata akumulasi poin FPL yang dihitung HANYA untuk pemain yang sudah bermain (Menit Bermain > 0).",
                        format="%.2f pts"
                    ),
                    "Pemain Aktif": st.column_config.NumberColumn(
                        "Pemain Aktif",
                        help="Jumlah pemain skuad yang telah mencatatkan menit bermain di EPL musim ini.",
                        format="%d"
                    ),
                    "Total Poin Skuad": st.column_config.NumberColumn(format="%d pts"),
                    "Total Gol": st.column_config.NumberColumn(format="%d"),
                    "Total xG": st.column_config.NumberColumn(format="%.2f"),
                    "Clean Sheet": st.column_config.NumberColumn(format="%d"),
                    "Total xGC": st.column_config.NumberColumn(format="%.2f"),
                    "Total Saves": st.column_config.NumberColumn(format="%d"),
                    "FDR1": st.column_config.NumberColumn(format="%.1f"),
                    "FDR3": st.column_config.NumberColumn(format="%.2f"),
                    "FDR5": st.column_config.NumberColumn(format="%.2f"),
                }
            )
            st.caption("💡 *Catatan: Rata-rata Poin Pemain dihitung khusus untuk pemain yang sudah bermain di EPL (Menit Bermain > 0) agar tidak bias oleh pemain cadangan atau akademi tanpa menit bermain. Indeks Kekuatan menggabungkan 20% Metrik Serangan, 20% Soliditas Pertahanan, 15% Efisiensi Poin Pemain Aktif, 30% Kekuatan Resmi Premier League, dan 15% Kemudahan Jadwal FDR.*")

            # 4. Visual Subtabs: Scatter Matrix, Bar Chart Comparison, Team Deep Dive
            st.divider()
            t_subtab1, t_subtab2, t_subtab3 = st.tabs([
                "📊 Matriks Serangan vs Pertahanan",
                "📈 Komparasi Bar Chart Kekuatan Tim & FDR",
                "🔍 Deep-Dive Analisis Klub & Top Aset FPL"
            ])

            # SUBTAB 1: ATTACK VS DEFENSE SCATTER PLOT
            with t_subtab1:
                st.markdown("##### 📊 Matriks Kuadran: Daya Gedor Serangan (xG) vs Soliditas Pertahanan (Clean Sheets)")
                st.write("Memetakan 20 klub Premier League ke dalam 4 kuadran performa untuk mengidentifikasi tim elit, tim menyerang rentan kebobolan, dan tim defensif.")

                avg_xg = df_teams['Total xG'].mean()
                avg_cs = df_teams['Clean Sheet'].mean()

                fig_quad = px.scatter(
                    df_teams,
                    x='Total xG',
                    y='Clean Sheet',
                    size='Rata-rata Poin Pemain',
                    color='Indeks Kekuatan',
                    text='Kode',
                    hover_name='Klub',
                    hover_data={
                        'Total Gol': True,
                        'Total xG': ':.2f',
                        'Clean Sheet': True,
                        'Total Poin Skuad': True,
                        'Rata-rata Poin Pemain': ':.2f',
                        'FDR3': ':.2f',
                        'Lawan Berikutnya': True,
                        'Kode': False
                    },
                    color_continuous_scale='Plasma',
                    title="Pemetaan Kuadran Tim: Serangan vs Pertahanan"
                )

                fig_quad.update_traces(
                    textposition='top center',
                    textfont=dict(size=11, family="Plus Jakarta Sans", color="#1e293b")
                )

                fig_quad.add_vline(x=avg_xg, line_width=1.5, line_dash="dash", line_color="#94a3b8")
                fig_quad.add_hline(y=avg_cs, line_width=1.5, line_dash="dash", line_color="#94a3b8")

                fig_quad.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                    margin=dict(l=30, r=30, t=50, b=30),
                    height=520,
                    xaxis=dict(gridcolor="#e2e8f0", title="Daya Gedor Serangan (Total xG Tim)"),
                    yaxis=dict(gridcolor="#e2e8f0", title="Soliditas Pertahanan (Total Clean Sheet Tim)")
                )

                st.plotly_chart(fig_quad, use_container_width=True)
                st.caption(f"💡 *Garis putus-putus abu-abu adalah rata-rata liga (xG rata-rata: {avg_xg:.2f}, Clean Sheet rata-rata: {avg_cs:.1f}). Kuadran kanan-atas mencerminkan tim paling seimbang dan dominan.*")

            # SUBTAB 2: BAR CHART COMPARISON
            with t_subtab2:
                st.markdown("##### 📈 Peringkat & Komparasi Antar Klub")
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    chart_metric_choice = st.selectbox(
                        "Pilih Metrik untuk Perbandingan Bar Chart:",
                        options=[
                            "Indeks Kekuatan", "Rata-rata Poin Pemain", "Total Poin Skuad",
                            "Total Gol", "Total xG", "Clean Sheet", "Nilai Skuad (£m)"
                        ],
                        index=0,
                        key="team_bar_metric_sel"
                    )
                with c_col2:
                    color_scale_choice = st.selectbox(
                        "Pewarnaan Bar:",
                        options=["Indeks Kekuatan", "FDR3 (Jadwal Mendatang)", "Total Poin Skuad"],
                        index=0,
                        key="team_bar_color_sel"
                    )

                df_sorted_bar = df_teams.sort_values(by=chart_metric_choice, ascending=False)

                fig_team_bar = px.bar(
                    df_sorted_bar,
                    x='Klub',
                    y=chart_metric_choice,
                    color=color_scale_choice,
                    color_continuous_scale='Turbo' if color_scale_choice == "FDR3 (Jadwal Mendatang)" else 'Viridis',
                    title=f"Peringkat 20 Klub Premier League berdasarkan {chart_metric_choice}",
                    hover_data=['Kode', 'Top Scorer', 'Top Aset FPL', 'FDR3', 'Lawan Berikutnya']
                )

                fig_team_bar.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Plus Jakarta Sans", color="#1e293b"),
                    margin=dict(l=20, r=20, t=50, b=30),
                    height=480,
                    xaxis=dict(gridcolor="#e2e8f0", title="Klub Premier League"),
                    yaxis=dict(gridcolor="#e2e8f0", title=chart_metric_choice)
                )

                st.plotly_chart(fig_team_bar, use_container_width=True)

            # SUBTAB 3: TEAM DEEP-DIVE & ASSETS
            with t_subtab3:
                st.markdown("##### 🔍 Analisis Mendalam Klub & Rekomendasi Aset FPL")
                
                selected_team_name = st.selectbox(
                    "Pilih Klub Premier League untuk Deep-Dive:",
                    options=sorted(df_teams['Klub'].tolist()),
                    key="deep_dive_team_sel"
                )

                team_matches = df_teams[df_teams['Klub'] == selected_team_name]
                team_row = team_matches.iloc[0] if not team_matches.empty else df_teams.iloc[0]
                t_id = team_row['team_id']

                dd1, dd2, dd3, dd4 = st.columns(4)
                with dd1:
                    st.metric("Indeks Kekuatan", f"{team_row['Indeks Kekuatan']}", team_row['Kategori Tim'])
                with dd2:
                    st.metric("Skor Serangan", f"{team_row['Skor Serangan']} / 100", f"{team_row['Total Gol']} Gol ({team_row['Total xG']:.2f} xG)")
                with dd3:
                    st.metric("Skor Pertahanan", f"{team_row['Skor Pertahanan']} / 100", f"{team_row['Clean Sheet']} CS · {team_row['Total Saves']} Saves")
                with dd4:
                    st.metric("Jadwal FDR3", f"{team_row['FDR3']:.2f}", f"Lawan: {team_row['Lawan Berikutnya']}")

                st.markdown(f"###### 🌟 Top 5 Aset FPL Utama di {selected_team_name}")
                club_players = players_df[players_df['Klub'] == selected_team_name]
                if not club_players.empty:
                    top_club_assets = club_players.sort_values(by=['Total Poin', 'xPoin'], ascending=False).head(5)
                    asset_cols = ['Nama Pemain', 'Posisi', 'Harga (£m)', 'Total Poin', 'xPoin', 'xG', 'xA', 'Form', 'Avg Mins (L5M)', 'Status']
                    st.dataframe(
                        top_club_assets[asset_cols],
                        use_container_width=True,
                        column_config={
                            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                            "Total Poin": st.column_config.NumberColumn(format="%d pts"),
                            "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
                            "xG": st.column_config.NumberColumn(format="%.2f"),
                            "xA": st.column_config.NumberColumn(format="%.2f"),
                            "Form": st.column_config.NumberColumn(format="%.2f"),
                            "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins")
                        }
                    )
                else:
                    st.info("Data pemain untuk klub ini belum tersedia.")

                st.markdown(f"###### 🗓️ Jadwal 5 Laga Mendatang ({selected_team_name})")
                upcoming_fxs = []
                for f in fixtures_data:
                    if not f.get('finished'):
                        h_id, a_id = f.get('team_h'), f.get('team_a')
                        if h_id == t_id or a_id == t_id:
                            is_home = (h_id == t_id)
                            opp_id = a_id if is_home else h_id
                            diff = f.get('team_h_difficulty', 3) if is_home else f.get('team_a_difficulty', 3)
                            opp_name = teams_dict.get(opp_id, f"Team {opp_id}")
                            diff_badge = {
                                1: "🟢 Sangat Mudah (1)",
                                2: "🟢 Mudah (2)",
                                3: "⚪ Sedang (3)",
                                4: "🔴 Sulit (4)",
                                5: "🔴 Sangat Sulit (5)"
                            }.get(diff, f"Rating {diff}")
                            upcoming_fxs.append({
                                'Gameweek': f"GW{f.get('event')}",
                                'Lawan': opp_name,
                                'Venue': "🏠 Home" if is_home else "✈️ Away",
                                'Tingkat Kesulitan (FDR)': diff,
                                'Status FDR': diff_badge,
                                'Kickoff': f.get('kickoff_time', '-')[:10] if f.get('kickoff_time') else '-'
                            })
                            if len(upcoming_fxs) >= 5:
                                break

                if upcoming_fxs:
                    df_up_fxs = pd.DataFrame(upcoming_fxs)
                    st.dataframe(
                        df_up_fxs,
                        use_container_width=True,
                        column_config={
                            "Tingkat Kesulitan (FDR)": st.column_config.NumberColumn(format="%d")
                        }
                    )
        else:
            st.warning("Data analisis kekuatan tim tidak dapat dihitung.")

    # -------------------------------------------------------------------------
    # TAB 4: FIXTURES & FDR SUMMARY
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("📅 Jadwal Pertandingan & Rating Kesulitan (FDR)")
        st.write("Evaluasi tingkat kesulitan lawan untuk setiap tim pada pertandingan mendatang.")

        # Create FDR Summary Table per Club
        fdr_table_data = []
        for t_id, t_name in sorted(teams_dict.items(), key=lambda x: x[1]):
            f_info = fdr_summary.get(t_id, {})
            fdr_table_data.append({
                'Klub': t_name,
                'FDR1 (LagaBerikutnya)': f_info.get('FDR1', 3.0),
                'FDR3 (Rata-rata 3 Laga)': f_info.get('FDR3', 3.0),
                'FDR5 (Rata-rata 5 Laga)': f_info.get('FDR5', 3.0),
                'Status Laga Berikutnya': '🏠 Home' if f_info.get('Next_Is_Home') == 1 else '✈️ Away'
            })

        fdr_df = pd.DataFrame(fdr_table_data).sort_values(by='FDR3 (Rata-rata 3 Laga)')

        st.dataframe(
            fdr_df,
            use_container_width=True,
            column_config={
                "FDR1 (LagaBerikutnya)": st.column_config.NumberColumn(format="%.1f"),
                "FDR3 (Rata-rata 3 Laga)": st.column_config.NumberColumn(format="%.2f"),
                "FDR5 (Rata-rata 5 Laga)": st.column_config.NumberColumn(format="%.2f")
            }
        )
        st.caption("💡 *Catatan FDR: Skala 1 (Sangat Mudah) hingga 5 (Sangat Sulit). Nilai FDR3 dan FDR5 yang lebih rendah menandakan jadwal pertandingan mendatang yang lebih menguntungkan.*")

    # -------------------------------------------------------------------------
    # TAB 5: OPTION B: COMPONENT MODEL XPOIN
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("🧮 Option B: Bottom-Up Component Model xPoin")
        st.write("Model perhitungan xPoin berdasarkan breakdown komponen individual: estimasi menit, xG/xA match regresi, kontribusi defensif, clean sheet, saves, dan bonus points.")

        # Summary Metrics Row for Option B
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Pemain Terfilter", len(filtered_players))
        with b_col2:
            top_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Top xPoin (Option B)", f"{top_optb['Nama Pemain']} ({top_optb['xPoin (Option B)']:.2f} pts)" if top_optb is not None else "-")
        with b_col3:
            top_xg_match = filtered_players.sort_values(by="xG Pred (Match)", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Top xG Pred (Match)", f"{top_xg_match['Nama Pemain']} ({top_xg_match['xG Pred (Match)']:.2f})" if top_xg_match is not None else "-")
        with b_col4:
            top_xa_match = filtered_players.sort_values(by="xA Pred (Match)", ascending=False).iloc[0] if not filtered_players.empty else None
            st.metric("Top xA Pred (Match)", f"{top_xa_match['Nama Pemain']} ({top_xa_match['xA Pred (Match)']:.2f})" if top_xa_match is not None else "-")

        with st.expander("🤖 Detail Model Regresi xG & xA (Match-Level Prediction)", expanded=False):
            m_tab1, m_tab2 = st.tabs(["⚽ Model Prediksi xG", "🎯 Model Prediksi xA"])
            with m_tab1:
                st.markdown("#### 📐 Regresi Linier Ekspektasi Gol (xG)")
                pos_sel_xg = st.selectbox("Pilih Posisi (xG):", ["FWD", "MID", "DEF"], key="optb_pos_xg_sel")
                p_stats_xg = stats_xg.get(pos_sel_xg, stats_xg)
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xg.get('r2', 0.0):.4f}")
                with mc2:
                    st.metric("Mean Absolute Error (MAE)", f"{p_stats_xg.get('mae', 0.0):.4f}")
                with mc3:
                    st.metric("Intercept (Konstanta β₀)", f"{p_stats_xg.get('intercept', 0.0):.4f}")
                st.dataframe(p_stats_xg.get('coef_df', pd.DataFrame()), use_container_width=True)
                st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
                st.dataframe(p_stats_xg.get('eval_df', pd.DataFrame()), use_container_width=True)
            with m_tab2:
                st.markdown("#### 📐 Regresi Linier Ekspektasi Asis (xA)")
                pos_sel_xa = st.selectbox("Pilih Posisi (xA):", ["FWD", "MID", "DEF"], key="optb_pos_xa_sel")
                p_stats_xa = stats_xa.get(pos_sel_xa, stats_xa)
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xa.get('r2', 0.0):.4f}")
                with mc2:
                    st.metric("Mean Absolute Error (MAE)", f"{p_stats_xa.get('mae', 0.0):.4f}")
                with mc3:
                    st.metric("Intercept (Konstanta β₀)", f"{p_stats_xa.get('intercept', 0.0):.4f}")
                st.dataframe(p_stats_xa.get('coef_df', pd.DataFrame()), use_container_width=True)
                st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
                st.dataframe(p_stats_xa.get('eval_df', pd.DataFrame()), use_container_width=True)

        opt_b_cols = [
            'Nama Pemain', 'Klub', 'Lawan GW Berikutnya', 'Posisi', 'Harga (£m)',
            'xPoin (Option B)', 'xG Pred (Match)', 'xA Pred (Match)',
            'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
            'FDR1', 'FDR3', 'FDR5'
        ]

        sorted_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False)

        st.dataframe(
            sorted_optb[opt_b_cols],
            use_container_width=True,
            height=560,
            column_config={
                "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                "xPoin (Option B)": st.column_config.NumberColumn(format="%.2f pts"),
                "xG Pred (Match)": st.column_config.NumberColumn(format="%.2f"),
                "xA Pred (Match)": st.column_config.NumberColumn(format="%.2f"),
                "xMins Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xG Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xA Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xSaves Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xDC Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xCS Pts": st.column_config.NumberColumn(format="%.2f pts"),
                "xBP": st.column_config.NumberColumn(format="%.2f pts"),
                "FDR1": st.column_config.NumberColumn(format="%.1f"),
                "FDR3": st.column_config.NumberColumn(format="%.2f"),
                "FDR5": st.column_config.NumberColumn(format="%.2f")
            }
        )
        st.caption("💡 *Bottom-Up Component Model (Option B) menghitung xPoin dari akumulasi individual komponen: xMins, xG, xA, xSaves, xDC (Poisson CDF), xCS, dan xBP.*")


    
    # -------------------------------------------------------------------------
    # TAB 6: OPTION C: CURRENT SEASON MODEL
    # -------------------------------------------------------------------------
    with tab6:
        st.subheader("🔮 Option C: Current Season (On-the-Fly Model)")
        st.write("Model regresi prediktif yang 100% dilatih menggunakan data API musim berjalan (tabel history). Model otomatis diremodel/update setiap 5 Gameweek (GW5, GW10, dst).")
        
        with st.spinner("Mengekstrak dan melatih model Option C dari histori API musim ini..."):
            try:
                df_train_c, df_view_c, models_c = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw)
            except Exception as e:
                st.error(f"Error Opt C: {e}")
                df_train_c, df_view_c, models_c = None, None, None
            
        if df_view_c is not None and not df_view_c.empty:
            st.success("✅ Model Option C berhasil diremodel menggunakan data histori terbaru.")
            
            # Show Models details
            with st.expander("🤖 Detail 4 Model Regresi Option C", expanded=False):
                ct1, ct2, ct3, ct4 = st.tabs(["⚽ FWD", "🎯 MID", "🛡️ DEF", "🧤 GK"])
                tabs_m = [(ct1, 'FWD'), (ct2, 'MID'), (ct3, 'DEF'), (ct4, 'GK')]
                
                for tab_obj, pos_key in tabs_m:
                    with tab_obj:
                        if pos_key in models_c:
                            m = models_c[pos_key]
                            st.write(f"**Fitur (Variabel Bebas) Posisi {pos_key}:**")
                            st.write(", ".join(m['features']))
                            coefs = m['model'].coef_
                            coef_df = pd.DataFrame({'Fitur': m['features'], 'Bobot (β)': coefs})
                            st.dataframe(coef_df, use_container_width=True)
                        else:
                            st.warning(f"Data latih untuk {pos_key} tidak cukup.")
                            
            # Show summary table
            st.markdown("##### 📋 Tabel Rangkuman Prediksi xPoin GW Selanjutnya (Option C)")
            
            # Filter search
            if search_query:
                df_view_c = df_view_c[df_view_c['Nama Pemain'].str.contains(search_query, case=False, na=False)]
                
            st.dataframe(
                df_view_c,
                use_container_width=True,
                height=560,
                column_config={
                    "xPoin (Option C)": st.column_config.NumberColumn(format="%.2f pts"),
                    "Total xG": st.column_config.NumberColumn(format="%.2f"),
                    "Total xA": st.column_config.NumberColumn(format="%.2f"),
                    "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins"),
                    "FDR1": st.column_config.NumberColumn(format="%.1f"),
                    "FDR3": st.column_config.NumberColumn(format="%.2f"),
                    "FDR5": st.column_config.NumberColumn(format="%.2f")
                }
            )
            
            st.caption("💡 *Tabel rangkuman prediksi ini digunakan untuk mengambil keputusan pemain yang akan dipakai pada Gameweek selanjutnya. Menggunakan data murni musim berjalan.*")
        else:
            st.warning("Belum ada data history musim berjalan yang cukup untuk melatih model Option C.")


if __name__ == "__main__":
    main()
