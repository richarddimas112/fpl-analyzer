import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & MAPS
# -----------------------------------------------------------------------------
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
    except Exception as e:
        st.error(f"Gagal mengambil data FPL: {e}")
        return None

@st.cache_data(ttl=86400)
def fetch_fixtures_data():
    """Fetch fixtures schedule from FPL API."""
    try:
        response = requests.get(FIXTURES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Gagal mengambil data jadwal: {e}")
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

# -----------------------------------------------------------------------------
# FIXTURE DIFFICULTY RATING (FDR1, FDR3, FDR5) CALCULATIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def calculate_team_fdrs(fixtures, teams_dict):
    """Calculate FDR1, FDR3, FDR5 and next match home status for every team."""
    team_upcoming = {t_id: [] for t_id in teams_dict.keys()}
    
    for f in fixtures:
        if not f.get('finished'):
            h_id = f.get('team_h')
            a_id = f.get('team_a')
            h_diff = f.get('team_h_difficulty', 3)
            a_diff = f.get('team_a_difficulty', 3)
            
            if h_id in team_upcoming:
                team_upcoming[h_id].append({'fdr': h_diff, 'is_home': 1, 'gw': f.get('event')})
            if a_id in team_upcoming:
                team_upcoming[a_id].append({'fdr': a_diff, 'is_home': 0, 'gw': f.get('event')})
                
    fdr_summary = {}
    for t_id, fxs in team_upcoming.items():
        if fxs:
            f1 = float(fxs[0]['fdr'])
            next_is_home = fxs[0]['is_home']
            f3 = float(np.mean([x['fdr'] for x in fxs[:3]])) if len(fxs) >= 3 else f1
            f5 = float(np.mean([x['fdr'] for x in fxs[:5]])) if len(fxs) >= 5 else f3
        else:
            f1 = 3.0
            f3 = 3.0
            f5 = 3.0
            next_is_home = 1
            
        fdr_summary[t_id] = {
            'FDR1': round(f1, 2),
            'FDR3': round(f3, 2),
            'FDR5': round(f5, 2),
            'Next_Is_Home': next_is_home
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

@st.cache_data(ttl=86400)
def train_xpoints_model(players_list, fdr_summary):
    """
    Train 4 separate LinearRegression models based on player position (FWD, MID, DEF, GK)
    using specific underlying metrics for each position.
    Fallback to realistic synthetic match dynamics if match history is empty (e.g. pre-season).
    """
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
def process_players(bootstrap_raw, fdr_summary, _models_dict):
    """Clean & format player dataset with extensive FPL metrics and positional xPoin predictions."""
    teams_list = bootstrap_raw.get('teams', [])
    team_dict = {t['id']: t['name'] for t in teams_list}
    
    elements = bootstrap_raw.get('elements', [])
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

    # 7. FDR Attachment
    df['FDR1'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR1', 3.0))
    df['FDR3'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR3', 3.0))
    df['FDR5'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('FDR5', 3.0))
    df['Next_Is_Home'] = df['team'].apply(lambda tid: fdr_summary.get(tid, {}).get('Next_Is_Home', 1))

    # 8. Predict xPoin using 4 Positional LinearRegression Models
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

    # Assign xPoin strictly >= 0.0 for players with >= 300 minutes played
    df['xPoin'] = np.where(df['Menit Bermain'] >= 300, df['xPoin_raw'], 0.0).round(2)

    cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin', 'Avg Mins (L5M)', 'Total Poin',
        'FDR1', 'FDR3', 'FDR5', 'Form', '% Ownership', 'Net Transfers GW',
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
    
    # Train 4 Positional Regression Models for xPoin
    models_dict = train_xpoints_model(
        fpl_data.get('elements', []), fdr_summary
    )

    # Process Player Dataset
    players_df, team_dict = process_players(fpl_data, fdr_summary, models_dict)

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
    tab1, tab2, tab3 = st.tabs(["📊 Player Stats & xPoin", "📈 Visualisasi Data & Korelasi", "📅 Fixtures & FDR"])

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
            selected_cols = st.multiselect(
                "Centang/Pilih kolom data FPL yang ingin dimunculkan di tabel:",
                options=list(players_df.columns),
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

    # -------------------------------------------------------------------------
    # TAB 2: VISUALIZATION & PEARSON CORRELATION (PLOTLY)
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("📈 Visualisasi Interaktif & Analisis Korelasi Pearson")
        st.write("Analisis hubungan antar variabel statistik pemain dengan Scatter Plot interaktif Plotly dan tren regresi linear.")

        chart_subtab1, chart_subtab2 = st.tabs(["🔵 Scatter Plot & Pearson Correlation", "📊 Bar Chart Agregat Klub"])

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

    # -------------------------------------------------------------------------
    # TAB 3: FIXTURES & FDR SUMMARY
    # -------------------------------------------------------------------------
    with tab3:
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

if __name__ == "__main__":
    main()
