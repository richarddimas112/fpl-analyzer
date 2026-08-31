"""
Hidden Gem & Haul Predictor Module (>10 Points Prediction with XGBoost, Feature Engineering, & Seaborn Isolation Plot).
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.constants import POSITION_MAP
from src.api import fetch_player_history_raw

@st.cache_data(ttl=86400)
def build_hidden_gem_and_haul_dataset(_fpl_data, fdr_summary, current_gw):
    """
    Ekstrak data match-by-match individual seluruh pemain aktif di musim berjalan:
    1. FEATURE ENGINEERING & DETEKSI UNDERPERFORMANCE:
       - actual_GI = goals_scored + assists
       - Rolling 5 laga terakhir: roll_xGI dan roll_actual_GI
       - underperformance_index = roll_xGI - roll_actual_GI
    2. VOLATILITAS & CEILING POIN:
       - std_points dan max_points
    3. TARGET LABEL:
       - is_haul = 1 jika total_points >= 10, else 0
    4. UPCOMING MATCH FEATURES (Untuk Prediksi GW Mendatang)
    """
    elements = _fpl_data.get('elements', []) if _fpl_data else []
    teams = _fpl_data.get('teams', []) if _fpl_data else []
    teams_dict = {t['id']: t['name'] for t in teams}

    # Filter pemain aktif (menit > 0)
    active_elements = [el for el in elements if int(el.get('minutes', 0)) > 0]
    
    # Hitung rata-rata defensif tim (xGC/90) untuk opp_roll5_xGC_per90
    team_xgc_map = {}
    for t in teams:
        t_id = t['id']
        t_players = [p for p in elements if p.get('team') == t_id]
        def_players = [p for p in t_players if p.get('element_type') in [1, 2]]
        tot_xgc = sum(float(p.get('expected_goals_conceded', 0.0) or 0.0) for p in def_players)
        tot_def_mins = sum(int(p.get('minutes', 0) or 0) for p in def_players)
        xgc_90 = (tot_xgc / tot_def_mins * 90.0) if tot_def_mins > 0 else 1.35
        team_xgc_map[t_id] = round(xgc_90, 2)

    def fetch_player_matches(el):
        p_id = el['id']
        hist = fetch_player_history_raw(p_id)
        return p_id, el, hist

    all_player_histories = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(fetch_player_matches, active_elements)
        for p_id, el, hist in results:
            if hist:
                all_player_histories[p_id] = (el, hist)

    train_rows = []
    upcoming_rows = []

    pos_numeric_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4, 1: 1, 2: 2, 3: 3, 4: 4}

    for p_id, (el, hist) in all_player_histories.items():
        sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
        if not sorted_hist:
            continue

        p_name = el.get('web_name', '')
        p_type = el.get('element_type', 3)
        pos_str = POSITION_MAP.get(p_type, 'MID')
        pos_num = pos_numeric_map.get(pos_str, 3)
        p_cost = float(el.get('now_cost', 50)) / 10.0
        p_team = el.get('team', 1)
        p_chance = float(el.get('chance_of_playing_next_round') if el.get('chance_of_playing_next_round') is not None else 100)
        p_ownership = float(el.get('selected_by_percent', 0.0))

        # Array metrik akumulatif match-by-match
        match_points = [int(m.get('total_points', 0)) for m in sorted_hist]
        match_mins = [int(m.get('minutes', 0)) for m in sorted_hist]
        
        # Volatilitas & Ceiling Poin
        std_points = float(np.std(match_points)) if len(match_points) > 1 else 0.0
        max_points = int(np.max(match_points)) if match_points else 0

        # Iterasi setiap match untuk membangun training set (Lagged / Rolling features)
        for i, m in enumerate(sorted_hist):
            pts_actual = int(m.get('total_points', 0))
            is_haul = 1 if pts_actual >= 10 else 0

            # Gunakan histori laga sebelumnya (sebelum match i) untuk mencegah data leakage
            past_matches = sorted_hist[:i]
            if not past_matches:
                # Cold start match 1
                past_matches = [m]

            last_5 = past_matches[-5:]
            
            # 1. Feature Engineering: Actual GI vs xGI
            # actual_GI = goals_scored + assists
            roll_goals = sum(int(x.get('goals_scored', 0)) for x in last_5)
            roll_assists = sum(int(x.get('assists', 0)) for x in last_5)
            roll_actual_GI = float(roll_goals + roll_assists)

            roll_xg = sum(float(x.get('expected_goals', 0.0) or 0.0) for x in last_5)
            roll_xa = sum(float(x.get('expected_assists', 0.0) or 0.0) for x in last_5)
            roll_xgi = float(roll_xg + roll_xa)

            # underperformance_index = roll_xGI - roll_actual_GI
            underperformance_index = float(roll_xgi - roll_actual_GI)

            # Rolling Minutes & Form
            mins_val = float(np.mean([int(x.get('minutes', 0)) for x in last_5]))
            player_form = float(np.mean([int(x.get('total_points', 0)) for x in last_5]))

            # Opponent features
            opp_id = m.get('opponent_team', 1)
            opp_fdr_info = fdr_summary.get(opp_id, {})
            fdr_val = float(opp_fdr_info.get('FDR1', m.get('difficulty', 3.0)))
            opp_roll5_xgc = float(team_xgc_map.get(opp_id, 1.35))

            # Running std & max points up to match i
            past_pts = [int(x.get('total_points', 0)) for x in past_matches]
            run_std = float(np.std(past_pts)) if len(past_pts) > 1 else 0.0
            run_max = int(np.max(past_pts)) if past_pts else pts_actual

            train_rows.append({
                'player_id': p_id,
                'Nama Pemain': p_name,
                'posisi': pos_num,
                'Posisi_Str': pos_str,
                'minutes': mins_val,
                'player_form': player_form,
                'FDR': fdr_val,
                'opp_roll5_xGC_per90': opp_roll5_xgc,
                'underperformance_index': underperformance_index,
                'roll_xGI': roll_xgi,
                'roll_actual_GI': roll_actual_GI,
                'std_points': run_std,
                'max_points': run_max,
                'total_points': pts_actual,
                'is_haul': is_haul
            })

        # Feature Upcoming Gameweek
        f_info = fdr_summary.get(p_team, {})
        next_fdr = float(f_info.get('FDR1', 3.0))
        next_opp_id = f_info.get('Next_Opponent_ID', 1)
        next_opp_xgc = float(team_xgc_map.get(next_opp_id, 1.35)) if next_opp_id else 1.35
        next_opp_fmt = f_info.get('Next_Opponent_Fmt', '-')

        last_5_all = sorted_hist[-5:]
        curr_goals_5 = sum(int(x.get('goals_scored', 0)) for x in last_5_all)
        curr_assists_5 = sum(int(x.get('assists', 0)) for x in last_5_all)
        curr_actual_gi_5 = float(curr_goals_5 + curr_assists_5)

        curr_xg_5 = sum(float(x.get('expected_goals', 0.0) or 0.0) for x in last_5_all)
        curr_xa_5 = sum(float(x.get('expected_assists', 0.0) or 0.0) for x in last_5_all)
        curr_xgi_5 = float(curr_xg_5 + curr_xa_5)

        curr_underperformance = float(curr_xgi_5 - curr_actual_gi_5)
        curr_mins = float(np.mean([int(x.get('minutes', 0)) for x in last_5_all]))
        curr_form = float(np.mean([int(x.get('total_points', 0)) for x in last_5_all]))

        upcoming_rows.append({
            'id': p_id,
            'Nama Pemain': p_name,
            'Klub': teams_dict.get(p_team, '-'),
            'Posisi': pos_str,
            'posisi': pos_num,
            'Harga (£m)': p_cost,
            '% Ownership': p_ownership,
            'Peluang Main GW (%)': p_chance,
            'Lawan GW Berikutnya': next_opp_fmt,
            'minutes': round(curr_mins, 1),
            'player_form': round(curr_form, 2),
            'FDR': round(next_fdr, 1),
            'opp_roll5_xGC_per90': round(next_opp_xgc, 2),
            'roll_xGI': round(curr_xgi_5, 2),
            'roll_actual_GI': round(curr_actual_gi_5, 1),
            'underperformance_index': round(curr_underperformance, 2),
            'std_points': round(std_points, 2),
            'max_points': int(max_points)
        })

    df_train = pd.DataFrame(train_rows)
    df_upcoming = pd.DataFrame(upcoming_rows)

    return df_train, df_upcoming


def train_xgboost_haul_model(df_train, df_upcoming):
    """
    Latih model XGBoost Classifier untuk memprediksi probabilitas Haul (is_haul = 1).
    Fitur input:
    ['posisi', 'minutes', 'player_form', 'FDR', 'opp_roll5_xGC_per90', 'underperformance_index', 'std_points', 'max_points']
    """
    feature_cols = [
        'posisi', 'minutes', 'player_form', 'FDR',
        'opp_roll5_xGC_per90', 'underperformance_index', 'std_points', 'max_points'
    ]

    feature_name_id = {
        'posisi': 'Posisi Pemain (Numeric)',
        'minutes': 'Avg Menit Bermain (L5M)',
        'player_form': 'Player Form (Avg Poin L5M)',
        'FDR': 'Tingkat Kesulitan Lawan (FDR)',
        'opp_roll5_xGC_per90': 'xGC/90 Lawan (Defensive Leakage)',
        'underperformance_index': 'Underperformance Index (xGI - GI)',
        'std_points': 'Volatilitas Poin (Std Dev)',
        'max_points': 'Ceiling Poin Tertinggi (Max Pts)'
    }

    if df_train.empty or len(df_train) < 20:
        return df_upcoming, None, {}

    X_train = df_train[feature_cols]
    y_train = df_train['is_haul']

    # Penanganan Class Imbalance (scale_pos_weight = neg_count / pos_count)
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_weight = float(num_neg) / float(max(1, num_pos))

    if xgb is not None:
        model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        # Fallback jika xgboost belum terinstall (menggunakan GradientBoosting)
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

    model.fit(X_train, y_train)

    # Prediksi train untuk evaluasi metrik
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_proba_train >= 0.5).astype(int)

    auc_score = roc_auc_score(y_train, y_pred_proba_train) if len(np.unique(y_train)) > 1 else 0.5
    prec = precision_score(y_train, y_pred_train, zero_division=0)
    rec = recall_score(y_train, y_pred_train, zero_division=0)
    f1 = f1_score(y_train, y_pred_train, zero_division=0)

    # Prediksi Upcoming Gameweek
    X_upcoming = df_upcoming[feature_cols]
    proba_upcoming = model.predict_proba(X_upcoming)[:, 1]

    # Kalibrasi probabilitas dengan peluang bermain (%)
    chance_factor = df_upcoming['Peluang Main GW (%)'] / 100.0
    calibrated_proba = proba_upcoming * chance_factor * 100.0

    df_res = df_upcoming.copy()
    df_res['Probability of Haul (%)'] = np.round(calibrated_proba, 1)
    df_res['Raw Haul Proba (%)'] = np.round(proba_upcoming * 100.0, 1)

    # Klasifikasi Potensi Ledakan / Hidden Gem
    def classify_gem(row):
        prob = row['Probability of Haul (%)']
        under = row['underperformance_index']
        own = row['% Ownership']
        
        if prob >= 30.0 and own <= 10.0 and under > 0.3:
            return "💎 Ultimate Hidden Gem (High Prob + Differential + Unlucky)"
        elif prob >= 25.0 and own <= 12.0:
            return "✨ Premium Differential (Low Own + High Haul)"
        elif prob >= 35.0:
            return "🔥 Elite Captaincy Candidate (High Haul)"
        elif under >= 0.8:
            return "💣 Bom Waktu (Extreme Unlucky, Due for Goals)"
        elif prob >= 15.0:
            return "📈 Steady Performer"
        else:
            return "⚪ Standard Asset"

    df_res['Kategori Gem'] = df_res.apply(classify_gem, axis=1)
    df_res = df_res.sort_values(by='Probability of Haul (%)', ascending=False)

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    feat_df = pd.DataFrame({
        'Kode Fitur': feature_cols,
        'Nama Fitur': [feature_name_id.get(f, f) for f in feature_cols],
        'Tingkat Kepentingan (%)': np.round(importances * 100.0, 2)
    }).sort_values(by='Tingkat Kepentingan (%)', ascending=False)

    metrics = {
        'auc': round(auc_score, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'total_samples': len(df_train),
        'total_hauls': int(num_pos),
        'scale_pos_weight': round(scale_weight, 2)
    }

    return df_res, feat_df, metrics


def render_seaborn_underperformance_plot(df_upcoming):
    """
    Buat visualisasi 2D Scatter Plot antara 'roll_xGI' vs 'roll_actual_GI' menggunakan Seaborn & Matplotlib
    untuk mengisolasi pemain di kuadran 'Unlucky / Bom Waktu' (High xGI, Low Actual GI).
    """
    if df_upcoming.empty:
        st.warning("Data pemain tidak mencukupi untuk visualisasi.")
        return

    # Filter pemain yang memiliki keterlibatan ofensif yang signifikan
    plot_df = df_upcoming[
        (df_upcoming['minutes'] >= 30.0) & 
        (df_upcoming['roll_xGI'] > 0.1)
    ].copy()

    if plot_df.empty:
        plot_df = df_upcoming.copy()

    # Hitung nilai rata-rata batas kuadran
    med_xgi = float(plot_df['roll_xGI'].median())
    med_gi = float(plot_df['roll_actual_GI'].median())

    # Kuadran Klasifikasi
    def assign_quadrant(r):
        if r['roll_xGI'] >= med_xgi and r['roll_actual_GI'] < med_gi:
            return "Bom Waktu / Unlucky (High xGI, Low GI)"
        elif r['roll_xGI'] >= med_xgi and r['roll_actual_GI'] >= med_gi:
            return "In-Form Star (High xGI, High GI)"
        elif r['roll_xGI'] < med_xgi and r['roll_actual_GI'] >= med_gi:
            return "Lucky / Overperforming (Low xGI, High GI)"
        else:
            return "Low Impact (Low xGI, Low GI)"

    plot_df['Kuadran'] = plot_df.apply(assign_quadrant, axis=1)

    # Konfigurasi Tema Seaborn
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    palette = {
        "Bom Waktu / Unlucky (High xGI, Low GI)": "#dc2626", # Merah
        "In-Form Star (High xGI, High GI)": "#16a34a",       # Hijau
        "Lucky / Overperforming (Low xGI, High GI)": "#f59e0b", # Kuning Amber
        "Low Impact (Low xGI, Low GI)": "#94a3b8"            # Abu-abu
    }

    # Seaborn Scatter Plot
    scatter = sns.scatterplot(
        data=plot_df,
        x='roll_xGI',
        y='roll_actual_GI',
        hue='Kuadran',
        size='Probability of Haul (%)',
        sizes=(40, 320),
        palette=palette,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8,
        ax=ax
    )

    # Garis Pembagi Kuadran
    ax.axvline(x=med_xgi, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(y=med_gi, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)

    # Garis Diagonal Keseimbangan (xGI == Actual GI)
    max_val = max(plot_df['roll_xGI'].max(), plot_df['roll_actual_GI'].max()) + 0.5
    ax.plot([0, max_val], [0, max_val], color='#cbd5e1', linestyle=':', linewidth=1.5, label='Keseimbangan Ideal (xGI = GI)')

    # Anotasi Pemain Paling Unlucky & High Probability
    unlucky_top = plot_df[plot_df['Kuadran'] == "Bom Waktu / Unlucky (High xGI, Low GI)"].sort_values(
        by=['underperformance_index', 'Probability of Haul (%)'], ascending=[False, False]
    ).head(8)

    for _, r in unlucky_top.iterrows():
        ax.annotate(
            f"{r['Nama Pemain']} (£{r['Harga (£m)']:.1f}m)\nΔ: +{r['underperformance_index']:.2f}",
            (r['roll_xGI'], r['roll_actual_GI']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#ef4444", alpha=0.85, lw=0.8)
        )

    # Anotasi Top In-Form
    inform_top = plot_df[plot_df['Kuadran'] == "In-Form Star (High xGI, High GI)"].sort_values(
        by='Probability of Haul (%)', ascending=False
    ).head(4)

    for _, r in inform_top.iterrows():
        ax.annotate(
            f"{r['Nama Pemain']}",
            (r['roll_xGI'], r['roll_actual_GI']),
            xytext=(6, -10),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color='#166534'
        )

    ax.set_title("2D Scatter Matrix: Rolling 5 Match xGI vs Actual GI (Deteksi Bom Waktu & Underperformance)", fontsize=13, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("Rolling 5 Match Expected Goal Involvement (roll_xGI)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Rolling 5 Match Actual Goal Involvement (roll_actual_GI = Goals + Assists)", fontsize=10.5, fontweight='bold', labelpad=8)
    
    # Label Kuadran
    ax.text(plot_df['roll_xGI'].max() * 0.75, 0.1, "[KUADRAN BOM WAKTU]\n(High xGI, Low GI - Siap Meledak)", 
            fontsize=9.5, fontweight='bold', color='#b91c1c', bbox=dict(boxstyle='square', fc='#fef2f2', ec='#f87171', alpha=0.8))

    ax.text(plot_df['roll_xGI'].max() * 0.75, plot_df['roll_actual_GI'].max() * 0.85, "[KUADRAN IN-FORM]\n(High xGI, High GI - Konsisten)", 
            fontsize=9.5, fontweight='bold', color='#15803d', bbox=dict(boxstyle='square', fc='#f0fdf4', ec='#86efac', alpha=0.8))

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=8.5)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
