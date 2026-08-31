"""
Data Processing, FDR calculations, Team Strength Analysis, and Player Transforms.
"""

import numpy as np
import pandas as pd
import streamlit as st
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import poisson

from src.constants import POSITION_MAP, STATUS_MAP
from src.api import fetch_player_history_raw

def get_current_gw(fpl_data):
    """Mendeteksi ID Gameweek yang sedang aktif/berjalan."""
    events = fpl_data.get('events', [])
    for ev in events:
        if ev.get('is_current'):
            return ev.get('id', 1)
    for ev in events:
        if ev.get('is_next'):
            return ev.get('id', 1)
    return 1

@st.cache_data(ttl=86400)
def load_historical_training_data():
    """Membaca dataset historis musim lalu untuk keperluan cold start training."""
    hist_file = "data/historical_train_data.csv"
    if os.path.exists(hist_file):
        try:
            return pd.read_csv(hist_file)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def format_setpiece_order(val):
    """Memformat urutan set-piece menjadi label yang jelas."""
    if pd.isnull(val) or val is None or str(val).strip() == "":
        return "-"
    try:
        v_int = int(float(val))
        return f"Pilihan ke-{v_int}"
    except Exception:
        return str(val)

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

@st.cache_data(ttl=86400)
def compute_all_l5m_avg_mins(elements):
    """
    Fetch match history for all active players concurrently using a thread pool.
    Returns a dict {player_id: avg_mins_l5m}
    """
    active_elements = [el for el in elements if int(el.get('minutes', 0)) > 0]
    
    def get_l5m_for_player(el):
        p_id = el['id']
        hist = fetch_player_history_raw(p_id)
        if hist:
            sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
            last_5 = sorted_hist[-5:]
            avg_m = sum(int(x.get('minutes', 0)) for x in last_5) / float(len(last_5))
            return p_id, round(avg_m, 1)
        else:
            fallback = min(90.0, float(el.get('minutes', 0)) / 5.0)
            return p_id, round(fallback, 1)

    l5m_map = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(get_l5m_for_player, active_elements)
        for p_id, val in results:
            l5m_map[p_id] = val

    # Fill default for non-active players
    for el in elements:
        if el['id'] not in l5m_map:
            l5m_map[el['id']] = 0.0

    return l5m_map

@st.cache_data(ttl=86400)
def process_players(fpl_data, fdr_summary, _models_dict, _opt_b_models=None):
    """Transform raw FPL JSON into a clean, feature-rich Pandas DataFrame with dual xPoin predictions."""
    elements = fpl_data.get('elements', [])
    teams = fpl_data.get('teams', [])
    
    if not elements:
        return pd.DataFrame(), {}
        
    team_dict = {t['id']: t['name'] for t in teams}
    team_xg_def_map = {}
    team_xg_att_map = {}
    
    for t in teams:
        t_id = t['id']
        t_players = [p for p in elements if p.get('team') == t_id]
        def_players = [p for p in t_players if p.get('element_type') in [1, 2]]
        
        tot_xgc = sum(float(p.get('expected_goals_conceded', 0.0) or 0.0) for p in def_players)
        tot_def_mins = sum(int(p.get('minutes', 0) or 0) for p in def_players)
        xgc_per_90 = (tot_xgc / tot_def_mins * 90.0) if tot_def_mins > 0 else 1.35
        team_xg_def_map[t_id] = xgc_per_90
        
        tot_team_xg = sum(float(p.get('expected_goals', 0.0) or 0.0) for p in t_players)
        tot_team_mins = sum(int(p.get('minutes', 0) or 0) for p in t_players)
        team_xg_att_per_90 = (tot_team_xg / tot_team_mins * 90.0) if tot_team_mins > 0 else 1.35
        team_xg_att_map[t_id] = team_xg_att_per_90

    # Fetch L5M Average Minutes for all players concurrently
    l5m_map = compute_all_l5m_avg_mins(elements)

    processed = []
    for el in elements:
        t_id = el.get('team')
        f_info = fdr_summary.get(t_id, {})
        fdr1 = f_info.get('FDR1', 3.0)
        fdr3 = f_info.get('FDR3', 3.0)
        fdr5 = f_info.get('FDR5', 3.0)
        next_is_home = f_info.get('Next_Is_Home', 1)
        next_opp_fmt = f_info.get('Next_Opponent_Fmt', '-')
        next_opp_id = f_info.get('Next_Opponent_ID')

        mins = int(el.get('minutes', 0))
        pts = int(el.get('total_points', 0))
        cost = el.get('now_cost', 50) / 10.0
        xg = float(el.get('expected_goals', 0.0) or 0.0)
        xa = float(el.get('expected_assists', 0.0) or 0.0)
        xgi = float(el.get('expected_goal_involvements', 0.0) or 0.0)
        xgc = float(el.get('expected_goals_conceded', 0.0) or 0.0)
        saves = int(el.get('saves', 0))
        form = float(el.get('form', 0.0) or 0.0)
        ict = float(el.get('ict_index', 0.0) or 0.0)
        threat = float(el.get('threat', 0.0) or 0.0)
        creativity = float(el.get('creativity', 0.0) or 0.0)
        bps = int(el.get('bps', 0))
        bonus = int(el.get('bonus', 0))
        goals = int(el.get('goals_scored', 0))
        assists = int(el.get('assists', 0))
        cs = int(el.get('clean_sheets', 0))
        pos_id = el.get('element_type', 1)
        pos_name = POSITION_MAP.get(pos_id, "MID")
        
        # Per 90 stats
        if mins > 0:
            xg90 = (xg / mins) * 90.0
            xa90 = (xa / mins) * 90.0
            xgi90 = (xgi / mins) * 90.0
            xgc90 = (xgc / mins) * 90.0
            saves90 = (saves / mins) * 90.0
            bps90 = (bps / mins) * 90.0
            ict90 = (ict / mins) * 90.0
            threat90 = (threat / mins) * 90.0
            creativity90 = (creativity / mins) * 90.0
        else:
            xg90 = float(el.get('expected_goals_per_90', 0.0) or 0.0)
            xa90 = float(el.get('expected_assists_per_90', 0.0) or 0.0)
            xgi90 = float(el.get('expected_goal_involvements_per_90', 0.0) or 0.0)
            xgc90 = float(el.get('expected_goals_conceded_per_90', 0.0) or 0.0)
            saves90 = float(el.get('saves_per_90', 0.0) or 0.0)
            bps90 = 0.0
            ict90 = 0.0
            threat90 = float(el.get('threat_rank_type', 0.0) or 0.0) / 10.0 if el.get('threat_rank_type') else 0.0
            creativity90 = float(el.get('creativity_rank_type', 0.0) or 0.0) / 10.0 if el.get('creativity_rank_type') else 0.0

        tackles = float(el.get('tackles', 0) or 0)
        interceptions = float(el.get('interceptions', 0) or 0)
        clearances = float(el.get('clearances_blocks_interceptions', el.get('clearances', 0)) or 0)
        recoveries = float(el.get('recoveries', 0) or 0)
        tot_def_actions = tackles + interceptions + clearances + recoveries
        def_contrib_90 = (tot_def_actions / mins * 90.0) if mins > 0 else float(el.get('defensive_contribution_per_90', 0.0) or 0.0)

        # L5M Average Minutes
        avg_mins_l5m = l5m_map.get(el['id'], 0.0)

        # Peluang Main GW (%)
        chance_gw = el.get('chance_of_playing_next_round')
        if chance_gw is None:
            status_val = el.get('status', 'a')
            if status_val == 'a':
                chance_gw = 100
            elif status_val == 'd':
                chance_gw = 75
            elif status_val == 'i':
                chance_gw = 0
            elif status_val == 's':
                chance_gw = 0
            elif status_val == 'u':
                chance_gw = 0
            else:
                chance_gw = 100
        else:
            chance_gw = int(chance_gw)

        # Opponent xGC / xG per 90
        opp_xg_def = team_xg_def_map.get(next_opp_id, 1.35)
        opp_xg_att = team_xg_att_map.get(next_opp_id, 1.35)

        # Set-piece order
        corner_order = el.get('corners_and_indirect_freekicks_order')
        fk_order = el.get('direct_freekicks_order')
        pen_order = el.get('penalties_order')
        
        is_sp_taker = 0
        if (corner_order is not None and str(corner_order).strip() not in ["", "None", "-"] and int(corner_order) <= 2) or \
           (fk_order is not None and str(fk_order).strip() not in ["", "None", "-"] and int(fk_order) <= 2):
            is_sp_taker = 1

        processed.append({
            'id': el['id'],
            'team': t_id,
            'Nama Pemain': el.get('web_name', 'Unknown'),
            'Nama Lengkap': f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
            'Klub': team_dict.get(t_id, "Unknown"),
            'Posisi': pos_name,
            'element_type': pos_id,
            'Harga (£m)': cost,
            'Total Poin': pts,
            'FDR1': fdr1,
            'FDR3': fdr3,
            'FDR5': fdr5,
            'Next_Is_Home': next_is_home,
            'Lawan GW Berikutnya': next_opp_fmt,
            'Opponent_xGC_per_90': opp_xg_def,
            'Opponent_xG_per_90_attack': opp_xg_att,
            'Form': form,
            '% Ownership': float(el.get('selected_by_percent', 0.0) or 0.0),
            'Net Transfers GW': int(el.get('transfers_in_event', 0)) - int(el.get('transfers_out_event', 0)),
            'Transfers In GW': int(el.get('transfers_in_event', 0)),
            'Transfers Out GW': int(el.get('transfers_out_event', 0)),
            'xG': xg,
            'xA': xa,
            'xGI': xgi,
            'xG per 90': round(xg90, 2),
            'xA per 90': round(xa90, 2),
            'xGI per 90': round(xgi90, 2),
            'xGC per 90': round(xgc90, 2),
            'Saves per 90': round(saves90, 2),
            'Defensive Contribution per 90': round(def_contrib_90, 2),
            'threat_per_90': round(threat90, 2),
            'creativity_per_90': round(creativity90, 2),
            'ICT Index': ict,
            'BPS': bps,
            'Bonus Poin': bonus,
            'Kartu Kuning': int(el.get('yellow_cards', 0)),
            'Kartu Merah': int(el.get('red_cards', 0)),
            'Saves': saves,
            'Status': STATUS_MAP.get(el.get('status', 'a'), 'Tersedia'),
            'Peluang Main GW (%)': chance_gw,
            'Berita Cedera': el.get('news', '') or 'Fit',
            'Menit Bermain': mins,
            'Gol': goals,
            'Asis': assists,
            'Clean Sheet': cs,
            'Penalti Order': format_setpiece_order(pen_order),
            'Free Kick Order': format_setpiece_order(fk_order),
            'Corner Order': format_setpiece_order(corner_order),
            'is_setpiece_taker': is_sp_taker,
            'Avg Mins (L5M)': avg_mins_l5m,
            'Poin per £m': round(pts / cost, 2) if cost > 0 else 0.0,
            'Kemudahan Jadwal': round(6.0 - fdr3, 1),
            'raw_xg90': xg90,
            'raw_xa90': xa90,
            'raw_xgc90': xgc90,
            'raw_saves90': saves90,
            'raw_bps90': bps90,
            'raw_ict90': ict90,
            'raw_threat90': threat90,
            'raw_creativity90': creativity90,
            'bps_per_90_calc': bps90
        })
        
    df = pd.DataFrame(processed)
    if df.empty:
        return df, team_dict

    # -------------------------------------------------------------------------
    # VECTORIZED PREDICTION FOR OPTION A
    # -------------------------------------------------------------------------
    xpoin_pred_all = np.zeros(len(df))
    for pos_key, model_info in _models_dict.items():
        pos_mask = (df['Posisi'] == pos_key)
        if not pos_mask.any():
            continue
        
        model = model_info['model']
        feature_cols = model_info.get('feature_cols', [])
        sub_df = df[pos_mask]
        
        feature_data = {}
        for col in feature_cols:
            if col == 'xG_per_90':
                feature_data[col] = sub_df['raw_xg90']
            elif col == 'xA_per_90':
                feature_data[col] = sub_df['raw_xa90']
            elif col == 'bps_per_90':
                feature_data[col] = sub_df['raw_bps90']
            elif col == 'form':
                feature_data[col] = sub_df['Form']
            elif col == 'was_home':
                feature_data[col] = sub_df['Next_Is_Home']
            elif col == 'FDR':
                feature_data[col] = sub_df['FDR1']
            elif col == 'last_minutes_5_match':
                feature_data[col] = sub_df['Avg Mins (L5M)']
            elif col == 'ict_index':
                feature_data[col] = sub_df['raw_ict90']
            elif col == 'Defensive_Contribution_per_90':
                feature_data[col] = sub_df['Defensive Contribution per 90']
            elif col == 'xGC_per_90':
                feature_data[col] = sub_df['raw_xgc90']
            elif col == 'Saves_per_90':
                feature_data[col] = sub_df['raw_saves90']
            else:
                feature_data[col] = sub_df.get(col, 0.0)
                
        X_pos = pd.DataFrame(feature_data, index=sub_df.index)
        preds = model.predict(X_pos)
        
        # Penyesuaian Peluang Main (Chance of Playing Next Round)
        chance_arr = sub_df['Peluang Main GW (%)'].values / 100.0
        preds = preds * chance_arr
        preds = np.clip(preds, 0.0, 24.0)
        xpoin_pred_all[pos_mask.values] = preds

    df['xPoin'] = np.round(xpoin_pred_all, 2)
    df['xPoin per £m'] = np.where(df['Harga (£m)'] > 0, (df['xPoin'] / df['Harga (£m)']).round(2), 0.0)

    # -------------------------------------------------------------------------
    # VECTORIZED PREDICTION FOR OPTION B
    # -------------------------------------------------------------------------
    opt_b_xg, opt_b_xa = _opt_b_models if _opt_b_models is not None else (None, None)
    
    raw_xg_match = np.zeros(len(df))
    raw_xa_match = np.zeros(len(df))

    # Support multiple positional regression models
    for pos_key in ['FWD', 'MID', 'DEF']:
        pos_mask = (df['Posisi'] == pos_key)
        if not pos_mask.any():
            continue
        sub_df = df[pos_mask]

        # Model xG
        pos_m_xg = opt_b_xg.get(pos_key) if isinstance(opt_b_xg, dict) else opt_b_xg
        if pos_m_xg is not None:
            X_xg = pd.DataFrame({
                'xG_per_90': sub_df['raw_xg90'],
                'Opponent_xGC_per_90': sub_df['Opponent_xGC_per_90'],
                'was_home': sub_df['Next_Is_Home'],
                'form': sub_df['Form'],
                'thread_per_90': sub_df['raw_threat90'],
                'FDR': sub_df['FDR1']
            }, index=sub_df.index)
            raw_xg_match[pos_mask.values] = pos_m_xg.predict(X_xg)
        else:
            raw_xg_match[pos_mask.values] = (sub_df['raw_xg90'] * (sub_df['Opponent_xGC_per_90'] / 1.35) * (1.1 if sub_df['Next_Is_Home'].mean() == 1 else 0.9)).values

        # Model xA
        pos_m_xa = opt_b_xa.get(pos_key) if isinstance(opt_b_xa, dict) else opt_b_xa
        if pos_m_xa is not None:
            X_xa = pd.DataFrame({
                'xA_per_90': sub_df['raw_xa90'],
                'Opponent_xGC_per_90': sub_df['Opponent_xGC_per_90'],
                'was_home': sub_df['Next_Is_Home'],
                'is_setpiece_taker': sub_df['is_setpiece_taker'],
                'form': sub_df['Form'],
                'creativity_per_90': sub_df['raw_creativity90'],
                'FDR': sub_df['FDR1']
            }, index=sub_df.index)
            raw_xa_match[pos_mask.values] = pos_m_xa.predict(X_xa)
        else:
            raw_xa_match[pos_mask.values] = (sub_df['raw_xa90'] * (sub_df['Opponent_xGC_per_90'] / 1.35) * (1.1 if sub_df['Next_Is_Home'].mean() == 1 else 0.9)).values

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
