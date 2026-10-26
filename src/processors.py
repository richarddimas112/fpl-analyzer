"""
Data Processing, FDR calculations, Team Strength Analysis, and Player Transforms.
"""

import numpy as np
import pandas as pd
import streamlit as st
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import poisson

from src.constants import POSITION_MAP, STATUS_MAP, POSITION_XG_MULTIPLIER, POSITION_XA_MULTIPLIER
from src.api import fetch_player_history

def bivariate_dixon_coles_cs_prob(lambda_team, lambda_opp, rho=-0.06):
    """
    Menghitung probabilitas Clean Sheet (P(Y = 0)) menggunakan Model Bivariat Dixon-Coles Poisson.
    Y adalah gol tim lawan, X adalah gol tim pemain.
    P(Clean Sheet) = sum_{x=0..inf} P(X=x, Y=0).
    Dixon-Coles mengoreksi korelasi skor rendah (0-0, 1-0, 0-1, 1-1) dengan parameter rho.
    """
    lam_team = np.maximum(0.05, np.asarray(lambda_team, dtype=float))
    lam_opp = np.maximum(0.05, np.asarray(lambda_opp, dtype=float))
    
    exp_opp = np.exp(-lam_opp)
    exp_team = np.exp(-lam_team)
    
    # Koreksi parameter Dixon-Coles untuk skor rendah
    tau_00 = np.clip(1.0 - lam_team * lam_opp * rho, 0.0, 2.0)
    tau_10 = np.clip(1.0 + lam_opp * rho, 0.0, 2.0)
    
    p_00 = tau_00 * exp_team * exp_opp
    p_10 = tau_10 * lam_team * exp_team * exp_opp
    p_rest = np.maximum(0.0, 1.0 - exp_team - lam_team * exp_team) * exp_opp
    
    prob_cs = np.clip(p_00 + p_10 + p_rest, 0.02, 0.95)
    return prob_cs

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


def compute_automatic_position_multipliers(players_data, fallback_xg=None, fallback_xa=None):
    """
    Menghitung multiplier posisi sebagai fallback xG dan sebagai multiplier xA:
      - xG fallback posisi = sum(Gol) / sum(xG)
      - xA multiplier posisi = sum(Asis) / sum(xA)
    Multiplier xG personal dihitung oleh compute_automatic_player_xg_multipliers.
    """
    if fallback_xg is None:
        fallback_xg = POSITION_XG_MULTIPLIER
    if fallback_xa is None:
        fallback_xa = POSITION_XA_MULTIPLIER

    if isinstance(players_data, pd.DataFrame):
        df = players_data.copy()
    else:
        df = pd.DataFrame(players_data)

    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    if 'Posisi' not in df.columns and 'element_type' in df.columns:
        df['Posisi'] = df['element_type'].map(pos_map)

    g_col = 'Gol' if 'Gol' in df.columns else ('goals_scored' if 'goals_scored' in df.columns else None)
    xg_col = 'xG' if 'xG' in df.columns else ('expected_goals' if 'expected_goals' in df.columns else None)
    a_col = 'Asis' if 'Asis' in df.columns else ('assists' if 'assists' in df.columns else None)
    xa_col = 'xA' if 'xA' in df.columns else ('expected_assists' if 'expected_assists' in df.columns else None)

    if not g_col or not xg_col or not a_col or not xa_col or 'Posisi' not in df.columns:
        return dict(fallback_xg), dict(fallback_xa), {}

    for c in [g_col, xg_col, a_col, xa_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    agg = df.groupby('Posisi')[[g_col, xg_col, a_col, xa_col]].sum()

    auto_xg = {}
    auto_xa = {}
    stats_dict = {}

    for pos in ['FWD', 'MID', 'DEF', 'GK']:
        if pos in agg.index:
            g = float(agg.loc[pos, g_col])
            xg = float(agg.loc[pos, xg_col])
            a = float(agg.loc[pos, a_col])
            xa = float(agg.loc[pos, xa_col])
        else:
            g, xg, a, xa = 0.0, 0.0, 0.0, 0.0

        if pos == 'GK':
            # Goalkeepers tidak mengambil tembakan ofensif reguler; proteksi 0.00x
            m_xg = 0.00
            m_xa = round(float(a / xa), 2) if xa > 0.1 else float(fallback_xa.get('GK', 0.50))
            m_xa = float(np.clip(m_xa, 0.20, 1.50))
        else:
            fb_g = float(fallback_xg.get(pos, 1.0))
            fb_a = float(fallback_xa.get(pos, 1.0))
            # Rasio konversi aktual terhadap ekspektasi statistik
            m_xg = round(float(g / xg), 2) if xg > 0.5 else fb_g
            m_xa = round(float(a / xa), 2) if xa > 0.5 else fb_a
            # Pembatasan realistis untuk mencegah anomali pembagian di awal gameweek
            m_xg = float(np.clip(m_xg, 0.50, 2.00))
            m_xa = float(np.clip(m_xa, 0.50, 2.00))

        auto_xg[pos] = m_xg
        auto_xa[pos] = m_xa
        stats_dict[pos] = {
            'goals': g,
            'xg': round(xg, 2),
            'assists': a,
            'xa': round(xa, 2),
            'xg_ratio': m_xg,
            'xa_ratio': m_xa
        }

    return auto_xg, auto_xa, stats_dict


def compute_automatic_player_xg_multipliers(players_data, fallback_xg=None, min_xg=0.5):
    """Return each player's goals/xG conversion multiplier, falling back to position at low xG."""
    if isinstance(players_data, pd.DataFrame):
        df = players_data.copy()
    else:
        df = pd.DataFrame(players_data)
    auto_xg_by_position, _, _ = compute_automatic_position_multipliers(df, fallback_xg=fallback_xg)

    goals_col = 'Gol' if 'Gol' in df.columns else ('goals_scored' if 'goals_scored' in df.columns else None)
    xg_col = 'xG' if 'xG' in df.columns else ('expected_goals' if 'expected_goals' in df.columns else None)
    if 'Posisi' in df.columns:
        positions = df['Posisi']
    elif 'element_type' in df.columns:
        positions = df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    else:
        positions = pd.Series(index=df.index, dtype=object)

    fallback = positions.map(auto_xg_by_position).fillna(1.0).astype(float)
    if not goals_col or not xg_col:
        return fallback

    goals = pd.to_numeric(df[goals_col], errors='coerce').fillna(0.0)
    xg = pd.to_numeric(df[xg_col], errors='coerce').fillna(0.0)
    personal_ratio = (goals / xg.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    eligible = xg >= float(min_xg)
    multiplier = fallback.copy()
    multiplier.loc[eligible] = personal_ratio.loc[eligible].fillna(fallback.loc[eligible]).clip(0.50, 2.00)
    multiplier.loc[positions == 'GK'] = 0.0
    return multiplier.astype(float)

def format_setpiece_order(val):
    """Memformat urutan set-piece menjadi label yang jelas."""
    if pd.isnull(val) or val is None or str(val).strip() == "":
        return "-"
    try:
        v_int = int(float(val))
        return f"Pilihan ke-{v_int}"
    except Exception:
        return str(val)

def get_team_short_map(fpl_data):
    """
    Extracts official 3-letter club short names directly from FPL API teams data.
    Returns a dictionary mapping both team ID and full team name to official short_name.
    """
    if not fpl_data or 'teams' not in fpl_data:
        return {}
    
    mapping = {}
    for t in fpl_data.get('teams', []):
        s_name = t.get('short_name', '')
        if s_name:
            mapping[t.get('id')] = s_name
            mapping[str(t.get('id'))] = s_name
            mapping[t.get('name')] = s_name
    return mapping

@st.cache_data(ttl=86400)
def calculate_team_fdrs(fixtures, teams_dict, teams_short_dict=None):
    """Calculate FDR1, FDR3, FDR5 and next match home status & opponent info for every team."""
    team_upcoming = {t_id: [] for t_id in teams_dict.keys()}
    
    for f in fixtures:
        # Hanya hitung pertandingan mendatang yang benar-benar belum berjalan (belum selesai, belum kickoff)
        is_already_played = bool(f.get('finished') or f.get('finished_provisional') or f.get('started'))
        if not is_already_played:
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
            f10 = float(np.mean([x['fdr'] for x in fxs[:10]])) if len(fxs) >= 10 else (float(np.mean([x['fdr'] for x in fxs])) if fxs else 3.0)
        else:
            f1 = 3.0
            f3 = 3.0
            f5 = 3.0
            f10 = 3.0
            next_is_home = 1
            next_opp_id = None

        opp_name = teams_dict.get(next_opp_id, 'TBD') if next_opp_id else 'TBD'
        opp_short = (teams_short_dict.get(next_opp_id) if teams_short_dict else None) or (teams_dict.get(next_opp_id, '')[:3].upper() if next_opp_id else 'TBD')
        opp_fmt = f"{opp_name} ({'🏠' if next_is_home == 1 else '✈️'})" if next_opp_id else "-"
        
        upcoming_10 = []
        for x in fxs[:10]:
            o_id = x.get('opp_id')
            o_name = teams_dict.get(o_id, f"Team {o_id}")
            o_short = (teams_short_dict.get(o_id) if teams_short_dict else None) or teams_dict.get(o_id, '')[:3].upper()
            o_home = x.get('is_home') == 1
            upcoming_10.append({
                'gw': x.get('gw'),
                'opp_id': o_id,
                'opp_name': o_name,
                'opp_short': o_short,
                'is_home': x.get('is_home'),
                'fdr': x.get('fdr'),
                'label': f"{o_name} ({'H' if o_home else 'A'}) [{x.get('fdr')}]"
            })

        upcoming_all = []
        for x in fxs:
            o_id = x.get('opp_id')
            o_name = teams_dict.get(o_id, f"Team {o_id}")
            o_short = (teams_short_dict.get(o_id) if teams_short_dict else None) or teams_dict.get(o_id, '')[:3].upper()
            o_home = x.get('is_home') == 1
            upcoming_all.append({
                'gw': x.get('gw'),
                'opp_id': o_id,
                'opp_name': o_name,
                'opp_short': o_short,
                'is_home': x.get('is_home'),
                'fdr': x.get('fdr'),
                'label': f"{o_name} ({'H' if o_home else 'A'}) [{x.get('fdr')}]"
            })

        fdr_summary[t_id] = {
            'FDR1': round(f1, 2),
            'FDR3': round(f3, 2),
            'FDR5': round(f5, 2),
            'FDR10': round(f10, 2),
            'Next_Is_Home': next_is_home,
            'Next_Opponent_ID': next_opp_id,
            'Next_Opponent_Name': opp_name,
            'Next_Opponent_Short': opp_short,
            'Next_Opponent_Fmt': opp_fmt,
            'upcoming_10': upcoming_10,
            'upcoming_all': upcoming_all
        }
        
    return fdr_summary

@st.cache_data(ttl=3600)
def collect_team_match_metrics(elements, fixtures_data, histories_by_player_id=None):
    """Aggregate player xG and goalkeeper xGC by the club they represented in each finished fixture."""
    fixture_by_id = {
        int(fixture['id']): fixture
        for fixture in (fixtures_data or [])
        if fixture.get('id') is not None
        and (fixture.get('finished') or fixture.get('finished_provisional'))
        and fixture.get('team_h_score') is not None
        and fixture.get('team_a_score') is not None
    }
    team_fixture_rows = {}
    for fixture_id, fixture in fixture_by_id.items():
        gw = fixture.get('event')
        if gw is None:
            continue
        home_id, away_id = int(fixture['team_h']), int(fixture['team_a'])
        home_score, away_score = int(fixture['team_h_score']), int(fixture['team_a_score'])
        for team_id, opponent_score, goals_for in (
            (home_id, away_score, home_score),
            (away_id, home_score, away_score),
        ):
            team_fixture_rows[(team_id, fixture_id)] = {
                'team_id': team_id,
                'fixture_id': fixture_id,
                'gw': int(gw),
                'xg': 0.0,
                'xgc': 0.0,
                'goals_for': goals_for,
                'goals_against': opponent_score,
                'clean_sheet': int(opponent_score == 0),
                'gk_xgc_rows': 0,
            }

    active_elements = [
        element for element in (elements or [])
        if int(element.get('minutes', 0) or 0) > 0 or int(element.get('total_points', 0) or 0) > 0
    ]
    position_by_player = {
        int(element['id']): int(element.get('element_type', 0) or 0)
        for element in active_elements
        if element.get('id') is not None
    }

    def load_history(element):
        player_id = int(element['id'])
        if histories_by_player_id is not None:
            history = histories_by_player_id.get(player_id, histories_by_player_id.get(str(player_id), []))
            return player_id, history
        return player_id, fetch_player_history(player_id)

    with ThreadPoolExecutor(max_workers=20) as executor:
        histories = list(executor.map(load_history, active_elements))

    for player_id, history in histories:
        is_goalkeeper = position_by_player.get(player_id) == 1
        for match in history:
            try:
                fixture_id = int(match['fixture'])
            except (KeyError, TypeError, ValueError):
                continue
            fixture = fixture_by_id.get(fixture_id)
            if fixture is None:
                continue
            if match.get('was_home') not in (True, False):
                continue
            is_home = match['was_home']
            team_id = int(fixture['team_h'] if is_home else fixture['team_a'])
            team_row = team_fixture_rows.get((team_id, fixture_id))
            if team_row is None:
                continue
            team_row['xg'] += float(match.get('expected_goals', 0.0) or 0.0)
            if is_goalkeeper:
                team_row['xgc'] += float(match.get('expected_goals_conceded', 0.0) or 0.0)
                team_row['gk_xgc_rows'] += 1

    rows = list(team_fixture_rows.values())
    for row in rows:
        # Use the official score as a transparent proxy only when no keeper xGC
        # is available for that team-fixture.
        if row['gk_xgc_rows'] == 0:
            row['xgc'] = float(row['goals_against'])
    return rows


def summarize_team_match_metrics(team_match_rows, target_gw=None, prior_matches=5.0):
    """Compute per-match team rates, optionally using only fixtures before a target GW."""
    prior_rows = [
        row for row in team_match_rows
        if target_gw is None or int(row['gw']) < int(target_gw)
    ]
    grouped = {}
    for row in prior_rows:
        stats = grouped.setdefault(int(row['team_id']), {
            'matches': 0, 'xg': 0.0, 'xgc': 0.0, 'goals_for': 0,
            'goals_against': 0, 'clean_sheets': 0,
        })
        stats['matches'] += 1
        for key in ('xg', 'xgc', 'goals_for', 'goals_against', 'clean_sheet'):
            stats[{'clean_sheet': 'clean_sheets'}.get(key, key)] += float(row[key])

    league_matches = sum(stats['matches'] for stats in grouped.values())
    prior_rates = {}
    for key in ('xg', 'xgc', 'goals_for', 'goals_against', 'clean_sheets'):
        total = sum(stats[key] for stats in grouped.values())
        prior_rates[key] = total / league_matches if league_matches else float('nan')

    result = {}
    team_ids = {int(row['team_id']) for row in team_match_rows}
    for team_id in team_ids:
        stats = grouped.get(team_id, {
            'matches': 0, 'xg': 0.0, 'xgc': 0.0, 'goals_for': 0,
            'goals_against': 0, 'clean_sheets': 0,
        })
        n = stats['matches']
        adjusted = {}
        for key in ('xg', 'xgc', 'goals_for', 'goals_against', 'clean_sheets'):
            prior_rate = prior_rates[key]
            if not np.isfinite(prior_rate):
                adjusted[key] = float('nan')
            else:
                adjusted[key] = (stats[key] + prior_rate * prior_matches) / (n + prior_matches)
        result[team_id] = {
            **stats,
            'xg_per_match': stats['xg'] / n if n else float('nan'),
            'xgc_per_match': stats['xgc'] / n if n else float('nan'),
            'goals_for_per_match': stats['goals_for'] / n if n else float('nan'),
            'goals_against_per_match': stats['goals_against'] / n if n else float('nan'),
            'clean_sheet_rate': stats['clean_sheets'] / n if n else float('nan'),
        'xg_per_match_adjusted': adjusted['xg'],
            'xgc_per_match_adjusted': adjusted['xgc'],
            'goals_for_per_match_adjusted': adjusted['goals_for'],
            'goals_against_per_match_adjusted': adjusted['goals_against'],
            'clean_sheet_rate_adjusted': adjusted['clean_sheets'],
        }
    return result


@st.cache_data(ttl=3600)
def calculate_team_strength_analysis(_fpl_data, players_df, fdr_summary, fixtures_data):
    """
    Aggregate comprehensive Premier League team data from player statistics and FPL APIs:
    - Average FPL points of players (and active players)
    - Total squad points and squad market value
    - Offensive metrics: Total Goals, Total Assists, Total xG, Total xA, Total xGI, Top Scorer
    - Defensive metrics: Clean Sheets, Total xGC, Total Saves, Defensive Contribution
    - Fixture Difficulty Ratings: FDR1, FDR3, FDR5, Next Opponent (Home/Away), Schedule Ease
    - Normalized Composite Team Strength Index (0 - 100) and Strength Tiers
    """
    if fixtures_data is None:
        raise ValueError("fixtures_data wajib diberikan agar metrik per laga dan clean sheet tim akurat.")

    teams = _fpl_data.get('teams', []) if _fpl_data else []
    records = []
    team_match_rows = collect_team_match_metrics(_fpl_data.get('elements', []) if _fpl_data else [], fixtures_data)
    team_season_metrics = summarize_team_match_metrics(team_match_rows)

    # Kelompokkan pemain sekali; tanpa ini seluruh DataFrame dipindai ulang
    # untuk setiap tim.
    group_column = 'team' if 'team' in players_df.columns else 'Klub'
    players_by_team = {
        team_key: team_players
        for team_key, team_players in players_df.groupby(group_column, sort=False)
    } if group_column in players_df.columns else {}
    
    for t in teams:
        t_id = t['id']
        t_name = t['name']
        t_short = t.get('short_name', '')
        
        # Filter players for this team
        lookup_key = t_id if group_column == 'team' else t_name
        tp = players_by_team.get(lookup_key, players_df.iloc[0:0])
            
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
            clean_sheets = 0
            
            # Hitung aktual kebobolan (Goals Conceded / GC) dari akumulasi kebobolan kiper
            gks = tp[tp['Posisi'].isin(['GK', 'GKP'])]
            if not gks.empty and 'Kebobolan' in gks.columns:
                total_gc = int(gks['Kebobolan'].sum())
            elif not gks.empty and 'goals_conceded' in gks.columns:
                total_gc = int(gks['goals_conceded'].sum())
            elif not def_gk.empty and 'Kebobolan' in def_gk.columns:
                total_gc = int(def_gk['Kebobolan'].max())
            elif not def_gk.empty and 'goals_conceded' in def_gk.columns:
                total_gc = int(def_gk['goals_conceded'].max())
            else:
                total_gc = 0

            # Fallback ke raw elements jika belum terhitung
            if total_gc == 0 and _fpl_data and 'elements' in _fpl_data:
                team_elements = [e for e in _fpl_data['elements'] if e.get('team') == t_id and e.get('element_type') == 1]
                if team_elements:
                    total_gc = sum(int(e.get('goals_conceded', 0) or 0) for e in team_elements)

            # Hitung xGC riil tim dari akumulasi xGC kiper yang bermain (atau defender jika kiper kosong)
            gk_xgc = float(pd.to_numeric(gks['xGC'], errors='coerce').fillna(0.0).sum()) if not gks.empty and 'xGC' in gks.columns else 0.0
            if gk_xgc > 0:
                total_xgc = gk_xgc
            elif not def_gk.empty and 'xGC' in def_gk.columns:
                total_xgc = float(pd.to_numeric(def_gk['xGC'], errors='coerce').fillna(0.0).max())
            elif not def_gk.empty and 'expected_goals_conceded' in def_gk.columns:
                total_xgc = float(pd.to_numeric(def_gk['expected_goals_conceded'], errors='coerce').fillna(0.0).max())
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
            top_asset_pts = int(top_asset_row['Total Poin']) if top_asset_row is not None else 0
            top_asset_raw_name = str(top_asset_row['Nama Pemain']) if top_asset_row is not None else "-"

            # Standar Deviasi & Analisis Single-Player Dependency (Pemain dengan Menit Bermain > 0)
            active_pts = active['Total Poin'] if not active.empty else pd.Series(dtype=float)
            num_active = len(active)
            if num_active > 1:
                std_dev_pts = float(active_pts.std(ddof=1))
            else:
                std_dev_pts = 0.0

            tot_active_pts = float(active_pts.sum()) if not active.empty else 0.0
            top_player_share_pct = round((top_asset_pts / tot_active_pts * 100.0), 1) if tot_active_pts > 0 else 0.0
            cv_pts = round(std_dev_pts / avg_points_played, 2) if avg_points_played > 0 else 0.0

            # Kategori Dependensi Single Player
            if top_player_share_pct >= 25.0 or (cv_pts >= 1.15 and top_player_share_pct >= 20.0):
                dependency_status = "⚠️ Ekstrem (One-Man Team)"
            elif top_player_share_pct >= 18.0 or cv_pts >= 0.95:
                dependency_status = "⚡ Tinggi (Talisman Reliant)"
            elif top_player_share_pct >= 14.0 or cv_pts >= 0.75:
                dependency_status = "⚖️ Moderat (Semi-Kolektif)"
            else:
                dependency_status = "🤝 Kolektif (Merata)"
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
            total_gc = 0
            total_xgc = 0.0
            total_saves = 0
            avg_form = 0.0
            total_squad_value = 0.0
            total_bps = 0
            top_scorer_name = "-"
            top_creator_name = "-"
            top_asset_name = "-"
            top_asset_pts = 0
            top_asset_raw_name = "-"
            std_dev_pts = 0.0
            top_player_share_pct = 0.0
            cv_pts = 0.0
            dependency_status = "Data Kosong"
            
        f_info = fdr_summary.get(t_id, {})
        fdr1 = float(f_info.get('FDR1', 3.0))
        fdr3 = float(f_info.get('FDR3', 3.0))
        fdr5 = float(f_info.get('FDR5', 3.0))
        fdr10 = float(f_info.get('FDR10', 3.0))
        next_opp = f_info.get('Next_Opponent_Fmt', '-')
        
        str_ovr_h = t.get('strength_overall_home', 3) or 3
        str_ovr_a = t.get('strength_overall_away', 3) or 3
        str_ovr = round((str_ovr_h + str_ovr_a) / 2.0, 1)
        
        team_metric = team_season_metrics.get(int(t_id), {})
        matches_played = int(team_metric.get('matches', 0))
        matches_completed = matches_played
        total_xg = float(team_metric.get('xg', 0.0))
        total_goals = int(team_metric.get('goals_for', 0))
        total_xgc = float(team_metric.get('xgc', 0.0))
        total_gc = int(team_metric.get('goals_against', 0))
        clean_sheets = int(team_metric.get('clean_sheets', 0))
        xg_per_match = team_metric.get('xg_per_match', float('nan'))
        goals_per_match = team_metric.get('goals_for_per_match', float('nan'))
        xgc_per_match = team_metric.get('xgc_per_match', float('nan'))
        goals_conceded_per_match = team_metric.get('goals_against_per_match', float('nan'))
        clean_sheet_rate = team_metric.get('clean_sheet_rate', float('nan'))

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
            'xG per Match': round(xg_per_match, 3),
            'Gol per Match': round(goals_per_match, 3),
            'xG per Match (Smoothed)': round(team_metric.get('xg_per_match_adjusted', float('nan')), 3),
            'Gol per Match (Smoothed)': round(team_metric.get('goals_for_per_match_adjusted', float('nan')), 3),
            'Total xA': round(total_xa, 2),
            'Total xGI': round(total_xgi, 2),
            'Clean Sheet': clean_sheets,
            'Clean Sheet Rate (%)': round(clean_sheet_rate * 100.0, 1),
            'Clean Sheet Rate (Smoothed %)': round(team_metric.get('clean_sheet_rate_adjusted', float('nan')) * 100.0, 1),
            'Matches Played': matches_played,
            'Matches Completed': matches_completed,
            'Kebobolan': total_gc,
            'Kebobolan (GC)': total_gc,
            'Kebobolan per Match': round(goals_conceded_per_match, 3),
            'Kebobolan per Match (Smoothed)': round(team_metric.get('goals_against_per_match_adjusted', float('nan')), 3),
            'Total xGC': round(total_xgc, 2),
            'xGC per Match': round(xgc_per_match, 3),
            'xGC per Match (Smoothed)': round(team_metric.get('xgc_per_match_adjusted', float('nan')), 3),
            'Total Saves': total_saves,
            'Form Rata-rata': round(avg_form, 2),
            'Nilai Skuad (£m)': round(total_squad_value, 1),
            'Total BPS': total_bps,
            'Top Scorer': top_scorer_name,
            'Top Creator': top_creator_name,
            'Top Aset FPL': top_asset_name,
            'Top Aset Poin': top_asset_pts,
            'Top Aset Nama': top_asset_raw_name,
            'Std Dev Poin Pemain': round(std_dev_pts, 2),
            'CV Poin Pemain': cv_pts,
            '% Poin Top Player': top_player_share_pct,
            'Status Dependensi': dependency_status,
            'FDR1': round(fdr1, 1),
            'FDR3': round(fdr3, 2),
            'FDR5': round(fdr5, 2),
            'FDR10': round(fdr10, 2),
            'Lawan Berikutnya': next_opp,
            'Official_Strength': str_ovr
        })
        
    df_teams = pd.DataFrame(records)
    if df_teams.empty:
        return df_teams
        
    def min_max_scale(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50.0, index=series.index).where(series.notna(), 50.0)
        scaled = (series - mn) / (mx - mn) * 100.0
        scaled = (100.0 - scaled) if invert else scaled
        # A club with no observed fixtures has no rate estimate; treat it as
        # neutral rather than as the best/worst team due to a synthetic zero.
        return scaled.fillna(50.0)

    # Rate metrics avoid rewarding teams for having played more fixtures.
    att_score = 0.6 * min_max_scale(df_teams['xG per Match (Smoothed)']) + 0.4 * min_max_scale(df_teams['Gol per Match (Smoothed)'])
    xgc_metric = 'xGC per Match (Smoothed)'
    gc_metric = 'Kebobolan per Match (Smoothed)'
    cs_metric = 'Clean Sheet Rate (Smoothed %)'
    # Skor pertahanan komparatif: 40% xGC/laga, 35% kebobolan/laga, 25% clean-sheet rate aktual.
    def_score = (
        0.40 * min_max_scale(df_teams[xgc_metric], invert=True) +
        0.35 * min_max_scale(df_teams[gc_metric], invert=True) +
        0.25 * min_max_scale(df_teams[cs_metric])
    )
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

@st.cache_data(ttl=3600)
def compute_all_l5m_option_b_features(elements, target_gw_by_player, histories_by_player_id=None):
    """Build leakage-safe rolling features and position-shrunk Threat/90."""
    active_elements = [el for el in elements if int(el.get('minutes', 0) or 0) > 0 or int(el.get('total_points', 0) or 0) > 0]

    def load_history(el):
        pid = int(el['id'])
        if histories_by_player_id is not None:
            return pid, histories_by_player_id.get(pid, histories_by_player_id.get(str(pid), []))
        return pid, fetch_player_history(pid)

    histories = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        for pid, history in executor.map(load_history, active_elements):
            histories[pid] = history or []

    player_by_id = {int(el['id']): el for el in active_elements}
    target_gws = {int(gw) for gw in target_gw_by_player.values() if gw is not None}
    position_priors = {}
    for target_gw in target_gws:
        totals = {}
        for pid, history in histories.items():
            pos = int(player_by_id[pid].get('element_type', 0) or 0)
            bucket = totals.setdefault(pos, [0.0, 0.0])
            for match in history:
                if int(match.get('round', match.get('event', 0)) or 0) < target_gw:
                    bucket[0] += float(match.get('threat', 0.0) or 0.0)
                    bucket[1] += float(match.get('minutes', 0) or 0)
        league_threat = sum(v[0] for v in totals.values())
        league_minutes = sum(v[1] for v in totals.values())
        league_rate = league_threat / league_minutes * 90.0 if league_minutes else 0.0
        position_priors[target_gw] = {
            pos: vals[0] / vals[1] * 90.0 if vals[1] else league_rate
            for pos, vals in totals.items()
        }

    feature_map = {}
    for el in active_elements:
        pid = int(el['id'])
        target_gw = target_gw_by_player.get(pid, target_gw_by_player.get(str(pid)))
        history = histories.get(pid, [])
        prior = [m for m in history if target_gw is None or int(m.get('round', m.get('event', 0)) or 0) < int(target_gw)]
        prior.sort(key=lambda m: (int(m.get('round', m.get('event', 0)) or 0), m.get('kickoff_time', '')))
        window = prior[-5:]
        n_matches = len(window)
        total_mins = sum(float(m.get('minutes', 0) or 0) for m in window)
        total_threat = sum(float(m.get('threat', 0.0) or 0.0) for m in window)
        pos = int(el.get('element_type', 0) or 0)
        pos_prior = position_priors.get(int(target_gw), {}).get(pos, 0.0) if target_gw is not None else 0.0
        threat90 = ((total_threat * 90.0) + pos_prior * 270.0) / (total_mins + 270.0)
        feature_map[pid] = {
            'avg_mins': round(total_mins / n_matches, 1) if n_matches else 0.0,
            'form': sum(float(m.get('total_points', 0) or 0) for m in window) / n_matches if n_matches else 0.0,
            'started_prev': int(float(prior[-1].get('minutes', 0) or 0) >= 60) if prior else 0,
            'threat_per_90': float(max(0.0, threat90)),
            'creativity_per_90': float(sum(float(m.get('creativity', 0.0) or 0.0) for m in window) / max(total_mins, 60.0) * 90.0),
        }

    # Preserve rows for inactive players so downstream output remains complete.
    for el in elements:
        feature_map.setdefault(int(el['id']), {
            'avg_mins': 0.0, 'form': 0.0, 'started_prev': 0,
            'threat_per_90': 0.0, 'creativity_per_90': 0.0,
        })
    return feature_map


@st.cache_data(ttl=3600)
def compute_team_differentials_asof(elements, fixtures_data, target_gws=None, histories_by_player_id=None):
    """Build leakage-safe, transfer-aware and shrunk team differentials per target GW."""
    team_match_rows = collect_team_match_metrics(elements, fixtures_data, histories_by_player_id)
    all_team_ids = sorted({
        int(f[key]) for f in (fixtures_data or []) for key in ('team_h', 'team_a')
        if f.get(key) is not None
    })
    fixtures_by_gw = {}
    for fixture in fixtures_data or []:
        if fixture.get('event') is not None:
            fixtures_by_gw.setdefault(int(fixture['event']), []).append(fixture)
    targets = sorted({
        int(gw) for gw in (target_gws if target_gws is not None else fixtures_by_gw)
        if gw is not None
    })
    result = {}

    def scale(metric_name, metrics, invert=False):
        values = np.asarray([
            float(metrics.get(team_id, {}).get(metric_name, np.nan)) for team_id in all_team_ids
        ], dtype=float)
        valid = np.isfinite(values)
        scaled = np.full(values.shape, 50.0)
        if valid.any():
            lo, hi = float(values[valid].min()), float(values[valid].max())
            if hi > lo:
                scaled[valid] = (values[valid] - lo) / (hi - lo) * 100.0
                if invert:
                    scaled[valid] = 100.0 - scaled[valid]
        return dict(zip(all_team_ids, scaled))

    for target_gw in targets:
        team_metrics = summarize_team_match_metrics(team_match_rows, target_gw=target_gw)
        att_xg = scale('xg_per_match_adjusted', team_metrics)
        att_goals = scale('goals_for_per_match_adjusted', team_metrics)
        def_xgc = scale('xgc_per_match_adjusted', team_metrics, invert=True)
        def_gc = scale('goals_against_per_match_adjusted', team_metrics, invert=True)
        def_cs = scale('clean_sheet_rate_adjusted', team_metrics)
        attack_scores = {tid: round(0.6 * att_xg[tid] + 0.4 * att_goals[tid], 1) for tid in all_team_ids}
        defense_scores = {tid: round(0.40 * def_xgc[tid] + 0.35 * def_gc[tid] + 0.25 * def_cs[tid], 1) for tid in all_team_ids}

        # Rebuild the Home vs Away simulator venue factors using only prior finished fixtures.
        prior_fixtures = [
            f for f in (fixtures_data or [])
            if f.get('event') is not None and int(f['event']) < target_gw
            and (f.get('finished') or f.get('finished_provisional'))
        ]
        venue = {}
        for team_id in all_team_ids:
            home_rows = [f for f in prior_fixtures if int(f.get('team_h', -1)) == team_id]
            away_rows = [f for f in prior_fixtures if int(f.get('team_a', -1)) == team_id]
            def avg(rows, field):
                return sum(float(f.get(field, 0) or 0) for f in rows) / max(len(rows), 1)
            home_gf, away_gf = avg(home_rows, 'team_h_score'), avg(away_rows, 'team_a_score')
            home_ga, away_ga = avg(home_rows, 'team_a_score'), avg(away_rows, 'team_h_score')
            home_pts = sum(
                3 if int(f.get('team_h_score', 0) or 0) > int(f.get('team_a_score', 0) or 0)
                else 1 if int(f.get('team_h_score', 0) or 0) == int(f.get('team_a_score', 0) or 0) else 0
                for f in home_rows
            ) / max(len(home_rows), 1)
            away_pts = sum(
                3 if int(f.get('team_a_score', 0) or 0) > int(f.get('team_h_score', 0) or 0)
                else 1 if int(f.get('team_a_score', 0) or 0) == int(f.get('team_h_score', 0) or 0) else 0
                for f in away_rows
            ) / max(len(away_rows), 1)
            perf_diff = (home_pts - away_pts) * 0.15 + (home_gf - away_gf) * 0.10 + (away_ga - home_ga) * 0.08
            venue[team_id] = {
                'att_boost': float(np.clip(4.0 + perf_diff * 4.0, 1.5, 9.0)),
                'def_boost': float(np.clip(4.0 + (away_ga - home_ga) * 3.0, 1.0, 8.0)),
            }

        def likely_score(lambda_home, lambda_away):
            lh, la, rho = max(0.1, float(lambda_home)), max(0.1, float(lambda_away)), -0.06
            matrix = np.zeros((6, 6), dtype=float)
            for home_goals in range(6):
                for away_goals in range(6):
                    adjustment = 1.0
                    if (home_goals, away_goals) == (0, 0): adjustment = 1.0 - lh * la * rho
                    elif (home_goals, away_goals) == (0, 1): adjustment = 1.0 + lh * rho
                    elif (home_goals, away_goals) == (1, 0): adjustment = 1.0 + la * rho
                    elif (home_goals, away_goals) == (1, 1): adjustment = 1.0 - rho
                    matrix[home_goals, away_goals] = max(
                        0.0, poisson.pmf(home_goals, lh) * poisson.pmf(away_goals, la) * adjustment
                    )
            matrix /= max(float(matrix.sum()), 1e-12)
            return tuple(int(value) for value in np.unravel_index(np.argmax(matrix), matrix.shape))

        for fixture in fixtures_by_gw.get(target_gw, []):
            home_id, away_id = int(fixture['team_h']), int(fixture['team_a'])
            home_venue = venue.get(home_id, {'att_boost': 4.5, 'def_boost': 4.0})
            away_venue = venue.get(away_id, {'att_boost': 4.5, 'def_boost': 4.0})
            home_edge = attack_scores.get(home_id, 50.0) + home_venue['att_boost'] - (
                defense_scores.get(away_id, 50.0) - away_venue['def_boost'] * 0.6
            )
            away_edge = (attack_scores.get(away_id, 50.0) - away_venue['att_boost'] * 0.6) - (
                defense_scores.get(home_id, 50.0) + home_venue['def_boost']
            )
            score = likely_score(1.48 * np.exp(home_edge / 40.0), 1.22 * np.exp(away_edge / 40.0))
            result[(target_gw, home_id)] = {
                'diff_attack': round(attack_scores.get(home_id, 50.0) - defense_scores.get(away_id, 50.0), 1),
                'diff_defense': round(defense_scores.get(home_id, 50.0) - attack_scores.get(away_id, 50.0), 1),
                'opponent_id': away_id,
                'is_home': True,
                'likely_team_goals': score[0],
                'likely_opponent_goals': score[1],
            }
            result[(target_gw, away_id)] = {
                'diff_attack': round(attack_scores.get(away_id, 50.0) - defense_scores.get(home_id, 50.0), 1),
                'diff_defense': round(defense_scores.get(away_id, 50.0) - attack_scores.get(home_id, 50.0), 1),
                'opponent_id': home_id,
                'is_home': False,
                'likely_team_goals': score[1],
                'likely_opponent_goals': score[0],
            }
    return result

@st.cache_data(ttl=3600)
def process_players(fpl_data, fdr_summary, _models_dict, _opt_b_models=None, histories_by_player_id=None):
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

    # Gunakan fitur rolling pra-pertandingan yang sama dengan Option B training.
    target_gw_by_player = {}
    for el in elements:
        upcoming = fdr_summary.get(el.get('team'), {}).get('upcoming_all', [])
        if upcoming and upcoming[0].get('gw') is not None:
            target_gw_by_player[el['id']] = int(upcoming[0]['gw'])
    model_features_map = compute_all_l5m_option_b_features(elements, target_gw_by_player, histories_by_player_id)

    processed = []
    for el in elements:
        t_id = el.get('team')
        f_info = fdr_summary.get(t_id, {})
        fdr1 = f_info.get('FDR1', 3.0)
        fdr3 = f_info.get('FDR3', 3.0)
        fdr5 = f_info.get('FDR5', 3.0)
        fdr10 = f_info.get('FDR10', 3.0)
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
        influence = float(el.get('influence', 0.0) or 0.0)
        creativity = float(el.get('creativity', 0.0) or 0.0)
        threat = float(el.get('threat', 0.0) or 0.0)
        bps = int(el.get('bps', 0))
        bonus = int(el.get('bonus', 0))
        goals = int(el.get('goals_scored', 0))
        assists = int(el.get('assists', 0))
        cs = int(el.get('clean_sheets', 0))
        gc = int(el.get('goals_conceded', 0) or 0)
        pos_id = el.get('element_type', 1)
        pos_name = POSITION_MAP.get(pos_id, "MID")
        
        # Damped effective minutes denominator (minimum 90 minutes baseline)
        # to eliminate extreme small-sample rate distortions (e.g. 1 min cameos having 90 DC/90 or 225 ICT/90)
        eff_mins = max(float(mins), 90.0)

        # Per 90 stats
        if mins > 0:
            xg90 = min(2.5, (xg / eff_mins) * 90.0)
            xa90 = min(2.0, (xa / eff_mins) * 90.0)
            xgi90 = min(3.5, (xgi / eff_mins) * 90.0)
            xgc90 = min(4.0, (xgc / eff_mins) * 90.0)
            saves90 = min(10.0, (saves / eff_mins) * 90.0)
            bps90 = min(50.0, (bps / eff_mins) * 90.0)
            ict90 = min(25.0, (ict / eff_mins) * 90.0)
            influence90 = min(120.0, (influence / eff_mins) * 90.0)
            threat90 = min(150.0, (threat / eff_mins) * 90.0)
            creativity90 = min(150.0, (creativity / eff_mins) * 90.0)
        else:
            xg90 = 0.0
            xa90 = 0.0
            xgi90 = 0.0
            xgc90 = 1.35
            saves90 = 0.0
            bps90 = 0.0
            ict90 = 0.0
            influence90 = 0.0
            threat90 = 0.0
            creativity90 = 0.0

        tackles = float(el.get('tackles', 0) or 0)
        cbi = float(el.get('clearances_blocks_interceptions', el.get('clearances', 0)) or 0)
        clearances = cbi
        interceptions = 0
        recoveries = float(el.get('recoveries', 0) or 0)
        raw_def_contrib = float(el.get('defensive_contribution', 0) or 0)
        tot_def_actions = (tackles + cbi + recoveries) if (tackles + cbi + recoveries) > 0 else raw_def_contrib
        def_contrib_90 = min(20.0, (tot_def_actions / eff_mins) * 90.0) if mins > 0 else 0.0
        tackles_90 = min(10.0, (tackles / eff_mins) * 90.0) if mins > 0 else 0.0
        cbi_90 = min(15.0, (cbi / eff_mins) * 90.0) if mins > 0 else 0.0
        clearances_90 = cbi_90
        recoveries_90 = min(15.0, (recoveries / eff_mins) * 90.0) if mins > 0 else 0.0

        # L5M Average Minutes
        rolling_features = model_features_map.get(el['id'], {})
        avg_mins_l5m = rolling_features.get('avg_mins', 0.0)

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
            'FDR10': fdr10,
            'Next_Is_Home': next_is_home,
            'Lawan GW Berikutnya': next_opp_fmt,
            'Opponent_xGC_per_90': opp_xg_def,
            'Opponent_xG_per_90_attack': opp_xg_att,
            'Form': form,
            'option_b_form_prior': rolling_features.get('form', 0.0),
            'option_b_threat_per_90': rolling_features.get('threat_per_90', 0.0),
            'option_b_started_prev': rolling_features.get('started_prev', 0),
            'option_b_creativity_per_90': rolling_features.get('creativity_per_90', 0.0),
            '% Ownership': float(el.get('selected_by_percent', 0.0) or 0.0),
            'Net Transfers GW': int(el.get('transfers_in_event', 0)) - int(el.get('transfers_out_event', 0)),
            'Transfers In GW': int(el.get('transfers_in_event', 0)),
            'Transfers Out GW': int(el.get('transfers_out_event', 0)),
            'xG': xg,
            'xA': xa,
            'xGI': xgi,
            'xGC': round(xgc, 2),
            'expected_goals_conceded': round(xgc, 2),
            'xG per 90': round(xg90, 2),
            'xA per 90': round(xa90, 2),
            'xGI per 90': round(xgi90, 2),
            'xGC per 90': round(xgc90, 2),
            'Saves per 90': round(saves90, 2),
            'Defensive Contribution per 90': round(def_contrib_90, 2),
            'threat_per_90': round(threat90, 2),
            'creativity_per_90': round(creativity90, 2),
            'influence_per_90': round(influence90, 2),
            'ICT Index': ict,
            'Influence': influence,
            'Creativity': creativity,
            'Threat': threat,
            'Tackles': int(tackles),
            'Tackles per 90': round(tackles_90, 2),
            'CBI': int(cbi),
            'CBI per 90': round(cbi_90, 2),
            'Interceptions': int(interceptions),
            'Clearances': int(clearances),
            'Recoveries': int(recoveries),
            'Recoveries per 90': round(recoveries_90, 2),
            'Defensive Contribution': round(tot_def_actions, 1),
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
            'Kebobolan': gc,
            'goals_conceded': gc,
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
                
        X_pos = pd.DataFrame(feature_data, index=sub_df.index).fillna(0.0)
        preds = model.predict(X_pos)
        
        # Penyesuaian Peluang Main (Chance of Playing Next Round)
        chance_arr = sub_df['Peluang Main GW (%)'].values / 100.0
        preds = preds * chance_arr
        preds = np.nan_to_num(preds, nan=0.0)
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

        # Option B memakai fitur pra-laga rolling yang sama dengan data latih.
        pos_m_xg = opt_b_xg.get(pos_key) if isinstance(opt_b_xg, dict) else opt_b_xg
        home_col = sub_df['Next_Is_Home'] if 'Next_Is_Home' in sub_df.columns else pd.Series(1, index=sub_df.index)
        form_col = sub_df['option_b_form_prior'] if 'option_b_form_prior' in sub_df.columns else pd.Series(0.0, index=sub_df.index)
        threat_col = sub_df['option_b_threat_per_90'] if 'option_b_threat_per_90' in sub_df.columns else pd.Series(0.0, index=sub_df.index)
        started_col = sub_df['option_b_started_prev'] if 'option_b_started_prev' in sub_df.columns else pd.Series(0, index=sub_df.index)
        likely_team_goals = sub_df.get('option_b_likely_team_goals', pd.Series(0.0, index=sub_df.index))
        likely_opponent_goals = sub_df.get('option_b_likely_opponent_goals', pd.Series(0.0, index=sub_df.index))
        fdr_col = sub_df['FDR1'] if 'FDR1' in sub_df.columns else pd.Series(3.0, index=sub_df.index)
        diff_att_col = sub_df['Diff Attack Team'] if 'Diff Attack Team' in sub_df.columns else pd.Series(0.0, index=sub_df.index)

        if pos_m_xg is not None:
            X_xg = pd.DataFrame({
                'form_points': form_col,
                'minutes_l5': sub_df['Avg Mins (L5M)'],
                'started_prev': started_col,
                'threat_per90_l5': threat_col,
                'attack_diff': diff_att_col,
                'is_home': home_col,
                'likely_team_goals': likely_team_goals,
                'likely_opponent_goals': likely_opponent_goals,
            }, index=sub_df.index).fillna(0.0)
            raw_xg_match[pos_mask.values] = pos_m_xg.predict(X_xg)
        else:
            raw_xg_match[pos_mask.values] = (threat_col / 50.0 * (1.1 if home_col.mean() == 1 else 0.9)).values

        # Tidak memakai is_setpiece_taker: riwayat status per GW tidak tersedia.
        pos_m_xa = opt_b_xa.get(pos_key) if isinstance(opt_b_xa, dict) else opt_b_xa
        creativity_col = sub_df['option_b_creativity_per_90'] if 'option_b_creativity_per_90' in sub_df.columns else pd.Series(0.0, index=sub_df.index)

        if pos_m_xa is not None:
            X_xa = pd.DataFrame({
                'was_home': home_col,
                'form': form_col,
                'creativity_per_90': creativity_col,
                'FDR': fdr_col,
                'Diff Attack Team': diff_att_col,
                'position': pos_key,
            }, index=sub_df.index).fillna(0.0)
            raw_xa_match[pos_mask.values] = pos_m_xa.predict(X_xa)
        else:
            raw_xa_match[pos_mask.values] = (creativity_col / 60.0 * (1.1 if home_col.mean() == 1 else 0.9)).values

    # KHUSUS UNTUK POSISI 'GK' (Kiper): Paksa (hardcode) nilai raw_xg_match = 0.0 dan raw_xa_match = 0.0
    gk_mask = (df['Posisi'] == 'GK')
    raw_xg_match[gk_mask] = 0.0
    raw_xa_match[gk_mask] = 0.0

    mins_ratio = df['Avg Mins (L5M)'] / 90.0
    expected_mins_ratio = mins_ratio * (df['Peluang Main GW (%)'].fillna(100.0) / 100.0)
    df['xG Pred (90)'] = np.maximum(0.0, raw_xg_match).round(3)
    df['xG Pred (Match)'] = (np.maximum(0.0, raw_xg_match) * mins_ratio).round(2)
    df['xA Pred (90)'] = np.maximum(0.0, raw_xa_match).round(3)
    df['xA Pred (Match)'] = (np.maximum(0.0, raw_xa_match) * expected_mins_ratio).round(2)

    # Component Calculations:
    # a. xMins_Pts
    chance_factor = df['Peluang Main GW (%)'] / 100.0
    avg_mins_l5m = df['Avg Mins (L5M)']
    df['xMins Pts'] = np.where(
        avg_mins_l5m >= 60.0, 2.0 * chance_factor,
        np.where(avg_mins_l5m > 0.0, 1.0 * chance_factor, 0.0)
    ).round(2)

    # b. xG_Pts & xA_Pts with Normalized Points Contribution Formula:
    # Weighted Multiplier Posisi dihitung OTOMATIS secara dinamis berbasis riwayat
    # agregasi rasio performa aktual: sum(Gol)/sum(xG) dan sum(Asis)/sum(xA) per posisi (FWD, MID, DEF, GK).
    # Formula Kontribusi Poin Ternormalisasi:
    # xG Pts = xG Pred (Match) * Poin Gol Posisi * Pos_xG_Multiplier
    # xA Pts = xA Pred (Match) * 3.0 * Pos_xA_Multiplier
    poin_gol_map = {'GK': 10.0, 'DEF': 6.0, 'MID': 5.0, 'FWD': 4.0}
    poin_gol = df['Posisi'].map(poin_gol_map).fillna(4.0)
    
    auto_xg_mult, auto_xa_mult, _ = compute_automatic_position_multipliers(df)
    pos_xg_mult = compute_automatic_player_xg_multipliers(df, fallback_xg=auto_xg_mult)
    pos_xa_mult = df['Posisi'].map(auto_xa_mult).fillna(1.0)

    df['xG Mult'] = pos_xg_mult.round(2)
    df['xA Mult'] = pos_xa_mult.round(2)
    df['xG Pts'] = (df['xG Pred (Match)'] * poin_gol * pos_xg_mult).round(2)
    df['xA Pts'] = (df['xA Pred (Match)'] * 3.0 * pos_xa_mult).round(2)

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
    # Menggunakan Model Bivariat Dixon-Coles / Poisson (P(Y_opp = 0)) pada level tim 90 menit penuh
    # Peluang CS (%) merefleksikan probabilitas clean sheet klub sehingga seragam untuk seluruh pemain di klub yang sama.
    # xCS Pts memperhitungkan syarat FPL minimal 60 menit bermain.
    poin_cs_map = {'GK': 4.0, 'DEF': 4.0, 'MID': 1.0, 'FWD': 0.0}
    poin_cs = df['Posisi'].map(poin_cs_map).fillna(0.0)
    base_opp_xg = df['Opponent_xG_per_90_attack'] if 'Opponent_xG_per_90_attack' in df.columns else pd.Series(1.35, index=df.index)
    base_team_xg = df['xG per 90'] if 'xG per 90' in df.columns else pd.Series(1.35, index=df.index)
    
    # Hitung probabilitas CS klub 90 menit penuh tanpa modulasi menit individual
    team_prob_cs = bivariate_dixon_coles_cs_prob(base_team_xg, base_opp_xg, rho=-0.06)
    df['Peluang CS (%)'] = np.round(team_prob_cs * 100.0, 1)
    
    # Syarat FPL: Pemain harus bermain >= 60 menit untuk meraih poin Clean Sheet
    # Probabilitas bermain >= 60 menit diperkirakan dari histori menit bermain (Avg Mins L5M)
    p_60_mins = np.clip((df['Avg Mins (L5M)'] - 30.0) / 30.0, 0.0, 1.0)
    raw_xcs = team_prob_cs * poin_cs * p_60_mins
    df['xCS Pts'] = np.where(df['Avg Mins (L5M)'] >= 60.0, raw_xcs, 0.0).round(2)

    # Inisialisasi default variabel differential tim
    df['Diff Attack Team'] = 0.0
    df['Diff Defense Team'] = 0.0

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
        'Diff Attack Team', 'Diff Defense Team', 'Peluang CS (%)',
        'xG Mult', 'xA Mult',
        'xG Pred (90)', 'xG Pred (Match)', 'xA Pred (90)', 'xA Pred (Match)', 'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
        'Avg Mins (L5M)', 'Total Poin', 'FDR1', 'FDR3', 'FDR5', 'FDR10', 'Form', '% Ownership', 'Net Transfers GW',
        'Transfers In GW', 'Transfers Out GW',
        'xG', 'xA', 'xGI', 'xGC', 'expected_goals_conceded', 'xG per 90', 'xA per 90', 'xGI per 90',
        'xGC per 90', 'Saves per 90', 'Defensive Contribution per 90',
        'ICT Index', 'Influence', 'Creativity', 'Threat',
        'influence_per_90', 'creativity_per_90', 'threat_per_90',
        'Tackles', 'Tackles per 90', 'CBI', 'CBI per 90', 'Clearances', 'Recoveries', 'Recoveries per 90', 'Interceptions',
        'Defensive Contribution',
        'BPS', 'Bonus Poin', 'Kartu Kuning', 'Kartu Merah', 'Saves',
        'Penalti Order', 'Free Kick Order', 'Corner Order', 'Status',
        'Peluang Main GW (%)', 'Berita Cedera', 'Menit Bermain', 'Gol', 'Asis',
        'Clean Sheet', 'Kebobolan', 'goals_conceded', 'Nama Lengkap', 'Poin per £m', 'xPoin per £m', 'Kemudahan Jadwal',
        'Opponent_xGC_per_90', 'Opponent_xG_per_90_attack', 'Next_Is_Home', 'is_setpiece_taker',
        'raw_threat90', 'raw_creativity90', 'raw_xg90', 'raw_xa90', 'raw_xgc90', 'raw_saves90',
        'raw_bps90', 'raw_ict90', 'bps_per_90_calc', 'xDC_calc'
    ]
    
    valid_cols = [c for c in cols if c in df.columns]
    return df[valid_cols], team_dict


def apply_team_differentials_and_recalc_option_b(players_df: pd.DataFrame, team_diffs_map: dict, _opt_b_models: tuple = None, custom_multipliers: tuple = None) -> pd.DataFrame:
    """
    Menyematkan 'Diff Attack Team' dan 'Diff Defense Team' ke setiap pemain berdasarkan klubnya,
    kemudian merekalibrasi model Option B (xG Pred, xA Pred, xDC Pts, xCS Pts, xBP, dan xPoin)
    menggunakan Weighted Multiplier Posisi yang dihitung OTOMATIS dari data histori performa aktual.
    
    Fitur Model Option B: venue, form prior L5, Threat/90 atau Creativity/90 prior L5, FDR, dan differential pra-GW.
    """
    if players_df.empty:
        return players_df

    df = players_df.copy()

    if not team_diffs_map:
        if 'Diff Attack Team' not in df.columns:
            df['Diff Attack Team'] = 0.0
        if 'Diff Defense Team' not in df.columns:
            df['Diff Defense Team'] = 0.0
        return df

    # Lookup sekali per pemain tanpa membangun Series untuk setiap baris
    # seperti yang dilakukan DataFrame.apply(axis=1).
    club_values = df['Klub'] if 'Klub' in df.columns else pd.Series([None] * len(df), index=df.index)
    team_values = df['team'] if 'team' in df.columns else pd.Series([None] * len(df), index=df.index)

    def get_diff_info(c_name, t_id):
        info = team_diffs_map.get(c_name) or team_diffs_map.get(str(c_name)) or team_diffs_map.get(t_id) or team_diffs_map.get(str(t_id), {})
        if isinstance(info, dict):
            return info
        return {}

    diff_infos = [get_diff_info(club, team) for club, team in zip(club_values, team_values)]
    df['Diff Attack Team'] = [float(info.get('diff_attack', 0.0)) for info in diff_infos]
    df['Diff Defense Team'] = [float(info.get('diff_defense', 0.0)) for info in diff_infos]
    df['option_b_likely_team_goals'] = [float(info.get('likely_team_goals', 0.0)) for info in diff_infos]
    df['option_b_likely_opponent_goals'] = [float(info.get('likely_opponent_goals', 0.0)) for info in diff_infos]

    mins_ratio = df['Avg Mins (L5M)'] / 90.0
    expected_mins_ratio = mins_ratio * (df['Peluang Main GW (%)'].fillna(100.0) / 100.0)

    # 1. Prediksi xG Pred (Match) dan xA Pred (Match) menggunakan Model Regresi Option B atau penyesuaian faktor
    if _opt_b_models is not None and any(m is not None for m in _opt_b_models):
        opt_b_xg, opt_b_xa = _opt_b_models
        raw_xg_match = np.zeros(len(df))
        raw_xa_match = np.zeros(len(df))

        was_home = df['Next_Is_Home'] if 'Next_Is_Home' in df.columns else pd.Series(1, index=df.index)
        form_series = df['option_b_form_prior'] if 'option_b_form_prior' in df.columns else pd.Series(0.0, index=df.index)
        threat_series = df['option_b_threat_per_90'] if 'option_b_threat_per_90' in df.columns else pd.Series(0.0, index=df.index)
        creativity_series = df['option_b_creativity_per_90'] if 'option_b_creativity_per_90' in df.columns else pd.Series(0.0, index=df.index)
        fdr_series = df['FDR1'] if 'FDR1' in df.columns else pd.Series(3.0, index=df.index)
        diff_att_series = df['Diff Attack Team'] if 'Diff Attack Team' in df.columns else pd.Series(0.0, index=df.index)

        for pos_key in ['FWD', 'MID', 'DEF']:
            pos_mask = (df['Posisi'] == pos_key)
            if not pos_mask.any():
                continue
            sub_df = df[pos_mask]
            sub_idx = sub_df.index

            pos_m_xg = opt_b_xg.get(pos_key) if isinstance(opt_b_xg, dict) else opt_b_xg
            if pos_m_xg is not None:
                X_xg = pd.DataFrame({
                    'form_points': form_series.loc[sub_idx],
                    'minutes_l5': sub_df['Avg Mins (L5M)'],
                    'started_prev': sub_df.get('option_b_started_prev', pd.Series(0, index=sub_idx)),
                    'threat_per90_l5': threat_series.loc[sub_idx],
                    'attack_diff': diff_att_series.loc[sub_idx],
                    'is_home': was_home.loc[sub_idx],
                    'likely_team_goals': sub_df.get('option_b_likely_team_goals', pd.Series(0.0, index=sub_idx)),
                    'likely_opponent_goals': sub_df.get('option_b_likely_opponent_goals', pd.Series(0.0, index=sub_idx)),
                }, index=sub_idx).fillna(0.0)
                raw_xg_match[pos_mask.values] = pos_m_xg.predict(X_xg)
            else:
                raw_xg_match[pos_mask.values] = (threat_series.loc[sub_idx] / 50.0).values

            pos_m_xa = opt_b_xa.get(pos_key) if isinstance(opt_b_xa, dict) else opt_b_xa
            if pos_m_xa is not None:
                X_xa = pd.DataFrame({
                    'was_home': was_home.loc[sub_idx],
                    'form': form_series.loc[sub_idx],
                    'creativity_per_90': creativity_series.loc[sub_idx],
                    'FDR': fdr_series.loc[sub_idx],
                    'Diff Attack Team': diff_att_series.loc[sub_idx],
                    'position': pos_key,
                }, index=sub_idx).fillna(0.0)
                raw_xa_match[pos_mask.values] = pos_m_xa.predict(X_xa)
            else:
                raw_xa_match[pos_mask.values] = (creativity_series.loc[sub_idx] / 60.0).values

        gk_mask = (df['Posisi'] == 'GK')
        raw_xg_match[gk_mask] = 0.0
        raw_xa_match[gk_mask] = 0.0

        df['xG Pred (90)'] = np.maximum(0.0, raw_xg_match).round(3)
        df['xG Pred (Match)'] = (np.maximum(0.0, raw_xg_match) * mins_ratio).round(2)
        df['xA Pred (90)'] = np.maximum(0.0, raw_xa_match).round(3)
        df['xA Pred (Match)'] = (np.maximum(0.0, raw_xa_match) * expected_mins_ratio).round(2)
    else:
        att_diff_factor = np.clip(1.0 + (df['Diff Attack Team'] / 100.0) * 0.5, 0.6, 1.6)
        gk_mask = (df['Posisi'] == 'GK')
        df['xG Pred (Match)'] = np.where(
            gk_mask,
            0.0,
            (df['xG Pred (Match)'] * att_diff_factor).round(2)
        )
        df['xA Pred (Match)'] = np.where(
            gk_mask,
            0.0,
            (df['xA Pred (Match)'] * att_diff_factor).round(2)
        )
        df['xA Pred (90)'] = np.divide(
            df['xA Pred (Match)'], expected_mins_ratio,
            out=np.zeros(len(df), dtype=float), where=expected_mins_ratio.to_numpy() > 0,
        ).round(3)

    if _opt_b_models is None or not any(m is not None for m in _opt_b_models):
        df['xG Pred (90)'] = np.divide(
            df['xG Pred (Match)'], mins_ratio,
            out=np.zeros(len(df), dtype=float), where=mins_ratio.to_numpy() > 0,
        ).round(3)

    # 2. xG Pts dan xA Pts with Normalized Points Contribution Formula:
    # Menggunakan Weighted Multiplier Posisi yang dihitung OTOMATIS secara dinamis dari histori xG vs G dan xA vs A
    # (atau kustom jika ditentukan secara eksplisit):
    # Formula Kontribusi Poin Ternormalisasi:
    # xG Pts = xG Pred (Match) * Poin Gol Posisi * Pos_xG_Multiplier
    # xA Pts = xA Pred (Match) * 3.0 * Pos_xA_Multiplier
    poin_gol_map = {'GK': 10.0, 'DEF': 6.0, 'MID': 5.0, 'FWD': 4.0}
    poin_gol = df['Posisi'].map(poin_gol_map).fillna(4.0)

    custom_xg_multiplier = bool(custom_multipliers and len(custom_multipliers) == 2)
    if custom_xg_multiplier:
        active_xg_mult, active_xa_mult = custom_multipliers
    else:
        active_xg_mult, active_xa_mult, _ = compute_automatic_position_multipliers(df)

    pos_xg_mult = (
        df['Posisi'].map(active_xg_mult).fillna(1.0)
        if custom_xg_multiplier
        else compute_automatic_player_xg_multipliers(df, fallback_xg=active_xg_mult)
    )
    pos_xa_mult = df['Posisi'].map(active_xa_mult).fillna(1.0)

    df['xG Mult'] = pos_xg_mult.round(2)
    df['xA Mult'] = pos_xa_mult.round(2)
    df['xG Pts'] = (df['xG Pred (Match)'] * poin_gol * pos_xg_mult).round(2)
    df['xA Pts'] = (df['xA Pred (Match)'] * 3.0 * pos_xa_mult).round(2)

    # 3. xDC Pts (Defensive Contribution Points)
    # Jika Diff Defense negatif (serangan lawan sangat kuat), aksi defensif pemain (tackle, blok, clearance) meningkat
    dc_thresh_map = {'DEF': 10, 'MID': 12, 'FWD': 12, 'GK': 0}
    thresholds = df['Posisi'].map(dc_thresh_map).fillna(0)
    dc_pressure_factor = np.clip(1.0 - (df['Diff Defense Team'] / 100.0) * 0.3, 0.7, 1.4)
    dc_col = df['Defensive Contribution per 90'] if 'Defensive Contribution per 90' in df.columns else (df['xDC_calc'] if 'xDC_calc' in df.columns else pd.Series(0.0, index=df.index))
    dc_pts = []
    for mu_base, t_val, p_factor in zip(dc_col * mins_ratio, thresholds, dc_pressure_factor):
        if t_val > 0 and mu_base > 0:
            mu_adj = mu_base * p_factor
            prob = 1.0 - float(poisson.cdf(t_val - 1, mu_adj))
            val = float(np.clip(2.0 * prob, 0.0, 2.0))
        else:
            val = 0.0
        dc_pts.append(val)
    df['xDC Pts'] = np.array(dc_pts).round(2)

    # 4. xCS Pts (Clean Sheet Points) menggunakan Model Bivariat Dixon-Coles / Poisson (P(Y_opp = 0)) pada level tim 90 menit penuh
    # Peluang Clean Sheet (%) merefleksikan probabilitas klub (seragam per klub pada matchday tersebut).
    # xCS Pts memperhitungkan probabilitas bermain >= 60 menit sesuai regulasi FPL.
    poin_cs_map = {'GK': 4.0, 'DEF': 4.0, 'MID': 1.0, 'FWD': 0.0}
    poin_cs = df['Posisi'].map(poin_cs_map).fillna(0.0)
    
    diff_def = df['Diff Defense Team'] if 'Diff Defense Team' in df.columns else pd.Series(0.0, index=df.index)
    diff_att = df['Diff Attack Team'] if 'Diff Attack Team' in df.columns else pd.Series(0.0, index=df.index)
    
    # Base goals Premier League per match ~ 1.35
    lambda_opp = 1.35 * np.exp(-diff_def / 45.0)
    lambda_team = 1.35 * np.exp(diff_att / 45.0)
    
    # Hitung probabilitas CS tim 90 menit penuh tanpa modulasi menit individual
    team_prob_cs = bivariate_dixon_coles_cs_prob(lambda_team, lambda_opp, rho=-0.06)
    df['Peluang CS (%)'] = np.round(team_prob_cs * 100.0, 1)
    
    avg_mins_col = df['Avg Mins (L5M)'] if 'Avg Mins (L5M)' in df.columns else pd.Series(90.0, index=df.index)
    # Probabilitas bermain >= 60 menit diperkirakan dari konsistensi menit bermain pemain
    p_60_mins = np.clip((avg_mins_col - 30.0) / 30.0, 0.0, 1.0)
    raw_xcs = team_prob_cs * poin_cs * p_60_mins
    df['xCS Pts'] = np.where(avg_mins_col >= 60.0, raw_xcs, 0.0).round(2)

    # 5. xBP (Bonus Points)
    if 'bps_per_90_calc' in df.columns:
        exp_bps_match = df['bps_per_90_calc'] * mins_ratio
    elif 'raw_bps90' in df.columns:
        exp_bps_match = df['raw_bps90'] * mins_ratio
    elif 'BPS' in df.columns:
        exp_bps_match = df['BPS'] / 90.0 * mins_ratio
    else:
        exp_bps_match = pd.Series(15.0, index=df.index) * mins_ratio
    raw_xbp = (exp_bps_match * 0.02) + ((df['xG Pred (Match)'] + df['xA Pred (Match)']) * 0.5)
    df['xBP'] = np.clip(raw_xbp, 0.0, 3.0).round(2)

    # 6. Total Option B xPoin
    mins_pts = df['xMins Pts'] if 'xMins Pts' in df.columns else np.clip(avg_mins_col / 45.0, 0.0, 2.0).round(2)
    saves_pts = df['xSaves Pts'] if 'xSaves Pts' in df.columns else pd.Series(0.0, index=df.index)
    df['xPoin (Option B)'] = (
        mins_pts + df['xG Pts'] + df['xA Pts'] + 
        saves_pts + df['xDC Pts'] + df['xCS Pts'] + df['xBP']
    ).round(2)

    return df

