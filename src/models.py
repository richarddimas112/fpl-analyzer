"""
Machine Learning Training Models: Option A (Positional Regression), Option B (Match Components), Classical Assumption Diagnostics, and Option C.
"""

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor

from src.constants import POS_MODEL_CONFIGS, POSITION_MAP, STATUS_MAP
from src.api import fetch_player_history_raw, fetch_player_history

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
def train_option_b_models(players_list, fdr_summary, current_gw, df_historical, _df_teams=None, fixtures_data=None):
    """
    Train separate Linear Regression models for xG and xA match-level prediction per position (FWD, MID, DEF).
    Model xG Features: Opponent_xGC_per_90, was_home, form, thread_per_90, FDR, Diff Attack Team
    Model xA Features: Opponent_xGC_per_90, was_home, is_setpiece_taker, form, creativity_per_90, FDR, Diff Attack Team
    """
    opt_b_models_xg = {}
    opt_b_models_xa = {}
    stats_xg = {}
    stats_xa = {}

    target_positions = ['FWD', 'MID', 'DEF']

    # Pre-build fixture FDR mapping
    fixture_fdr_map = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_fdr_map[(f_id, True)] = float(f.get('team_h_difficulty', 3.0))
                fixture_fdr_map[(f_id, False)] = float(f.get('team_a_difficulty', 3.0))

    # Pre-build team strength dictionary to optimize calculation
    team_att_dict = {}
    team_def_dict = {}
    if _df_teams is not None and not _df_teams.empty:
        for _, row in _df_teams.iterrows():
            t_id = row.get('team_id')
            if t_id:
                team_att_dict[t_id] = float(row.get('Skor Serangan', 50.0))
                team_def_dict[t_id] = float(row.get('Skor Pertahanan', 50.0))

    for pos_key in target_positions:
        cfg = POS_MODEL_CONFIGS[pos_key]
        pos_el_type = cfg['element_type']

        pos_players = [p for p in players_list if p.get('element_type') == pos_el_type]
        top_pos_players = sorted(pos_players, key=lambda p: (p.get('total_points', 0), p.get('minutes', 0)), reverse=True)[:30]

        rows_xg = []
        rows_xa = []

        for p in top_pos_players:
            p_form = float(p.get('form', 0.0) or 0.0)
            corner_ord = p.get('corners_and_indirect_freekicks_order')
            fk_ord = p.get('direct_freekicks_order')
            is_sp = check_setpiece_taker(corner_ord, fk_ord)
            p_team_id = p.get('team')

            p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: m.get('round', m.get('event', 0)))
                for m in sorted_hist:
                    mins = int(m.get('minutes', 0))
                    if mins > 0:
                        was_home = 1 if m.get('was_home') else 0
                        f_id = m.get('fixture')
                        opp_id = m.get('opponent_team', 1)
                        opp_fdr_info = fdr_summary.get(opp_id, {})

                        # Ambil FDR resmi match historis dari fixtures_data
                        fdr_val = fixture_fdr_map.get((f_id, bool(m.get('was_home'))))
                        if fdr_val is None:
                            fdr_val = float(m.get('difficulty') or opp_fdr_info.get('FDR1', 3.0))

                        opp_xgc90 = float(fdr_val) / 2.22
                        
                        threat_val = float(m.get('threat', 0.0) or 0.0)
                        threat90 = (threat_val / mins) * 90.0
                        
                        creativity_val = float(m.get('creativity', 0.0) or 0.0)
                        creativity90 = (creativity_val / mins) * 90.0

                        actual_xg = float(m.get('expected_goals', 0.0) or 0.0)
                        actual_xa = float(m.get('expected_assists', 0.0) or 0.0)

                        player_name = p.get('web_name', f"Pemain {p.get('id')}")

                        # Hitung Diff Attack Team setiap laga historis dengan modulasi venue matchday
                        diff_attack_val = 0.0
                        if team_att_dict and team_def_dict:
                            p_att = team_att_dict.get(p_team_id, 50.0) + (4.0 if was_home else -4.0)
                            opp_def = team_def_dict.get(opp_id, 50.0) + (-4.0 if was_home else 4.0)
                            diff_attack_val = round(p_att - opp_def, 1)
                        else:
                            diff_attack_val = round((3.5 - fdr_val) * 12.0 + (5.0 if was_home else -5.0), 1)

                        rows_xg.append({
                            'player_name': player_name,
                            'Opponent_xGC_per_90': opp_xgc90,
                            'was_home': was_home,
                            'form': p_form,
                            'thread_per_90': threat90,
                            'FDR': fdr_val,
                            'Diff Attack Team': diff_attack_val,
                            'actual_xg': actual_xg
                        })

                        rows_xa.append({
                            'player_name': player_name,
                            'Opponent_xGC_per_90': opp_xgc90,
                            'was_home': was_home,
                            'is_setpiece_taker': is_sp,
                            'form': p_form,
                            'creativity_per_90': creativity90,
                            'FDR': fdr_val,
                            'Diff Attack Team': diff_attack_val,
                            'actual_xa': actual_xa
                        })

        if len(rows_xg) >= 20:
            df_xg_train = pd.DataFrame(rows_xg)
            df_xa_train = pd.DataFrame(rows_xa)
        else:
            np.random.seed(101 + pos_el_type)
            N = 400
            opp_xgc_s = np.random.uniform(0.7, 2.3, size=N)
            home_s = np.random.choice([0, 1], size=N)
            form_s = np.random.uniform(0.5, 8.5, size=N)
            sp_s = np.random.choice([0, 1], p=[0.75, 0.25], size=N)
            threat90_s = np.random.uniform(5.0, 60.0 if pos_key in ['FWD', 'MID'] else 15.0, size=N)
            creativity90_s = np.random.uniform(5.0, 70.0 if pos_key in ['FWD', 'MID'] else 20.0, size=N)
            fdr_s = np.random.uniform(1.0, 5.0, size=N)
            diff_att_s = np.random.uniform(-35.0, 35.0, size=N)

            noise_xg = np.random.normal(0, 0.04, size=N)
            noise_xa = np.random.normal(0, 0.03, size=N)

            actual_xg_s = np.maximum(0.0, (
                0.08 * opp_xgc_s + 
                0.06 * home_s + 
                0.02 * form_s + 
                0.0035 * threat90_s - 
                0.02 * (fdr_s - 3.0) + 
                0.003 * diff_att_s + 
                noise_xg
            ))
            actual_xa_s = np.maximum(0.0, (
                0.06 * opp_xgc_s + 
                0.05 * home_s + 
                0.09 * sp_s + 
                0.015 * form_s + 
                0.003 * creativity90_s - 
                0.02 * (fdr_s - 3.0) + 
                0.0025 * diff_att_s + 
                noise_xa
            ))

            df_xg_train = pd.DataFrame({
                'player_name': [f"Simulated {pos_key} {i+1}" for i in range(N)],
                'Opponent_xGC_per_90': opp_xgc_s,
                'was_home': home_s,
                'form': form_s,
                'thread_per_90': threat90_s,
                'FDR': fdr_s,
                'Diff Attack Team': diff_att_s,
                'actual_xg': actual_xg_s
            })

            df_xa_train = pd.DataFrame({
                'player_name': [f"Simulated {pos_key} {i+1}" for i in range(N)],
                'Opponent_xGC_per_90': opp_xgc_s,
                'was_home': home_s,
                'is_setpiece_taker': sp_s,
                'form': form_s,
                'creativity_per_90': creativity90_s,
                'FDR': fdr_s,
                'Diff Attack Team': diff_att_s,
                'actual_xa': actual_xa_s
            })

        # --- INCREMENTAL TRAINING LOGIC (OPTION B) ---
        if current_gw <= 10 and not df_historical.empty:
            hist_pos = df_historical[df_historical['element_type'] == pos_el_type].copy()
            if not hist_pos.empty:
                if 'Diff Attack Team' not in hist_pos.columns:
                    hist_pos['Diff Attack Team'] = 0.0
                req_xg = ['Opponent_xGC_per_90', 'was_home', 'form', 'thread_per_90', 'FDR', 'Diff Attack Team', 'actual_xg']
                if all(c in hist_pos.columns for c in req_xg):
                    df_xg_train = pd.concat([df_xg_train, hist_pos[req_xg]], ignore_index=True)
                
                req_xa = ['Opponent_xGC_per_90', 'was_home', 'is_setpiece_taker', 'form', 'creativity_per_90', 'FDR', 'Diff Attack Team', 'actual_xa']
                if all(c in hist_pos.columns for c in req_xa):
                    df_xa_train = pd.concat([df_xa_train, hist_pos[req_xa]], ignore_index=True)

        # Fit Model xG: Opponent_xGC_per_90, was_home, form, thread_per_90, FDR, Diff Attack Team
        feature_cols_xg = ['Opponent_xGC_per_90', 'was_home', 'form', 'thread_per_90', 'FDR', 'Diff Attack Team']
        X_xg = df_xg_train[feature_cols_xg]
        y_xg = df_xg_train['actual_xg']
        model_xg = LinearRegression()
        model_xg.fit(X_xg, y_xg)
        pred_xg = model_xg.predict(X_xg)
        r2_xg = round(r2_score(y_xg, pred_xg), 4)
        mae_xg = round(mean_absolute_error(y_xg, pred_xg), 4)

        opt_b_models_xg[pos_key] = model_xg

        # Fit Model xA: Opponent_xGC_per_90, was_home, is_setpiece_taker, form, creativity_per_90, FDR, Diff Attack Team
        feature_cols_xa = ['Opponent_xGC_per_90', 'was_home', 'is_setpiece_taker', 'form', 'creativity_per_90', 'FDR', 'Diff Attack Team']
        X_xa = df_xa_train[feature_cols_xa]
        y_xa = df_xa_train['actual_xa']
        model_xa = LinearRegression()
        model_xa.fit(X_xa, y_xa)
        pred_xa = model_xa.predict(X_xa)
        r2_xa = round(r2_score(y_xa, pred_xa), 4)
        mae_xa = round(mean_absolute_error(y_xa, pred_xa), 4)

        opt_b_models_xa[pos_key] = model_xa

        # Evaluation DataFrames for Streamlit UI Inspection
        eval_df_xg = pd.DataFrame({
            'Pemain': df_xg_train['player_name'] if 'player_name' in df_xg_train else f"{pos_key} Sample",
            'Lawan xGC/90': df_xg_train['Opponent_xGC_per_90'].round(2),
            'Home': df_xg_train['was_home'],
            'Form': df_xg_train['form'].round(1),
            'Threat/90': df_xg_train['thread_per_90'].round(2),
            'FDR': df_xg_train['FDR'].round(1),
            'Diff Attack Team': df_xg_train['Diff Attack Team'].round(1),
            'y_actual': df_xg_train['actual_xg'].round(2),
            'y_predicted': np.round(pred_xg, 2),
            'residual (e)': np.round(df_xg_train['actual_xg'] - pred_xg, 2)
        }).sort_values(by='y_actual', ascending=False).head(10)

        eval_df_xa = pd.DataFrame({
            'Pemain': df_xa_train['player_name'] if 'player_name' in df_xa_train else f"{pos_key} Sample",
            'Lawan xGC/90': df_xa_train['Opponent_xGC_per_90'].round(2),
            'Home': df_xa_train['was_home'],
            'SetPiece': df_xa_train['is_setpiece_taker'],
            'Form': df_xa_train['form'].round(1),
            'Creativity/90': df_xa_train['creativity_per_90'].round(2),
            'FDR': df_xa_train['FDR'].round(1),
            'Diff Attack Team': df_xa_train['Diff Attack Team'].round(1),
            'y_actual': df_xa_train['actual_xa'].round(2),
            'y_predicted': np.round(pred_xa, 2),
            'residual (e)': np.round(df_xa_train['actual_xa'] - pred_xa, 2)
        }).sort_values(by='y_actual', ascending=False).head(10)

        # Standardized Beta Coefficients & Importance Calculation
        # Beta_std = Beta_raw * (std_X / std_y) -> mengukur kontribusi relatif variabel
        std_y_xg = float(np.std(y_xg)) if float(np.std(y_xg)) > 0 else 1.0
        std_x_xg = np.array([float(np.std(X_xg[col])) if float(np.std(X_xg[col])) > 0 else 1.0 for col in X_xg.columns])
        beta_std_xg = model_xg.coef_ * (std_x_xg / std_y_xg)
        abs_beta_xg = np.abs(beta_std_xg)
        total_abs_xg = np.sum(abs_beta_xg) if np.sum(abs_beta_xg) > 0 else 1.0
        pct_importance_xg = np.round((abs_beta_xg / total_abs_xg) * 100.0, 1)

        coef_df_xg = pd.DataFrame({
            'Variabel Fitur': list(X_xg.columns),
            'Koefisien Mentah (β)': np.round(model_xg.coef_, 4),
            'Beta Standar (Std β)': np.round(beta_std_xg, 4),
            'Kontribusi Relatif (%)': pct_importance_xg,
            'Arah Pengaruh': ['Positif (+)' if c > 0 else ('Negatif (-)' if c < 0 else 'Netral') for c in model_xg.coef_]
        }).sort_values(by='Kontribusi Relatif (%)', ascending=False)

        top_feature_xg = coef_df_xg.iloc[0]['Variabel Fitur'] if not coef_df_xg.empty else '-'
        top_contrib_pct_xg = coef_df_xg.iloc[0]['Kontribusi Relatif (%)'] if not coef_df_xg.empty else 0.0

        std_y_xa = float(np.std(y_xa)) if float(np.std(y_xa)) > 0 else 1.0
        std_x_xa = np.array([float(np.std(X_xa[col])) if float(np.std(X_xa[col])) > 0 else 1.0 for col in X_xa.columns])
        beta_std_xa = model_xa.coef_ * (std_x_xa / std_y_xa)
        abs_beta_xa = np.abs(beta_std_xa)
        total_abs_xa = np.sum(abs_beta_xa) if np.sum(abs_beta_xa) > 0 else 1.0
        pct_importance_xa = np.round((abs_beta_xa / total_abs_xa) * 100.0, 1)

        coef_df_xa = pd.DataFrame({
            'Variabel Fitur': list(X_xa.columns),
            'Koefisien Mentah (β)': np.round(model_xa.coef_, 4),
            'Beta Standar (Std β)': np.round(beta_std_xa, 4),
            'Kontribusi Relatif (%)': pct_importance_xa,
            'Arah Pengaruh': ['Positif (+)' if c > 0 else ('Negatif (-)' if c < 0 else 'Netral') for c in model_xa.coef_]
        }).sort_values(by='Kontribusi Relatif (%)', ascending=False)

        top_feature_xa = coef_df_xa.iloc[0]['Variabel Fitur'] if not coef_df_xa.empty else '-'
        top_contrib_pct_xa = coef_df_xa.iloc[0]['Kontribusi Relatif (%)'] if not coef_df_xa.empty else 0.0

        stats_xg[pos_key] = {
            'r2': r2_xg,
            'mae': mae_xg,
            'intercept': round(float(model_xg.intercept_), 4),
            'coef_df': coef_df_xg,
            'top_feature': top_feature_xg,
            'top_pct': top_contrib_pct_xg,
            'eval_df': eval_df_xg
        }

        stats_xa[pos_key] = {
            'r2': r2_xa,
            'mae': mae_xa,
            'intercept': round(float(model_xa.intercept_), 4),
            'coef_df': coef_df_xa,
            'top_feature': top_feature_xa,
            'top_pct': top_contrib_pct_xa,
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
def train_xpoints_model(players_list, fdr_summary, current_gw, df_historical, fixtures_data=None):
    models_dict = {}

    # Pre-build fixture FDR mapping
    fixture_fdr_map = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_fdr_map[(f_id, True)] = float(f.get('team_h_difficulty', 3.0))
                fixture_fdr_map[(f_id, False)] = float(f.get('team_a_difficulty', 3.0))

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
                        eff_m_mins = max(float(mins), 60.0)
                        prev_5 = sorted_hist[max(0, i-4):i+1]
                        avg_mins_l5m = sum(int(x.get('minutes', 0)) for x in prev_5) / float(len(prev_5))

                        xg90 = min(2.5, (float(m.get('expected_goals', 0.0)) / eff_m_mins) * 90.0)
                        xa90 = min(2.0, (float(m.get('expected_assists', 0.0)) / eff_m_mins) * 90.0)
                        xgc90 = min(4.0, (float(m.get('expected_goals_conceded', 0.0)) / eff_m_mins) * 90.0)
                        saves90 = min(10.0, (float(m.get('saves', 0)) / eff_m_mins) * 90.0)
                        bps90 = min(50.0, (float(m.get('bps', 0)) / eff_m_mins) * 90.0)
                        ict90 = min(25.0, (float(m.get('ict_index', 0.0)) / eff_m_mins) * 90.0)
                        
                        tackles = float(m.get('tackles', 0))
                        interceptions = float(m.get('interceptions', 0))
                        clearances = float(m.get('clearances_blocks_interceptions', m.get('clearances', 0)))
                        recoveries = float(m.get('recoveries', 0))
                        tot_def_actions = tackles + interceptions + clearances + recoveries
                        def_contrib_90 = min(20.0, (tot_def_actions / eff_m_mins) * 90.0) if tot_def_actions > 0 else min(10.0, p_def_contrib)

                        was_home = 1 if m.get('was_home') else 0
                        f_id = m.get('fixture')
                        opp_id = m.get('opponent_team', 1)
                        opp_fdr_info = fdr_summary.get(opp_id, {})
                        
                        # Fix bug FDR: Ekstraksi nilai resmi FDR (1-5), bukan opponent_team ID (1-20)
                        fdr_val = fixture_fdr_map.get((f_id, bool(m.get('was_home'))))
                        if fdr_val is None:
                            fdr_val = float(m.get('difficulty') or opp_fdr_info.get('FDR1', 3.0))

                        history_rows.append({
                            'xG_per_90': xg90,
                            'xA_per_90': xa90,
                            'bps_per_90': bps90,
                            'form': p_form,
                            'was_home': was_home,
                            'FDR': fdr_val,
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

def run_purged_walk_forward_cv(df_pos, feature_cols, target_col='target_points'):
    """
    Menjalankan Purged Walk-Forward Time-Series Cross Validation.
    Data diurutkan secara temporal berdasarkan 'round' (Gameweek).
    Pada setiap fold:
      Train: round <= t (hanya informasi masa lampau)
      Test: round == t + 1 (strictly out-of-sample forward step)
    Menghitung MAE, RMSE, dan R2 out-of-sample murni tanpa lookahead leakage.
    """
    unique_rounds = sorted([int(r) for r in df_pos['round'].dropna().unique()])
    if len(unique_rounds) < 2:
        return None, []
        
    fold_details = []
    oof_preds_lr = []
    oof_preds_rd = []
    oof_preds_gb = []
    oof_preds_ens = []
    oof_y_true = []
    
    for t in unique_rounds[:-1]:
        train_df = df_pos[df_pos['round'] <= t]
        test_df = df_pos[df_pos['round'] == t + 1]
        
        if len(train_df) < 5 or len(test_df) < 2:
            continue
            
        X_tr = train_df[feature_cols].fillna(0.0)
        y_tr = train_df[target_col].fillna(0.0)
        X_te = test_df[feature_cols].fillna(0.0)
        y_te = test_df[target_col].fillna(0.0)
        
        m_lr = LinearRegression()
        m_rd = Ridge(alpha=1.0)
        m_gb = GradientBoostingRegressor(n_estimators=60, learning_rate=0.05, max_depth=3, random_state=42)
        
        m_lr.fit(X_tr, y_tr)
        m_rd.fit(X_tr, y_tr)
        m_gb.fit(X_tr, y_tr)
        
        p_lr = np.clip(m_lr.predict(X_te), 0.0, 24.0)
        p_rd = np.clip(m_rd.predict(X_te), 0.0, 24.0)
        p_gb = np.clip(m_gb.predict(X_te), 0.0, 24.0)
        p_ens = (0.40 * p_gb) + (0.35 * p_rd) + (0.25 * p_lr)
        
        f_mae_ens = mean_absolute_error(y_te, p_ens)
        f_rmse_ens = np.sqrt(np.mean((y_te - p_ens) ** 2))
        
        fold_details.append({
            'fold': f"GW 1-{t} → GW {t+1}",
            'train_info': f"GW <= {t} ({len(train_df)} sampel)",
            'test_info': f"GW {t+1} ({len(test_df)} sampel)",
            'mae_ens': round(float(f_mae_ens), 4),
            'rmse_ens': round(float(f_rmse_ens), 4),
            'mae_gb': round(float(mean_absolute_error(y_te, p_gb)), 4),
            'mae_rd': round(float(mean_absolute_error(y_te, p_rd)), 4),
            'mae_lr': round(float(mean_absolute_error(y_te, p_lr)), 4)
        })
        
        oof_preds_lr.extend(p_lr.tolist())
        oof_preds_rd.extend(p_rd.tolist())
        oof_preds_gb.extend(p_gb.tolist())
        oof_preds_ens.extend(p_ens.tolist())
        oof_y_true.extend(y_te.tolist())
        
    if not oof_y_true:
        return None, []
        
    y_true_arr = np.array(oof_y_true)
    cv_summary = {
        'Linear Regression': {
            'mae': round(float(mean_absolute_error(y_true_arr, oof_preds_lr)), 4),
            'rmse': round(float(np.sqrt(np.mean((y_true_arr - np.array(oof_preds_lr)) ** 2))), 4),
            'r2': round(float(r2_score(y_true_arr, oof_preds_lr)), 4)
        },
        'Ridge Regression': {
            'mae': round(float(mean_absolute_error(y_true_arr, oof_preds_rd)), 4),
            'rmse': round(float(np.sqrt(np.mean((y_true_arr - np.array(oof_preds_rd)) ** 2))), 4),
            'r2': round(float(r2_score(y_true_arr, oof_preds_rd)), 4)
        },
        'Gradient Boosting': {
            'mae': round(float(mean_absolute_error(y_true_arr, oof_preds_gb)), 4),
            'rmse': round(float(np.sqrt(np.mean((y_true_arr - np.array(oof_preds_gb)) ** 2))), 4),
            'r2': round(float(r2_score(y_true_arr, oof_preds_gb)), 4)
        },
        'Ensemble': {
            'mae': round(float(mean_absolute_error(y_true_arr, oof_preds_ens)), 4),
            'rmse': round(float(np.sqrt(np.mean((y_true_arr - np.array(oof_preds_ens)) ** 2))), 4),
            'r2': round(float(r2_score(y_true_arr, oof_preds_ens)), 4)
        }
    }
    return cv_summary, fold_details

@st.cache_data(ttl=86400)
def build_option_c_model_and_view(fpl_data, fdr_summary, current_gw, fixtures_data=None):
    """
    Mengambil data match history dari seluruh pemain aktif di musim berjalan (Current Season Only),
    mengekstraksi fitur rolling (form L3M, minutes L5M, rolling xG/xA/xGC, Home/Away, FDR lawan),
    kemudian melatih model terpisah per posisi (FWD, MID, DEF, GK) dengan 3 algoritma Machine Learning:
    1. Multiple Linear Regression
    2. Ridge Regression (L2 Regularization)
    3. Gradient Boosting Regressor (Tree-based Non-linear Ensemble)
    
    Mengevaluasi model menggunakan Purged Walk-Forward Time-Series Cross Validation (tanpa lookahead leakage)
    dan menghasilkan prediksi xPoin GW selanjutnya.
    """
    elements = fpl_data.get('elements', [])
    teams = fpl_data.get('teams', [])
    teams_dict = {t['id']: t['name'] for t in teams}

    # Pre-build fixture FDR mapping
    fixture_fdr_map = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_fdr_map[(f_id, True)] = float(f.get('team_h_difficulty', 3.0))
                fixture_fdr_map[(f_id, False)] = float(f.get('team_a_difficulty', 3.0))

    # Ambil pemain aktif yang telah bermain minimal 1 menit
    active_elements = [el for el in elements if int(el.get('minutes', 0)) > 0]
    
    # Ambil data histori secara concurrent
    def fetch_player_dataset(el):
        p_id = el['id']
        p_name = el.get('web_name', 'Unknown')
        p_type = el.get('element_type', 3)
        p_cost = el.get('now_cost', 50) / 10.0
        p_team = el.get('team', 1)
        p_chance = el.get('chance_of_playing_next_round')
        if p_chance is None:
            p_chance = 100 if el.get('status') == 'a' else (75 if el.get('status') == 'd' else 0)
        else:
            p_chance = int(p_chance)
            
        hist = fetch_player_history_raw(p_id)
        return {
            'id': p_id,
            'name': p_name,
            'type': p_type,
            'cost': p_cost,
            'team': p_team,
            'chance': p_chance,
            'history': hist
        }

    with ThreadPoolExecutor(max_workers=20) as executor:
        player_histories = list(executor.map(fetch_player_dataset, active_elements))

    # Bangun Dataset Baris Pertandingan (Match-Level Records)
    train_records = []
    current_prediction_features = []

    for p in player_histories:
        hist = p['history']
        p_type = p['type']
        p_id = p['id']
        p_name = p['name']
        p_team = p['team']
        p_cost = p['cost']
        p_chance = p['chance']

        if not hist:
            continue

        sorted_hist = sorted(hist, key=lambda m: m.get('round', m.get('event', 0)))
        
        # 1. Bangun Data Training dari Pertandingan yang Telah Selesai
        for i in range(len(sorted_hist)):
            m = sorted_hist[i]
            pts_actual = int(m.get('total_points', 0))
            m_round = int(m.get('round', m.get('event', 1)))
            
            # Hitung Rolling Features dari Laga-laga sebelumnya (lagged history)
            past_matches = sorted_hist[:i] # Pertandingan sebelum laga ini
            
            if len(past_matches) == 0:
                # Cold start untuk match pertama musim ini
                roll_mins_5 = float(m.get('minutes', 0))
                roll_pts_3 = float(pts_actual)
                roll_xg_3 = float(m.get('expected_goals', 0.0))
                roll_xa_3 = float(m.get('expected_assists', 0.0))
                roll_xgc_3 = float(m.get('expected_goals_conceded', 0.0))
                roll_bps_3 = float(m.get('bps', 0))
                roll_ict_3 = float(m.get('ict_index', 0.0))
            else:
                last_5 = past_matches[-5:]
                last_3 = past_matches[-3:]
                roll_mins_5 = sum(int(x.get('minutes', 0)) for x in last_5) / float(len(last_5))
                roll_pts_3 = sum(int(x.get('total_points', 0)) for x in last_3) / float(len(last_3))
                roll_xg_3 = sum(float(x.get('expected_goals', 0.0)) for x in last_3) / float(len(last_3))
                roll_xa_3 = sum(float(x.get('expected_assists', 0.0)) for x in last_3) / float(len(last_3))
                roll_xgc_3 = sum(float(x.get('expected_goals_conceded', 0.0)) for x in last_3) / float(len(last_3))
                roll_bps_3 = sum(int(x.get('bps', 0)) for x in last_3) / float(len(last_3))
                roll_ict_3 = sum(float(x.get('ict_index', 0.0)) for x in last_3) / float(len(last_3))

            was_home = 1 if m.get('was_home') else 0
            f_id = m.get('fixture')
            opp_id = m.get('opponent_team', 1)
            opp_fdr_info = fdr_summary.get(opp_id, {})
            
            fdr_match = fixture_fdr_map.get((f_id, bool(m.get('was_home'))))
            if fdr_match is None:
                fdr_match = float(m.get('difficulty') or opp_fdr_info.get('FDR1', 3.0))

            train_records.append({
                'player_id': p_id,
                'round': m_round,
                'element_type': p_type,
                'cost': p_cost,
                'was_home': was_home,
                'fdr': fdr_match,
                'roll_mins_5': roll_mins_5,
                'roll_pts_3': roll_pts_3,
                'roll_xg_3': roll_xg_3,
                'roll_xa_3': roll_xa_3,
                'roll_xgc_3': roll_xgc_3,
                'roll_bps_3': roll_bps_3,
                'roll_ict_3': roll_ict_3,
                'target_points': pts_actual
            })

        # 2. Bangun Fitur untuk Prediksi Gameweek Selanjutnya (Upcoming Match)
        f_info = fdr_summary.get(p_team, {})
        next_is_home = f_info.get('Next_Is_Home', 1)
        next_fdr = float(f_info.get('FDR1', 3.0))
        next_opp_name = f_info.get('Next_Opponent_Fmt', '-')
        
        last_5_all = sorted_hist[-5:]
        last_3_all = sorted_hist[-3:]
        
        curr_mins_5 = sum(int(x.get('minutes', 0)) for x in last_5_all) / float(len(last_5_all))
        curr_pts_3 = sum(int(x.get('total_points', 0)) for x in last_3_all) / float(len(last_3_all))
        curr_xg_3 = sum(float(x.get('expected_goals', 0.0)) for x in last_3_all) / float(len(last_3_all))
        curr_xa_3 = sum(float(x.get('expected_assists', 0.0)) for x in last_3_all) / float(len(last_3_all))
        curr_xgc_3 = sum(float(x.get('expected_goals_conceded', 0.0)) for x in last_3_all) / float(len(last_3_all))
        curr_bps_3 = sum(int(x.get('bps', 0)) for x in last_3_all) / float(len(last_3_all))
        curr_ict_3 = sum(float(x.get('ict_index', 0.0)) for x in last_3_all) / float(len(last_3_all))

        current_prediction_features.append({
            'id': p_id,
            'Nama Pemain': p_name,
            'Klub': teams_dict.get(p_team, '-'),
            'Posisi': POSITION_MAP.get(p_type, 'MID'),
            'Harga (£m)': p_cost,
            'Peluang Main GW (%)': p_chance,
            'Lawan GW Berikutnya': next_opp_name,
            'FDR1': next_fdr,
            'FDR3': float(f_info.get('FDR3', 3.0)),
            'FDR5': float(f_info.get('FDR5', 3.0)),
            'element_type': p_type,
            'cost': p_cost,
            'was_home': next_is_home,
            'fdr': next_fdr,
            'roll_mins_5': curr_mins_5,
            'roll_pts_3': curr_pts_3,
            'roll_xg_3': curr_xg_3,
            'roll_xa_3': curr_xa_3,
            'roll_xgc_3': curr_xgc_3,
            'roll_bps_3': curr_bps_3,
            'roll_ict_3': curr_ict_3
        })

    df_train_all = pd.DataFrame(train_records)
    df_pred_all = pd.DataFrame(current_prediction_features)

    if df_train_all.empty or len(df_train_all) < 30:
        return df_pred_all, {}

    # Konfigurasi Fitur Spesifik per Posisi (Tanpa parameter Harga/Cost)
    pos_configs = {
        'FWD': {
            'element_type': 4,
            'features': ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'roll_bps_3', 'roll_ict_3'],
            'labels': ['Home', 'FDR Lawan', 'Avg Menit L5M', 'Form Poin L3M', 'Avg xG L3M', 'Avg xA L3M', 'Avg BPS L3M', 'Avg ICT Index L3M']
        },
        'MID': {
            'element_type': 3,
            'features': ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'roll_xgc_3', 'roll_bps_3', 'roll_ict_3'],
            'labels': ['Home', 'FDR Lawan', 'Avg Menit L5M', 'Form Poin L3M', 'Avg xG L3M', 'Avg xA L3M', 'Avg xGC L3M', 'Avg BPS L3M', 'Avg ICT Index L3M']
        },
        'DEF': {
            'element_type': 2,
            'features': ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'roll_xgc_3', 'roll_bps_3', 'roll_ict_3'],
            'labels': ['Home', 'FDR Lawan', 'Avg Menit L5M', 'Form Poin L3M', 'Avg xG L3M', 'Avg xA L3M', 'Avg xGC L3M', 'Avg BPS L3M', 'Avg ICT Index L3M']
        },
        'GK': {
            'element_type': 1,
            'features': ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xgc_3', 'roll_bps_3'],
            'labels': ['Home', 'FDR Lawan', 'Avg Menit L5M', 'Form Poin L3M', 'Avg xGC L3M', 'Avg BPS L3M']
        }
    }

    # Kolom output prediksi
    df_pred_all['xPoin (Linear Reg)'] = 0.0
    df_pred_all['xPoin (Ridge Reg)'] = 0.0
    df_pred_all['xPoin (Gradient Boosting)'] = 0.0
    df_pred_all['xPoin (Option C Ensemble)'] = 0.0

    models_performance = {}

    # Latih model spesifik per posisi
    for pos_key, p_cfg in pos_configs.items():
        el_type = p_cfg['element_type']
        f_cols = p_cfg['features']
        f_labels = p_cfg['labels']

        pos_train = df_train_all[df_train_all['element_type'] == el_type]
        pos_pred = df_pred_all[df_pred_all['element_type'] == el_type]

        if pos_train.empty or len(pos_train) < 10:
            # Fallback ke seluruh dataset jika data posisi terlalu sedikit
            pos_train = df_train_all
            common_cols = [c for c in f_cols if c in pos_train.columns]
            f_cols = common_cols
            f_labels = [f_labels[i] for i, c in enumerate(p_cfg['features']) if c in common_cols]

        # 1. Jalankan Purged Walk-Forward Time-Series Cross Validation
        cv_summary, fold_details = run_purged_walk_forward_cv(pos_train, f_cols)

        # 2. Latih Model Final pada seluruh data historis posisi ini
        X_train = pos_train[f_cols].fillna(0.0)
        y_train = pos_train['target_points'].fillna(0.0)

        lr = LinearRegression()
        ridge = Ridge(alpha=1.0)
        gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42)

        lr.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        gbr.fit(X_train, y_train)

        # In-sample metrics
        p_lr_tr = lr.predict(X_train)
        p_rd_tr = ridge.predict(X_train)
        p_gb_tr = gbr.predict(X_train)
        p_ens_tr = (0.40 * p_gb_tr) + (0.35 * p_rd_tr) + (0.25 * p_lr_tr)

        in_sample = {
            'Linear Regression': {'mae': round(float(mean_absolute_error(y_train, p_lr_tr)), 4), 'rmse': round(float(np.sqrt(np.mean((y_train - p_lr_tr)**2))), 4), 'r2': round(float(r2_score(y_train, p_lr_tr)), 4)},
            'Ridge Regression': {'mae': round(float(mean_absolute_error(y_train, p_rd_tr)), 4), 'rmse': round(float(np.sqrt(np.mean((y_train - p_rd_tr)**2))), 4), 'r2': round(float(r2_score(y_train, p_rd_tr)), 4)},
            'Gradient Boosting': {'mae': round(float(mean_absolute_error(y_train, p_gb_tr)), 4), 'rmse': round(float(np.sqrt(np.mean((y_train - p_gb_tr)**2))), 4), 'r2': round(float(r2_score(y_train, p_gb_tr)), 4)},
            'Ensemble': {'mae': round(float(mean_absolute_error(y_train, p_ens_tr)), 4), 'rmse': round(float(np.sqrt(np.mean((y_train - p_ens_tr)**2))), 4), 'r2': round(float(r2_score(y_train, p_ens_tr)), 4)}
        }

        # Predict upcoming match jika ada pemain di posisi ini
        if not pos_pred.empty:
            X_up = pos_pred[f_cols].fillna(0.0)
            chance_factor = (pos_pred['Peluang Main GW (%)'] / 100.0).values

            p_lr_up = np.clip(lr.predict(X_up), 0.0, 24.0) * chance_factor
            p_rd_up = np.clip(ridge.predict(X_up), 0.0, 24.0) * chance_factor
            p_gb_up = np.clip(gbr.predict(X_up), 0.0, 24.0) * chance_factor
            p_ens_up = (0.40 * p_gb_up) + (0.35 * p_rd_up) + (0.25 * p_lr_up)

            sub_idx = pos_pred.index
            df_pred_all.loc[sub_idx, 'xPoin (Linear Reg)'] = np.round(p_lr_up, 2)
            df_pred_all.loc[sub_idx, 'xPoin (Ridge Reg)'] = np.round(p_rd_up, 2)
            df_pred_all.loc[sub_idx, 'xPoin (Gradient Boosting)'] = np.round(p_gb_up, 2)
            df_pred_all.loc[sub_idx, 'xPoin (Option C Ensemble)'] = np.round(p_ens_up, 2)

        # Feature Importance DataFrames
        imp_df = pd.DataFrame({
            'Fitur': f_labels,
            'Tingkat Kepentingan (%)': np.round(gbr.feature_importances_ * 100, 2)
        }).sort_values(by='Tingkat Kepentingan (%)', ascending=False)

        coef_df = pd.DataFrame({
            'Fitur': f_labels,
            'Bobot LR (β)': np.round(lr.coef_, 4),
            'Bobot Ridge (β)': np.round(ridge.coef_, 4)
        })

        models_performance[pos_key] = {
            'cv_metrics': cv_summary if cv_summary else in_sample,
            'in_sample': in_sample,
            'importance_df': imp_df,
            'coef_df': coef_df,
            'n_samples': len(pos_train),
            'folds_evaluated': len(fold_details),
            'fold_details': fold_details,
            'has_walk_forward_cv': bool(cv_summary)
        }

    # Summary keseluruhan untuk kompatibilitas view
    overall_f_cols = ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'roll_xgc_3', 'roll_bps_3', 'roll_ict_3']
    overall_labels = ['Laga Kandang (Home)', 'FDR Lawan Mendatang', 'Avg Menit L5M', 'Avg Poin L3M (Form)', 'Avg xG L3M', 'Avg xA L3M', 'Avg xGC L3M', 'Avg BPS L3M', 'Avg ICT Index L3M']
    
    cv_all, folds_all = run_purged_walk_forward_cv(df_train_all, overall_f_cols)
    
    # Global fallback model untuk interpretasi umum
    g_lr = LinearRegression().fit(df_train_all[overall_f_cols], df_train_all['target_points'])
    g_rd = Ridge(alpha=1.0).fit(df_train_all[overall_f_cols], df_train_all['target_points'])
    g_gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42).fit(df_train_all[overall_f_cols], df_train_all['target_points'])

    global_in_sample = {
        'Linear Regression': {'mae': round(float(mean_absolute_error(df_train_all['target_points'], g_lr.predict(df_train_all[overall_f_cols]))), 4), 'rmse': round(float(np.sqrt(np.mean((df_train_all['target_points'] - g_lr.predict(df_train_all[overall_f_cols]))**2))), 4), 'r2': round(float(r2_score(df_train_all['target_points'], g_lr.predict(df_train_all[overall_f_cols]))), 4)},
        'Ridge Regression': {'mae': round(float(mean_absolute_error(df_train_all['target_points'], g_rd.predict(df_train_all[overall_f_cols]))), 4), 'rmse': round(float(np.sqrt(np.mean((df_train_all['target_points'] - g_rd.predict(df_train_all[overall_f_cols]))**2))), 4), 'r2': round(float(r2_score(df_train_all['target_points'], g_rd.predict(df_train_all[overall_f_cols]))), 4)},
        'Gradient Boosting': {'mae': round(float(mean_absolute_error(df_train_all['target_points'], g_gb.predict(df_train_all[overall_f_cols]))), 4), 'rmse': round(float(np.sqrt(np.mean((df_train_all['target_points'] - g_gb.predict(df_train_all[overall_f_cols]))**2))), 4), 'r2': round(float(r2_score(df_train_all['target_points'], g_gb.predict(df_train_all[overall_f_cols]))), 4)}
    }

    models_performance['ALL'] = {
        'cv_metrics': cv_all if cv_all else global_in_sample,
        'in_sample': global_in_sample,
        'importance_df': pd.DataFrame({'Fitur': overall_labels, 'Tingkat Kepentingan Fitur (%)': np.round(g_gb.feature_importances_ * 100, 2)}).sort_values(by='Tingkat Kepentingan Fitur (%)', ascending=False),
        'coef_df': pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_rd.coef_, 4)}),
        'n_samples': len(df_train_all),
        'folds_evaluated': len(folds_all),
        'fold_details': folds_all,
        'has_walk_forward_cv': bool(cv_all)
    }

    # Aliases untuk backwards compatibility
    models_performance['Linear Regression'] = cv_all['Linear Regression'] if cv_all else global_in_sample['Linear Regression']
    models_performance['Linear Regression']['coef_df'] = pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_lr.coef_, 4)})
    
    models_performance['Ridge Regression'] = cv_all['Ridge Regression'] if cv_all else global_in_sample['Ridge Regression']
    models_performance['Ridge Regression']['coef_df'] = pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_rd.coef_, 4)})
    
    models_performance['Gradient Boosting'] = cv_all['Gradient Boosting'] if cv_all else global_in_sample['Gradient Boosting']
    models_performance['Gradient Boosting']['importance_df'] = models_performance['ALL']['importance_df']

    return df_pred_all, models_performance
