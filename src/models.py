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
def train_option_b_models(players_list, fdr_summary, current_gw, df_historical):
    """
    Train separate Linear Regression models for xG and xA match-level prediction per position (FWD, MID, DEF).
    """
    opt_b_models_xg = {}
    opt_b_models_xa = {}
    stats_xg = {}
    stats_xa = {}

    target_positions = ['FWD', 'MID', 'DEF']

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

            p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: m.get('round', m.get('event', 0)))
                for m in sorted_hist:
                    mins = int(m.get('minutes', 0))
                    if mins > 0:
                        xg90 = (float(m.get('expected_goals', 0.0) or 0.0) / mins) * 90.0
                        xa90 = (float(m.get('expected_assists', 0.0) or 0.0) / mins) * 90.0
                        was_home = 1 if m.get('was_home') else 0
                        opp_id = m.get('opponent_team', 1)
                        opp_fdr_info = fdr_summary.get(opp_id, {})
                        opp_xgc90 = float(opp_fdr_info.get('FDR1', 3.0)) / 2.22
                        fdr_val = float(opp_fdr_info.get('FDR1', m.get('difficulty', 3.0)))
                        
                        threat_val = float(m.get('threat', 0.0) or 0.0)
                        threat90 = (threat_val / mins) * 90.0
                        
                        creativity_val = float(m.get('creativity', 0.0) or 0.0)
                        creativity90 = (creativity_val / mins) * 90.0

                        actual_xg = float(m.get('expected_goals', 0.0) or 0.0)
                        actual_xa = float(m.get('expected_assists', 0.0) or 0.0)

                        player_name = p.get('web_name', f"Pemain {p.get('id')}")

                        rows_xg.append({
                            'player_name': player_name,
                            'xG_per_90': xg90,
                            'Opponent_xGC_per_90': opp_xgc90,
                            'was_home': was_home,
                            'form': p_form,
                            'thread_per_90': threat90,
                            'FDR': fdr_val,
                            'actual_xg': actual_xg
                        })

                        rows_xa.append({
                            'player_name': player_name,
                            'xA_per_90': xa90,
                            'Opponent_xGC_per_90': opp_xgc90,
                            'was_home': was_home,
                            'is_setpiece_taker': is_sp,
                            'form': p_form,
                            'creativity_per_90': creativity90,
                            'FDR': fdr_val,
                            'actual_xa': actual_xa
                        })

        if len(rows_xg) >= 20:
            df_xg_train = pd.DataFrame(rows_xg)
            df_xa_train = pd.DataFrame(rows_xa)
        else:
            np.random.seed(101 + pos_el_type)
            N = 400
            xg90_s = np.random.exponential(0.35 if pos_key in ['FWD', 'MID'] else 0.08, size=N)
            xa90_s = np.random.exponential(0.25 if pos_key in ['FWD', 'MID'] else 0.10, size=N)
            opp_xgc_s = np.random.uniform(0.7, 2.3, size=N)
            home_s = np.random.choice([0, 1], size=N)
            form_s = np.random.uniform(0.5, 8.5, size=N)
            sp_s = np.random.choice([0, 1], p=[0.75, 0.25], size=N)
            threat90_s = np.random.uniform(5.0, 60.0 if pos_key in ['FWD', 'MID'] else 15.0, size=N)
            creativity90_s = np.random.uniform(5.0, 70.0 if pos_key in ['FWD', 'MID'] else 20.0, size=N)
            fdr_s = np.random.uniform(1.0, 5.0, size=N)

            noise_xg = np.random.normal(0, 0.05, size=N)
            noise_xa = np.random.normal(0, 0.04, size=N)

            actual_xg_s = np.maximum(0.0, (0.45 * xg90_s * (opp_xgc_s / 1.35) + 0.10 * home_s + 0.02 * form_s + 0.003 * threat90_s - 0.02 * (fdr_s - 3.0) + noise_xg))
            actual_xa_s = np.maximum(0.0, (0.40 * xa90_s * (opp_xgc_s / 1.35) + 0.08 * home_s + 0.12 * sp_s + 0.02 * form_s + 0.003 * creativity90_s - 0.02 * (fdr_s - 3.0) + noise_xa))

            df_xg_train = pd.DataFrame({
                'player_name': [f"Simulated {pos_key} {i+1}" for i in range(N)],
                'xG_per_90': xg90_s,
                'Opponent_xGC_per_90': opp_xgc_s,
                'was_home': home_s,
                'form': form_s,
                'thread_per_90': threat90_s,
                'FDR': fdr_s,
                'actual_xg': actual_xg_s
            })

            df_xa_train = pd.DataFrame({
                'player_name': [f"Simulated {pos_key} {i+1}" for i in range(N)],
                'xA_per_90': xa90_s,
                'Opponent_xGC_per_90': opp_xgc_s,
                'was_home': home_s,
                'is_setpiece_taker': sp_s,
                'form': form_s,
                'creativity_per_90': creativity90_s,
                'FDR': fdr_s,
                'actual_xa': actual_xa_s
            })

        # --- INCREMENTAL TRAINING LOGIC (OPTION B) ---
        if current_gw <= 10 and not df_historical.empty:
            hist_pos = df_historical[df_historical['element_type'] == pos_el_type]
            if not hist_pos.empty:
                req_xg = ['xG_per_90', 'Opponent_xGC_per_90', 'was_home', 'form', 'thread_per_90', 'FDR', 'actual_xg']
                if all(c in hist_pos.columns for c in req_xg):
                    df_xg_train = pd.concat([df_xg_train, hist_pos[req_xg]], ignore_index=True)
                
                req_xa = ['xA_per_90', 'Opponent_xGC_per_90', 'was_home', 'is_setpiece_taker', 'form', 'creativity_per_90', 'FDR', 'actual_xa']
                if all(c in hist_pos.columns for c in req_xa):
                    df_xa_train = pd.concat([df_xa_train, hist_pos[req_xa]], ignore_index=True)

        # Fit Model xG (FWD, MID, DEF include thread_per_90 and FDR)
        feature_cols_xg = ['xG_per_90', 'Opponent_xGC_per_90', 'was_home', 'form', 'thread_per_90', 'FDR']
        X_xg = df_xg_train[feature_cols_xg]
        y_xg = df_xg_train['actual_xg']
        model_xg = LinearRegression()
        model_xg.fit(X_xg, y_xg)
        pred_xg = model_xg.predict(X_xg)
        r2_xg = round(r2_score(y_xg, pred_xg), 4)
        mae_xg = round(mean_absolute_error(y_xg, pred_xg), 4)

        opt_b_models_xg[pos_key] = model_xg

        # Fit Model xA (FWD, MID, DEF include creativity_per_90 and FDR)
        feature_cols_xa = ['xA_per_90', 'Opponent_xGC_per_90', 'was_home', 'is_setpiece_taker', 'form', 'creativity_per_90', 'FDR']
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
            'xG/90': df_xg_train['xG_per_90'].round(2),
            'Threat/90': df_xg_train['thread_per_90'].round(2),
            'FDR': df_xg_train['FDR'].round(1),
            'Lawan xGC/90': df_xg_train['Opponent_xGC_per_90'].round(2),
            'Home': df_xg_train['was_home'],
            'Form': df_xg_train['form'].round(1),
            'y_actual': df_xg_train['actual_xg'].round(2),
            'y_predicted': np.round(pred_xg, 2),
            'residual (e)': np.round(df_xg_train['actual_xg'] - pred_xg, 2)
        }).sort_values(by='y_actual', ascending=False).head(10)

        eval_df_xa = pd.DataFrame({
            'Pemain': df_xa_train['player_name'] if 'player_name' in df_xa_train else f"{pos_key} Sample",
            'xA/90': df_xa_train['xA_per_90'].round(2),
            'Creativity/90': df_xa_train['creativity_per_90'].round(2),
            'FDR': df_xa_train['FDR'].round(1),
            'Lawan xGC/90': df_xa_train['Opponent_xGC_per_90'].round(2),
            'Home': df_xa_train['was_home'],
            'SetPiece': df_xa_train['is_setpiece_taker'],
            'Form': df_xa_train['form'].round(1),
            'y_actual': df_xa_train['actual_xa'].round(2),
            'y_predicted': np.round(pred_xa, 2),
            'residual (e)': np.round(df_xa_train['actual_xa'] - pred_xa, 2)
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

@st.cache_data(ttl=86400)
def build_option_c_model_and_view(fpl_data, fdr_summary, current_gw):
    """
    Mengambil data match history dari seluruh pemain aktif di musim berjalan (Current Season Only),
    mengekstraksi fitur rolling (form L3M, minutes L5M, rolling xG/xA/xGC, Home/Away, FDR lawan),
    kemudian melatih 3 algoritma Machine Learning:
    1. Multiple Linear Regression
    2. Ridge Regression (L2 Regularization)
    3. Gradient Boosting Regressor (Tree-based Non-linear Ensemble)
    
    Menghitung metrik performa (MAE, RMSE, R²) dan menghasilkan prediksi xPoin GW selanjutnya.
    """
    elements = fpl_data.get('elements', [])
    teams = fpl_data.get('teams', [])
    teams_dict = {t['id']: t['name'] for t in teams}

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
            opp_id = m.get('opponent_team', 1)
            opp_fdr_info = fdr_summary.get(opp_id, {})
            fdr_match = float(opp_fdr_info.get('FDR1', 3.0))

            train_records.append({
                'player_id': p_id,
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

    feature_cols = [
        'cost', 'was_home', 'fdr', 'roll_mins_5', 'roll_pts_3',
        'roll_xg_3', 'roll_xa_3', 'roll_xgc_3', 'roll_bps_3', 'roll_ict_3'
    ]

    feature_labels = [
        'Harga Pemain (£m)',
        'Laga Kandang (Home)',
        'FDR Lawan Mendatang',
        'Avg Menit L5M',
        'Avg Poin L3M (Form)',
        'Avg xG L3M',
        'Avg xA L3M',
        'Avg xGC L3M',
        'Avg BPS L3M',
        'Avg ICT Index L3M'
    ]

    models_performance = {}
    
    # 1. Model: Multiple Linear Regression
    lr = LinearRegression()
    # 2. Model: Ridge Regression (L2)
    ridge = Ridge(alpha=1.0)
    # 3. Model: Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

    X_train = df_train_all[feature_cols]
    y_train = df_train_all['target_points']
    X_upcoming = df_pred_all[feature_cols]

    # Training Linear Regression
    lr.fit(X_train, y_train)
    p_lr_train = lr.predict(X_train)
    p_lr_up = np.clip(lr.predict(X_upcoming), 0.0, 24.0) * (df_pred_all['Peluang Main GW (%)'] / 100.0)

    mae_lr = mean_absolute_error(y_train, p_lr_train)
    rmse_lr = np.sqrt(np.mean((y_train - p_lr_train) ** 2))
    r2_lr = r2_score(y_train, p_lr_train)

    models_performance['Linear Regression'] = {
        'mae': round(mae_lr, 4),
        'rmse': round(rmse_lr, 4),
        'r2': round(r2_lr, 4),
        'coef_df': pd.DataFrame({'Fitur': feature_labels, 'Bobot Koefisien (β)': np.round(lr.coef_, 4)})
    }

    # Training Ridge Regression
    ridge.fit(X_train, y_train)
    p_ridge_train = ridge.predict(X_train)
    p_ridge_up = np.clip(ridge.predict(X_upcoming), 0.0, 24.0) * (df_pred_all['Peluang Main GW (%)'] / 100.0)

    mae_rd = mean_absolute_error(y_train, p_ridge_train)
    rmse_rd = np.sqrt(np.mean((y_train - p_ridge_train) ** 2))
    r2_rd = r2_score(y_train, p_ridge_train)

    models_performance['Ridge Regression'] = {
        'mae': round(mae_rd, 4),
        'rmse': round(rmse_rd, 4),
        'r2': round(r2_rd, 4),
        'coef_df': pd.DataFrame({'Fitur': feature_labels, 'Bobot Koefisien (β)': np.round(ridge.coef_, 4)})
    }

    # Training Gradient Boosting Regressor
    gbr.fit(X_train, y_train)
    p_gbr_train = gbr.predict(X_train)
    p_gbr_up = np.clip(gbr.predict(X_upcoming), 0.0, 24.0) * (df_pred_all['Peluang Main GW (%)'] / 100.0)

    mae_gb = mean_absolute_error(y_train, p_gbr_train)
    rmse_gb = np.sqrt(np.mean((y_train - p_gbr_train) ** 2))
    r2_gb = r2_score(y_train, p_gbr_train)

    models_performance['Gradient Boosting'] = {
        'mae': round(mae_gb, 4),
        'rmse': round(rmse_gb, 4),
        'r2': round(r2_gb, 4),
        'importance_df': pd.DataFrame({'Fitur': feature_labels, 'Tingkat Kepentingan Fitur (%)': np.round(gbr.feature_importances_ * 100, 2)}).sort_values(by='Tingkat Kepentingan Fitur (%)', ascending=False)
    }

    # Assign Prediksi ke DataFrame Rangkuman
    df_pred_all['xPoin (Linear Reg)'] = np.round(p_lr_up, 2)
    df_pred_all['xPoin (Ridge Reg)'] = np.round(p_ridge_up, 2)
    df_pred_all['xPoin (Gradient Boosting)'] = np.round(p_gbr_up, 2)

    # Ensemble Weighted Average Prediction
    df_pred_all['xPoin (Option C Ensemble)'] = np.round(
        (0.35 * df_pred_all['xPoin (Gradient Boosting)']) + 
        (0.35 * df_pred_all['xPoin (Ridge Reg)']) + 
        (0.30 * df_pred_all['xPoin (Linear Reg)']), 
        2
    )

    return df_pred_all, models_performance
