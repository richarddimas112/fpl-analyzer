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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from concurrent.futures import ThreadPoolExecutor

from src.constants import POS_MODEL_CONFIGS, POSITION_MAP, STATUS_MAP
from src.api import fetch_player_history_raw, fetch_player_history
from src.processors import compute_team_differentials_asof

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

@st.cache_data(ttl=3600)
def train_option_b_models(players_list, fdr_summary, current_gw, df_historical, _df_teams=None, fixtures_data=None, histories_by_player_id=None):
    """
    Train separate xG/90 models and one pooled xA/90 model with position as a categorical feature.
    xA target rows require at least 30 minutes; all features are strictly pre-match.
    """
    opt_b_models_xg = {}
    opt_b_models_xa = {}
    stats_xg = {}
    stats_xa = {}

    target_positions = ['FWD', 'MID', 'DEF']

    if histories_by_player_id is None:
        histories_by_player_id = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {p['id']: executor.submit(fetch_player_history, p['id']) for p in players_list if p.get('id') is not None}
            for player_id, future in futures.items():
                try:
                    histories_by_player_id[int(player_id)] = future.result()
                except Exception:
                    histories_by_player_id[int(player_id)] = []

    # Pre-build fixture FDR mapping
    fixture_fdr_map = {}
    fixture_by_id = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_by_id[f_id] = f
                fixture_fdr_map[(f_id, True)] = float(f.get('team_h_difficulty', 3.0))
                fixture_fdr_map[(f_id, False)] = float(f.get('team_a_difficulty', 3.0))

    # Build all historical features from one as-of-GW team-strength calculation.
    all_fixture_gws = sorted({int(f['event']) for f in (fixtures_data or []) if f.get('event') is not None})
    team_diff_by_gw_team = compute_team_differentials_asof(
        players_list, fixtures_data, target_gws=all_fixture_gws,
        histories_by_player_id=histories_by_player_id,
    )
    position_threat_rate_by_gw = {}
    for target_gw in all_fixture_gws:
        for pos_key in target_positions:
            pos_el_type = POS_MODEL_CONFIGS[pos_key]['element_type']
            mins_sum = threat_sum = 0.0
            for player in players_list:
                if player.get('element_type') != pos_el_type:
                    continue
                history = histories_by_player_id.get(player['id'], histories_by_player_id.get(str(player['id']), []))
                for historical_match in history:
                    if int(historical_match.get('round', historical_match.get('event', 1)) or 1) < target_gw:
                        mins_sum += float(historical_match.get('minutes', 0) or 0)
                        threat_sum += float(historical_match.get('threat', 0) or 0)
            position_threat_rate_by_gw[(target_gw, pos_key)] = threat_sum * 90.0 / mins_sum if mins_sum > 0 else 0.0

    pooled_xa_rows = []

    for pos_key in target_positions:
        cfg = POS_MODEL_CONFIGS[pos_key]
        pos_el_type = cfg['element_type']

        # Avoid selecting historical training players using current-season minutes/points.
        # A target row is included only when its own historical match and prior features exist.
        pos_players = [p for p in players_list if p.get('element_type') == pos_el_type]

        rows_xg = []
        rows_xa = []

        for p in pos_players:
            if histories_by_player_id is not None:
                p_hist = histories_by_player_id.get(p['id'], histories_by_player_id.get(str(p['id']), []))
            else:
                p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: (int(m.get('round', m.get('event', 0)) or 0), m.get('kickoff_time', '')))
                for m in sorted_hist:
                    target_gw = int(m.get('round', m.get('event', 1)) or 1)
                    fixture = fixture_by_id.get(m.get('fixture'))
                    if fixture is None or not (fixture.get('finished') or fixture.get('finished_provisional')):
                        continue
                    target_gw = int(fixture.get('event') or target_gw)

                    # Bebas kebocoran: HANYA gunakan riwayat pertandingan sebelum Gameweek target (GW < target_gw).
                    # Mengeluarkan laga pada Gameweek yang sama (termasuk laga pertama pada DGW).
                    past_matches = [pm for pm in sorted_hist if int(pm.get('round', pm.get('event', 1)) or 1) < target_gw]
                    if len(past_matches) == 0:
                        continue

                    # Jendela 5 laga terakhir sebelum Gameweek target
                    w = past_matches[-5:]
                    tot_mins = sum(int(x.get('minutes', 0)) for x in w)
                    eff_mins = max(float(tot_mins), 60.0)

                    # Threat/90 dan Creativity/90 murni dari laga sebelum GW target
                    threat90 = float(min(150.0, max(0.0, (sum(float(x.get('threat', 0.0) or 0.0) for x in w) / eff_mins) * 90.0)))
                    creativity90 = float(min(150.0, max(0.0, (sum(float(x.get('creativity', 0.0) or 0.0) for x in w) / eff_mins) * 90.0)))

                    # Form historis sebelum laga target (bukan live form masa depan)
                    hist_form = sum(int(x.get('total_points', 0)) for x in w) / float(len(w))

                    was_home = 1 if m.get('was_home') else 0
                    f_id = m.get('fixture')
                    team_id = fixture.get('team_h') if was_home else fixture.get('team_a')
                    diff_info = team_diff_by_gw_team.get((target_gw, int(team_id))) if team_id is not None else None
                    if diff_info is None:
                        continue
                    diff_attack_val = float(diff_info['diff_attack'])
                    fdr_val = fixture_fdr_map.get((f_id, bool(m.get('was_home'))))
                    if fdr_val is None:
                        fdr_val = float(m.get('difficulty') or 3.0)

                    actual_xg = float(m.get('expected_goals', 0.0) or 0.0)
                    actual_xa = float(m.get('expected_assists', 0.0) or 0.0)

                    player_name = p.get('web_name', f"Pemain {p.get('id')}")

                    # xG target per 90 from this target match only; exclude cameo minutes.
                    target_minutes = int(m.get('minutes', 0) or 0)
                    if target_minutes >= 30:
                        threat_total = sum(float(x.get('threat', 0.0) or 0.0) for x in w)
                        position_rate = position_threat_rate_by_gw.get((target_gw, pos_key), 0.0)
                        shrunk_threat90 = (threat_total * 90.0 + position_rate * 270.0) / (tot_mins + 270.0)
                        likely_team_goals = float(diff_info.get('likely_team_goals', 0.0))
                        likely_opponent_goals = float(diff_info.get('likely_opponent_goals', 0.0))
                        rows_xg.append({
                            'player_name': player_name, 'round': target_gw, 'was_home': was_home,
                            'form_points': hist_form, 'minutes_l5': tot_mins / float(len(w)),
                            'started_prev': int(bool(past_matches[-1].get('minutes', 0) and int(past_matches[-1].get('minutes', 0)) >= 60)),
                            'threat_per90_l5': shrunk_threat90, 'attack_diff': diff_attack_val,
                            'is_home': was_home, 'likely_team_goals': likely_team_goals,
                            'likely_opponent_goals': likely_opponent_goals,
                            'target_minutes': target_minutes,
                            'actual_xg90': (actual_xg / target_minutes) * 90.0,
                        })

                    # Catatan: is_setpiece_taker dikeluarkan dari riwayat laga karena status pengambil bola mati historis
                    # tidak tersedia per gameweek masa lalu dan mengambil nilai live akan memicu kebocoran retrospektif.
                    xa_row = {
                        'player_id': p.get('id'),
                        'player_name': player_name,
                        'position': pos_key,
                        'round': target_gw,
                        'was_home': was_home,
                        'form': hist_form,
                        'creativity_per_90': creativity90,
                        'FDR': fdr_val,
                        'Diff Attack Team': diff_attack_val,
                        'target_minutes': target_minutes,
                        'actual_xa': actual_xa,
                        'actual_xa90': (actual_xa / target_minutes) * 90.0 if target_minutes >= 30 else np.nan
                    }
                    rows_xa.append(xa_row)
                    pooled_xa_rows.append(xa_row)

        # Data xG musim sebelumnya dengan schema lama tidak dicampur karena target/model sudah xG per 90.
        hist_rows_xg = []
        hist_rows_xa = []
        # Gabungkan CSV historis hanya bila schema-nya benar-benar sepadan dengan
        # fitur pra-laga ini. xG/90 bukan pengganti Threat/90, begitu pula xA/90
        # bukan pengganti Creativity/90.
        required_hist_cols = {
            'element_type', 'round', 'was_home', 'form', 'FDR',
            'thread_per_90', 'creativity_per_90', 'Diff Attack Team', 'actual_xg', 'actual_xa'
        }
        if not df_historical.empty and required_hist_cols.issubset(df_historical.columns):
            hist_pos = df_historical[df_historical['element_type'] == pos_el_type].copy()
            if not hist_pos.empty:
                for _, h_row in hist_pos.iterrows():
                    home_val = int(h_row.get('was_home', 0))
                    form_val = float(h_row.get('form', 1.0))
                    fdr_val = float(h_row.get('FDR', 3.0))
                    th90 = float(np.clip(h_row.get('thread_per_90', 0.0), 0.0, 150.0))
                    cr90 = float(np.clip(h_row.get('creativity_per_90', 0.0), 0.0, 150.0))
                    diff_att = float(h_row.get('Diff Attack Team', 0.0))
                    act_xg = float(h_row.get('actual_xg', 0.0))
                    act_xa = float(h_row.get('actual_xa', 0.0))

                    hist_rows_xa.append({
                        'player_name': 'Historical (Musim Lalu)',
                        'round': 0,
                        'was_home': home_val,
                        'form': form_val,
                        'creativity_per_90': cr90,
                        'FDR': fdr_val,
                        'Diff Attack Team': diff_att,
                        'actual_xa': act_xa
                    })

        # Gabungkan seluruh data empiris (data historis + musim berjalan bebas bocor)
        all_rows_xg = rows_xg
        all_rows_xa = hist_rows_xa + rows_xa

        df_xg_train = pd.DataFrame(all_rows_xg) if all_rows_xg else pd.DataFrame()
        df_xa_train = pd.DataFrame(all_rows_xa) if all_rows_xa else pd.DataFrame()

        feature_cols_xg = ['form_points', 'minutes_l5', 'started_prev', 'threat_per90_l5', 'attack_diff', 'is_home', 'likely_team_goals', 'likely_opponent_goals']
        feature_cols_xa = ['was_home', 'form', 'creativity_per_90', 'FDR', 'Diff Attack Team']

        # 2. Evaluasi Kronologis Out-of-Sample (Walk-Forward Expanding Window)
        eval_available_xg = False
        eval_available_xa = False
        mae_xg, rmse_xg, r2_xg = None, None, None
        mae_xa, rmse_xa, r2_xa = None, None, None
        eval_df_xg = pd.DataFrame()
        eval_df_xa = pd.DataFrame()

        y_test_xg, pred_test_xg, eval_rows_xg = [], [], []
        y_test_xa, pred_test_xa, eval_rows_xa = [], [], []

        if not df_xg_train.empty and not df_xa_train.empty:
            gws = sorted(df_xg_train['round'].unique())
            # Evaluasi pada gameweek uji (round > 0 atau round >= 2)
            test_gws = [t for t in gws if t > 0]
            if not test_gws and len(gws) >= 2:
                test_gws = gws[1:]

            for t in test_gws:
                train_mask_xg = (df_xg_train['round'] < t)
                test_mask_xg = (df_xg_train['round'] == t)
                y_tr_xg = df_xg_train.loc[train_mask_xg, 'actual_xg90']

                # Fold xG valid jika sampel latih >= 15, target tidak nol semua (std dev > 0 dan sum > 0)
                if train_mask_xg.sum() >= 15 and float(y_tr_xg.sum()) > 0 and float(np.std(y_tr_xg)) > 1e-4 and test_mask_xg.sum() > 0:
                    fold_m_xg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                    fold_m_xg.fit(df_xg_train.loc[train_mask_xg, feature_cols_xg], y_tr_xg,
                                  ridge__sample_weight=np.minimum(df_xg_train.loc[train_mask_xg, 'target_minutes'].to_numpy(), 90.0))
                    raw_pred_xg = fold_m_xg.predict(df_xg_train.loc[test_mask_xg, feature_cols_xg])
                    fold_p_xg = np.maximum(0.0, raw_pred_xg)
                    y_test_xg.extend(df_xg_train.loc[test_mask_xg, 'actual_xg90'])
                    pred_test_xg.extend(fold_p_xg)

                    for p_name, (_, r), pred_val in zip(df_xg_train.loc[test_mask_xg, 'player_name'], df_xg_train.loc[test_mask_xg].iterrows(), fold_p_xg):
                        eval_rows_xg.append({
                            'Pemain': p_name,
                            'GW Target': int(r['round']),
                            'Home': int(r['was_home']),
                            'Form (Prior L5)': round(r['form_points'], 1),
                            'Minutes (Prior L5)': round(r['minutes_l5'], 1),
                            'Started (Prior Match)': int(r['started_prev']),
                            'Threat/90 (Prior L5)': round(r['threat_per90_l5'], 2),
                            'Attack Diff (Prior GW)': round(r['attack_diff'], 1),
                            'Likely Score': f"{r['likely_team_goals']:.0f}-{r['likely_opponent_goals']:.0f}",
                            'y_actual (xG/90)': round(r['actual_xg90'], 2),
                            'y_predicted (Out-of-Sample)': round(pred_val, 2),
                            'residual (e)': round(r['actual_xg90'] - pred_val, 2)
                        })

                train_mask_xa = (df_xa_train['round'] < t)
                test_mask_xa = (df_xa_train['round'] == t)
                y_tr_xa = df_xa_train.loc[train_mask_xa, 'actual_xa']

                # Fold xA valid jika sampel latih >= 15, target tidak nol semua (std dev > 0 dan sum > 0)
                if train_mask_xa.sum() >= 15 and float(y_tr_xa.sum()) > 0 and float(np.std(y_tr_xa)) > 1e-4 and test_mask_xa.sum() > 0:
                    fold_m_xa = LinearRegression().fit(df_xa_train.loc[train_mask_xa, feature_cols_xa], y_tr_xa)
                    raw_pred_xa = fold_m_xa.predict(df_xa_train.loc[test_mask_xa, feature_cols_xa])
                    fold_p_xa = np.maximum(0.0, raw_pred_xa)
                    y_test_xa.extend(df_xa_train.loc[test_mask_xa, 'actual_xa'])
                    pred_test_xa.extend(fold_p_xa)

                    for p_name, (_, r), pred_val in zip(df_xa_train.loc[test_mask_xa, 'player_name'], df_xa_train.loc[test_mask_xa].iterrows(), fold_p_xa):
                        eval_rows_xa.append({
                            'Pemain': p_name,
                            'GW Target': int(r['round']),
                            'Home': int(r['was_home']),
                            'Form (Prior)': round(r['form'], 1),
                            'Creativity/90 (Prior)': round(r['creativity_per_90'], 2),
                            'FDR': round(r['FDR'], 1),
                            'Diff Attack Team (Prior GW)': round(r['Diff Attack Team'], 1),
                            'y_actual (xA)': round(r['actual_xa'], 2),
                            'y_predicted (Out-of-Sample)': round(pred_val, 2),
                            'residual (e)': round(r['actual_xa'] - pred_val, 2)
                        })

            if len(y_test_xg) >= 10:
                mae_xg = round(mean_absolute_error(y_test_xg, pred_test_xg), 4)
                rmse_xg = round(float(np.sqrt(mean_squared_error(y_test_xg, pred_test_xg))), 4)
                r2_xg = round(r2_score(y_test_xg, pred_test_xg), 4)
                eval_df_xg = pd.DataFrame(eval_rows_xg).sort_values(by='y_actual (xG/90)', ascending=False).head(10)
                eval_available_xg = True

            if len(y_test_xa) >= 10:
                mae_xa = round(mean_absolute_error(y_test_xa, pred_test_xa), 4)
                rmse_xa = round(float(np.sqrt(mean_squared_error(y_test_xa, pred_test_xa))), 4)
                r2_xa = round(r2_score(y_test_xa, pred_test_xa), 4)
                eval_df_xa = pd.DataFrame(eval_rows_xa).sort_values(by='y_actual (xA)', ascending=False).head(10)
                eval_available_xa = True

        # 3. Latih model final MURNI pada data empiris bebas bocor (Tanpa Data Sintetis)
        if not df_xg_train.empty and len(df_xg_train) >= 15 and float(df_xg_train['actual_xg90'].sum()) > 0 and float(np.std(df_xg_train['actual_xg90'])) > 1e-4:
            X_xg = df_xg_train[feature_cols_xg]
            y_xg = df_xg_train['actual_xg90']
            model_xg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model_xg.fit(X_xg, y_xg, ridge__sample_weight=np.minimum(df_xg_train['target_minutes'].to_numpy(), 90.0))
            opt_b_models_xg[pos_key] = model_xg

            std_y_xg = float(np.std(y_xg)) if float(np.std(y_xg)) > 0 else 1.0
            std_x_xg = np.array([float(np.std(X_xg[col])) if float(np.std(X_xg[col])) > 0 else 1.0 for col in X_xg.columns])
            ridge_xg = model_xg.named_steps['ridge']
            beta_std_xg = ridge_xg.coef_ / std_y_xg
            coef_raw_xg = ridge_xg.coef_ / std_x_xg
            abs_beta_xg = np.abs(beta_std_xg)
            total_abs_xg = np.sum(abs_beta_xg) if np.sum(abs_beta_xg) > 0 else 1.0
            pct_importance_xg = np.round((abs_beta_xg / total_abs_xg) * 100.0, 1)

            coef_df_xg = pd.DataFrame({
                'Variabel Fitur': list(X_xg.columns),
                'Koefisien Mentah (β)': np.round(coef_raw_xg, 4),
                'Beta Standar (Std β)': np.round(beta_std_xg, 4),
                'Kontribusi Relatif (%)': pct_importance_xg,
                'Arah Pengaruh': ['Positif (+)' if c > 0 else ('Negatif (-)' if c < 0 else 'Netral') for c in coef_raw_xg]
            }).sort_values(by='Kontribusi Relatif (%)', ascending=False)

            top_feature_xg = coef_df_xg.iloc[0]['Variabel Fitur'] if not coef_df_xg.empty else '-'
            top_contrib_pct_xg = coef_df_xg.iloc[0]['Kontribusi Relatif (%)'] if not coef_df_xg.empty else 0.0
            intercept_xg = round(float(ridge_xg.intercept_ - np.dot(coef_raw_xg, X_xg.mean().to_numpy())), 4)
        else:
            opt_b_models_xg[pos_key] = None
            coef_df_xg = pd.DataFrame()
            top_feature_xg = 'Data Empiris Belum Mencukupi'
            top_contrib_pct_xg = 0.0
            intercept_xg = None

        if not df_xa_train.empty and len(df_xa_train) >= 15 and float(df_xa_train['actual_xa'].sum()) > 0 and float(np.std(df_xa_train['actual_xa'])) > 1e-4:
            X_xa = df_xa_train[feature_cols_xa]
            y_xa = df_xa_train['actual_xa']
            model_xa = LinearRegression().fit(X_xa, y_xa)
            opt_b_models_xa[pos_key] = model_xa

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
            intercept_xa = round(float(model_xa.intercept_), 4)
        else:
            opt_b_models_xa[pos_key] = None
            coef_df_xa = pd.DataFrame()
            top_feature_xa = 'Data Empiris Belum Mencukupi'
            top_contrib_pct_xa = 0.0
            intercept_xa = None

        stats_xg[pos_key] = {
            'r2': r2_xg,
            'mae': mae_xg,
            'rmse': rmse_xg,
            'eval_available': eval_available_xg,
            'training_rows': len(df_xg_train),
            'match_history_rows': len(rows_xg),
            'evaluated_rows': len(y_test_xg),
            'historical_rows_used': len(hist_rows_xg),
            'intercept': intercept_xg,
            'coef_df': coef_df_xg,
            'top_feature': top_feature_xg,
            'top_pct': top_contrib_pct_xg,
            'eval_df': eval_df_xg
        }

        stats_xa[pos_key] = {
            'r2': r2_xa,
            'mae': mae_xa,
            'rmse': rmse_xa,
            'eval_available': eval_available_xa,
            'training_rows': len(df_xa_train),
            'match_history_rows': len(rows_xa),
            'evaluated_rows': len(y_test_xa),
            'historical_rows_used': len(hist_rows_xa),
            'intercept': intercept_xa,
            'coef_df': coef_df_xa,
            'top_feature': top_feature_xa,
            'top_pct': top_contrib_pct_xa,
            'eval_df': eval_df_xa
        }

    # Replace the legacy per-position xA regressions with one pooled xA/90 model.
    # Position is encoded inside the estimator; diagnostics stay split by position for the UI.
    xa_numeric_features = ['was_home', 'form', 'creativity_per_90', 'FDR', 'Diff Attack Team']
    xa_categorical_features = ['position']
    xa_feature_cols = xa_numeric_features + xa_categorical_features
    xa_train_all = pd.DataFrame(pooled_xa_rows)
    xa_train_all = xa_train_all[
        xa_train_all['target_minutes'].ge(30)
        & xa_train_all['actual_xa90'].notna()
    ].copy() if not xa_train_all.empty else pd.DataFrame()

    def make_xa_model():
        preprocessor = ColumnTransformer([
            ('numeric', 'passthrough', xa_numeric_features),
            ('position', OneHotEncoder(handle_unknown='ignore'), xa_categorical_features),
        ])
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression()),
        ])

    xa_oof_rows = []
    if not xa_train_all.empty:
        for target_gw in sorted(xa_train_all['round'].unique()):
            train_xa = xa_train_all[xa_train_all['round'] < target_gw]
            test_xa = xa_train_all[xa_train_all['round'] == target_gw]
            if len(train_xa) < 30 or float(train_xa['actual_xa90'].sum()) <= 0 or float(train_xa['actual_xa90'].std()) <= 1e-4 or test_xa.empty:
                continue
            fold_xa = make_xa_model().fit(train_xa[xa_feature_cols], train_xa['actual_xa90'])
            fold_predictions = np.maximum(0.0, fold_xa.predict(test_xa[xa_feature_cols]))
            for (_, row), prediction in zip(test_xa.iterrows(), fold_predictions):
                xa_oof_rows.append({
                    'Pemain': row['player_name'],
                    'GW Target': int(row['round']),
                    'Posisi': row['position'],
                    'Home': int(row['was_home']),
                    'Form (Prior)': round(float(row['form']), 1),
                    'Creativity/90 (Prior)': round(float(row['creativity_per_90']), 2),
                    'FDR': round(float(row['FDR']), 1),
                    'Diff Attack Team (Prior GW)': round(float(row['Diff Attack Team']), 1),
                    'Actual Minutes': int(row['target_minutes']),
                    'y_actual (xA/90)': round(float(row['actual_xa90']), 3),
                    'y_predicted (Out-of-Sample)': round(float(prediction), 3),
                    'residual (e)': round(float(row['actual_xa90'] - prediction), 3),
                })

    pooled_xa_model = None
    xa_coef_df = pd.DataFrame()
    xa_intercept = None
    xa_top_feature = 'Data empiris belum mencukupi'
    xa_top_pct = 0.0
    if len(xa_train_all) >= 30 and float(xa_train_all['actual_xa90'].sum()) > 0 and float(xa_train_all['actual_xa90'].std()) > 1e-4:
        pooled_xa_model = make_xa_model().fit(xa_train_all[xa_feature_cols], xa_train_all['actual_xa90'])
        transformed = pooled_xa_model.named_steps['preprocessor'].transform(xa_train_all[xa_feature_cols])
        encoder = pooled_xa_model.named_steps['preprocessor'].named_transformers_['position']
        feature_names = xa_numeric_features + [f"position_{value}" for value in encoder.categories_[0]]
        std_x = np.asarray(transformed.std(axis=0)).reshape(-1)
        std_y = float(xa_train_all['actual_xa90'].std()) or 1.0
        coefficients = pooled_xa_model.named_steps['regressor'].coef_
        beta_std = coefficients * std_x / std_y
        abs_beta = np.abs(beta_std)
        importance = abs_beta / abs_beta.sum() * 100 if abs_beta.sum() > 0 else np.zeros_like(abs_beta)
        xa_coef_df = pd.DataFrame({
            'Variabel Fitur': feature_names,
            'Koefisien Mentah (β)': np.round(coefficients, 4),
            'Beta Standar (Std β)': np.round(beta_std, 4),
            'Kontribusi Relatif (%)': np.round(importance, 1),
            'Arah Pengaruh': ['Positif (+)' if value > 0 else ('Negatif (-)' if value < 0 else 'Netral') for value in coefficients],
        }).sort_values('Kontribusi Relatif (%)', ascending=False)
        xa_top_feature = xa_coef_df.iloc[0]['Variabel Fitur'] if not xa_coef_df.empty else '-'
        xa_top_pct = float(xa_coef_df.iloc[0]['Kontribusi Relatif (%)']) if not xa_coef_df.empty else 0.0
        xa_intercept = round(float(pooled_xa_model.named_steps['regressor'].intercept_), 4)
        opt_b_models_xa = {pos: pooled_xa_model for pos in target_positions}
    else:
        opt_b_models_xa = {pos: None for pos in target_positions}

    xa_eval_df = pd.DataFrame(xa_oof_rows)
    for pos in target_positions:
        pos_eval = xa_eval_df[xa_eval_df['Posisi'] == pos] if not xa_eval_df.empty else pd.DataFrame()
        actual = pos_eval['y_actual (xA/90)'] if not pos_eval.empty else pd.Series(dtype=float)
        predicted = pos_eval['y_predicted (Out-of-Sample)'] if not pos_eval.empty else pd.Series(dtype=float)
        n_eval = len(pos_eval)
        stats_xa[pos] = {
            'r2': round(float(r2_score(actual, predicted)), 4) if n_eval >= 2 and float(np.std(actual)) > 0 else None,
            'mae': round(float(mean_absolute_error(actual, predicted)), 4) if n_eval else None,
            'rmse': round(float(np.sqrt(mean_squared_error(actual, predicted))), 4) if n_eval else None,
            'eval_available': n_eval >= 10,
            'training_rows': int((xa_train_all['position'] == pos).sum()) if not xa_train_all.empty else 0,
            'match_history_rows': int((pd.DataFrame(pooled_xa_rows)['position'] == pos).sum()) if pooled_xa_rows else 0,
            'evaluated_rows': n_eval,
            'historical_rows_used': 0,
            'intercept': xa_intercept,
            'coef_df': xa_coef_df,
            'top_feature': xa_top_feature,
            'top_pct': xa_top_pct,
            'eval_df': pos_eval.sort_values('y_actual (xA/90)', ascending=False).head(10) if n_eval else pd.DataFrame(),
            'model_scope': 'pooled_with_position_feature',
            'target': 'xA per 90 (minimal 30 menit)',
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
    fixture_by_id = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_by_id[f_id] = f
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
            p_hist = fetch_player_history(p['id'])
            if p_hist:
                sorted_hist = sorted(p_hist, key=lambda m: (m.get('round', m.get('event', 0)), m.get('kickoff_time', '')))
                for m in sorted_hist:
                    target_gw = int(m.get('round', m.get('event', 1)))
                    
                    # Cegah kebocoran data: HANYA gunakan riwayat pertandingan sebelum Gameweek target (GW < target_gw).
                    # Ini mencegah in-match leakage dan kebocoran antar-laga pada Double Gameweek (DGW) yang sama.
                    past_matches = [past_m for past_m in sorted_hist if int(past_m.get('round', past_m.get('event', 1))) < target_gw]
                    
                    # Keluarkan baris tanpa riwayat sebelumnya dari data latih
                    if len(past_matches) == 0:
                        continue

                    # Ambil window maksimal 5 pertandingan terakhir sebelum Gameweek target
                    w = past_matches[-5:]
                    tot_mins = sum(int(x.get('minutes', 0)) for x in w)
                    avg_mins_l5m = tot_mins / float(len(w))
                    eff_m_mins = max(float(tot_mins), 60.0)

                    # Fitur statistik rolling per 90 menit dari riwayat masa lalu (benar-benar diketahui pra-pertandingan)
                    xg90 = min(2.5, (sum(float(x.get('expected_goals', 0.0)) for x in w) / eff_m_mins) * 90.0)
                    xa90 = min(2.0, (sum(float(x.get('expected_assists', 0.0)) for x in w) / eff_m_mins) * 90.0)
                    xgc90 = min(4.0, (sum(float(x.get('expected_goals_conceded', 0.0)) for x in w) / eff_m_mins) * 90.0)
                    saves90 = min(10.0, (sum(float(x.get('saves', 0)) for x in w) / eff_m_mins) * 90.0)
                    bps90 = min(50.0, (sum(float(x.get('bps', 0)) for x in w) / eff_m_mins) * 90.0)
                    ict90 = min(25.0, (sum(float(x.get('ict_index', 0.0)) for x in w) / eff_m_mins) * 90.0)
                    
                    # Form historis murni: rata-rata poin per laga pada window sebelum target (bukan form live saat ini)
                    hist_form = sum(int(x.get('total_points', 0)) for x in w) / float(len(w))

                    tackles = sum(float(x.get('tackles', 0)) for x in w)
                    interceptions = sum(float(x.get('interceptions', 0)) for x in w)
                    clearances = sum(float(x.get('clearances_blocks_interceptions', x.get('clearances', 0))) for x in w)
                    recoveries = sum(float(x.get('recoveries', 0)) for x in w)
                    tot_def_actions = tackles + interceptions + clearances + recoveries
                    def_contrib_90 = min(20.0, (tot_def_actions / eff_m_mins) * 90.0)

                    # Fitur target pertandingan yang sudah diketahui pra-kickoff (jadwal & tingkat kesulitan)
                    was_home = 1 if m.get('was_home') else 0
                    f_id = m.get('fixture')
                    opp_id = m.get('opponent_team', 1)
                    opp_fdr_info = fdr_summary.get(opp_id, {})
                    
                    fdr_val = fixture_fdr_map.get((f_id, bool(m.get('was_home'))))
                    if fdr_val is None:
                        fdr_val = float(m.get('difficulty') or opp_fdr_info.get('FDR1', 3.0))

                    history_rows.append({
                        'xG_per_90': xg90,
                        'xA_per_90': xa90,
                        'bps_per_90': bps90,
                        'form': hist_form,
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

def run_walk_forward_cv(df_pos, feature_cols, target_col='target_points'):
    """
    Menjalankan Walk-Forward Time-Series Cross Validation.
    Data diurutkan secara temporal berdasarkan 'round' (Gameweek).
    Pada setiap fold:
      Train: round <= t (hanya informasi masa lampau)
      Test: gameweek berikutnya yang memiliki sampel (strictly out-of-sample forward step)
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
    
    for t, next_round in zip(unique_rounds[:-1], unique_rounds[1:]):
        train_df = df_pos[df_pos['round'] <= t]
        test_df = df_pos[df_pos['round'] == next_round]
        
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
            'fold': f"GW <= {t} → GW {next_round}",
            'train_info': f"GW <= {t} ({len(train_df)} sampel)",
            'test_info': f"GW {next_round} ({len(test_df)} sampel)",
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
    
    Mengevaluasi model menggunakan Walk-Forward Time-Series Cross Validation (tanpa lookahead leakage)
    dan menghasilkan prediksi xPoin GW selanjutnya.
    """
    elements = fpl_data.get('elements', [])
    teams = fpl_data.get('teams', [])
    teams_dict = {t['id']: t['name'] for t in teams}

    # Pre-build fixture metadata for FDR and completion checks.
    fixture_fdr_map = {}
    fixture_by_id = {}
    if fixtures_data:
        for f in fixtures_data:
            f_id = f.get('id')
            if f_id:
                fixture_by_id[f_id] = f
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

        # Training and inference only use completed matches from prior GWs. This
        # excludes partial live-GW stats as well as postponed fixtures not yet played.
        completed_hist = []
        for match in hist:
            fixture = fixture_by_id.get(match.get('fixture'))
            match_round = int((fixture or {}).get('event') or match.get('round', match.get('event', 0)) or 0)
            is_finished = (
                bool(fixture.get('finished') or fixture.get('finished_provisional'))
                if fixture is not None
                else match_round < int(current_gw)
            )
            if is_finished and match_round < int(current_gw):
                completed_hist.append((match_round, match))
        completed_hist.sort(key=lambda item: (item[0], item[1].get('kickoff_time', ''), item[1].get('fixture', 0)))
        sorted_hist = [match for _, match in completed_hist]
        if not sorted_hist:
            continue
        history_rounds = np.asarray([round_no for round_no, _ in completed_hist], dtype=int)
        
        # Siapkan nilai numerik sekali, lalu gunakan prefix sum untuk rolling
        # features. Ini menghindari pemotongan dan penjumlahan ulang histori.
        history_fields = {
            'minutes': lambda match: int(match.get('minutes', 0)),
            'total_points': lambda match: int(match.get('total_points', 0)),
            'expected_goals': lambda match: float(match.get('expected_goals', 0.0)),
            'expected_assists': lambda match: float(match.get('expected_assists', 0.0)),
            'expected_goals_conceded': lambda match: float(match.get('expected_goals_conceded', 0.0)),
            'bps': lambda match: int(match.get('bps', 0)),
            'ict_index': lambda match: float(match.get('ict_index', 0.0)),
        }
        history_values = {
            field: np.asarray([convert(match) for match in sorted_hist], dtype=float)
            for field, convert in history_fields.items()
        }
        history_prefix = {
            field: np.concatenate(([0.0], np.cumsum(values)))
            for field, values in history_values.items()
        }

        def history_mean(field, end, window):
            start = max(0, end - window)
            count = end - start
            return float((history_prefix[field][end] - history_prefix[field][start]) / count)

        # 1. Bangun Data Training dari Pertandingan yang Telah Selesai
        for i in range(len(sorted_hist)):
            m = sorted_hist[i]
            pts_actual = int(m.get('total_points', 0))
            fixture = fixture_by_id.get(m.get('fixture'))
            m_round = int((fixture or {}).get('event') or m.get('round', m.get('event', 1)) or 1)

            # Every row for a target GW uses only appearances from earlier GWs.
            # This keeps both first appearances and every fixture in a DGW causal.
            prior_end = int(np.searchsorted(history_rounds, m_round, side='left'))
            if prior_end == 0:
                continue
            
            roll_mins_5 = history_mean('minutes', prior_end, 5)
            roll_pts_3 = history_mean('total_points', prior_end, 3)
            roll_xg_3 = history_mean('expected_goals', prior_end, 3)
            roll_xa_3 = history_mean('expected_assists', prior_end, 3)
            roll_xgc_3 = history_mean('expected_goals_conceded', prior_end, 3)
            roll_bps_3 = history_mean('bps', prior_end, 3)
            roll_ict_3 = history_mean('ict_index', prior_end, 3)

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
        
        history_len = len(sorted_hist)
        curr_mins_5 = history_mean('minutes', history_len, 5)
        curr_pts_3 = history_mean('total_points', history_len, 3)
        curr_xg_3 = history_mean('expected_goals', history_len, 3)
        curr_xa_3 = history_mean('expected_assists', history_len, 3)
        curr_xgc_3 = history_mean('expected_goals_conceded', history_len, 3)
        curr_bps_3 = history_mean('bps', history_len, 3)
        curr_ict_3 = history_mean('ict_index', history_len, 3)

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
        used_global_position_fallback = False

        if pos_train.empty or len(pos_train) < 10:
            # Fallback ke seluruh dataset jika data posisi terlalu sedikit
            pos_train = df_train_all
            used_global_position_fallback = True
            common_cols = [c for c in f_cols if c in pos_train.columns]
            f_cols = common_cols
            f_labels = [f_labels[i] for i, c in enumerate(p_cfg['features']) if c in common_cols]

        # 1. Jalankan Walk-Forward Time-Series Cross Validation
        cv_summary, fold_details = run_walk_forward_cv(pos_train, f_cols)

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
            'cv_metrics': cv_summary or {},
            'in_sample': in_sample,
            'importance_df': imp_df,
            'coef_df': coef_df,
            'n_samples': len(pos_train),
            'folds_evaluated': len(fold_details),
            'fold_details': fold_details,
            'has_walk_forward_cv': bool(cv_summary),
            'used_global_position_fallback': used_global_position_fallback
        }

    # Summary keseluruhan untuk kompatibilitas view
    overall_f_cols = ['was_home', 'fdr', 'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'roll_xgc_3', 'roll_bps_3', 'roll_ict_3']
    overall_labels = ['Laga Kandang (Home)', 'FDR Lawan Mendatang', 'Avg Menit L5M', 'Avg Poin L3M (Form)', 'Avg xG L3M', 'Avg xA L3M', 'Avg xGC L3M', 'Avg BPS L3M', 'Avg ICT Index L3M']
    
    cv_all, folds_all = run_walk_forward_cv(df_train_all, overall_f_cols)
    
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
        'cv_metrics': cv_all or {},
        'in_sample': global_in_sample,
        'importance_df': pd.DataFrame({'Fitur': overall_labels, 'Tingkat Kepentingan Fitur (%)': np.round(g_gb.feature_importances_ * 100, 2)}).sort_values(by='Tingkat Kepentingan Fitur (%)', ascending=False),
        'coef_df': pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_rd.coef_, 4)}),
        'n_samples': len(df_train_all),
        'folds_evaluated': len(folds_all),
        'fold_details': folds_all,
        'has_walk_forward_cv': bool(cv_all)
    }

    # Aliases untuk backwards compatibility
    models_performance['Linear Regression'] = cv_all.get('Linear Regression', {}) if cv_all else {}
    models_performance['Linear Regression']['coef_df'] = pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_lr.coef_, 4)})
    
    models_performance['Ridge Regression'] = cv_all.get('Ridge Regression', {}) if cv_all else {}
    models_performance['Ridge Regression']['coef_df'] = pd.DataFrame({'Fitur': overall_labels, 'Bobot Koefisien (β)': np.round(g_rd.coef_, 4)})
    
    models_performance['Gradient Boosting'] = cv_all.get('Gradient Boosting', {}) if cv_all else {}
    models_performance['Gradient Boosting']['importance_df'] = models_performance['ALL']['importance_df']

    return df_pred_all, models_performance
