"""
Fixtures and FDR Table View Tab.
Ultra-polished, responsive Fixture Matrix, Interactive Ticker, Matchday Hub with Home/Away Differentials, and Swing Insights.
"""

from datetime import datetime, timedelta
import math
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from src.processors import get_team_short_map, bivariate_dixon_coles_cs_prob

def simulate_bivariate_poisson_match(lambda_h, lambda_a, max_goals=6, rho=-0.06):
    """
    Simulasi distribusi probabilitas skor pertandingan menggunakan Bivariate Poisson / Dixon-Coles Model.
    Menghitung matriks probabilitas P(H=x, A=y), serta probabilitas Home Win, Draw, Away Win,
    Over/Under 2.5 Goals, Most Likely Score, dan Clean Sheet H & A.
    """
    lh = max(0.1, float(lambda_h))
    la = max(0.1, float(lambda_a))
    
    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_ind = poisson.pmf(x, lh) * poisson.pmf(y, la)
            # Dixon-Coles adjustment for low scores
            adj = 1.0
            if x == 0 and y == 0:
                adj = 1.0 - (lh * la * rho)
            elif x == 0 and y == 1:
                adj = 1.0 + (lh * rho)
            elif x == 1 and y == 0:
                adj = 1.0 + (la * rho)
            elif x == 1 and y == 1:
                adj = 1.0 - rho
            score_matrix[x, y] = max(0.0, p_ind * adj)

    # Normalize total probability
    total_p = np.sum(score_matrix)
    if total_p > 0:
        score_matrix = score_matrix / total_p

    p_home_win = float(np.sum(np.tril(score_matrix, -1)))
    p_draw = float(np.sum(np.diag(score_matrix)))
    p_away_win = float(np.sum(np.triu(score_matrix, 1)))

    p_cs_h = float(np.sum(score_matrix[:, 0]))  # Away scores 0
    p_cs_a = float(np.sum(score_matrix[0, :]))  # Home scores 0

    # Over / Under 2.5
    p_under_25 = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            if x + y <= 2:
                p_under_25 += score_matrix[x, y]
    p_over_25 = max(0.0, 1.0 - p_under_25)

    # Most likely score
    best_idx = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
    most_likely_score = f"{best_idx[0]}-{best_idx[1]}"
    most_likely_prob = float(score_matrix[best_idx]) * 100.0

    return {
        'matrix': score_matrix,
        'p_home_win': round(p_home_win * 100.0, 1),
        'p_draw': round(p_draw * 100.0, 1),
        'p_away_win': round(p_away_win * 100.0, 1),
        'p_cs_h': round(p_cs_h * 100.0, 1),
        'p_cs_a': round(p_cs_a * 100.0, 1),
        'p_over_25': round(p_over_25 * 100.0, 1),
        'p_under_25': round(p_under_25 * 100.0, 1),
        'most_likely_score': most_likely_score,
        'most_likely_prob': round(most_likely_prob, 1),
        'lambda_h': round(lh, 2),
        'lambda_a': round(la, 2)
    }


FDR_PALETTE = {
    1: {'bg': '#15803d', 'text': '#ffffff', 'label': 'Sangat Mudah', 'dot': '🟢'},
    2: {'bg': '#10b981', 'text': '#ffffff', 'label': 'Mudah', 'dot': '🟢'},
    3: {'bg': '#64748b', 'text': '#ffffff', 'label': 'Netral', 'dot': '⚪'},
    4: {'bg': '#f59e0b', 'text': '#ffffff', 'label': 'Sulit', 'dot': '🟠'},
    5: {'bg': '#ef4444', 'text': '#ffffff', 'label': 'Sangat Sulit', 'dot': '🔴'},
}

def format_kickoff_wib(iso_str):
    """Mengubah timestamp ISO UTC fixture menjadi string representatif WIB."""
    if not iso_str:
        return "Jadwal Belum Ditentukan"
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        dt_wib = dt + timedelta(hours=7)
        days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
        return f"{days[dt_wib.weekday()]}, {dt_wib.day} {months[dt_wib.month - 1]} {dt_wib.year} • {dt_wib.strftime('%H:%M')} WIB"
    except Exception:
        return str(iso_str)

def get_diff_pill_html(diff_val, label):
    """Menghasilkan pill HTML berwarna untuk visualisasi differential metrik."""
    try:
        v = float(diff_val)
    except Exception:
        v = 0.0
    
    sign = "+" if v > 0 else ""
    if v >= 10.0:
        bg = "#dcfce7"
        text = "#15803d"
        border = "#86efac"
        icon = "🟢"
    elif v >= 3.0:
        bg = "#ecfdf5"
        text = "#047857"
        border = "#a7f3d0"
        icon = "🟢"
    elif v >= -3.0:
        bg = "#f1f5f9"
        text = "#334155"
        border = "#cbd5e1"
        icon = "⚪"
    elif v >= -10.0:
        bg = "#fff7ed"
        text = "#c2410c"
        border = "#fed7aa"
        icon = "🟠"
    else:
        bg = "#fef2f2"
        text = "#b91c1c"
        border = "#fecaca"
        icon = "🔴"
        
    return (
        f"<div style='background: {bg}; border: 1px solid {border}; border-radius: 6px; padding: 4px 8px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem;'>"
        f"<span style='color: #475569; font-weight: 500;'>{label}</span>"
        f"<span style='color: {text}; font-weight: 700; font-family: monospace;'>{icon} {sign}{v:.1f}</span>"
        f"</div>"
    )

@st.dialog("🏟️ Profil Klub & Analisis Skuad FPL", width="large")
def show_club_profile_dialog(
    initial_club,
    match_opp_club=None,
    all_club_names=None,
    df_teams=None,
    players_df=None,
    fdr_summary=None,
    teams_dict=None,
    club_short_map=None
):
    """
    Pop-up dialog modal untuk melihat profil lengkap klub, metrik kekuatan, top 5 aset FPL,
    dan jadwal pertandingan mendatang tanpa berpindah tab/halaman.
    Pengguna dapat dengan mudah beralih melihat klub lain langsung di dalam dialog ini.
    """
    if all_club_names is None and df_teams is not None and not df_teams.empty:
        all_club_names = sorted(df_teams['Klub'].unique().tolist())
    elif all_club_names is None:
        all_club_names = []

    # Reset active modal club jika dialog dipanggil untuk klub awal yang berbeda
    if st.session_state.get('dlg_last_initial') != initial_club:
        st.session_state['dlg_last_initial'] = initial_club
        st.session_state['dlg_modal_club'] = initial_club

    curr_club = st.session_state.get('dlg_modal_club', initial_club)
    if curr_club not in all_club_names and all_club_names:
        curr_club = all_club_names[0]
        st.session_state['dlg_modal_club'] = curr_club

    # 1. Bar Pengalih Klub (Switcher) yang Sangat Mudah Digunakan
    st.markdown("""
    <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
        🔄 Beralih Melihat Profil Klub:
    </div>
    """, unsafe_allow_html=True)

    if match_opp_club and match_opp_club in all_club_names:
        sw_col1, sw_col2, sw_col3 = st.columns([1.1, 1.1, 2.6])
        with sw_col1:
            h_is_active = curr_club == initial_club
            if st.button(f"🏠 {initial_club}", key=f"dlg_btn_h_{initial_club}", type="primary" if h_is_active else "secondary", use_container_width=True):
                st.session_state['dlg_modal_club'] = initial_club
                st.rerun()
        with sw_col2:
            a_is_active = curr_club == match_opp_club
            if st.button(f"✈️ {match_opp_club}", key=f"dlg_btn_a_{match_opp_club}", type="primary" if a_is_active else "secondary", use_container_width=True):
                st.session_state['dlg_modal_club'] = match_opp_club
                st.rerun()
        with sw_col3:
            curr_idx = all_club_names.index(curr_club) if curr_club in all_club_names else 0
            chosen_club = st.selectbox(
                "Pilih Klub Lain (20 Klub PL):",
                options=all_club_names,
                index=curr_idx,
                key=f"dlg_sel_box_{curr_club}",
                label_visibility="collapsed"
            )
            if chosen_club != curr_club:
                st.session_state['dlg_modal_club'] = chosen_club
                st.rerun()
    else:
        curr_idx = all_club_names.index(curr_club) if curr_club in all_club_names else 0
        chosen_club = st.selectbox(
            "Pilih Klub Premier League (20 Klub PL):",
            options=all_club_names,
            index=curr_idx,
            key=f"dlg_sel_box_{curr_club}"
        )
        if chosen_club != curr_club:
            st.session_state['dlg_modal_club'] = chosen_club
            st.rerun()

    # 2. Ambil Statistik Klub dari df_teams
    c_info = None
    if df_teams is not None and not df_teams.empty:
        c_match = df_teams[df_teams['Klub'] == curr_club]
        if not c_match.empty:
            c_info = c_match.iloc[0]

    if c_info is not None:
        t_id = c_info.get('team_id')
        code = c_info.get('Kode', curr_club[:3].upper())
        kategori = c_info.get('Kategori Tim', 'Premier League')
        indeks = c_info.get('Indeks Kekuatan', 50.0)
        lawan_gw = c_info.get('Lawan Berikutnya', '-')
        fdr1_val = c_info.get('FDR1', 3.0)
        skor_att = c_info.get('Skor Serangan', 50.0)
        skor_def = c_info.get('Skor Pertahanan', 50.0)
        total_gol = c_info.get('Total Gol', 0)
        total_xg = c_info.get('Total xG', 0.0)
        clean_sheet = c_info.get('Clean Sheet', 0)
        gc_val = c_info.get('Kebobolan (GC)', 0)
        total_saves = c_info.get('Total Saves', 0)
        total_pts = c_info.get('Total Poin Skuad', 0)
        avg_pts = c_info.get('Rata-rata Poin Pemain', 0.0)
        squad_val = c_info.get('Nilai Skuad (£m)', 0.0)
        fdr3_val = c_info.get('FDR3', 3.0)
        top_scorer = c_info.get('Top Scorer', '-')
        top_creator = c_info.get('Top Creator', '-')
        top_fpl = c_info.get('Top Aset FPL', '-')

        # Club Badge Header Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 10px; padding: 14px 18px; color: #ffffff; margin: 10px 0 14px 0; border-left: 5px solid #2563eb; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">{curr_club} ({code})</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">
                    Kategori: <b style="color: #60a5fa;">{kategori}</b> &nbsp;·&nbsp; Lawan GW Berikutnya: <b style="color: #f1f5f9;">{lawan_gw}</b> (FDR {fdr1_val})
                </div>
            </div>
            <div style="background: rgba(37, 99, 235, 0.25); border: 1px solid rgba(96, 165, 250, 0.5); padding: 6px 14px; border-radius: 18px; font-weight: 700; font-size: 0.92rem; color: #93c5fd;">
                Indeks Kekuatan: {indeks:.1f}/100
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Metrik Ringkasan Klub
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Skor Serangan", f"{skor_att:.1f}", f"{total_gol} Gol · {total_xg:.1f} xG")
        with m2:
            st.metric("Skor Pertahanan", f"{skor_def:.1f}", f"{clean_sheet} CS · {gc_val} GC · {total_saves} Sv")
        with m3:
            st.metric("Total Poin Skuad", f"{total_pts} pts", f"Avg: {avg_pts:.1f} pts")
        with m4:
            st.metric("Nilai Skuad (£m)", f"£{squad_val:.1f}m", f"FDR 3 Laga: {fdr3_val:.2f}")

        # 3. Top 5 Aset Utama FPL Klub Ini
        if players_df is not None and not players_df.empty:
            if 'team' in players_df.columns and t_id is not None:
                club_players = players_df[players_df['team'] == t_id].copy()
            elif 'Klub' in players_df.columns:
                club_players = players_df[players_df['Klub'] == curr_club].copy()
            else:
                club_players = pd.DataFrame()
        else:
            club_players = pd.DataFrame()

        if not club_players.empty:
            st.markdown(f"###### ⭐ Top 5 Aset Utama FPL: **{curr_club}**")
            top_5_assets = club_players.sort_values(by=['Total Poin', 'xPoin'], ascending=False).head(5)

            top_asset_cols = [
                'Nama Pemain', 'Posisi', 'Harga (£m)', 'Total Poin', 'xPoin', 'Form',
                'Avg Mins (L5M)', '% Ownership', 'Peluang Main GW (%)', 'Status'
            ]
            valid_cols = [c for c in top_asset_cols if c in top_5_assets.columns]

            st.dataframe(
                top_5_assets[valid_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nama Pemain": st.column_config.TextColumn("Nama Pemain", pinned=True, width="medium"),
                    "Posisi": st.column_config.TextColumn("Posisi", width="small"),
                    "Harga (£m)": st.column_config.NumberColumn("Harga", format="£%.1fm", width="small"),
                    "Total Poin": st.column_config.NumberColumn("Total Pts", format="%d pts", width="small"),
                    "xPoin": st.column_config.ProgressColumn(
                        "xPoin",
                        min_value=0.0,
                        max_value=12.0,
                        format="%.2f",
                        width="medium",
                        help="Prediksi poin Gameweek berikutnya."
                    ),
                    "Form": st.column_config.ProgressColumn(
                        "Form",
                        min_value=0.0,
                        max_value=12.0,
                        format="%.1f",
                        width="medium",
                        help="Rata-rata poin per laga dalam 30 hari terakhir."
                    ),
                    "% Ownership": st.column_config.ProgressColumn(
                        "Kepemilikan",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                        width="medium"
                    ),
                    "Peluang Main GW (%)": st.column_config.ProgressColumn(
                        "Peluang Main",
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                        width="small"
                    ),
                    "Avg Mins (L5M)": st.column_config.NumberColumn("Mins L5M", format="%.0f'", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                }
            )

            st.caption(f"💡 **Top Scorer:** {top_scorer} &nbsp;·&nbsp; **Top Creator:** {top_creator} &nbsp;·&nbsp; **Talisman Utama:** {top_fpl}")

        # 4. Jadwal 5 Pertandingan Mendatang & FDR
        if fdr_summary and t_id in fdr_summary:
            t_fdr = fdr_summary.get(t_id, {})
            up5 = t_fdr.get('upcoming_10', [])[:5]
            if up5:
                st.markdown("###### 🗓️ Jadwal 5 Laga Mendatang & Tingkat Kesulitan (FDR)")
                f_cols = st.columns(len(up5))
                for i, m in enumerate(up5):
                    fdr_val = m.get('fdr', m.get('difficulty', 3))
                    fdr_cfg = FDR_PALETTE.get(fdr_val, FDR_PALETTE[3])
                    opp_id = m.get('opp_id', m.get('opponent_id'))
                    opp_name = teams_dict.get(opp_id, 'TBD') if teams_dict else 'TBD'
                    opp_code = club_short_map.get(opp_id) or club_short_map.get(opp_name, opp_name[:3].upper()) if club_short_map else opp_name[:3].upper()
                    ha = "H" if m.get('is_home') == 1 else "A"
                    gw_label = f"GW {m.get('gw', i+1)}"
                    with f_cols[i]:
                        st.markdown(f"""
                        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 6px 8px; text-align: center;">
                            <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">{gw_label}</div>
                            <div style="font-size: 0.85rem; font-weight: 800; color: #0f172a; margin: 2px 0;">{opp_code} ({ha})</div>
                            <div style="background: {fdr_cfg['bg']}; color: {fdr_cfg['text']}; border-radius: 4px; font-size: 0.68rem; font-weight: 700; padding: 2px 6px; display: inline-block;">
                                FDR {fdr_val}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info(f"Informasi detail statistik untuk klub {curr_club} belum tersedia.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_btn_close, _ = st.columns([1.5, 4.5])
    with c_btn_close:
        if st.button("✕ Tutup Pop-up", key=f"btn_close_dlg_{curr_club}", use_container_width=True):
            st.rerun()

def render_tab_fixtures(fixtures_data, teams_dict, fdr_summary, fpl_data=None, df_teams=None, players_df=None):
    """
    Renders Fixtures, Schedule and Fixture Difficulty Rating (FDR) Matrix & Ticker.
    """
    # Dynamic short club names extracted directly from FPL API
    club_short_map = get_team_short_map(fpl_data)

    # Header Banner Card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 20px 24px; color: #ffffff; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 0.76rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.08em;">Official Premier League Fixtures & Scout Hub</span>
                <h2 style="margin: 4px 0 6px 0; font-size: 1.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">📅 Analisis Jadwal, Gameweek & Ticker FDR</h2>
                <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.5; max-width: 860px;">
                    Tinjau 10 pertandingan Gameweek mendatang secara otomatis lengkap dengan <b>informasi differential tim Home & Away</b>, 
                    matriks FDR 20 klub, serta komparasi head-to-head untuk memandu keputusan transfer aset FPL.
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Klub</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">20 Tim</div>
                </div>
                <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Per Gameweek</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #10b981;">10 Match</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not fdr_summary or not fixtures_data:
        st.warning("Data FDR dan jadwal pertandingan tidak tersedia.")
        return

    # Build lookup dictionary for team strengths, attack/defense scores & top assets
    team_stats_map = {}
    if df_teams is not None and not df_teams.empty:
        for _, row in df_teams.iterrows():
            c_name = str(row['Klub'])
            info = {
                'club': c_name,
                'code': str(row.get('Kode', '')),
                'att': float(row.get('Skor Serangan', 50.0)),
                'def': float(row.get('Skor Pertahanan', 50.0)),
                'overall': float(row.get('Indeks Kekuatan', 50.0)),
                'top_asset': str(row.get('Top Aset FPL', '-')),
                'top_scorer': str(row.get('Top Scorer', '-')),
                'top_creator': str(row.get('Top Creator', '-')),
                'category': str(row.get('Kategori Tim', 'Menengah')),
                'team_id': row.get('team_id')
            }
            team_stats_map[c_name] = info
            if 'team_id' in row and pd.notna(row['team_id']):
                team_stats_map[row['team_id']] = info
                try:
                    team_stats_map[int(row['team_id'])] = info
                except Exception:
                    pass

    # Daftar seluruh 20 nama klub Premier League untuk pop-up modal & switcher
    all_club_names = sorted(df_teams['Klub'].unique().tolist()) if df_teams is not None and not df_teams.empty else (sorted(list(teams_dict.values())) if teams_dict else [])

    # Compile structured data for matrix and tickers
    raw_matrix_items = []
    detected_gw_labels = []
    for t_id, f_data in fdr_summary.items():
        up10 = f_data.get('upcoming_10', [])
        for idx, m in enumerate(up10):
            gw_val = m.get('gw')
            if gw_val:
                while len(detected_gw_labels) <= idx:
                    detected_gw_labels.append(None)
                if detected_gw_labels[idx] is None:
                    detected_gw_labels[idx] = gw_val

    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        up10 = f_data.get('upcoming_10', [])
        up_all = f_data.get('upcoming_all', up10)
        fdr1_val = f_data.get('FDR1', 3.0)
        fdr3_val = f_data.get('FDR3', 3.0)
        fdr5_val = f_data.get('FDR5', 3.0)
        fdr10_val = f_data.get('FDR10', 3.0)

        home_count_10 = sum(1 for m in up10 if m.get('is_home') == 1)

        club_matches = []
        for idx, m in enumerate(up10):
            opp_id = m.get('opp_id')
            opp_full = teams_dict.get(opp_id, 'TBD')
            opp_short = club_short_map.get(opp_id) or club_short_map.get(opp_full, opp_full[:3].upper())
            ha = 'H' if m.get('is_home') == 1 else 'A'
            fdr_diff = m.get('fdr', 3)
            gw_num = m.get('gw')
            club_matches.append({
                'idx': idx,
                'opp': opp_full,
                'opp_short': opp_short,
                'opp_id': opp_id,
                'ha': ha,
                'diff': fdr_diff,
                'gw': gw_num
            })

        all_club_matches = []
        for idx, m in enumerate(up_all):
            opp_id = m.get('opp_id')
            opp_full = teams_dict.get(opp_id, 'TBD')
            opp_short = club_short_map.get(opp_id) or club_short_map.get(opp_full, opp_full[:3].upper())
            ha = 'H' if m.get('is_home') == 1 else 'A'
            fdr_diff = m.get('fdr', 3)
            gw_num = m.get('gw')
            all_club_matches.append({
                'idx': idx,
                'opp': opp_full,
                'opp_short': opp_short,
                'opp_id': opp_id,
                'ha': ha,
                'diff': fdr_diff,
                'gw': gw_num
            })

        dots_3 = "".join([FDR_PALETTE.get(m['diff'], {}).get('dot', '⚪') for m in club_matches[:3]])
        dots_5 = "".join([FDR_PALETTE.get(m['diff'], {}).get('dot', '⚪') for m in club_matches[:5]])
        dots_10 = "".join([FDR_PALETTE.get(m['diff'], {}).get('dot', '⚪') for m in club_matches[:10]])

        raw_matrix_items.append({
            't_id': t_id,
            'club': t_name,
            'short_name': club_short_map.get(t_id) or club_short_map.get(t_name, t_name[:3].upper()),
            'fdr1': fdr1_val,
            'fdr3': fdr3_val,
            'fdr5': fdr5_val,
            'fdr10': fdr10_val,
            'home_count_10': home_count_10,
            'matches': club_matches,
            'all_matches': all_club_matches,
            'dots_3': dots_3,
            'dots_5': dots_5,
            'dots_10': dots_10
        })

    all_club_names = sorted(list({item['club'] for item in raw_matrix_items}))

    # -------------------------------------------------------------------------
    # SUB-TAB NAVIGATION
    # Tab 1: Matchday Hub (Otomatis Gameweek Berikutnya & 10 Match Differentials)
    # Tab 2: Matriks & Ticker FDR 10 Match
    # Tab 3: Komparasi Head-to-Head 2 Klub
    # Tab 4: Analisis Dampak Home vs Away (FDR, Serang & Bertahan)
    # -------------------------------------------------------------------------
    tab_matchday, tab_matrix, tab_h2h, tab_home_away = st.tabs([
        "⚔️ Matchday Gameweek Berikutnya (10 Match & Differentials)",
        "📅 Matriks & Ticker FDR 10 Match",
        "👥 Komparasi Head-to-Head 2 Klub",
        "🏟️ Analisis Pengaruh Home vs Away (FDR, Serang & Bertahan)"
    ])

    # =========================================================================
    # TAB 1: MATCHDAY GAMEWEEK BERIKUTNYA (+-10 MATCH DGN INFO DIFFERENTIAL)
    # =========================================================================
    with tab_matchday:
        # 1. Otomatis Deteksi Gameweek Berikutnya (pekan unplayed pertama)
        all_gw_events = sorted(list({f.get('event') for f in fixtures_data if f.get('event') is not None}))
        unplayed_gw_events = sorted(list({
            f.get('event') for f in fixtures_data 
            if not (f.get('finished') or f.get('started')) and f.get('event') is not None
        }))
        auto_next_gw = unplayed_gw_events[0] if unplayed_gw_events else (all_gw_events[0] if all_gw_events else 1)

        # 2. Gameweek Selector Bar & Layout Controls
        st.markdown("#### ⚡ Hub Pertandingan Gameweek Mendatang & Analisis Differential")
        st.write(
            "Secara otomatis menampilkan seluruh pertandingan (10 match) di **Gameweek berikutnya**. "
            "Setiap pertandingan dilengkapi metrik **Differential Serangan vs Pertahanan** untuk tim Home & Away guna memandu transfer FPL."
        )

        c_gw_ctrl, c_view_ctrl, c_filter_ctrl = st.columns([1.6, 1.4, 1.4])

        with c_gw_ctrl:
            def gw_label_formatter(g):
                if g == auto_next_gw:
                    return f"🔥 Gameweek {g} (Pekan Berikutnya Otomatis)"
                return f"Gameweek {g}"

            default_idx = all_gw_events.index(auto_next_gw) if auto_next_gw in all_gw_events else 0
            selected_gw = st.selectbox(
                "Pilih Gameweek:",
                options=all_gw_events,
                index=default_idx,
                format_func=gw_label_formatter,
                key="matchday_selected_gw_dropdown"
            )

        with c_view_ctrl:
            md_view_format = st.radio(
                "Format Tampilan:",
                options=["🗂️ Kartu Visual Match (10 Laga)", "📋 Tabel Komparasi Interaktif"],
                horizontal=True,
                key="matchday_view_format_radio"
            )

        with c_filter_ctrl:
            match_filter_type = st.selectbox(
                "Filter Match:",
                options=[
                    "Semua Pertandingan (10 Match)",
                    "🟢 Potensi Pesta Gol Home (Diff Serang > +5)",
                    "🛡️ Peluang Clean Sheet Tinggi (Diff Def > +5)",
                    "✈️ Potensi Poin Tim Away (Diff Serang > 0)",
                    "⚔️ Big Match / Duel Sengit"
                ],
                index=0,
                key="matchday_filter_dropdown"
            )

        # Ambil seluruh pertandingan untuk gameweek terpilih
        gw_raw_matches = [f for f in fixtures_data if f.get('event') == selected_gw]
        gw_raw_matches = sorted(gw_raw_matches, key=lambda x: x.get('kickoff_time') or '')

        if not gw_raw_matches:
            st.info(f"Belum ada jadwal pertandingan resmi untuk Gameweek {selected_gw}.")
        else:
            # 3. Proses Data 10 Match & Hitung Differential Home vs Away
            processed_matches = []
            for m in gw_raw_matches:
                h_id = m.get('team_h')
                a_id = m.get('team_a')
                h_name = teams_dict.get(h_id, f"Team {h_id}")
                a_name = teams_dict.get(a_id, f"Team {a_id}")
                h_short = club_short_map.get(h_id) or club_short_map.get(h_name, h_name[:3].upper())
                a_short = club_short_map.get(a_id) or club_short_map.get(a_name, a_name[:3].upper())

                fdr_h = int(m.get('team_h_difficulty', 3))
                fdr_a = int(m.get('team_a_difficulty', 3))

                kickoff_str = m.get('kickoff_time')
                kickoff_formatted = format_kickoff_wib(kickoff_str)

                # Fetch team stats
                h_stats = team_stats_map.get(h_id) or team_stats_map.get(h_name, {})
                a_stats = team_stats_map.get(a_id) or team_stats_map.get(a_name, {})

                h_att = h_stats.get('att', 50.0)
                h_def = h_stats.get('def', 50.0)
                a_att = a_stats.get('att', 50.0)
                a_def = a_stats.get('def', 50.0)

                # Differentials Serang & Bertahan (Menggunakan Modulasi Keunggulan Venue Pertandingan):
                # 1. Home Attack vs Away Defense (Dengan bonus atmosfer kandang tuan rumah)
                h_att_diff = round((h_att + 4.0) - (a_def - 4.0), 1)
                # 2. Home Defense vs Away Attack (Positive = Home Clean Sheet potential high)
                h_def_diff = round((h_def + 4.0) - (a_att - 4.0), 1)
                # 3. Away Attack vs Home Defense (Dengan penalti tandang tim tamu)
                a_att_diff = round((a_att - 4.0) - (h_def + 4.0), 1)
                # 4. Away Defense vs Home Attack (Positive = Away Clean Sheet potential high)
                a_def_diff = round((a_def - 4.0) - (h_att + 4.0), 1)

                # Scout Verdict Generation
                h_top_asset = h_stats.get('top_asset', '-')
                a_top_asset = a_stats.get('top_asset', '-')

                verdict_bullets = []
                if h_att_diff >= 6.0:
                    verdict_bullets.append(f"🎯 Target Aset Serang {h_short} ({h_top_asset})")
                elif h_att_diff <= -6.0:
                    verdict_bullets.append(f"⚠️ Serangan {h_short} Berpotensi Tertahan")

                if h_def_diff >= 6.0:
                    verdict_bullets.append(f"🛡️ Clean Sheet {h_short} Sangat Potensial (+{h_def_diff:.1f})")

                if a_att_diff >= 6.0:
                    verdict_bullets.append(f"🎯 Target Aset Serang {a_short} ({a_top_asset})")

                if a_def_diff >= 6.0:
                    verdict_bullets.append(f"🛡️ Clean Sheet {a_short} Sangat Potensial (+{a_def_diff:.1f})")

                if h_att_diff >= 2.0 and a_att_diff >= 2.0:
                    verdict_bullets.append("🔥 Potensi Laga Hujan Gol (Kedua Tim Saling Bobol)")
                elif h_att_diff <= -2.0 and a_att_diff <= -2.0:
                    verdict_bullets.append("🧱 Potensi Laga Ketat & Minim Gol")

                if not verdict_bullets:
                    verdict_bullets.append("⚖️ Laga Cukup Berimbang (Peluang Merata)")

                scout_verdict_text = " • ".join(verdict_bullets)

                # Match status check
                is_finished = bool(m.get('finished') or m.get('finished_provisional'))
                is_started = bool(m.get('started'))
                h_score = m.get('team_h_score')
                a_score = m.get('team_a_score')

                if is_finished:
                    status_text = f"Selesai: {h_score} - {a_score}"
                    status_badge = "🏁 Selesai"
                elif is_started:
                    status_text = f"Live: {h_score} - {a_score}"
                    status_badge = "🔴 Sedang Berlangsung"
                else:
                    status_text = "Mendatang"
                    status_badge = "⏳ Belum Dimulai"

                processed_matches.append({
                    'id': m.get('id'),
                    'kickoff': kickoff_formatted,
                    'kickoff_raw': kickoff_str,
                    'status_text': status_text,
                    'status_badge': status_badge,
                    'is_finished': is_finished,
                    'is_started': is_started,
                    'home_id': h_id,
                    'away_id': a_id,
                    'home_name': h_name,
                    'away_name': a_name,
                    'home_short': h_short,
                    'away_short': a_short,
                    'fdr_h': fdr_h,
                    'fdr_a': fdr_a,
                    'h_att': h_att,
                    'h_def': h_def,
                    'a_att': a_att,
                    'a_def': a_def,
                    'h_att_diff': h_att_diff,
                    'h_def_diff': h_def_diff,
                    'a_att_diff': a_att_diff,
                    'a_def_diff': a_def_diff,
                    'h_top_asset': h_top_asset,
                    'a_top_asset': a_top_asset,
                    'verdict': scout_verdict_text
                })

            # 4. Metric Highlights Header Gameweek Ini
            top_h_att = max(processed_matches, key=lambda x: x['h_att_diff']) if processed_matches else None
            top_clean_sheet = max(processed_matches, key=lambda x: max(x['h_def_diff'], x['a_def_diff'])) if processed_matches else None
            cs_team_name = top_clean_sheet['home_name'] if top_clean_sheet and top_clean_sheet['h_def_diff'] >= top_clean_sheet['a_def_diff'] else (top_clean_sheet['away_name'] if top_clean_sheet else "-")
            cs_team_val = max(top_clean_sheet['h_def_diff'], top_clean_sheet['a_def_diff']) if top_clean_sheet else 0.0

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Gameweek Terpilih", f"GW {selected_gw}", f"Total {len(processed_matches)} Pertandingan")
            with m_col2:
                kickoff_range = f"{processed_matches[0]['kickoff'].split('•')[0].strip()} s/d {processed_matches[-1]['kickoff'].split('•')[0].strip()}" if len(processed_matches) > 1 else "-"
                st.metric("Rentang Jadwal Laga", f"{len(processed_matches)} Laga", kickoff_range)
            with m_col3:
                if top_h_att:
                    st.metric("Potensi Gol Tertinggi", f"{top_h_att['home_name']}", f"Att Diff: +{top_h_att['h_att_diff']:.1f}")
                else:
                    st.metric("Potensi Gol Tertinggi", "-", "-")
            with m_col4:
                st.metric("Peluang CS Terbaik", f"{cs_team_name}", f"Def Diff: +{cs_team_val:.1f}")

            # Guidance Note Callout
            st.markdown("""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: 8px; padding: 10px 14px; margin: 12px 0 16px 0; font-size: 0.82rem; color: #334155; line-height: 1.45;">
                <b>💡 Panduan Analisis Differential Matchday FPL:</b><br/>
                • <b>Differential Serang (+) Tinggi</b>: Lini serang klub jauh melampaui pertahanan lawan ➔ <b>Prioritaskan Penyerang/Gelandang (FWD/MID)</b> klub tersebut sebagai pencetak gol & calon kapten.<br/>
                • <b>Differential Bertahan (+) Tinggi</b>: Lini belakang klub sangat solid menghadapi serangan lawan ➔ <b>Prioritaskan Bek/Kiper (DEF/GKP)</b> untuk target Clean Sheet 6 poin.
            </div>
            """, unsafe_allow_html=True)

            # Apply Match Filters if requested
            filtered_match_list = list(processed_matches)
            if match_filter_type == "🟢 Potensi Pesta Gol Home (Diff Serang > +5)":
                filtered_match_list = [m for m in filtered_match_list if m['h_att_diff'] >= 5.0]
            elif match_filter_type == "🛡️ Peluang Clean Sheet Tinggi (Diff Def > +5)":
                filtered_match_list = [m for m in filtered_match_list if m['h_def_diff'] >= 5.0 or m['a_def_diff'] >= 5.0]
            elif match_filter_type == "✈️ Potensi Poin Tim Away (Diff Serang > 0)":
                filtered_match_list = [m for m in filtered_match_list if m['a_att_diff'] > 0]
            elif match_filter_type == "⚔️ Big Match / Duel Sengit":
                filtered_match_list = [m for m in filtered_match_list if m['fdr_h'] >= 4 or m['fdr_a'] >= 4]

            # -----------------------------------------------------------------
            # TAMPILAN 1: KARTU VISUAL MATCHDAY (10 MATCH GRID)
            # -----------------------------------------------------------------
            if "Kartu Visual" in md_view_format:
                st.markdown(f"##### 🗂️ 10 Pertandingan Gameweek {selected_gw} & Rincian Differential")

                # Render in 2-column responsive layout
                for idx in range(0, len(filtered_match_list), 2):
                    c_left, c_right = st.columns(2)
                    pair = [filtered_match_list[idx]]
                    if idx + 1 < len(filtered_match_list):
                        pair.append(filtered_match_list[idx + 1])

                    for col, match_item in zip([c_left, c_right], pair):
                        with col:
                            # Palette colors for FDR
                            h_fdr_style = FDR_PALETTE.get(match_item['fdr_h'], FDR_PALETTE[3])
                            a_fdr_style = FDR_PALETTE.get(match_item['fdr_a'], FDR_PALETTE[3])

                            h_pill_att = get_diff_pill_html(match_item['h_att_diff'], "⚔️ Serang vs Def Lawan")
                            h_pill_def = get_diff_pill_html(match_item['h_def_diff'], "🛡️ Def vs Serang (CS)")
                            a_pill_att = get_diff_pill_html(match_item['a_att_diff'], "⚔️ Serang vs Def Lawan")
                            a_pill_def = get_diff_pill_html(match_item['a_def_diff'], "🛡️ Def vs Serang (CS)")

                            card_html = f"""
                            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 10px;">
                                    <span style="font-size: 0.76rem; color: #64748b; font-weight: 600;">⏰ {match_item['kickoff']}</span>
                                    <span style="font-size: 0.72rem; background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 700;">{match_item['status_badge']}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="flex: 1; text-align: left;">
                                        <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a;">{match_item['home_name']}</div>
                                        <div style="display: flex; gap: 6px; align-items: center; margin-top: 4px;">
                                            <span style="background: #eff6ff; color: #1d4ed8; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">🏠 Kandang</span>
                                            <span style="background: {h_fdr_style['bg']}; color: {h_fdr_style['text']}; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">FDR {match_item['fdr_h']}</span>
                                        </div>
                                    </div>
                                    <div style="padding: 0 12px; text-align: center;">
                                        <div style="font-size: 0.82rem; font-weight: 800; color: #94a3b8; background: #f8fafc; border: 1px solid #e2e8f0; padding: 4px 10px; border-radius: 20px;">VS</div>
                                    </div>
                                    <div style="flex: 1; text-align: right;">
                                        <div style="font-size: 0.98rem; font-weight: 800; color: #0f172a;">{match_item['away_name']}</div>
                                        <div style="display: flex; gap: 6px; align-items: center; justify-content: flex-end; margin-top: 4px;">
                                            <span style="background: {a_fdr_style['bg']}; color: {a_fdr_style['text']}; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">FDR {match_item['fdr_a']}</span>
                                            <span style="background: #f8fafc; color: #475569; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">✈️ Tandang</span>
                                        </div>
                                    </div>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px;">
                                        <div style="font-size: 0.74rem; font-weight: 700; color: #1e293b; margin-bottom: 6px; text-transform: uppercase;">Differential {match_item['home_short']}</div>
                                        {h_pill_att}
                                        {h_pill_def}
                                        <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">⭐ Top: <b>{match_item['h_top_asset']}</b></div>
                                    </div>
                                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px;">
                                        <div style="font-size: 0.74rem; font-weight: 700; color: #1e293b; margin-bottom: 6px; text-transform: uppercase;">Differential {match_item['away_short']}</div>
                                        {a_pill_att}
                                        {a_pill_def}
                                        <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">⭐ Top: <b>{match_item['a_top_asset']}</b></div>
                                    </div>
                                </div>
                                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px 10px; font-size: 0.78rem; color: #166534; line-height: 1.4; margin-bottom: 8px;">
                                    <b>Scout Verdict:</b> {match_item['verdict']}
                                </div>
                            </div>
                            """
                            if hasattr(st, 'html'):
                                st.html(card_html)
                            else:
                                st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)

                            # Quick Pop-up Modal Launchers (Buka di halaman yg sama, tidak pindah halaman)
                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                if st.button(f"🔍 Profil {match_item['home_short']}", key=f"btn_card_h_{match_item['id']}", use_container_width=True):
                                    st.session_state['dlg_last_initial'] = match_item['home_name']
                                    st.session_state['dlg_modal_club'] = match_item['home_name']
                                    show_club_profile_dialog(
                                        match_item['home_name'],
                                        match_item['away_name'],
                                        all_club_names=all_club_names,
                                        df_teams=df_teams,
                                        players_df=players_df,
                                        fdr_summary=fdr_summary,
                                        teams_dict=teams_dict,
                                        club_short_map=club_short_map
                                    )
                            with col_b2:
                                if st.button(f"🔍 Profil {match_item['away_short']}", key=f"btn_card_a_{match_item['id']}", use_container_width=True):
                                    st.session_state['dlg_last_initial'] = match_item['away_name']
                                    st.session_state['dlg_modal_club'] = match_item['away_name']
                                    show_club_profile_dialog(
                                        match_item['away_name'],
                                        match_item['home_name'],
                                        all_club_names=all_club_names,
                                        df_teams=df_teams,
                                        players_df=players_df,
                                        fdr_summary=fdr_summary,
                                        teams_dict=teams_dict,
                                        club_short_map=club_short_map
                                    )

            # -----------------------------------------------------------------
            # TAMPILAN 2: TABEL DATA KOMPARASI INTERAKTIF (10 MATCH)
            # -----------------------------------------------------------------
            else:
                st.markdown(f"##### 📋 Tabel Komparasi Seluruh Pertandingan Gameweek {selected_gw}")
                
                table_rows = []
                for m in filtered_match_list:
                    table_rows.append({
                        'Pertandingan': f"{m['home_name']} vs {m['away_name']}",
                        'Jadwal Kickoff': m['kickoff'],
                        'FDR (H vs A)': f"FDR {m['fdr_h']} vs FDR {m['fdr_a']}",
                        'Diff Serang Home': m['h_att_diff'],
                        'Diff CS Home': m['h_def_diff'],
                        'Diff Serang Away': m['a_att_diff'],
                        'Diff CS Away': m['a_def_diff'],
                        'Top Aset Home': m['h_top_asset'],
                        'Top Aset Away': m['a_top_asset'],
                        'Scout Recommendation': m['verdict']
                    })

                df_match_summary = pd.DataFrame(table_rows)
                st.dataframe(
                    df_match_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pertandingan": st.column_config.TextColumn("Match (Home vs Away)", pinned=True, width="medium"),
                        "Jadwal Kickoff": st.column_config.TextColumn("Kickoff (WIB)", width="medium"),
                        "FDR (H vs A)": st.column_config.TextColumn("FDR Tingkat Kesulitan", width="small"),
                        "Diff Serang Home": st.column_config.NumberColumn("Diff Serang Home", format="%+.1f", width="small", help="Selisih Serangan Home vs Pertahanan Lawan"),
                        "Diff CS Home": st.column_config.NumberColumn("Diff CS Home", format="%+.1f", width="small", help="Selisih Pertahanan Home vs Serangan Lawan (Peluang Clean Sheet)"),
                        "Diff Serang Away": st.column_config.NumberColumn("Diff Serang Away", format="%+.1f", width="small", help="Selisih Serangan Away vs Pertahanan Lawan"),
                        "Diff CS Away": st.column_config.NumberColumn("Diff CS Away", format="%+.1f", width="small", help="Selisih Pertahanan Away vs Serangan Lawan"),
                        "Top Aset Home": st.column_config.TextColumn("Aset Utama Home", width="small"),
                        "Top Aset Away": st.column_config.TextColumn("Aset Utama Away", width="small"),
                        "Scout Recommendation": st.column_config.TextColumn("Rekomendasi Taktis FPL", width="large")
                    }
                )

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.markdown("###### 🔍 Buka Pop-up Profil Klub Langsung dari Tabel:")
                c_tb1, c_tb2 = st.columns([3, 1.5])
                with c_tb1:
                    tbl_selected_club = st.selectbox(
                        "Pilih klub untuk membuka pop-up profil & aset FPL:",
                        options=all_club_names,
                        key="tbl_view_club_sel"
                    )
                with c_tb2:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button(f"🔍 Buka Profil {tbl_selected_club}", key="btn_open_tbl_club", use_container_width=True):
                        st.session_state['dlg_last_initial'] = tbl_selected_club
                        st.session_state['dlg_modal_club'] = tbl_selected_club
                        show_club_profile_dialog(
                            tbl_selected_club,
                            None,
                            all_club_names=all_club_names,
                            df_teams=df_teams,
                            players_df=players_df,
                            fdr_summary=fdr_summary,
                            teams_dict=teams_dict,
                            club_short_map=club_short_map
                        )

    # =========================================================================
    # TAB 2: MATRIKS & TICKER FDR 10 MATCH
    # =========================================================================
    with tab_matrix:
        # Strategic Fixture Insights & Swing Highlights
        clubs_ranked_fdr3 = sorted(
            [{'id': tid, 'name': teams_dict.get(tid, f"Team {tid}"), 'fdr': float(f.get('FDR3', 3.0) or 3.0), 'fdr5': float(f.get('FDR5', 3.0) or 3.0), 'fdr10': float(f.get('FDR10', 3.0) or 3.0), 'next': f.get('Next_Opponent_Fmt', '-')} 
             for tid, f in fdr_summary.items()],
            key=lambda x: x['fdr']
        )

        top_easy_3 = clubs_ranked_fdr3[:3] if clubs_ranked_fdr3 else []
        top_hard_3 = clubs_ranked_fdr3[-3:][::-1] if len(clubs_ranked_fdr3) >= 3 else clubs_ranked_fdr3[::-1]

        st.markdown("##### ⚡ Ringkasan Strategi Fixture Run (3-5 Laga Mendatang)")
        c_easy, c_hard, c_rot = st.columns([1.2, 1.2, 1.6])

        with c_easy:
            easy_list_html = "".join([
                f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem;'>"
                f"<div><strong style='color: #0f172a;'>{c['name']}</strong><div style='font-size: 0.75rem; color: #64748b;'>Lawan: {c.get('next', '-')}</div></div>"
                f"<div style='background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;'>FDR3: {c['fdr']:.2f}</div>"
                f"</div>"
                for c in top_easy_3
            ])
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem;">🟢</span>
                    <strong style="color: #166534; font-size: 0.92rem;">Jadwal Paling Menguntungkan</strong>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 8px;">Target transfer utama untuk aset ofensif & defensif:</div>
                {easy_list_html if easy_list_html else "<div style='color: #94a3b8; font-size: 0.8rem;'>Data tidak tersedia</div>"}
            </div>
            """, unsafe_allow_html=True)

        with c_hard:
            hard_list_html = "".join([
                f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem;'>"
                f"<div><strong style='color: #0f172a;'>{c['name']}</strong><div style='font-size: 0.75rem; color: #64748b;'>Lawan: {c.get('next', '-')}</div></div>"
                f"<div style='background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;'>FDR3: {c['fdr']:.2f}</div>"
                f"</div>"
                for c in top_hard_3
            ])
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #fecaca; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem;">🔴</span>
                    <strong style="color: #991b1b; font-size: 0.92rem;">Jadwal Paling Menantang</strong>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 8px;">Waspadai penurunan poin / pertimbangkan rotasi:</div>
                {hard_list_html if hard_list_html else "<div style='color: #94a3b8; font-size: 0.8rem;'>Data tidak tersedia</div>"}
            </div>
            """, unsafe_allow_html=True)

        with c_rot:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem;">💡</span>
                    <strong style="color: #1e293b; font-size: 0.92rem;">Strategi Rotasi FPL (Home/Away Pairs)</strong>
                </div>
                <div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">
                    Memilih 2 pemain bertahan murah (£4.5m atau £4.0m) dari 2 klub dengan rotasi kandang/tandang ideal 
                    memungkinkan Anda selalu memainkan bek yang bertanding <b>di kandang sendiri (Home)</b> dengan FDR rendah di setiap Gameweek!
                </div>
                <div style="margin-top: 10px; padding: 6px 10px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 0.78rem; color: #334155;">
                    <span style="color: #2563eb; font-weight: 700;">Pasangan Populer:</span> FUL + CRY, BRE + BOU, EVE + NFO
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Interactive Matrix Controls
        st.markdown("#### 🗓️ Matriks & Ticker Interaktif FDR")
        st.write("Sesuaikan cakupan laga (3, 5, atau 10 match), filter klub berdasarkan preset kemudahan jadwal, dan periksa detail kesulitan laga.")

        c_hor, c_pre, c_club = st.columns([1.5, 2, 2.5])

        with c_hor:
            horizon_choice = st.pills(
                "Cakupan Laga (Horizon):",
                options=["3 Laga (Transfer Dekat)", "5 Laga (Wildcard)", "10 Laga (Penuh)"],
                default="5 Laga (Wildcard)",
                key="fdr_horizon_pills"
            )
            # Safe fallback if user deselects the pill
            if not horizon_choice:
                horizon_choice = "5 Laga (Wildcard)"
            horizon_n = 3 if "3 Laga" in str(horizon_choice) else (10 if "10 Laga" in str(horizon_choice) else 5)
            active_fdr_key = f"fdr{horizon_n}"
            active_fdr_label = f"FDR{horizon_n}"

        with c_pre:
            preset_choice = st.selectbox(
                "Filter Cepat (Preset):",
                options=[
                    "Semua 20 Klub",
                    "🟢 Top 5 Jadwal Paling Mudah (Run Hijau)",
                    "🔴 Top 5 Jadwal Paling Sulit (Run Merah)",
                    "👑 Big Six Premier League",
                    "🏠 Klub Terbanyak Laga Kandang (Home Run)"
                ],
                index=0,
                key="fdr_preset_sel"
            )

        with c_club:
            selected_clubs_filter = st.multiselect(
                "Pilih Klub Tertentu (Opsional):",
                options=all_club_names,
                default=[],
                placeholder="Semua 20 Klub",
                key="fdr_matrix_club_filter"
            )

        c_sort, c_view, c_dummy = st.columns([2.5, 2, 1.5])
        with c_sort:
            sort_matrix_opt = st.selectbox(
                "Urutkan Matriks:",
                options=[
                    f"Tingkat Kemudahan ({active_fdr_label} Terendah)",
                    f"Tingkat Kesulitan ({active_fdr_label} Tertinggi)",
                    "Paling Banyak Laga Kandang (Home Matches)",
                    "Nama Klub (A-Z)"
                ],
                index=0,
                key=f"sort_matrix_sel_{active_fdr_key}"
            )
            if not sort_matrix_opt:
                sort_matrix_opt = f"Tingkat Kemudahan ({active_fdr_label} Terendah)"

        with c_view:
            view_mode = st.radio(
                "Format Tampilan:",
                options=["🎨 Blok Warna Visual (Ticker)", "📊 Dataframe Interaktif"],
                horizontal=True,
                key="fdr_matrix_view_mode"
            )

        filtered_items = list(raw_matrix_items)
        big_six = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Man City", "Manchester United", "Man Utd", "Tottenham", "Tottenham Hotspur", "Spurs"]

        if preset_choice == "🟢 Top 5 Jadwal Paling Mudah (Run Hijau)":
            filtered_items = sorted(filtered_items, key=lambda x: float(x.get(active_fdr_key, 3.0) or 3.0))[:5]
        elif preset_choice == "🔴 Top 5 Jadwal Paling Sulit (Run Merah)":
            filtered_items = sorted(filtered_items, key=lambda x: float(x.get(active_fdr_key, 3.0) or 3.0), reverse=True)[:5]
        elif preset_choice == "👑 Big Six Premier League":
            filtered_items = [it for it in filtered_items if any(b.lower() in str(it.get('club', '')).lower() for b in big_six)]
        elif preset_choice == "🏠 Klub Terbanyak Laga Kandang (Home Run)":
            filtered_items = sorted(
                filtered_items, 
                key=lambda x: sum(1 for m in x.get('matches', [])[:horizon_n] if m.get('ha') == 'H'), 
                reverse=True
            )[:7]

        if selected_clubs_filter:
            filtered_items = [it for it in filtered_items if it.get('club') in selected_clubs_filter]

        if sort_matrix_opt and "Terendah" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: float(x.get(active_fdr_key, 3.0) or 3.0))
        elif sort_matrix_opt and "Tertinggi" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: float(x.get(active_fdr_key, 3.0) or 3.0), reverse=True)
        elif sort_matrix_opt and "Paling Banyak Laga Kandang" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: sum(1 for m in x.get('matches', [])[:horizon_n] if m.get('ha') == 'H'), reverse=True)
        elif sort_matrix_opt and "Nama Klub" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: str(x.get('club', '')))

        legend_html = (
            '<div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 14px 0; padding: 10px 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; font-size: 0.8rem;">'
            '<strong style="color: #1e293b;">Legenda Tingkat Kesulitan (FDR):</strong> '
            '<span style="background: #15803d; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">1 Sangat Mudah</span> '
            '<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">2 Mudah</span> '
            '<span style="background: #64748b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">3 Netral</span> '
            '<span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">4 Sulit</span> '
            '<span style="background: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">5 Sangat Sulit</span> '
            '<span style="color: #64748b; margin-left: auto; font-style: italic;">Huruf Kapital (H) = Home / Kandang, (A) = Away / Tandang</span>'
            '</div>'
        )
        if hasattr(st, 'html'):
            st.html(legend_html)
        else:
            st.markdown(legend_html, unsafe_allow_html=True)

        if not filtered_items:
            st.info("⚠️ Tidak ada klub yang sesuai dengan filter yang dipilih. Silakan ubah filter klub atau preset di atas.")
        elif "Blok Warna Visual" in view_mode:
            table_header_cols = ""
            for idx in range(horizon_n):
                gw_val = detected_gw_labels[idx] if idx < len(detected_gw_labels) and detected_gw_labels[idx] is not None else None
                col_header = f"GW{gw_val}" if gw_val is not None else f"M{idx+1}"
                table_header_cols += f"<th style='text-align: center; padding: 8px 4px; font-size: 0.78rem; color: #475569; min-width: 60px;'>{col_header}</th>"

            table_rows_html = ""
            for rank, item in enumerate(filtered_items, 1):
                club_name = item.get('club', 'TBD')
                short_code = item.get('short_name', club_name[:3].upper())
                fdr_val = float(item.get(active_fdr_key, 3.0) or 3.0)
                
                fdr_badge_color = "#10b981" if fdr_val <= 2.6 else ("#64748b" if fdr_val <= 3.2 else "#ef4444")
                fdr_badge_bg = "#ecfdf5" if fdr_val <= 2.6 else ("#f1f5f9" if fdr_val <= 3.2 else "#fef2f2")

                match_blocks_html = ""
                matches_list = item.get('matches', [])[:horizon_n]
                for m in matches_list:
                    try:
                        fdr_diff = int(round(float(m.get('diff', 3) or 3)))
                    except Exception:
                        fdr_diff = 3
                    fdr_meta = FDR_PALETTE.get(fdr_diff, {'bg': '#94a3b8', 'text': '#ffffff'})
                    ha_badge = str(m.get('ha', '-'))
                    opp_label = str(m.get('opp_short', 'TBD'))
                    opp_full = str(m.get('opp', 'TBD'))
                    gw_num = m.get('gw')
                    gw_tag = f"GW{gw_num}" if gw_num else ""
                    block_tooltip = f"{club_name} vs {opp_full} ({'Home' if ha_badge == 'H' else 'Away'}) | {gw_tag} | FDR {fdr_diff}"
                    
                    match_blocks_html += (
                        f"<td style='padding: 4px 3px; text-align: center;'>"
                        f"<div title='{block_tooltip}' style='background-color: {fdr_meta['bg']}; color: {fdr_meta['text']}; font-weight: 700; border-radius: 6px; padding: 5px 2px; font-size: 0.76rem; box-shadow: 0 1px 2px rgba(0,0,0,0.08); cursor: default;'>"
                        f"<div>{opp_label}</div>"
                        f"<div style='font-size: 0.65rem; opacity: 0.95; font-weight: 600;'>{ha_badge}{' · ' + gw_tag if gw_tag else ''}</div>"
                        f"</div>"
                        f"</td>"
                    )

                # Pad missing matches if team has fewer matches than horizon_n
                for _ in range(horizon_n - len(matches_list)):
                    match_blocks_html += (
                        "<td style='padding: 4px 3px; text-align: center;'>"
                        "<div style='background-color: #f1f5f9; color: #94a3b8; font-weight: 600; border-radius: 6px; padding: 6px 2px; font-size: 0.76rem;'>-</div>"
                        "</td>"
                    )
                
                home_in_horizon = sum(1 for m in matches_list if m.get('ha') == 'H')

                table_rows_html += (
                    f"<tr style='border-bottom: 1px solid #f1f5f9;'>"
                    f"<td style='padding: 8px 12px; font-size: 0.85rem; font-weight: 700; color: #0f172a; white-space: nowrap;'>"
                    f"<span style='color: #94a3b8; font-size: 0.75rem; font-weight: 600; margin-right: 6px;'>#{rank}</span>"
                    f"{club_name} "
                    f"<span style='font-size: 0.72rem; color: #64748b; font-weight: 600; margin-left: 4px;'>({short_code})</span>"
                    f"</td>"
                    f"<td style='padding: 8px 6px; text-align: center; white-space: nowrap;'>"
                    f"<span style='background: {fdr_badge_bg}; color: {fdr_badge_color}; border: 1px solid {fdr_badge_color}33; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;'>"
                    f"{fdr_val:.2f}"
                    f"</span>"
                    f"</td>"
                    f"<td style='padding: 8px 6px; text-align: center; font-size: 0.8rem; font-weight: 600; color: #334155;'>"
                    f"<span style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 2px 6px; border-radius: 4px;'>{home_in_horizon}H / {horizon_n - home_in_horizon}A</span>"
                    f"</td>"
                    f"{match_blocks_html}"
                    f"</tr>"
                )

            full_matrix_html = (
                f"<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 16px;'>"
                f"<table style='width: 100%; border-collapse: collapse; min-width: 650px;'>"
                f"<thead>"
                f"<tr style='background: #f8fafc; border-bottom: 1px solid #e2e8f0;'>"
                f"<th style='text-align: left; padding: 10px 12px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;'>Klub Premier League</th>"
                f"<th style='text-align: center; padding: 10px 6px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;'>{active_fdr_label} Avg</th>"
                f"<th style='text-align: center; padding: 10px 6px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;'>Venue</th>"
                f"{table_header_cols}"
                f"</tr>"
                f"</thead>"
                f"<tbody>"
                f"{table_rows_html}"
                f"</tbody>"
                f"</table>"
                f"</div>"
            )
            if hasattr(st, 'html'):
                st.html(full_matrix_html)
            else:
                st.markdown(full_matrix_html, unsafe_allow_html=True)
        else:
            df_rows = []
            for item in filtered_items:
                row_data = {
                    'Klub': item.get('club', 'TBD'),
                    'Kode': item.get('short_name', ''),
                    f'{active_fdr_label} Avg': float(item.get(active_fdr_key, 3.0) or 3.0),
                    'Laga Kandang': sum(1 for m in item.get('matches', [])[:horizon_n] if m.get('ha') == 'H'),
                }
                for idx, m in enumerate(item.get('matches', [])[:horizon_n]):
                    gw_val = detected_gw_labels[idx] if idx < len(detected_gw_labels) and detected_gw_labels[idx] is not None else None
                    col_name = f"GW{gw_val}" if gw_val is not None else f"M{idx+1}"
                    try:
                        diff_val = int(round(float(m.get('diff', 3) or 3)))
                    except Exception:
                        diff_val = 3
                    row_data[col_name] = f"{m.get('opp_short', 'TBD')} ({m.get('ha', '-')}) [{diff_val}]"
                df_rows.append(row_data)

            matrix_df = pd.DataFrame(df_rows)
            match_cols = [c for c in matrix_df.columns if (c.startswith('GW') or c.startswith('M')) and c != f'{active_fdr_label} Avg']
            
            def style_fdr_cell(val):
                if not isinstance(val, str) or '[' not in val:
                    return ''
                try:
                    diff_num = int(val.split('[')[1].replace(']', '').strip())
                    p = FDR_PALETTE.get(diff_num, {})
                    return f"background-color: {p.get('bg', '#fff')}; color: {p.get('text', '#000')}; font-weight: 700; text-align: center;"
                except Exception:
                    return ''

            def style_fdr_avg(val):
                try:
                    v = float(val)
                    if v <= 2.6:
                        return 'background-color: #dcfce7; color: #166534; font-weight: 700; text-align: center;'
                    elif v <= 3.2:
                        return 'background-color: #f1f5f9; color: #334155; font-weight: 700; text-align: center;'
                    else:
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: 700; text-align: center;'
                except Exception:
                    return ''

            try:
                styled_matrix = matrix_df.style
                if match_cols:
                    styled_matrix = styled_matrix.map(style_fdr_cell, subset=match_cols)
                if f'{active_fdr_label} Avg' in matrix_df.columns:
                    styled_matrix = styled_matrix.map(style_fdr_avg, subset=[f'{active_fdr_label} Avg']).format({f'{active_fdr_label} Avg': '{:.2f}'})

                st.dataframe(
                    styled_matrix,
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                st.dataframe(
                    matrix_df,
                    use_container_width=True,
                    hide_index=True
                )

        # Quick Club Profile & Assets Dialog Opener from Matrix
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("###### 🔍 Buka Pop-up Profil Klub Langsung dari Matriks Ticker:")
        c_m_tb1, c_m_tb2 = st.columns([3, 1.5])
        with c_m_tb1:
            m_tbl_selected_club = st.selectbox(
                "Pilih klub untuk membuka profil & analisis aset FPL:",
                options=all_club_names,
                key="matrix_view_club_sel"
            )
        with c_m_tb2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button(f"🔍 Buka Profil {m_tbl_selected_club}", key="btn_open_matrix_club", use_container_width=True):
                st.session_state['dlg_last_initial'] = m_tbl_selected_club
                st.session_state['dlg_modal_club'] = m_tbl_selected_club
                show_club_profile_dialog(
                    m_tbl_selected_club,
                    None,
                    all_club_names=all_club_names,
                    df_teams=df_teams,
                    players_df=players_df,
                    fdr_summary=fdr_summary,
                    teams_dict=teams_dict,
                    club_short_map=club_short_map
                )

    # =========================================================================
    # TAB 3: KOMPARASI HEAD-TO-HEAD 2 KLUB (DUEL FIXTURES)
    # =========================================================================
    with tab_h2h:
        st.markdown("#### ⚔️ Komparasi Head-to-Head Jadwal 2 Klub (Duel Fixtures)")
        st.markdown(
            "Bandingkan seluruh jadwal mendatang antara dua klub Premier League secara komparatif. "
            "Dilengkapi **analisis differential metrik ofensif vs defensif** per laga untuk memandu "
            "keputusan transfer pemain menyerang (*attacking*) vs bertahan (*defensive*)."
        )
        
        c_cmp1, c_cmp2 = st.columns(2)
        with c_cmp1:
            cmp_club1 = st.selectbox("Pilih Klub Pertama:", options=all_club_names, index=0, key="cmp_club_1")
        with c_cmp2:
            default_c2_idx = 1 if len(all_club_names) > 1 else 0
            cmp_club2 = st.selectbox("Pilih Klub Kedua:", options=all_club_names, index=default_c2_idx, key="cmp_club_2")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(f"🔍 Deep Dive Profil & Aset: **{cmp_club1}**", key=f"btn_deepdive_{cmp_club1}", use_container_width=True):
                st.session_state['selected_deepdive_club'] = cmp_club1
                st.session_state['active_nav_id'] = "team_strength"
                st.rerun()
        with col_btn2:
            if st.button(f"🔍 Deep Dive Profil & Aset: **{cmp_club2}**", key=f"btn_deepdive_{cmp_club2}", use_container_width=True):
                st.session_state['selected_deepdive_club'] = cmp_club2
                st.session_state['active_nav_id'] = "team_strength"
                st.rerun()

        c1_item = next((it for it in raw_matrix_items if it['club'] == cmp_club1), None)
        c2_item = next((it for it in raw_matrix_items if it['club'] == cmp_club2), None)

        if c1_item and c2_item:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(f"Rerata FDR3: {c1_item['club']}", f"{c1_item['fdr3']:.2f}", f"FDR5: {c1_item['fdr5']:.2f}")
            with m2:
                diff_fdr = c2_item['fdr3'] - c1_item['fdr3']
                easier_name = c1_item['club'] if diff_fdr > 0 else c2_item['club']
                st.metric("Jadwal Lebih Menguntungkan", easier_name, f"Selisih: {abs(diff_fdr):.2f} pts")
            with m3:
                st.metric(f"Rerata FDR3: {c2_item['club']}", f"{c2_item['fdr3']:.2f}", f"FDR5: {c2_item['fdr5']:.2f}")

            matches_1 = c1_item.get('all_matches', c1_item.get('matches', []))
            matches_2 = c2_item.get('all_matches', c2_item.get('matches', []))
            total_matches = max(len(matches_1), len(matches_2))

            def compute_match_differential(club_name, m_info):
                if not m_info or m_info.get('opp') == '-':
                    return None
                
                opp_id = m_info.get('opp_id')
                opp_name = m_info.get('opp', '')
                is_home = (m_info.get('ha') == 'H')
                
                home_team_name = club_name if is_home else opp_name
                away_team_name = opp_name if is_home else club_name
                
                home_stats = team_stats_map.get(home_team_name) or (team_stats_map.get(c1_item['t_id']) if home_team_name == c1_item['club'] else team_stats_map.get(c2_item['t_id']) if home_team_name == c2_item['club'] else None)
                away_stats = team_stats_map.get(away_team_name) or team_stats_map.get(opp_id)
                
                if not home_stats or not away_stats:
                    return None
                    
                h_att = home_stats.get('att', 50.0)
                h_def = home_stats.get('def', 50.0)
                a_att = away_stats.get('att', 50.0)
                a_def = away_stats.get('def', 50.0)
                
                return {
                    'h_att': h_att,
                    'h_def': h_def,
                    'a_att': a_att,
                    'a_def': a_def,
                    'h_att_vs_a_def': h_att - a_def,
                    'h_def_vs_a_att': h_def - a_att,
                    'is_home': is_home
                }

            cmp_rows = []
            for i in range(total_matches):
                m_a = matches_1[i] if i < len(matches_1) else {'opp': '-', 'opp_short': '-', 'ha': '-', 'diff': 3, 'gw': None}
                m_b = matches_2[i] if i < len(matches_2) else {'opp': '-', 'opp_short': '-', 'ha': '-', 'diff': 3, 'gw': None}
                
                gw_val = m_a.get('gw') or m_b.get('gw') or f"{i+1}"
                gw_label = f"GW{gw_val}" if not str(gw_val).startswith("GW") else str(gw_val)
                
                if m_a.get('opp') != '-':
                    t1_match_str = f"{m_a['opp_short']} ({m_a['ha']}) [FDR {m_a['diff']}]"
                    diff_a = compute_match_differential(cmp_club1, m_a)
                    if diff_a:
                        if diff_a['is_home']:
                            att_diff = diff_a['h_att_vs_a_def']
                            def_diff = diff_a['h_def_vs_a_att']
                        else:
                            att_diff = diff_a['a_att'] - diff_a['h_def']
                            def_diff = diff_a['a_def'] - diff_a['h_att']
                        
                        t1_diff_str = f"Serang: {'+' if att_diff > 0 else ''}{att_diff:.1f} | Bertahan: {'+' if def_diff > 0 else ''}{def_diff:.1f}"
                    else:
                        t1_diff_str = "Data Strength N/A"
                else:
                    t1_match_str = "-"
                    t1_diff_str = "-"
                
                if m_b.get('opp') != '-':
                    t2_match_str = f"{m_b['opp_short']} ({m_b['ha']}) [FDR {m_b['diff']}]"
                    diff_b = compute_match_differential(cmp_club2, m_b)
                    if diff_b:
                        if diff_b['is_home']:
                            att_diff = diff_b['h_att_vs_a_def']
                            def_diff = diff_b['h_def_vs_a_att']
                        else:
                            att_diff = diff_b['a_att'] - diff_b['h_def']
                            def_diff = diff_b['a_def'] - diff_b['h_att']
                        
                        t2_diff_str = f"Serang: {'+' if att_diff > 0 else ''}{att_diff:.1f} | Bertahan: {'+' if def_diff > 0 else ''}{def_diff:.1f}"
                    else:
                        t2_diff_str = "Data Strength N/A"
                else:
                    t2_match_str = "-"
                    t2_diff_str = "-"

                if m_a.get('opp') == '-' and m_b.get('opp') == '-':
                    keuntungan = "-"
                elif m_a.get('opp') == '-':
                    keuntungan = f"🔵 {c2_item['club']}"
                elif m_b.get('opp') == '-':
                    keuntungan = f"🟢 {c1_item['club']}"
                elif m_a['diff'] < m_b['diff']:
                    keuntungan = f"🟢 {c1_item['club']} (FDR {m_a['diff']} vs {m_b['diff']})"
                elif m_b['diff'] < m_a['diff']:
                    keuntungan = f"🔵 {c2_item['club']} (FDR {m_b['diff']} vs {m_a['diff']})"
                else:
                    keuntungan = "⚖️ Seimbang"

                cmp_rows.append({
                    'Gameweek': gw_label,
                    f'Jadwal {c1_item["club"]}': t1_match_str,
                    f'Differential {c1_item["club"]}': t1_diff_str,
                    f'Jadwal {c2_item["club"]}': t2_match_str,
                    f'Differential {c2_item["club"]}': t2_diff_str,
                    'Keuntungan Jadwal': keuntungan
                })

            st.markdown(f"###### 📋 Seluruh Jadwal Mendatang & Analisis Differential ({len(cmp_rows)} Pertandingan)")
            st.caption(
                "💡 **Panduan Membaca Differential:**\n"
                "- **Serang (+) tinggi**: Serangan tim jauh melampaui pertahanan lawan ➔ **Prioritaskan pemain penyerang/gelandang serang (Attacking Assets)**.\n"
                "- **Bertahan (+) tinggi**: Pertahanan tim jauh lebih kokoh dibanding serangan lawan ➔ **Prioritaskan bek/kiper (Defensive/Clean Sheet Assets)**."
            )
            
            st.dataframe(
                pd.DataFrame(cmp_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Gameweek": st.column_config.TextColumn("Gameweek", pinned=True, width="small"),
                    f'Jadwal {c1_item["club"]}': st.column_config.TextColumn(f"Laga {c1_item['club']}", width="medium"),
                    f'Differential {c1_item["club"]}': st.column_config.TextColumn(f"Differential {c1_item['club']}", width="medium", help="Selisih Serangan vs Pertahanan Lawan & Pertahanan vs Serangan Lawan"),
                    f'Jadwal {c2_item["club"]}': st.column_config.TextColumn(f"Laga {c2_item['club']}", width="medium"),
                    f'Differential {c2_item["club"]}': st.column_config.TextColumn(f"Differential {c2_item['club']}", width="medium", help="Selisih Serangan vs Pertahanan Lawan & Pertahanan vs Serangan Lawan"),
                    "Keuntungan Jadwal": st.column_config.TextColumn("Keuntungan Jadwal", width="medium")
                }
            )

    # =========================================================================
    # TAB 4: ANALISIS PENGARUH HOME VS AWAY (FDR, SKOR SERANGAN & PERTAHANAN)
    # =========================================================================
    with tab_home_away:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%); padding: 18px 22px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #10b98133; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="max-width: 750px;">
                    <h3 style="margin: 0; color: #ffffff; font-size: 1.25rem; font-weight: 800;">
                        🏟️ Analisis Efek Home vs Away: Dampak Venue terhadap Hasil Match, FDR, Serangan & Pertahanan
                    </h3>
                    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.85rem; line-height: 1.45;">
                        Di Premier League, faktor kandang (<strong>Home Advantage</strong>) sangat memengaruhi efisiensi serangan, soliditas nirbobol (Clean Sheet), dan validitas tingkat kesulitan <strong>FDR</strong>. Sub-tab ini membedah data empiris hasil pertandingan, mengorelasikannya dengan level FDR, serta menguji ketangguhan <strong>Skor Serangan</strong> dan <strong>Skor Pertahanan</strong> 20 klub.
                    </p>
                </div>
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 8px 14px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #a7f3d0; text-transform: uppercase; font-weight: 700;">Venue Impact</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #34d399;">Empirical Analytics</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 1. Hitung Statistik Laga Selesai Liga & Performa Klub Home vs Away
        # ---------------------------------------------------------------------
        finished_fixtures = [f for f in fixtures_data if f.get('finished')]
        total_finished = len(finished_fixtures)

        if total_finished == 0:
            st.info("ℹ️ Belum ada pertandingan yang selesai di database musim ini untuk menghasilkan statistik Home vs Away historis.")
        else:
            # Hitung metrik liga keseluruhan
            total_h_wins = 0
            total_a_wins = 0
            total_draws = 0
            total_h_goals = 0
            total_a_goals = 0
            total_h_cs = 0
            total_a_cs = 0

            # Statistik FDR agregat berdasarkan Venue
            fdr_venue_stats = {
                2: {'h_m': 0, 'h_w': 0, 'h_d': 0, 'h_l': 0, 'h_gf': 0, 'h_ga': 0, 'h_cs': 0, 'a_m': 0, 'a_w': 0, 'a_d': 0, 'a_l': 0, 'a_gf': 0, 'a_ga': 0, 'a_cs': 0},
                3: {'h_m': 0, 'h_w': 0, 'h_d': 0, 'h_l': 0, 'h_gf': 0, 'h_ga': 0, 'h_cs': 0, 'a_m': 0, 'a_w': 0, 'a_d': 0, 'a_l': 0, 'a_gf': 0, 'a_ga': 0, 'a_cs': 0},
                4: {'h_m': 0, 'h_w': 0, 'h_d': 0, 'h_l': 0, 'h_gf': 0, 'h_ga': 0, 'h_cs': 0, 'a_m': 0, 'a_w': 0, 'a_d': 0, 'a_l': 0, 'a_gf': 0, 'a_ga': 0, 'a_cs': 0},
                5: {'h_m': 0, 'h_w': 0, 'h_d': 0, 'h_l': 0, 'h_gf': 0, 'h_ga': 0, 'h_cs': 0, 'a_m': 0, 'a_w': 0, 'a_d': 0, 'a_l': 0, 'a_gf': 0, 'a_ga': 0, 'a_cs': 0}
            }

            # Statistik Home vs Away per Klub
            club_ha_map = {}
            for t_id, t_name in teams_dict.items():
                t_short = club_short_map.get(t_id) or club_short_map.get(t_name, t_name[:3].upper())
                t_info = team_stats_map.get(t_id, {})
                club_ha_map[t_id] = {
                    't_id': t_id,
                    'club': t_name,
                    'short': t_short,
                    'category': t_info.get('category', 'Menengah'),
                    'att_score': t_info.get('att', 50.0),
                    'def_score': t_info.get('def', 50.0),
                    'overall_score': t_info.get('overall', 50.0),
                    'top_asset': t_info.get('top_asset', '-'),
                    # Home
                    'h_pld': 0, 'h_w': 0, 'h_d': 0, 'h_l': 0, 'h_gf': 0, 'h_ga': 0, 'h_cs': 0, 'h_pts': 0,
                    # Away
                    'a_pld': 0, 'a_w': 0, 'a_d': 0, 'a_l': 0, 'a_gf': 0, 'a_ga': 0, 'a_cs': 0, 'a_pts': 0
                }

            # Parsing seluruh finished match
            for f in finished_fixtures:
                th_id = f.get('team_h')
                ta_id = f.get('team_a')
                sh = int(f.get('team_h_score', 0) or 0)
                sa = int(f.get('team_a_score', 0) or 0)
                dh = min(5, max(2, int(round(float(f.get('team_h_difficulty', 3) or 3)))))
                da = min(5, max(2, int(round(float(f.get('team_a_difficulty', 3) or 3)))))

                total_h_goals += sh
                total_a_goals += sa
                if sa == 0: total_h_cs += 1
                if sh == 0: total_a_cs += 1

                if sh > sa:
                    total_h_wins += 1
                elif sa > sh:
                    total_a_wins += 1
                else:
                    total_draws += 1

                # Update FDR Breakdown
                st_h = fdr_venue_stats[dh]
                st_h['h_m'] += 1
                st_h['h_gf'] += sh
                st_h['h_ga'] += sa
                if sa == 0: st_h['h_cs'] += 1
                if sh > sa: st_h['h_w'] += 1
                elif sh == sa: st_h['h_d'] += 1
                else: st_h['h_l'] += 1

                st_a = fdr_venue_stats[da]
                st_a['a_m'] += 1
                st_a['a_gf'] += sa
                st_a['a_ga'] += sh
                if sh == 0: st_a['a_cs'] += 1
                if sa > sh: st_a['a_w'] += 1
                elif sa == sh: st_a['a_d'] += 1
                else: st_a['a_l'] += 1

                # Update Klub Home
                if th_id in club_ha_map:
                    c = club_ha_map[th_id]
                    c['h_pld'] += 1
                    c['h_gf'] += sh
                    c['h_ga'] += sa
                    if sa == 0: c['h_cs'] += 1
                    if sh > sa:
                        c['h_w'] += 1
                        c['h_pts'] += 3
                    elif sh == sa:
                        c['h_d'] += 1
                        c['h_pts'] += 1
                    else:
                        c['h_l'] += 1

                # Update Klub Away
                if ta_id in club_ha_map:
                    c = club_ha_map[ta_id]
                    c['a_pld'] += 1
                    c['a_gf'] += sa
                    c['a_ga'] += sh
                    if sh == 0: c['a_cs'] += 1
                    if sa > sh:
                        c['a_w'] += 1
                        c['a_pts'] += 3
                    elif sa == sh:
                        c['a_d'] += 1
                        c['a_pts'] += 1
                    else:
                        c['a_l'] += 1

            # Rata-rata liga
            avg_h_goals = total_h_goals / max(total_finished, 1)
            avg_a_goals = total_a_goals / max(total_finished, 1)
            pct_h_win = (total_h_wins / max(total_finished, 1)) * 100
            pct_a_win = (total_a_wins / max(total_finished, 1)) * 100
            pct_draw = (total_draws / max(total_finished, 1)) * 100
            pct_h_cs = (total_h_cs / max(total_finished, 1)) * 100
            pct_a_cs = (total_a_cs / max(total_finished, 1)) * 100
            goal_adv_pct = ((avg_h_goals - avg_a_goals) / max(avg_a_goals, 0.01)) * 100

            # -----------------------------------------------------------------
            # 2. Key Metrik Cards (Ringkasan Keunggulan Venue Liga)
            # -----------------------------------------------------------------
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Hasil Pertandingan</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 4px 0;">{pct_h_win:.1f}% Home Win</div>
                    <div style="font-size: 0.74rem; color: #475569;">Draw: <b>{pct_draw:.1f}%</b> | Away: <b>{pct_a_win:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col2:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Produktivitas Gol / Laga</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #10b981; margin: 4px 0;">{avg_h_goals:.2f} vs {avg_a_goals:.2f}</div>
                    <div style="font-size: 0.74rem; color: #059669;">Keunggulan Tuan Rumah: <b>+{goal_adv_pct:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col3:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Rasio Nirbobol (Clean Sheet)</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #3b82f6; margin: 4px 0;">{pct_h_cs:.1f}% vs {pct_a_cs:.1f}%</div>
                    <div style="font-size: 0.74rem; color: #1e40af;">Total Nirbobol: <b>{total_h_cs} H / {total_a_cs} A</b></div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col4:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Sampel Pertandingan</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #6366f1; margin: 4px 0;">{total_finished} Laga Selesai</div>
                    <div style="font-size: 0.74rem; color: #4338ca;">Total Gol: <b>{total_h_goals + total_a_goals} gol</b> ({((total_h_goals + total_a_goals)/max(total_finished,1)):.2f}/m)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # 3. Sub-bagian: Korelasi Empiris FDR x Venue (Home vs Away)
            # -----------------------------------------------------------------
            st.markdown("##### 📊 1. Korelasi Empiris: Tingkat Kesulitan FDR vs Venue (Home vs Away)")
            st.caption(
                "Menganalisis seberapa akurat level FDR memprediksi hasil pertandingan ketika tim bertanding di kandang sendiri (Home) dibandingkan saat bertandang (Away)."
            )

            fdr_chart_data = []
            fdr_table_data = []
            fdr_labels_map = {2: 'FDR 2 (Mudah)', 3: 'FDR 3 (Netral)', 4: 'FDR 4 (Sulit)', 5: 'FDR 5 (Sangat Sulit)'}

            for diff in [2, 3, 4, 5]:
                dt = fdr_venue_stats[diff]
                h_m = dt['h_m']
                a_m = dt['a_m']
                h_win_rate = (dt['h_w'] / h_m * 100) if h_m else 0
                a_win_rate = (dt['a_w'] / a_m * 100) if a_m else 0
                h_gf_avg = (dt['h_gf'] / h_m) if h_m else 0
                a_gf_avg = (dt['a_gf'] / a_m) if a_m else 0
                h_ga_avg = (dt['h_ga'] / h_m) if h_m else 0
                a_ga_avg = (dt['a_ga'] / a_m) if a_m else 0
                h_cs_rate = (dt['h_cs'] / h_m * 100) if h_m else 0
                a_cs_rate = (dt['a_cs'] / a_m * 100) if a_m else 0

                fdr_table_data.append({
                    'Tingkat FDR': fdr_labels_map[diff],
                    'Laga (H / A)': f"{h_m} / {a_m}",
                    'Win Rate Kandang': f"{h_win_rate:.1f}%",
                    'Win Rate Tandang': f"{a_win_rate:.1f}%",
                    'Rata Gol Dicetak (H vs A)': f"{h_gf_avg:.2f} vs {a_gf_avg:.2f}",
                    'Rata Kebobolan (H vs A)': f"{h_ga_avg:.2f} vs {a_ga_avg:.2f}",
                    'Peluang Clean Sheet (H vs A)': f"{h_cs_rate:.1f}% vs {a_cs_rate:.1f}%"
                })

                if h_m > 0:
                    fdr_chart_data.append({'FDR': f"FDR {diff}", 'Venue': 'Kandang (Home)', 'Rata Gol Dicetak': round(h_gf_avg, 2), 'Rata Kebobolan': round(h_ga_avg, 2), 'Win Rate (%)': round(h_win_rate, 1)})
                if a_m > 0:
                    fdr_chart_data.append({'FDR': f"FDR {diff}", 'Venue': 'Tandang (Away)', 'Rata Gol Dicetak': round(a_gf_avg, 2), 'Rata Kebobolan': round(a_ga_avg, 2), 'Win Rate (%)': round(a_win_rate, 1)})

            fdr_col_left, fdr_col_right = st.columns([3, 2])
            with fdr_col_left:
                if fdr_chart_data:
                    df_fdr_chart = pd.DataFrame(fdr_chart_data)
                    fig_fdr = px.bar(
                        df_fdr_chart,
                        x='FDR',
                        y='Rata Gol Dicetak',
                        color='Venue',
                        barmode='group',
                        color_discrete_map={'Kandang (Home)': '#10b981', 'Tandang (Away)': '#64748b'},
                        title="Rata-rata Gol Dicetak Tim Berdasarkan Level FDR & Venue",
                        text_auto=True
                    )
                    fig_fdr.update_layout(
                        margin=dict(l=10, r=10, t=35, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=280
                    )
                    st.plotly_chart(fig_fdr, use_container_width=True)

            with fdr_col_right:
                st.markdown("""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; font-size: 0.82rem; height: 100%;">
                    <div style="font-weight: 800; color: #1e293b; margin-bottom: 8px;">💡 Rekomendasi Transfer & Lineup FPL:</div>
                    <ul style="margin: 0; padding-left: 18px; color: #475569; line-height: 1.5;">
                        <li><strong>FDR 3 di Home = FDR 2 di Away:</strong> Penyerang dengan jadwal FDR 3 di kandang sering kali mencetak gol setara atau lebih tinggi dari penyerang dengan FDR 2 di tandang.</li>
                        <li><strong>Defensive Penalty di Laga Tandang:</strong> Probabilitas nirbobol (Clean Sheet) tim tandang anjlok drastis terutama saat menghadapi tim tuan rumah dengan Skor Serangan di atas 45.</li>
                        <li><strong>Diskon FDR untuk Tuan Rumah:</strong> Dalam perencanaan jangka panjang, pertimbangkan untuk memberi bobot ekstra (+0.4 poin xP) pada aset yang memiliki rentetan laga kandang.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(fdr_table_data), use_container_width=True, hide_index=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # 4. Sub-bagian: Pengaruh Skor Serangan & Pertahanan terhadap Venue
            # -----------------------------------------------------------------
            st.markdown("##### ⚔️ 2. Hubungan Skor Serangan & Pertahanan terhadap Efisiensi Venue")
            st.caption(
                "Memvisualisasikan bagaimana Skor Serangan (Attack Rating) dan Skor Pertahanan (Defense Rating) berinteraksi dengan atmosfer kandang/tandang."
            )

            # Buat dataset klub untuk visualisasi scatter & bar
            club_chart_rows = []
            for t_id, c in club_ha_map.items():
                h_p = max(c['h_pld'], 1)
                a_p = max(c['a_pld'], 1)
                h_gf_avg = c['h_gf'] / h_p
                a_gf_avg = c['a_gf'] / a_p
                h_ga_avg = c['h_ga'] / h_p
                a_ga_avg = c['a_ga'] / a_p
                h_ppg = c['h_pts'] / h_p
                a_ppg = c['a_pts'] / a_p

                # Home Advantage Index = Keuntungan Gol + Keuntungan Soliditas Pertahanan + Keuntungan Poin
                ha_index = round((h_ppg - a_ppg) * 10 + (h_gf_avg - a_gf_avg) * 5 + (a_ga_avg - h_ga_avg) * 5, 1)

                # Klasifikasi Profil Venue Klub
                if h_ppg >= 1.8 and (h_ppg - a_ppg) >= 0.8:
                    archetype = "🏰 Home Fortress"
                    rec_fpl = "Wajib Kapten/Mainkan saat Home, Cadangkan saat Away"
                elif h_ppg >= 1.7 and a_ppg >= 1.4:
                    archetype = "⚔️ All-Weather Elite"
                    rec_fpl = "Aset Inti Jangka Panjang (Set-and-Forget)"
                elif a_ppg >= h_ppg and a_gf_avg >= 1.2:
                    archetype = "✈️ Away Counter Specialist"
                    rec_fpl = "Sangat Bahaya saat Tandang (Transisi Cepat)"
                elif a_ga_avg >= 2.0:
                    archetype = "⚠️ Fragile Travellers"
                    rec_fpl = "Hindari Aset Bertahan saat Tandang"
                else:
                    archetype = "⚖️ Moderat / Seimbang"
                    rec_fpl = "Rotasi Fleksibel Berdasarkan FDR Lawan"

                club_chart_rows.append({
                    'Klub': c['club'],
                    'Kode': c['short'],
                    'Kategori Tim': c['category'],
                    'Skor Serangan': round(c['att_score'], 1),
                    'Skor Pertahanan': round(c['def_score'], 1),
                    'Indeks Kekuatan': round(c['overall_score'], 1),
                    'Top Aset': c['top_asset'],
                    'Laga (H / A)': f"{c['h_pld']} / {c['a_pld']}",
                    'Home W-D-L': f"{c['h_w']}-{c['h_d']}-{c['h_l']}",
                    'Away W-D-L': f"{c['a_w']}-{c['a_d']}-{c['a_l']}",
                    'Gol Home (Avg)': round(h_gf_avg, 2),
                    'Gol Away (Avg)': round(a_gf_avg, 2),
                    'Selisih Gol (H - A)': round(h_gf_avg - a_gf_avg, 2),
                    'Kebobolan Home (Avg)': round(h_ga_avg, 2),
                    'Kebobolan Away (Avg)': round(a_ga_avg, 2),
                    'CS (H / A)': f"{c['h_cs']} / {c['a_cs']}",
                    'PPG (Home vs Away)': f"{h_ppg:.2f} vs {a_ppg:.2f}",
                    'Home Advantage Index': ha_index,
                    'Profil Venue': archetype,
                    'Rekomendasi FPL': rec_fpl
                })

            df_club_ha = pd.DataFrame(club_chart_rows)

            col_scat, col_bar = st.columns(2)
            with col_scat:
                # Scatter Plot: Skor Serangan vs Produksi Gol Home vs Away
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=df_club_ha['Skor Serangan'],
                    y=df_club_ha['Gol Home (Avg)'],
                    mode='markers+text',
                    name='Laga Kandang (Home)',
                    text=df_club_ha['Kode'],
                    textposition='top center',
                    marker=dict(size=11, color='#10b981', line=dict(width=1, color='#047857'))
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=df_club_ha['Skor Serangan'],
                    y=df_club_ha['Gol Away (Avg)'],
                    mode='markers+text',
                    name='Laga Tandang (Away)',
                    text=df_club_ha['Kode'],
                    textposition='bottom center',
                    marker=dict(size=9, color='#64748b', line=dict(width=1, color='#334155'))
                ))
                fig_scatter.update_layout(
                    title="Korelasi Skor Serangan vs Rata-rata Gol Dicetak",
                    xaxis_title="Skor Serangan Klub (Attack Rating)",
                    yaxis_title="Rata-rata Gol per Pertandingan",
                    margin=dict(l=10, r=10, t=35, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=320
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_bar:
                # Bar Chart Kebobolan Home vs Away untuk Tim Berdasarkan Skor Pertahanan
                df_sorted_def = df_club_ha.sort_values(by='Skor Pertahanan', ascending=False).head(10)
                fig_def = go.Figure()
                fig_def.add_trace(go.Bar(
                    name='Kebobolan Home (Avg)',
                    x=df_sorted_def['Kode'],
                    y=df_sorted_def['Kebobolan Home (Avg)'],
                    marker_color='#3b82f6'
                ))
                fig_def.add_trace(go.Bar(
                    name='Kebobolan Away (Avg)',
                    x=df_sorted_def['Kode'],
                    y=df_sorted_def['Kebobolan Away (Avg)'],
                    marker_color='#ef4444'
                ))
                fig_def.update_layout(
                    title="Soliditas Pertahanan: Kebobolan Kandang vs Tandang (Top 10 Klub)",
                    xaxis_title="Klub",
                    yaxis_title="Rata-rata Kebobolan / Match",
                    barmode='group',
                    margin=dict(l=10, r=10, t=35, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=320
                )
                st.plotly_chart(fig_def, use_container_width=True)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # 5. Sub-bagian: Tabel Analisis 20 Klub: "Home Monsters vs Away Warriors"
            # -----------------------------------------------------------------
            st.markdown("##### 🏰 3. Klasifikasi Profil 20 Klub Premier League: Home Monsters vs Away Warriors")
            st.caption(
                "Gunakan tabel ini untuk melihat klub mana yang wajib Anda targetkan saat bermain di kandang, dan klub mana yang pertahanannya runtuh saat bermain tandang."
            )

            flt_col1, flt_col2, flt_col3 = st.columns([2, 2, 2])
            with flt_col1:
                archetype_filter = st.selectbox(
                    "Filter Profil Venue:",
                    ["Semua Profil", "🏰 Home Fortress", "⚔️ All-Weather Elite", "✈️ Away Counter Specialist", "⚠️ Fragile Travellers", "⚖️ Moderat / Seimbang"],
                    key="sb_archetype_filter"
                )
            with flt_col2:
                sort_col_opt = st.selectbox(
                    "Urutkan Berdasarkan:",
                    ["Home Advantage Index (Tertinggi)", "Skor Serangan (Tertinggi)", "Skor Pertahanan (Tertinggi)", "Gol Home Terbanyak", "Kebobolan Away Terbanyak"],
                    key="sb_ha_sort_col"
                )
            with flt_col3:
                search_ha_club = st.text_input("Cari Klub Spesifik:", "", placeholder="Ketik nama klub...", key="txt_search_ha_club")

            # Filter data
            df_display_ha = df_club_ha.copy()
            if archetype_filter and archetype_filter != "Semua Profil":
                df_display_ha = df_display_ha[df_display_ha['Profil Venue'] == archetype_filter]
            if search_ha_club.strip():
                kw = search_ha_club.strip().lower()
                df_display_ha = df_display_ha[df_display_ha['Klub'].str.lower().str.contains(kw) | df_display_ha['Kode'].str.lower().str.contains(kw)]

            # Sorting
            if "Home Advantage Index" in sort_col_opt:
                df_display_ha = df_display_ha.sort_values(by='Home Advantage Index', ascending=False)
            elif "Skor Serangan" in sort_col_opt:
                df_display_ha = df_display_ha.sort_values(by='Skor Serangan', ascending=False)
            elif "Skor Pertahanan" in sort_col_opt:
                df_display_ha = df_display_ha.sort_values(by='Skor Pertahanan', ascending=False)
            elif "Gol Home Terbanyak" in sort_col_opt:
                df_display_ha = df_display_ha.sort_values(by='Gol Home (Avg)', ascending=False)
            elif "Kebobolan Away Terbanyak" in sort_col_opt:
                df_display_ha = df_display_ha.sort_values(by='Kebobolan Away (Avg)', ascending=False)

            st.dataframe(
                df_display_ha[[
                    'Klub', 'Kode', 'Profil Venue', 'Home Advantage Index', 'Skor Serangan', 'Skor Pertahanan',
                    'Laga (H / A)', 'Home W-D-L', 'Away W-D-L', 'Gol Home (Avg)', 'Gol Away (Avg)',
                    'Kebobolan Home (Avg)', 'Kebobolan Away (Avg)', 'CS (H / A)', 'Rekomendasi FPL'
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Klub": st.column_config.TextColumn("Klub", pinned=True, width="medium"),
                    "Kode": st.column_config.TextColumn("Kode", width="small"),
                    "Profil Venue": st.column_config.TextColumn("Karakteristik Venue", width="medium"),
                    "Home Advantage Index": st.column_config.NumberColumn("Indeks Home Bias", format="%.1f", help="Semakin tinggi skor, semakin dominan klub saat bermain di kandang sendiri dibanding saat tandang."),
                    "Skor Serangan": st.column_config.NumberColumn("Skor Serang", format="%.1f"),
                    "Skor Pertahanan": st.column_config.NumberColumn("Skor Bertahan", format="%.1f"),
                    "Gol Home (Avg)": st.column_config.NumberColumn("Gol Home", format="%.2f"),
                    "Gol Away (Avg)": st.column_config.NumberColumn("Gol Away", format="%.2f"),
                    "Kebobolan Home (Avg)": st.column_config.NumberColumn("GA Home", format="%.2f"),
                    "Kebobolan Away (Avg)": st.column_config.NumberColumn("GA Away", format="%.2f"),
                    "Rekomendasi FPL": st.column_config.TextColumn("Panduan Strategi FPL", width="large")
                }
            )

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # 6. Sub-bagian: Simulator Matchday Mendatang dengan Dynamic Club-Specific Venue Factor & Poisson Bivariat
            # -----------------------------------------------------------------
            st.markdown("##### 🔮 4. Matchday Impact Simulator: Dynamic Club Venue Factor & Model Poisson Bivariat")
            st.caption(
                "Menerapkan penyesuaian **Dynamic Club-Specific Venue Factor** (disesuaikan dengan rasio kekuatan kandang/tandang historis masing-masing klub) "
                "dan **Model Bivariat Dixon-Coles Poisson** untuk mensimulasikan ekspektasi skor gol ($\lambda$), probabilitas hasil (Home/Draw/Away), Clean Sheet, serta Over/Under 2.5."
            )

            # Hitung Dynamic Club-Specific Venue Factor dictionary
            # home_boost: seberapa besar klub ini terangkat performanya di kandang vs rata-rata liga (skala -0.15 s/d +0.35)
            club_dynamic_venue = {}
            for t_id, c_data in club_ha_map.items():
                h_p = max(c_data['h_pld'], 1)
                a_p = max(c_data['a_pld'], 1)
                h_pts_avg = c_data['h_pts'] / h_p
                a_pts_avg = c_data['a_pts'] / a_p
                h_gf_avg = c_data['h_gf'] / h_p
                a_gf_avg = c_data['a_gf'] / a_p
                h_ga_avg = c_data['h_ga'] / h_p
                a_ga_avg = c_data['a_ga'] / a_p

                # Dynamic Venue Factor per klub dihitung dari selisih performa home vs away
                # Klub benteng kokoh (seperti Aston Villa / Newcastle / Liverpool) memiliki boost lebih tinggi
                perf_diff = (h_pts_avg - a_pts_avg) * 0.15 + (h_gf_avg - a_gf_avg) * 0.10 + (a_ga_avg - h_ga_avg) * 0.08
                dynamic_fdr_adj = float(np.clip(0.30 + perf_diff * 0.15, 0.15, 0.60))
                dynamic_att_boost = float(np.clip(4.0 + perf_diff * 4.0, 1.5, 9.0))
                dynamic_def_boost = float(np.clip(4.0 + (a_ga_avg - h_ga_avg) * 3.0, 1.0, 8.0))

                club_dynamic_venue[t_id] = {
                    'fdr_adj': dynamic_fdr_adj,
                    'att_boost': dynamic_att_boost,
                    'def_boost': dynamic_def_boost,
                    'ha_index': round((h_pts_avg - a_pts_avg) * 10 + (h_gf_avg - a_gf_avg) * 5, 1)
                }

            # Dapatkan gameweek mendatang
            all_gw_events = sorted(list({f.get('event') for f in fixtures_data if f.get('event') is not None}))
            unplayed_gw_events = sorted(list({
                f.get('event') for f in fixtures_data 
                if f.get('event') is not None and not f.get('finished', False)
            }))
            sim_default_gw = unplayed_gw_events[0] if unplayed_gw_events else (all_gw_events[-1] if all_gw_events else 1)

            sim_gw_col, sim_view_col = st.columns([2, 3])
            with sim_gw_col:
                selected_sim_gw = st.selectbox(
                    "Pilih Gameweek untuk Disimulasikan:",
                    unplayed_gw_events if unplayed_gw_events else all_gw_events,
                    index=0,
                    key="sb_selected_sim_gw"
                )
            with sim_view_col:
                sim_mode = st.radio(
                    "Mode Tampilan Simulator:",
                    ["📊 Ringkasan Matriks Laga & Poisson", "🎲 Detail Distribusi Skor Probabilistik"],
                    horizontal=True,
                    key="sim_view_mode_radio"
                )

            # Ambil seluruh pertandingan di Gameweek ini
            sim_fixtures = [f for f in fixtures_data if f.get('event') == selected_sim_gw]

            if not sim_fixtures:
                st.info(f"Tidak ada jadwal pertandingan ditemukan untuk Gameweek {selected_sim_gw}.")
            else:
                sim_rows = []
                poisson_details = []

                for fix in sim_fixtures:
                    h_id = fix.get('team_h')
                    a_id = fix.get('team_a')
                    h_name = teams_dict.get(h_id, f"Team {h_id}")
                    a_name = teams_dict.get(a_id, f"Team {a_id}")
                    h_short = club_short_map.get(h_id) or club_short_map.get(h_name, h_name[:3].upper())
                    a_short = club_short_map.get(a_id) or club_short_map.get(a_name, a_name[:3].upper())

                    # FDR Asli
                    fdr_h_raw = fix.get('team_h_difficulty', 3) or 3
                    fdr_a_raw = fix.get('team_a_difficulty', 3) or 3

                    # Dynamic Club-Specific Venue Adjustment
                    h_dyn = club_dynamic_venue.get(h_id, {'fdr_adj': 0.35, 'att_boost': 4.5, 'def_boost': 4.0})
                    a_dyn = club_dynamic_venue.get(a_id, {'fdr_adj': 0.35, 'att_boost': 4.5, 'def_boost': 4.0})

                    # FDR disesuaikan dengan profil spesifik kandang tuan rumah
                    adj_fdr_h = max(1.0, min(5.0, fdr_h_raw - h_dyn['fdr_adj']))
                    adj_fdr_a = max(1.0, min(5.0, fdr_a_raw + h_dyn['fdr_adj']))

                    # Skor Serangan & Pertahanan
                    h_info = team_stats_map.get(h_id, {})
                    a_info = team_stats_map.get(a_id, {})
                    h_att = h_info.get('att', 50.0)
                    h_def = h_info.get('def', 50.0)
                    a_att = a_info.get('att', 50.0)
                    a_def = a_info.get('def', 50.0)

                    # Dynamic Venue Mismatch (Tuan rumah mendapat boost spesifik klubnya)
                    h_att_effective = h_att + h_dyn['att_boost']
                    h_def_effective = h_def + h_dyn['def_boost']
                    a_att_effective = a_att - (a_dyn['att_boost'] * 0.6)
                    a_def_effective = a_def - (a_dyn['def_boost'] * 0.6)

                    # Differentials Serang terkalibrasi konsisten
                    h_attack_edge = round(h_att_effective - a_def_effective, 1)
                    a_attack_edge = round(a_att_effective - h_def_effective, 1)

                    # Hitung Lambda Poisson Gol Pertandingan
                    # Base gol rata-rata Premier League ~ 1.50 (Home) vs 1.25 (Away)
                    lambda_home = 1.48 * np.exp(h_attack_edge / 40.0)
                    lambda_away = 1.22 * np.exp(a_attack_edge / 40.0)

                    # Jalankan Model Bivariat Poisson / Dixon-Coles
                    sim_res = simulate_bivariate_poisson_match(lambda_home, lambda_away, max_goals=5, rho=-0.06)

                    # Outcome tag
                    p_hw = sim_res['p_home_win']
                    p_dr = sim_res['p_draw']
                    p_aw = sim_res['p_away_win']
                    if p_hw >= 50.0:
                        outcome_tag = f"🏠 Unggul Tuan Rumah ({p_hw:.0f}%)"
                    elif p_aw >= 45.0:
                        outcome_tag = f"✈️ Unggul Tim Tamu ({p_aw:.0f}%)"
                    elif p_dr >= 30.0 or abs(p_hw - p_aw) < 10.0:
                        outcome_tag = f"⚖️ Ketat / Imbang ({p_dr:.0f}%)"
                    else:
                        outcome_tag = f"🏠 Condong Home ({p_hw:.0f}%)"

                    # Rekomendasi Aset FPL
                    if h_attack_edge >= 8.0:
                        rec_target = f"⭐ Penyerang {h_short} ({h_info.get('top_asset', '-')})"
                    elif a_attack_edge >= 7.0:
                        rec_target = f"⭐ Penyerang {a_short} ({a_info.get('top_asset', '-')})"
                    elif sim_res['p_cs_h'] >= 42.0:
                        rec_target = f"🛡️ Pertahanan {h_short} ({sim_res['p_cs_h']:.0f}% CS)"
                    elif sim_res['p_cs_a'] >= 38.0:
                        rec_target = f"🛡️ Pertahanan {a_short} ({sim_res['p_cs_a']:.0f}% CS)"
                    else:
                        rec_target = f"🎯 Aset Kunci: {h_short} vs {a_short}"

                    sim_rows.append({
                        'Pertandingan': f"{h_name} vs {a_name}",
                        'FDR Asli (H vs A)': f"{fdr_h_raw} vs {fdr_a_raw}",
                        'Venue-Adjusted FDR': f"{adj_fdr_h:.2f} (H) vs {adj_fdr_a:.2f} (A)",
                        'Dynamic Home Boost': f"+{h_dyn['fdr_adj']:.2f} FDR / +{h_dyn['att_boost']:.1f} Att",
                        'Diff Serang Home': f"{'+' if h_attack_edge > 0 else ''}{h_attack_edge:.1f}",
                        'Diff Serang Away': f"{'+' if a_attack_edge > 0 else ''}{a_attack_edge:.1f}",
                        'Proyeksi Skor (xG)': f"{sim_res['lambda_h']:.2f} - {sim_res['lambda_a']:.2f}",
                        'Peluang Hasil (H/D/A)': f"{p_hw:.0f}% / {p_dr:.0f}% / {p_aw:.0f}%",
                        'Clean Sheet (H / A)': f"{h_short} {sim_res['p_cs_h']:.0f}% | {a_short} {sim_res['p_cs_a']:.0f}%",
                        'Skor Terfavorit': f"{sim_res['most_likely_score']} ({sim_res['most_likely_prob']:.1f}%)",
                        'Over 2.5 Gol': f"{sim_res['p_over_25']:.0f}%",
                        'Rekomendasi Aset FPL': rec_target
                    })

                    poisson_details.append({
                        'match_name': f"{h_name} vs {a_name}",
                        'h_name': h_name,
                        'a_name': a_name,
                        'h_short': h_short,
                        'a_short': a_short,
                        'sim_res': sim_res,
                        'rec_target': rec_target,
                        'h_dyn': h_dyn
                    })

                df_sim_summary = pd.DataFrame(sim_rows)

                if "Ringkasan" in sim_mode:
                    st.dataframe(
                        df_sim_summary[[
                            'Pertandingan', 'FDR Asli (H vs A)', 'Venue-Adjusted FDR', 'Dynamic Home Boost',
                            'Diff Serang Home', 'Diff Serang Away', 'Proyeksi Skor (xG)',
                            'Peluang Hasil (H/D/A)', 'Clean Sheet (H / A)', 'Skor Terfavorit', 'Over 2.5 Gol', 'Rekomendasi Aset FPL'
                        ]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Pertandingan": st.column_config.TextColumn("Pertandingan (Kandang vs Tandang)", pinned=True, width="large"),
                            "FDR Asli (H vs A)": st.column_config.TextColumn("FDR Asli", width="small"),
                            "Venue-Adjusted FDR": st.column_config.TextColumn("Dynamic FDR", width="medium", help="FDR yang dikoreksi faktor venue dinamis spesifik keunggulan kandang klub"),
                            "Dynamic Home Boost": st.column_config.TextColumn("Dynamic Factor", width="medium"),
                            "Diff Serang Home": st.column_config.TextColumn("Diff Serang (H)", width="small"),
                            "Diff Serang Away": st.column_config.TextColumn("Diff Serang (A)", width="small"),
                            "Proyeksi Skor (xG)": st.column_config.TextColumn("Exp Goals (λH-λA)", width="small"),
                            "Peluang Hasil (H/D/A)": st.column_config.TextColumn("Peluang (H/D/A)", width="medium"),
                            "Clean Sheet (H / A)": st.column_config.TextColumn("Dixon-Coles CS %", width="medium"),
                            "Skor Terfavorit": st.column_config.TextColumn("Skor Paling Mungkin", width="medium"),
                            "Over 2.5 Gol": st.column_config.TextColumn("Over 2.5", width="small"),
                            "Rekomendasi Aset FPL": st.column_config.TextColumn("Panduan Aset Unggulan", width="large")
                        }
                    )
                else:
                    # Tampilan Detail Probabilistik Grid per Match
                    st.markdown("###### 🎲 Matriks Distribusi Skor Pertandingan Bivariat Dixon-Coles Poisson:")
                    for p_item in poisson_details:
                        s_res = p_item['sim_res']
                        with st.expander(f"⚽ {p_item['match_name']} — Proyeksi xG: {s_res['lambda_h']:.2f} vs {s_res['lambda_a']:.2f} | Skor Terfavorit: {s_res['most_likely_score']} ({s_res['most_likely_prob']}%)", expanded=False):
                            pc1, pc2, pc3, pc4 = st.columns(4)
                            with pc1:
                                st.metric("Menang Tuan Rumah (Home)", f"{s_res['p_home_win']}%", f"λ Home = {s_res['lambda_h']:.2f}")
                            with pc2:
                                st.metric("Imbang (Draw)", f"{s_res['p_draw']}%")
                            with pc3:
                                st.metric("Menang Tim Tamu (Away)", f"{s_res['p_away_win']}%", f"λ Away = {s_res['lambda_a']:.2f}")
                            with pc4:
                                st.metric("Peluang Over 2.5 Gol", f"{s_res['p_over_25']}%", f"Under 2.5: {s_res['p_under_25']}%")

                            sc_col1, sc_col2 = st.columns([3, 2])
                            with sc_col1:
                                # Heatmap matriks skor
                                matrix_data = np.round(s_res['matrix'][:5, :5] * 100, 1)
                                fig_mat = px.imshow(
                                    matrix_data,
                                    labels=dict(x=f"Gol {p_item['a_short']} (Away)", y=f"Gol {p_item['h_short']} (Home)", color="Peluang (%)"),
                                    x=[f"{g} Gol" for g in range(5)],
                                    y=[f"{g} Gol" for g in range(5)],
                                    color_continuous_scale="Blues",
                                    text_auto=True,
                                    title=f"Distribusi Probabilitas Skor {p_item['h_short']} vs {p_item['a_short']} (%)"
                                )
                                fig_mat.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                                st.plotly_chart(fig_mat, use_container_width=True)

                            with sc_col2:
                                st.markdown(f"""
                                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.82rem; height: 100%;">
                                    <div style="font-weight: 700; color: #1e293b; margin-bottom: 6px;">📋 Rangkuman Simulasi Poisson:</div>
                                    <ul style="margin: 0; padding-left: 16px; color: #475569; line-height: 1.55;">
                                        <li><strong>Clean Sheet {p_item['h_short']}:</strong> {s_res['p_cs_h']:.1f}%</li>
                                        <li><strong>Clean Sheet {p_item['a_short']}:</strong> {s_res['p_cs_a']:.1f}%</li>
                                        <li><strong>Dynamic Venue Multiplier:</strong> +{p_item['h_dyn']['att_boost']:.1f} Serang, +{p_item['h_dyn']['def_boost']:.1f} Def</li>
                                        <li><strong>Rekomendasi FPL:</strong> {p_item['rec_target']}</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)

