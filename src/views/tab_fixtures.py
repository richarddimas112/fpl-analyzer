"""
Fixtures and FDR Table View Tab.
Ultra-polished, responsive Fixture Matrix, Interactive Ticker, Matchday Hub with Home/Away Differentials, and Swing Insights.
"""

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from src.processors import get_team_short_map

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

def render_tab_fixtures(fixtures_data, teams_dict, fdr_summary, fpl_data=None, df_teams=None):
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
    # -------------------------------------------------------------------------
    tab_matchday, tab_matrix, tab_h2h = st.tabs([
        "⚔️ Matchday Gameweek Berikutnya (10 Match & Differentials)",
        "📅 Matriks & Ticker FDR 10 Match",
        "👥 Komparasi Head-to-Head 2 Klub"
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

                # Differentials:
                # 1. Home Attack vs Away Defense (Positive = Home Attack heavily favored)
                h_att_diff = round(h_att - a_def, 1)
                # 2. Home Defense vs Away Attack (Positive = Home Clean Sheet potential high)
                h_def_diff = round(h_def - a_att, 1)
                # 3. Away Attack vs Home Defense (Positive = Away Attack heavily favored)
                a_att_diff = round(a_att - h_def, 1)
                # 4. Away Defense vs Home Attack (Positive = Away Clean Sheet potential high)
                a_def_diff = round(a_def - h_att, 1)

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
                            st.markdown(card_html, unsafe_allow_html=True)

                            # Quick Launchers Deep Dive
                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                if st.button(f"🔍 Profil {match_item['home_short']}", key=f"btn_card_h_{match_item['id']}", use_container_width=True):
                                    st.session_state['selected_deepdive_club'] = match_item['home_name']
                                    st.session_state['active_nav_id'] = "team_strength"
                                    st.rerun()
                            with col_b2:
                                if st.button(f"🔍 Profil {match_item['away_short']}", key=f"btn_card_a_{match_item['id']}", use_container_width=True):
                                    st.session_state['selected_deepdive_club'] = match_item['away_name']
                                    st.session_state['active_nav_id'] = "team_strength"
                                    st.rerun()

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
                        'Diff Serang Home': f"{'+' if m['h_att_diff'] > 0 else ''}{m['h_att_diff']:.1f}",
                        'Diff CS Home': f"{'+' if m['h_def_diff'] > 0 else ''}{m['h_def_diff']:.1f}",
                        'Diff Serang Away': f"{'+' if m['a_att_diff'] > 0 else ''}{m['a_att_diff']:.1f}",
                        'Diff CS Away': f"{'+' if m['a_def_diff'] > 0 else ''}{m['a_def_diff']:.1f}",
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
                        "Diff Serang Home": st.column_config.TextColumn("Diff Serang Home", width="small", help="Selisih Serangan Home vs Pertahanan Lawan"),
                        "Diff CS Home": st.column_config.TextColumn("Diff CS Home", width="small", help="Selisih Pertahanan Home vs Serangan Lawan (Peluang Clean Sheet)"),
                        "Diff Serang Away": st.column_config.TextColumn("Diff Serang Away", width="small", help="Selisih Serangan Away vs Pertahanan Lawan"),
                        "Diff CS Away": st.column_config.TextColumn("Diff CS Away", width="small", help="Selisih Pertahanan Away vs Serangan Lawan"),
                        "Top Aset Home": st.column_config.TextColumn("Aset Utama Home", width="small"),
                        "Top Aset Away": st.column_config.TextColumn("Aset Utama Away", width="small"),
                        "Scout Recommendation": st.column_config.TextColumn("Rekomendasi Taktis FPL", width="large")
                    }
                )

    # =========================================================================
    # TAB 2: MATRIKS & TICKER FDR 10 MATCH
    # =========================================================================
    with tab_matrix:
        # Strategic Fixture Insights & Swing Highlights
        clubs_ranked_fdr3 = sorted(
            [{'id': tid, 'name': teams_dict.get(tid, f"Team {tid}"), 'fdr': f['FDR3'], 'fdr5': f['FDR5'], 'fdr10': f['FDR10'], 'next': f.get('Next_Opponent_Fmt', '-')} 
             for tid, f in fdr_summary.items()],
            key=lambda x: x['fdr']
        )

        top_easy_3 = clubs_ranked_fdr3[:3]
        top_hard_3 = clubs_ranked_fdr3[-3:][::-1]

        st.markdown("##### ⚡ Ringkasan Strategi Fixture Run (3-5 Laga Mendatang)")
        c_easy, c_hard, c_rot = st.columns([1.2, 1.2, 1.6])

        with c_easy:
            easy_list_html = "".join([
                f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem;'>"
                f"<div><strong style='color: #0f172a;'>{c['name']}</strong><div style='font-size: 0.75rem; color: #64748b;'>Lawan: {c['next']}</div></div>"
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
                {easy_list_html}
            </div>
            """, unsafe_allow_html=True)

        with c_hard:
            hard_list_html = "".join([
                f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem;'>"
                f"<div><strong style='color: #0f172a;'>{c['name']}</strong><div style='font-size: 0.75rem; color: #64748b;'>Lawan: {c['next']}</div></div>"
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
                {hard_list_html}
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
            horizon_n = 3 if "3 Laga" in horizon_choice else (5 if "5 Laga" in horizon_choice else 10)
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
            filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key])[:5]
        elif preset_choice == "🔴 Top 5 Jadwal Paling Sulit (Run Merah)":
            filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key], reverse=True)[:5]
        elif preset_choice == "👑 Big Six Premier League":
            filtered_items = [it for it in filtered_items if any(b.lower() in it['club'].lower() for b in big_six)]
        elif preset_choice == "🏠 Klub Terbanyak Laga Kandang (Home Run)":
            filtered_items = sorted(
                filtered_items, 
                key=lambda x: sum(1 for m in x['matches'][:horizon_n] if m.get('ha') == 'H'), 
                reverse=True
            )[:7]

        if selected_clubs_filter:
            filtered_items = [it for it in filtered_items if it['club'] in selected_clubs_filter]

        if "Terendah" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key])
        elif "Tertinggi" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key], reverse=True)
        elif "Paling Banyak Laga Kandang" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: sum(1 for m in x['matches'][:horizon_n] if m.get('ha') == 'H'), reverse=True)
        elif "Nama Klub" in sort_matrix_opt:
            filtered_items = sorted(filtered_items, key=lambda x: x['club'])

        legend_html = f"""
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 14px 0; padding: 10px 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; font-size: 0.8rem;">
            <strong style="color: #1e293b;">Legenda Tingkat Kesulitan (FDR):</strong>
            <span style="background: #15803d; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">1 Sangat Mudah</span>
            <span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">2 Mudah</span>
            <span style="background: #64748b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">3 Netral</span>
            <span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">4 Sulit</span>
            <span style="background: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;">5 Sangat Sulit</span>
            <span style="color: #64748b; margin-left: auto; font-style: italic;">Huruf Kapital (H) = Home / Kandang, (A) = Away / Tandang</span>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        if "Blok Warna Visual" in view_mode:
            table_header_cols = ""
            for idx in range(horizon_n):
                gw_val = detected_gw_labels[idx] if idx < len(detected_gw_labels) and detected_gw_labels[idx] is not None else None
                col_header = f"GW{gw_val}" if gw_val is not None else f"M{idx+1}"
                table_header_cols += f"<th style='text-align: center; padding: 8px 4px; font-size: 0.78rem; color: #475569; min-width: 60px;'>{col_header}</th>"

            table_rows_html = ""
            for rank, item in enumerate(filtered_items, 1):
                club_name = item['club']
                short_code = item['short_name']
                fdr_val = item[active_fdr_key]
                
                fdr_badge_color = "#10b981" if fdr_val <= 2.6 else ("#64748b" if fdr_val <= 3.2 else "#ef4444")
                fdr_badge_bg = "#ecfdf5" if fdr_val <= 2.6 else ("#f1f5f9" if fdr_val <= 3.2 else "#fef2f2")

                match_blocks_html = ""
                for m in item['matches'][:horizon_n]:
                    fdr_diff = m['diff']
                    fdr_meta = FDR_PALETTE.get(fdr_diff, {'bg': '#94a3b8', 'text': '#ffffff'})
                    ha_badge = m['ha']
                    opp_label = m['opp_short']
                    block_tooltip = f"{club_name} vs {m['opp']} ({'Home' if ha_badge == 'H' else 'Away'}) | FDR {fdr_diff}"
                    
                    match_blocks_html += f"""
                    <td style="padding: 4px 3px; text-align: center;">
                        <div title="{block_tooltip}" style="background-color: {fdr_meta['bg']}; color: {fdr_meta['text']}; font-weight: 700; border-radius: 6px; padding: 6px 2px; font-size: 0.76rem; box-shadow: 0 1px 2px rgba(0,0,0,0.08); transition: transform 0.1s ease; cursor: default;">
                            <div>{opp_label}</div>
                            <div style="font-size: 0.65rem; opacity: 0.9; font-weight: 600;">({ha_badge})</div>
                        </div>
                    </td>
                    """
                
                home_in_horizon = sum(1 for m in item['matches'][:horizon_n] if m.get('ha') == 'H')

                table_rows_html += f"""
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 8px 12px; font-size: 0.85rem; font-weight: 700; color: #0f172a; white-space: nowrap;">
                        <span style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; margin-right: 6px;">#{rank}</span>
                        {club_name}
                        <span style="font-size: 0.72rem; color: #64748b; font-weight: 600; margin-left: 4px;">({short_code})</span>
                    </td>
                    <td style="padding: 8px 6px; text-align: center; white-space: nowrap;">
                        <span style="background: {fdr_badge_bg}; color: {fdr_badge_color}; border: 1px solid {fdr_badge_color}33; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">
                            {fdr_val:.2f}
                        </span>
                    </td>
                    <td style="padding: 8px 6px; text-align: center; font-size: 0.8rem; font-weight: 600; color: #334155;">
                        <span style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 2px 6px; border-radius: 4px;">{home_in_horizon}H / {horizon_n - home_in_horizon}A</span>
                    </td>
                    {match_blocks_html}
                </tr>
                """

            full_matrix_html = f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse; min-width: 650px;">
                    <thead>
                        <tr style="background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                            <th style="text-align: left; padding: 10px 12px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">Klub Premier League</th>
                            <th style="text-align: center; padding: 10px 6px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">{active_fdr_label} Avg</th>
                            <th style="text-align: center; padding: 10px 6px; font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">Venue</th>
                            {table_header_cols}
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
            """
            st.markdown(full_matrix_html, unsafe_allow_html=True)
        else:
            df_rows = []
            for item in filtered_items:
                row_data = {
                    'Klub': item['club'],
                    'Kode': item['short_name'],
                    f'{active_fdr_label} Avg': item[active_fdr_key],
                    'Laga Kandang': sum(1 for m in item['matches'][:horizon_n] if m.get('ha') == 'H'),
                }
                for idx, m in enumerate(item['matches'][:horizon_n]):
                    gw_val = detected_gw_labels[idx] if idx < len(detected_gw_labels) and detected_gw_labels[idx] is not None else None
                    col_name = f"GW{gw_val}" if gw_val is not None else f"M{idx+1}"
                    row_data[col_name] = f"{m['opp_short']} ({m['ha']}) [{m['diff']}]"
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

            styled_matrix = (
                matrix_df.style
                .map(style_fdr_cell, subset=match_cols)
                .map(style_fdr_avg, subset=[f'{active_fdr_label} Avg'])
                .format({f'{active_fdr_label} Avg': '{:.2f}'})
            )

            st.dataframe(
                styled_matrix,
                use_container_width=True,
                hide_index=True
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
