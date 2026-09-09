"""
Fixtures and FDR Table View Tab.
Ultra-polished, responsive Fixture Matrix, Interactive Ticker, and Swing Insights.
"""

import pandas as pd
import streamlit as st
from src.processors import get_team_short_map

SHORT_CLUB_NAMES = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Brighton and Hove Albion': 'BHA', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Ipswich': 'IPS', 'Ipswich Town': 'IPS', 'Leicester': 'LEI',
    'Leicester City': 'LEI', 'Liverpool': 'LIV', 'Man City': 'MCI', 'Manchester City': 'MCI',
    'Man Utd': 'MUN', 'Manchester Utd': 'MUN', 'Manchester United': 'MUN', 'Newcastle': 'NEW',
    'Newcastle United': 'NEW', "Nott'm Forest": 'NFO', 'Nottingham Forest': 'NFO',
    'Southampton': 'SOU', 'Spurs': 'TOT', 'Tottenham': 'TOT', 'Tottenham Hotspur': 'TOT',
    'West Ham': 'WHU', 'West Ham United': 'WHU', 'Wolves': 'WOL', 'Wolverhampton': 'WOL',
    'Wolverhampton Wanderers': 'WOL'
}

FDR_PALETTE = {
    1: {'bg': '#15803d', 'text': '#ffffff', 'label': 'Sangat Mudah', 'dot': '🟢'},
    2: {'bg': '#10b981', 'text': '#ffffff', 'label': 'Mudah', 'dot': '🟢'},
    3: {'bg': '#64748b', 'text': '#ffffff', 'label': 'Netral', 'dot': '⚪'},
    4: {'bg': '#f59e0b', 'text': '#ffffff', 'label': 'Sulit', 'dot': '🟠'},
    5: {'bg': '#ef4444', 'text': '#ffffff', 'label': 'Sangat Sulit', 'dot': '🔴'},
}

def render_tab_fixtures(fixtures_data, teams_dict, fdr_summary, fpl_data=None):
    """
    Renders Fixtures, Schedule and Fixture Difficulty Rating (FDR) Matrix & Ticker.
    """
    # Dynamic short club names extracted directly from FPL API
    club_short_map = get_team_short_map(fpl_data) if fpl_data else SHORT_CLUB_NAMES
    # Header Banner Card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 20px 24px; color: #ffffff; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 0.76rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.08em;">Official Premier League Fixture Difficulty Rating</span>
                <h2 style="margin: 4px 0 6px 0; font-size: 1.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">📅 Matriks Fixtures & Ticker FDR 10 Match</h2>
                <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.5; max-width: 820px;">
                    Jadwal komprehensif 20 klub Premier League dengan indikator visual tingkat kesulitan (FDR), 
                    analisis horizon 3/5/10 laga, deteksi jadwal kandang/tandang, serta rekomendasi rotasi pertahanan.
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Klub</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">20 Tim</div>
                </div>
                <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Horizon Jadwal</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #10b981;">10 Laga</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not fdr_summary:
        st.warning("Data FDR dan jadwal pertandingan tidak tersedia.")
        return

    # -------------------------------------------------------------------------
    # 1. STRATEGIC FIXTURE INSIGHTS & SWING HIGHLIGHTS
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. RAW STRUCTURED DATA COMPILATION
    # -------------------------------------------------------------------------
    raw_matrix_items = []
    
    # Deteksi Gameweek aktual dari fixture pertama yang valid
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
        fdr1_val = f_data.get('FDR1', 3.0)
        fdr3_val = f_data.get('FDR3', 3.0)
        fdr5_val = f_data.get('FDR5', 3.0)
        fdr10_val = f_data.get('FDR10', 3.0)

        club_matches = []
        home_count_10 = 0

        for idx in range(10):
            if idx < len(up10):
                m = up10[idx]
                opp = m.get('opp_name', 'TBD')
                opp_short = m.get('opp_short') or club_short_map.get(m.get('opp_id')) or club_short_map.get(opp, opp[:3].upper())
                ha = "H" if m.get('is_home') == 1 else "A"
                if ha == "H":
                    home_count_10 += 1
                diff = int(m.get('fdr', 3)) if m.get('fdr') is not None else 3
                gw_num = m.get('gw') or (detected_gw_labels[idx] if idx < len(detected_gw_labels) else idx + 1)
                club_matches.append({
                    'opp': opp,
                    'opp_short': opp_short,
                    'ha': ha,
                    'diff': diff,
                    'gw': gw_num
                })
            else:
                club_matches.append({
                    'opp': '-',
                    'opp_short': '-',
                    'ha': '-',
                    'diff': 3,
                    'gw': None
                })

        # Calculate streak dot string
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
            'dots_3': dots_3,
            'dots_5': dots_5,
            'dots_10': dots_10
        })

    # -------------------------------------------------------------------------
    # 3. INTERACTIVE CONTROLS BAR (HORIZON, PRESETS, SORT, VIEW)
    # -------------------------------------------------------------------------
    st.markdown("#### 🗓️ Matriks & Ticker Interaktif FDR")
    st.write("Sesuaikan cakupan laga (3, 5, atau 10 match), filter klub berdasarkan preset kemudahan jadwal, dan periksa detail kesulitan laga.")

    # Control Row 1: Horizon & Quick Filter Presets
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
        all_club_names = sorted(list({item['club'] for item in raw_matrix_items}))
        selected_clubs_filter = st.multiselect(
            "Pilih Klub Tertentu (Opsional):",
            options=all_club_names,
            default=[],
            placeholder="Semua 20 Klub",
            key="fdr_matrix_club_filter"
        )

    # Control Row 2: Sort & View Mode
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

    # Apply Presets & Filters
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
        )[:8]

    # Apply manual club multiselect if specified
    if selected_clubs_filter:
        filtered_items = [it for it in filtered_items if it['club'] in selected_clubs_filter]

    # Apply Sorting
    if "Terendah" in sort_matrix_opt:
        filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key])
    elif "Tertinggi" in sort_matrix_opt:
        filtered_items = sorted(filtered_items, key=lambda x: x[active_fdr_key], reverse=True)
    elif "Kandang" in sort_matrix_opt:
        filtered_items = sorted(
            filtered_items, 
            key=lambda x: (sum(1 for m in x['matches'][:horizon_n] if m.get('ha') == 'H'), -x[active_fdr_key]), 
            reverse=True
        )
    else:
        filtered_items = sorted(filtered_items, key=lambda x: x['club'])

    # FDR Legend Display
    st.markdown("""
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; margin-bottom: 14px; font-size: 0.82rem; align-items: center; background: #ffffff; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0;">
        <span style="font-weight: 700; color: #1e293b; margin-right: 4px;">Legenda FDR:</span>
        <span style="background: #15803d; color: white; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 0.78rem;">FDR 1 (Sangat Mudah)</span>
        <span style="background: #10b981; color: white; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 0.78rem;">FDR 2 (Mudah)</span>
        <span style="background: #64748b; color: white; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 0.78rem;">FDR 3 (Netral)</span>
        <span style="background: #f59e0b; color: white; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 0.78rem;">FDR 4 (Sulit)</span>
        <span style="background: #ef4444; color: white; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 0.78rem;">FDR 5 (Sangat Sulit)</span>
        <span style="margin-left: auto; color: #64748b; font-size: 0.78rem;">
            <span style="background: #0f172a; color: white; padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">H</span> Kandang &nbsp;
            <span style="background: #e2e8f0; color: #334155; padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">A</span> Tandang
        </span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. VIEW RENDERING: 🎨 BLOK WARNA VISUAL (TICKER GRID)
    # -------------------------------------------------------------------------
    if "Blok Warna Visual" in view_mode:
        html_blocks = [
            '<div style="overflow-x: auto; width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 6px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">'
            '<table style="width: 100%; border-collapse: separate; border-spacing: 0; font-family: \'Plus Jakarta Sans\', -apple-system, sans-serif; font-size: 0.84rem; background: #ffffff;">'
            '<thead><tr style="background: #f8fafc; border-bottom: 2px solid #cbd5e1;">'
            '<th style="padding: 12px 16px; text-align: left; font-weight: 800; color: #0f172a; min-width: 170px; position: sticky; left: 0; background: #f8fafc; z-index: 3; border-bottom: 2px solid #cbd5e1; border-right: 1px solid #e2e8f0;">Klub</th>'
            f'<th style="padding: 12px 10px; text-align: center; font-weight: 800; color: #0f172a; min-width: 90px; border-bottom: 2px solid #cbd5e1; border-right: 1px solid #e2e8f0;">Rerata {active_fdr_label}</th>'
            f'<th style="padding: 12px 10px; text-align: center; font-weight: 700; color: #475569; min-width: 80px; border-bottom: 2px solid #cbd5e1; border-right: 1px solid #e2e8f0;">H / A</th>'
        ]

        # Column Headers for Match 1 to horizon_n with actual Gameweek label
        for idx in range(horizon_n):
            gw_val = detected_gw_labels[idx] if idx < len(detected_gw_labels) and detected_gw_labels[idx] else None
            gw_title = f"GW {gw_val}" if gw_val else f"Match +{idx + 1}"
            sub_title = f"Laga +{idx + 1}" if gw_val else ""
            html_blocks.append(
                f'<th style="padding: 10px 8px; text-align: center; font-weight: 700; color: #1e293b; min-width: 88px; border-bottom: 2px solid #cbd5e1;">'
                f'<div style="font-weight: 800; font-size: 0.84rem;">{gw_title}</div>'
                f'<div style="font-size: 0.7rem; color: #64748b; font-weight: 600;">{sub_title}</div>'
                f'</th>'
            )

        html_blocks.append('</tr></thead><tbody>')

        for row in filtered_items:
            fdr_score = row[active_fdr_key]
            
            # Color coding for average badge
            if fdr_score <= 2.6:
                badge_bg = '#dcfce7'
                badge_color = '#15803d'
                badge_border = '#86efac'
            elif fdr_score <= 3.2:
                badge_bg = '#f1f5f9'
                badge_color = '#334155'
                badge_border = '#cbd5e1'
            else:
                badge_bg = '#fee2e2'
                badge_color = '#b91c1c'
                badge_border = '#fca5a5'

            # Calculate Home/Away counts in current horizon
            home_in_horizon = sum(1 for m in row['matches'][:horizon_n] if m.get('ha') == 'H')
            away_in_horizon = horizon_n - home_in_horizon

            html_blocks.append('<tr style="border-bottom: 1px solid #f1f5f9; transition: background 0.15s ease;">')
            
            # 1. Sticky Club Name Cell with Run Streak
            streak_dots = "".join([FDR_PALETTE.get(m['diff'], {}).get('dot', '⚪') for m in row['matches'][:horizon_n]])
            html_blocks.append(
                f'<td style="padding: 10px 16px; text-align: left; position: sticky; left: 0; background: #ffffff; z-index: 2; border-bottom: 1px solid #f1f5f9; border-right: 1px solid #e2e8f0; white-space: nowrap;">'
                f'<div style="font-weight: 800; color: #0f172a; font-size: 0.88rem;">{row["club"]}</div>'
                f'<div style="font-size: 0.7rem; margin-top: 2px; letter-spacing: 1px;" title="Rangkaian kesulitan: {streak_dots}">{streak_dots}</div>'
                f'</td>'
            )

            # 2. Average FDR Badge
            html_blocks.append(
                f'<td style="padding: 8px 8px; text-align: center; border-bottom: 1px solid #f1f5f9; border-right: 1px solid #e2e8f0;">'
                f'<span style="background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 3px 8px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">{fdr_score:.2f}</span>'
                f'</td>'
            )

            # 3. Home / Away tally in horizon
            html_blocks.append(
                f'<td style="padding: 8px 8px; text-align: center; border-bottom: 1px solid #f1f5f9; border-right: 1px solid #e2e8f0; font-size: 0.78rem; font-weight: 600; color: #475569;">'
                f'<span style="color: #0f172a; font-weight: 700;">{home_in_horizon}H</span> / {away_in_horizon}A'
                f'</td>'
            )

            # 4. Fixture Cells for the chosen horizon
            for m in row['matches'][:horizon_n]:
                diff = m.get('diff', 3)
                opp_short = m.get('opp_short', '-')
                ha = m.get('ha', '-')
                opp_full = m.get('opp', '-')
                gw_val = m.get('gw', '')

                if opp_short == '-':
                    html_blocks.append('<td style="padding: 6px 4px; text-align: center; border-bottom: 1px solid #f1f5f9; color: #94a3b8;">-</td>')
                else:
                    pal = FDR_PALETTE.get(diff, FDR_PALETTE[3])
                    bg = pal['bg']
                    ha_desc = 'Kandang' if ha == 'H' else 'Tandang'
                    ha_badge_style = (
                        'background: #0f172a; color: #ffffff;' if ha == 'H'
                        else 'background: rgba(255,255,255,0.25); color: #ffffff; border: 1px solid rgba(255,255,255,0.4);'
                    )

                    tooltip = f"{row['club']} vs {opp_full} ({ha_desc}) · GW {gw_val} · Kesulitan FDR {diff} ({pal['label']})"

                    html_blocks.append(
                        f'<td style="padding: 6px 4px; text-align: center; border-bottom: 1px solid #f1f5f9;">'
                        f'<div title="{tooltip}" style="background-color: {bg}; color: #ffffff; border-radius: 8px; padding: 7px 4px; font-size: 0.8rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); line-height: 1.25; cursor: pointer; transition: transform 0.12s ease, box-shadow 0.12s ease;">'
                        f'<div style="font-weight: 800; letter-spacing: -0.01em; display: flex; align-items: center; justify-content: center; gap: 4px;">'
                        f'<span>{opp_short}</span>'
                        f'<span style="{ha_badge_style} font-size: 0.65rem; font-weight: 800; padding: 1px 4px; border-radius: 3px;">{ha}</span>'
                        f'</div>'
                        f'<div style="font-size: 0.68rem; opacity: 0.92; font-weight: 600; margin-top: 2px;">FDR {diff}</div>'
                        f'</div>'
                        f'</td>'
                    )

            html_blocks.append('</tr>')

        html_blocks.append('</tbody></table></div>')
        st.markdown(''.join(html_blocks), unsafe_allow_html=True)
        st.caption("💡 *Arahkan kursor ke setiap kotak pertandingan untuk melihat informasi lengkap klub lawan, status venue (Kandang/Tandang), dan nomor Gameweek.*")

    # -------------------------------------------------------------------------
    # 5. VIEW RENDERING: 📊 DATAFRAME INTERAKTIF
    # -------------------------------------------------------------------------
    else:
        df_rows = []
        match_cols = [f"Match +{i + 1}" for i in range(horizon_n)]

        for item in filtered_items:
            row_dict = {
                'Klub': item['club'],
                f'{active_fdr_label} Avg': item[active_fdr_key],
                'Kandang / Tandang': f"{sum(1 for m in item['matches'][:horizon_n] if m.get('ha') == 'H')}H / {horizon_n - sum(1 for m in item['matches'][:horizon_n] if m.get('ha') == 'H')}A"
            }
            for idx in range(horizon_n):
                m = item['matches'][idx]
                gw_val = m.get('gw')
                gw_txt = f"GW{gw_val} " if gw_val else ""
                row_dict[match_cols[idx]] = f"{gw_txt}{m['opp_short']} ({m['ha']}) [{m['diff']}]"
            df_rows.append(row_dict)

        matrix_df = pd.DataFrame(df_rows)

        def style_fdr_cell(val):
            if isinstance(val, str) and '[' in val and ']' in val:
                try:
                    fdr_str = val.split('[')[1].split(']')[0].strip()
                    diff = int(fdr_str)
                    pal = FDR_PALETTE.get(diff, FDR_PALETTE[3])
                    return f'background-color: {pal["bg"]}; color: {pal["text"]}; font-weight: 700; text-align: center;'
                except Exception:
                    return ''
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

    # -------------------------------------------------------------------------
    # 6. DIRECT CLUB-VS-CLUB FIXTURE COMPARISON ACCORDION
    # -------------------------------------------------------------------------
    with st.expander("⚔️ Komparasi Head-to-Head Jadwal 2 Klub (Duel Fixtures)", expanded=False):
        st.markdown("Bandingkan jadwal langsung antara dua klub Premier League untuk memutuskan aset tim mana yang harus Anda prioritaskan.")
        
        c_cmp1, c_cmp2 = st.columns(2)
        with c_cmp1:
            cmp_club1 = st.selectbox("Pilih Klub Pertama:", options=all_club_names, index=0, key="cmp_club_1")
        with c_cmp2:
            default_c2_idx = 1 if len(all_club_names) > 1 else 0
            cmp_club2 = st.selectbox("Pilih Klub Kedua:", options=all_club_names, index=default_c2_idx, key="cmp_club_2")

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

            # Mini visual comparison table
            cmp_rows = []
            for i in range(5):
                m_a = c1_item['matches'][i]
                m_b = c2_item['matches'][i]
                gw_label = f"GW{m_a.get('gw', i+1)}"
                cmp_rows.append({
                    'Gameweek': gw_label,
                    f'{c1_item["club"]}': f"{m_a['opp_short']} ({m_a['ha']}) [FDR {m_a['diff']}]",
                    f'{c2_item["club"]}': f"{m_b['opp_short']} ({m_b['ha']}) [FDR {m_b['diff']}]",
                    'Keuntungan': f"🟢 {c1_item['club']}" if m_a['diff'] < m_b['diff'] else (f"🔵 {c2_item['club']}" if m_b['diff'] < m_a['diff'] else "⚖️ Seimbang")
                })
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
