"""
Fixtures and FDR Table View Tab.
"""

import pandas as pd
import streamlit as st

def render_tab_fixtures(fixtures_data, teams_dict, fdr_summary):
    """
    Renders Tab 4: Fixtures, Schedule and Fixture Difficulty Rating (FDR).
    """
    st.subheader("📅 Jadwal Pertandingan & Analisis Fixture Difficulty Rating (FDR)")
    st.write("Jadwal lengkap pertandingan mendatang beserta tingkat kesulitan (FDR) resmi Premier League untuk memudahkan perencanaan transfer jangka pendek (FDR3/FDR5) dan jangka panjang (FDR10).")

    # 1. Main FDR Comparison Table (FDR1, FDR3, FDR5, FDR10)
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown("#### 🏆 Peringkat Kemudahan Jadwal Klub (FDR1 - FDR10)")
    with col_hdr2:
        sort_fdr = st.selectbox(
            "Urutkan Berdasarkan:",
            options=["FDR3 (Rata-rata 3 Laga)", "FDR5 (Rata-rata 5 Laga)", "FDR10 (Rata-rata 10 Laga)", "FDR1 (Laga Terdekat)"],
            index=0,
            key="sort_fdr_sel"
        )

    fdr_table_data = []
    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        fdr_table_data.append({
            'Klub': t_name,
            'Lawan Laga Berikutnya': f_data.get('Next_Opponent_Fmt', '-'),
            'FDR1 (Laga Terdekat)': f_data.get('FDR1', 3.0),
            'FDR3 (Rata-rata 3 Laga)': f_data.get('FDR3', 3.0),
            'FDR5 (Rata-rata 5 Laga)': f_data.get('FDR5', 3.0),
            'FDR10 (Rata-rata 10 Laga)': f_data.get('FDR10', 3.0)
        })

    fdr_df = pd.DataFrame(fdr_table_data).sort_values(by=sort_fdr, ascending=True)

    st.dataframe(
        fdr_df,
        use_container_width=True,
        column_config={
            "FDR1 (Laga Terdekat)": st.column_config.NumberColumn(format="%.1f"),
            "FDR3 (Rata-rata 3 Laga)": st.column_config.NumberColumn(format="%.2f"),
            "FDR5 (Rata-rata 5 Laga)": st.column_config.NumberColumn(format="%.2f"),
            "FDR10 (Rata-rata 10 Laga)": st.column_config.NumberColumn(format="%.2f")
        }
    )
    st.caption("💡 *Catatan FDR: Skala 1 (Sangat Mudah) hingga 5 (Sangat Sulit). Nilai FDR yang lebih rendah menandakan jadwal pertandingan mendatang yang lebih menguntungkan.*")

    st.divider()

    # 2. Comprehensive 10-Match Fixture Matrix
    st.markdown("#### 🗓️ Matriks & Ticker 10 Pertandingan Mendatang (20 Klub Premier League)")
    st.write("Rincian lawan dan tingkat kesulitan (FDR) untuk 10 pertandingan mendatang setiap klub dengan **fill warna berdasarkan nilai FDR** untuk memudahkan identifikasi jadwal mudah vs sulit serta strategi rotasi pemain.")

    # Complete FDR Legend
    st.markdown("""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; margin-bottom: 14px; font-size: 0.84rem; align-items: center;">
        <span style="font-weight: 700; color: #475569; margin-right: 2px;">Indikator Warna FDR:</span>
        <span style="background: #15803d; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">FDR 1 (Sangat Mudah)</span>
        <span style="background: #22c55e; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">FDR 2 (Mudah)</span>
        <span style="background: #94a3b8; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">FDR 3 (Netral)</span>
        <span style="background: #f97316; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">FDR 4 (Sulit)</span>
        <span style="background: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; box-shadow: 0 1px 2px rgba(0,0,0,0.06);">FDR 5 (Sangat Sulit)</span>
    </div>
    """, unsafe_allow_html=True)

    SHORT_CLUB_NAMES = {
        'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
        'Brighton': 'BHA', 'Brighton and Hove Albion': 'BHA', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
        'Everton': 'EVE', 'Fulham': 'FUL', 'Ipswich': 'IPS', 'Ipswich Town': 'IPS', 'Leicester': 'LEI',
        'Leicester City': 'LEI', 'Liverpool': 'LIV', 'Man City': 'MCI', 'Manchester City': 'MCI',
        'Man Utd': 'MUN', 'Manchester Utd': 'MUN', 'Manchester United': 'MUN', 'Newcastle': 'NEW',
        'Newcastle United': 'NEW', 'Nott\'m Forest': 'NFO', 'Nottingham Forest': 'NFO',
        'Southampton': 'SOU', 'Spurs': 'TOT', 'Tottenham': 'TOT', 'Tottenham Hotspur': 'TOT',
        'West Ham': 'WHU', 'West Ham United': 'WHU', 'Wolves': 'WOL', 'Wolverhampton': 'WOL',
        'Wolverhampton Wanderers': 'WOL'
    }

    # Prepare raw structured data
    raw_matrix_items = []
    match_column_names = [f"Match +{i}" for i in range(1, 11)]

    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        up10 = f_data.get('upcoming_10', [])
        fdr10_val = f_data.get('FDR10', 3.0)

        row_df = {
            'Klub': t_name,
            'FDR10 Avg': fdr10_val
        }
        club_matches = []

        for idx in range(10):
            col_name = f"Match +{idx + 1}"
            if idx < len(up10):
                m = up10[idx]
                opp = m.get('opp_name', 'TBD')
                opp_short = SHORT_CLUB_NAMES.get(opp, opp[:3].upper())
                ha = "H" if m.get('is_home') == 1 else "A"
                diff = int(m.get('fdr', 3)) if m.get('fdr') is not None else 3
                row_df[col_name] = f"{opp_short} ({ha}) [{diff}]"
                club_matches.append({
                    'opp': opp,
                    'opp_short': opp_short,
                    'ha': ha,
                    'diff': diff,
                    'gw': m.get('gw')
                })
            else:
                row_df[col_name] = "-"
                club_matches.append({
                    'opp': '-',
                    'opp_short': '-',
                    'ha': '-',
                    'diff': 3,
                    'gw': None
                })

        raw_matrix_items.append({
            'club': t_name,
            'fdr10': fdr10_val,
            'matches': club_matches,
            'df_row': row_df
        })

    # Controls: Sorting and Filtering
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 3.5, 2])
    with ctrl_col1:
        sort_matrix_opt = st.selectbox(
            "Urutkan Matriks:",
            options=[
                "FDR10 Terendah (Jadwal Paling Mudah)",
                "FDR10 Tertinggi (Jadwal Paling Sulit)",
                "Nama Klub (A-Z)"
            ],
            index=0,
            key="sort_matrix_sel"
        )
    with ctrl_col2:
        all_club_names = sorted(list({item['club'] for item in raw_matrix_items}))
        selected_clubs_filter = st.multiselect(
            "Filter Klub Tertentu (Opsional):",
            options=all_club_names,
            default=[],
            placeholder="Semua 20 Klub",
            key="fdr_matrix_club_filter"
        )
    with ctrl_col3:
        view_mode = st.radio(
            "Format Tampilan:",
            options=["🎨 Blok Warna Visual", "📊 Dataframe"],
            horizontal=True,
            key="fdr_matrix_view_mode"
        )

    # Apply Filter
    filtered_items = raw_matrix_items
    if selected_clubs_filter:
        filtered_items = [it for it in filtered_items if it['club'] in selected_clubs_filter]

    # Apply Sort
    if sort_matrix_opt == "FDR10 Terendah (Jadwal Paling Mudah)":
        filtered_items = sorted(filtered_items, key=lambda x: x['fdr10'])
    elif sort_matrix_opt == "FDR10 Tertinggi (Jadwal Paling Sulit)":
        filtered_items = sorted(filtered_items, key=lambda x: x['fdr10'], reverse=True)
    else:
        filtered_items = sorted(filtered_items, key=lambda x: x['club'])

    # 1. VIEW MODE: BLOK WARNA VISUAL (TICKER GRID)
    if view_mode == "🎨 Blok Warna Visual":
        fdr_hex_map = {
            1: '#15803d',
            2: '#22c55e',
            3: '#94a3b8',
            4: '#f97316',
            5: '#ef4444'
        }

        html_blocks = [
            '<div style="overflow-x: auto; width: 100%; border-radius: 10px; border: 1px solid #e2e8f0; margin-top: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">'
            '<table style="width: 100%; border-collapse: separate; border-spacing: 0; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; font-size: 0.82rem; background: #ffffff;">'
            '<thead><tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">'
            '<th style="padding: 10px 14px; text-align: left; font-weight: 700; color: #1e293b; min-width: 140px; position: sticky; left: 0; background: #f8fafc; z-index: 2; border-bottom: 2px solid #e2e8f0;">Klub</th>'
            '<th style="padding: 10px 10px; text-align: center; font-weight: 700; color: #1e293b; min-width: 80px; border-bottom: 2px solid #e2e8f0;">FDR10 Avg</th>'
        ]

        for i in range(1, 11):
            html_blocks.append(f'<th style="padding: 10px 6px; text-align: center; font-weight: 700; color: #475569; min-width: 84px; border-bottom: 2px solid #e2e8f0;">Match +{i}</th>')

        html_blocks.append('</tr></thead><tbody>')

        for row in filtered_items:
            fdr10 = row['fdr10']
            badge_bg = '#dcfce7' if fdr10 <= 2.8 else ('#f1f5f9' if fdr10 <= 3.2 else '#fee2e2')
            badge_color = '#166534' if fdr10 <= 2.8 else ('#334155' if fdr10 <= 3.2 else '#991b1b')

            html_blocks.append('<tr style="border-bottom: 1px solid #f1f5f9;">')
            # Sticky Club Name
            html_blocks.append(
                f'<td style="padding: 9px 14px; text-align: left; font-weight: 700; color: #0f172a; position: sticky; left: 0; background: #ffffff; z-index: 1; border-bottom: 1px solid #f1f5f9; border-right: 1px solid #f1f5f9; white-space: nowrap;">'
                f'{row["club"]}'
                f'</td>'
            )
            # FDR10 Badge
            html_blocks.append(
                f'<td style="padding: 8px 8px; text-align: center; border-bottom: 1px solid #f1f5f9; border-right: 1px solid #f1f5f9;">'
                f'<span style="background-color: {badge_bg}; color: {badge_color}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">{fdr10:.2f}</span>'
                f'</td>'
            )

            # 10 Match Blocks
            for m in row['matches']:
                diff = m.get('diff', 3)
                opp_short = m.get('opp_short', '-')
                ha = m.get('ha', '-')
                opp_full = m.get('opp', '-')

                if opp_short == '-':
                    html_blocks.append('<td style="padding: 6px 4px; text-align: center; border-bottom: 1px solid #f1f5f9; color: #94a3b8;">-</td>')
                else:
                    bg = fdr_hex_map.get(diff, '#94a3b8')
                    ha_text = 'Kandang' if ha == 'H' else 'Tandang'
                    gw_str = f"GW {m['gw']}" if m.get('gw') else ""
                    html_blocks.append(
                        f'<td style="padding: 5px 4px; text-align: center; border-bottom: 1px solid #f1f5f9;">'
                        f'<div title="{opp_full} ({ha_text}) {gw_str} · Kesulitan FDR {diff}" style="background-color: {bg}; color: #ffffff; font-weight: 700; border-radius: 6px; padding: 6px 2px; font-size: 0.77rem; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.07); line-height: 1.25; cursor: default;">'
                        f'<div>{opp_short} ({ha})</div>'
                        f'<div style="font-size: 0.68rem; opacity: 0.95; font-weight: 600; margin-top: 1px;">FDR {diff}</div>'
                        f'</div>'
                        f'</td>'
                    )

            html_blocks.append('</tr>')

        html_blocks.append('</tbody></table></div>')
        st.markdown(''.join(html_blocks), unsafe_allow_html=True)
        st.caption("💡 *Arahkan kursor ke kotak pertandingan untuk melihat nama lengkap klub lawan dan gameweek.*")

    # 2. VIEW MODE: DATAFRAME INTERAKTIF (STYLED)
    else:
        df_rows = [it['df_row'] for it in filtered_items]
        matrix_df = pd.DataFrame(df_rows)

        def style_fdr_cell(val):
            if isinstance(val, str) and '[' in val and ']' in val:
                try:
                    fdr_str = val.split('[')[1].split(']')[0].strip()
                    diff = int(fdr_str)
                    colors = {
                        1: 'background-color: #15803d; color: #ffffff; font-weight: 700; text-align: center;',
                        2: 'background-color: #22c55e; color: #ffffff; font-weight: 700; text-align: center;',
                        3: 'background-color: #94a3b8; color: #ffffff; font-weight: 700; text-align: center;',
                        4: 'background-color: #f97316; color: #ffffff; font-weight: 700; text-align: center;',
                        5: 'background-color: #ef4444; color: #ffffff; font-weight: 700; text-align: center;'
                    }
                    return colors.get(diff, '')
                except Exception:
                    return ''
            return ''

        def style_fdr_avg(val):
            try:
                v = float(val)
                if v <= 2.8:
                    return 'background-color: #dcfce7; color: #166534; font-weight: 700; text-align: center;'
                elif v <= 3.2:
                    return 'background-color: #f1f5f9; color: #334155; font-weight: 700; text-align: center;'
                else:
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: 700; text-align: center;'
            except Exception:
                return ''

        styled_matrix = (
            matrix_df.style
            .map(style_fdr_cell, subset=match_column_names)
            .map(style_fdr_avg, subset=['FDR10 Avg'])
            .format({'FDR10 Avg': '{:.2f}'})
        )

        st.dataframe(
            styled_matrix,
            use_container_width=True,
            hide_index=True
        )

