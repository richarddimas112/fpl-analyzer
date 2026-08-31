"""
Team Strength Analysis Tab View.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.processors import calculate_team_strength_analysis

def render_tab_team_strength(fpl_data, players_df, fdr_summary, fixtures_data=None, teams_dict=None):
    """
    Renders Tab 3: Comprehensive Premier League Team Strength Analysis.
    Includes:
    1. KPI Summary Cards & Tier Filtering Table
    2. Interactive Bar Chart with customizable comparison metrics
    3. Offensive vs Defensive Quadrant Scatter Plot with Reference Crosshairs & Quadrant Labels
    4. Squad Value vs Points Efficiency Plot
    5. Deep-Dive Club Analysis & Top 5 FPL Assets with Upcoming Fixtures Schedule
    """
    st.subheader("🛡️ Analisis Komprehensif Kekuatan Tim Premier League (Team Strength Analysis)")
    st.write("Modul agregasi statistik 20 klub Premier League: mengevaluasi rata-rata poin FPL pemain, daya gedor ofensif (Gol, xG, xA), soliditas pertahanan (Clean Sheet, xGC, Saves), serta tingkat kemudahan jadwal pertandingan mendatang (FDR).")

    df_teams = calculate_team_strength_analysis(fpl_data, players_df, fdr_summary)

    if not df_teams.empty:
        # 1. Highlights / Summary KPI Row
        top_strength_team = df_teams.sort_values(by="Indeks Kekuatan", ascending=False).iloc[0]
        top_attack_team = df_teams.sort_values(by="Total xG", ascending=False).iloc[0]
        top_defense_team = df_teams.sort_values(by="Clean Sheet", ascending=False).iloc[0]
        easiest_fdr_team = df_teams.sort_values(by="FDR3", ascending=True).iloc[0]

        tk1, tk2, tk3, tk4 = st.columns(4)
        with tk1:
            st.metric(
                "👑 Tim Terkuat (Indeks Tertinggi)",
                f"{top_strength_team['Klub']} ({top_strength_team['Indeks Kekuatan']})",
                top_strength_team['Kategori Tim']
            )
        with tk2:
            st.metric(
                "⚔️ Serangan Tertajam",
                f"{top_attack_team['Klub']} ({top_attack_team['Total xG']:.2f} xG)",
                f"{top_attack_team['Total Gol']} Gol dicetak"
            )
        with tk3:
            st.metric(
                "🛡️ Pertahanan Terkokoh",
                f"{top_defense_team['Klub']} ({top_defense_team['Clean Sheet']} CS)",
                f"{top_defense_team['Total Saves']} Saves"
            )
        with tk4:
            st.metric(
                "🗓️ Jadwal Termudah (FDR3)",
                f"{easiest_fdr_team['Klub']} (FDR: {easiest_fdr_team['FDR3']:.2f})",
                f"Lawan: {easiest_fdr_team['Lawan Berikutnya']}"
            )

        # 2. Controls: Filter & Sort
        st.markdown("##### 🔍 Filter & Urutkan Data Tim")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            team_search = st.text_input("Cari Nama Klub", "", placeholder="Misal: Arsenal, Liverpool...", key="team_search_input")
        with t_col2:
            tier_filter = st.selectbox(
                "Filter Kategori Kekuatan:",
                options=["Semua Kategori", "🏆 Elite Contender", "🌟 Top Tier Challenger", "⚖️ Mid-Table Stable", "⚠️ Underdogs / Rebuilding"],
                key="team_tier_filter"
            )
        with t_col3:
            sort_col = st.selectbox(
                "Urutkan Berdasarkan:",
                options=[
                    "Indeks Kekuatan", "Rata-rata Poin Pemain", "Total Poin Skuad",
                    "Total Gol", "Total xG", "Clean Sheet", "Total xGC", "FDR3", "FDR5", "Kemudahan Jadwal (%)"
                ],
                index=0,
                key="team_sort_col"
            )

        filtered_teams = df_teams.copy()
        if team_search:
            filtered_teams = filtered_teams[filtered_teams['Klub'].str.contains(team_search, case=False, na=False)]
        if tier_filter != "Semua Kategori":
            filtered_teams = filtered_teams[filtered_teams['Kategori Tim'] == tier_filter]

        is_asc = (sort_col in ["FDR3", "FDR5", "Total xGC"])
        filtered_teams = filtered_teams.sort_values(by=sort_col, ascending=is_asc)

        # 3. Comprehensive Sortable Table
        st.markdown("##### 📋 Tabel Agregasi & Pemeringkatan Kekuatan Tim")
        team_display_cols = [
            'Klub', 'Indeks Kekuatan', 'Kategori Tim', 'Rata-rata Poin Pemain', 'Pemain Aktif', 'Total Poin Skuad',
            'Skor Serangan', 'Total Gol', 'Total xG', 'Total xA', 'Top Scorer', 'Top Creator',
            'Skor Pertahanan', 'Clean Sheet', 'Total xGC', 'Total Saves', 'Top Aset FPL',
            'FDR1', 'FDR3', 'FDR5', 'Lawan Berikutnya', 'Nilai Skuad (£m)'
        ]

        st.dataframe(
            filtered_teams[team_display_cols],
            use_container_width=True,
            height=460,
            column_config={
                "Indeks Kekuatan": st.column_config.ProgressColumn(
                    "Indeks Kekuatan",
                    help="Indeks Komposit (0 - 100): 20% Serangan, 20% Pertahanan, 15% Rata-rata Poin, 30% Official EPL Strength, 15% Kemudahan FDR.",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "Skor Serangan": st.column_config.NumberColumn(format="%.1f"),
                "Skor Pertahanan": st.column_config.NumberColumn(format="%.1f"),
                "Rata-rata Poin Pemain": st.column_config.NumberColumn(format="%.2f pts", help="Rata-rata poin per pemain yang sudah pernah bermain"),
                "Total Poin Skuad": st.column_config.NumberColumn(format="%d pts"),
                "Total xG": st.column_config.NumberColumn(format="%.2f"),
                "Total xA": st.column_config.NumberColumn(format="%.2f"),
                "Total xGC": st.column_config.NumberColumn(format="%.2f"),
                "Nilai Skuad (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                "FDR1": st.column_config.NumberColumn("FDR1", format="%.1f", help="Tingkat Kesulitan Lawan GW Terdekat (1 Laga)"),
                "FDR3": st.column_config.NumberColumn("FDR3", format="%.2f", help="Rata-rata FDR 3 Pertandingan Mendatang"),
                "FDR5": st.column_config.NumberColumn("FDR5", format="%.2f", help="Rata-rata FDR 5 Pertandingan Mendatang")
            }
        )

        st.caption("💡 *Formula Indeks Kekuatan: 20% Metrik Serangan (xG & Gol) + 20% Pertahanan (Clean Sheet & xGC) + 15% Efisiensi Poin Pemain + 30% Official EPL Strength + 15% Kemudahan Jadwal (FDR3).*")

        # 4. Interactive Visualizations & Deep-Dive for Teams
        st.markdown("---")
        st.markdown("##### 📊 Visualisasi Perbandingan & Analisis Klub")

        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
            "🏆 Komparasi Bar Chart Tim",
            "⚔️ vs 🛡️ Matriks Ofensif vs Defensif (Kuadran)",
            "💰 Efisiensi Nilai Skuad (Poin vs Harga)",
            "🔍 Deep-Dive Analisis Klub & Top Aset FPL"
        ])

        # TAB 1: CUSTOMIZABLE BAR CHART COMPARISON
        with v_tab1:
            st.markdown("###### 📊 Komparasi Antar Klub Berdasarkan Pilihan Metrik")
            
            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                chosen_bar_metric = st.selectbox(
                    "Pilih Metrik untuk Komparasi Bar Chart:",
                    options=[
                        "Indeks Kekuatan",
                        "Rata-rata Poin Pemain",
                        "Total Poin Skuad",
                        "Total Gol",
                        "Total xG",
                        "Clean Sheet",
                        "Total xA",
                        "Total Saves",
                        "Skor Serangan",
                        "Skor Pertahanan",
                        "Nilai Skuad (£m)"
                    ],
                    index=0,
                    key="team_bar_metric_sel"
                )
            with b_col2:
                sort_bar_order = st.radio(
                    "Urutan Tampilan:",
                    options=["Tertinggi ke Terendah", "Terendah ke Tertinggi"],
                    horizontal=True,
                    key="team_bar_sort_order"
                )

            is_asc_bar = (sort_bar_order == "Terendah ke Tertinggi")
            df_bar_sorted = df_teams.sort_values(by=chosen_bar_metric, ascending=is_asc_bar)

            # Palette selection based on metric type
            if "Pertahanan" in chosen_bar_metric or "Clean" in chosen_bar_metric or "Saves" in chosen_bar_metric:
                c_scale = "Tealgrn"
            elif "Gol" in chosen_bar_metric or "xG" in chosen_bar_metric or "Serangan" in chosen_bar_metric:
                c_scale = "OrRd"
            elif "Nilai" in chosen_bar_metric:
                c_scale = "Purp"
            else:
                c_scale = "Blues"

            fig_bar = px.bar(
                df_bar_sorted,
                x=chosen_bar_metric,
                y="Klub",
                orientation='h',
                color=chosen_bar_metric,
                color_continuous_scale=c_scale,
                text=chosen_bar_metric,
                title=f"Perbandingan 20 Tim Premier League: {chosen_bar_metric}"
            )
            fig_bar.update_traces(
                texttemplate='%{text:.2f}' if df_bar_sorted[chosen_bar_metric].dtype == float else '%{text}',
                textposition='outside'
            )
            fig_bar.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                height=560,
                xaxis=dict(gridcolor="#e2e8f0", title=chosen_bar_metric),
                yaxis=dict(gridcolor="#e2e8f0", title="", categoryorder='total ascending' if not is_asc_bar else 'total descending'),
                margin=dict(l=20, r=40, t=50, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 2: OFFENSIVE VS DEFENSIVE QUADRANT MATRIX WITH DIVIDER LINES
        with v_tab2:
            st.markdown("###### ⚔️ vs 🛡️ Matriks Kuadran Skor Serangan vs Skor Pertahanan")

            avg_att = float(df_teams['Skor Serangan'].mean())
            avg_def = float(df_teams['Skor Pertahanan'].mean())

            fig_matrix = px.scatter(
                df_teams,
                x="Skor Serangan",
                y="Skor Pertahanan",
                size="Rata-rata Poin Pemain",
                color="Kategori Tim",
                text="Kode",
                hover_name="Klub",
                hover_data=["Indeks Kekuatan", "Total Gol", "Clean Sheet", "Total xG", "Total xGC", "Top Scorer"],
                title="Matriks Skor Serangan vs Skor Pertahanan (Garis Pemisah Kuadran Rata-rata Liga)"
            )
            fig_matrix.update_traces(textposition='top center')

            # 1. Garis Pemisah Kuadran Vertikal (Rata-rata Serangan)
            fig_matrix.add_vline(
                x=avg_att,
                line_dash="dash",
                line_color="#94a3b8",
                line_width=1.5,
                annotation_text=f"Rata-rata Serangan ({avg_att:.1f})",
                annotation_position="top",
                annotation_font=dict(size=11, color="#64748b")
            )

            # 2. Garis Pemisah Kuadran Horizontal (Rata-rata Pertahanan)
            fig_matrix.add_hline(
                y=avg_def,
                line_dash="dash",
                line_color="#94a3b8",
                line_width=1.5,
                annotation_text=f"Rata-rata Pertahanan ({avg_def:.1f})",
                annotation_position="right",
                annotation_font=dict(size=11, color="#64748b")
            )

            # 3. Anotasi Label 4 Kuadran
            min_x, max_x = df_teams['Skor Serangan'].min(), df_teams['Skor Serangan'].max()
            min_y, max_y = df_teams['Skor Pertahanan'].min(), df_teams['Skor Pertahanan'].max()
            pad_x = (max_x - min_x) * 0.08
            pad_y = (max_y - min_y) * 0.08

            fig_matrix.add_annotation(
                x=max_x + pad_x, y=max_y + pad_y,
                text="<b>👑 KUADRAN I: ELITE CONTENDER</b><br><span style='font-size:10px;'>Serangan Mematikan & Pertahanan Kokoh</span>",
                showarrow=False,
                align="right",
                font=dict(size=11, color="#15803d"),
                bgcolor="rgba(240, 253, 244, 0.8)",
                bordercolor="#86efac",
                borderwidth=1,
                borderpad=4
            )
            fig_matrix.add_annotation(
                x=min_x - pad_x, y=max_y + pad_y,
                text="<b>🛡️ KUADRAN II: DEFENSIVE SOLID</b><br><span style='font-size:10px;'>Pertahanan Kuat, Daya Gedor Terbatas</span>",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#0369a1"),
                bgcolor="rgba(240, 249, 255, 0.8)",
                bordercolor="#7dd3fc",
                borderwidth=1,
                borderpad=4
            )
            fig_matrix.add_annotation(
                x=max_x + pad_x, y=min_y - pad_y,
                text="<b>⚔️ KUADRAN IV: ULTRA ATTACKING</b><br><span style='font-size:10px;'>Serangan Tajam, Pertahanan Terbuka</span>",
                showarrow=False,
                align="right",
                font=dict(size=11, color="#b45309"),
                bgcolor="rgba(254, 243, 199, 0.8)",
                bordercolor="#fde68a",
                borderwidth=1,
                borderpad=4
            )
            fig_matrix.add_annotation(
                x=min_x - pad_x, y=min_y - pad_y,
                text="<b>⚠️ KUADRAN III: REBUILDING / LOW</b><br><span style='font-size:10px;'>Produktivitas Rendah & Rentan Kebobolan</span>",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#b91c1c"),
                bgcolor="rgba(254, 242, 242, 0.8)",
                bordercolor="#fca5a5",
                borderwidth=1,
                borderpad=4
            )

            fig_matrix.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                height=560,
                xaxis=dict(gridcolor="#e2e8f0", title="Skor Serangan (Ofensif)"),
                yaxis=dict(gridcolor="#e2e8f0", title="Skor Pertahanan (Defensif)"),
                legend=dict(bordercolor="#e2e8f0", borderwidth=1)
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
            st.caption("💡 *Garis putus-putus abu-abu membagi kuadran berdasarkan nilai rata-rata liga untuk Skor Serangan dan Skor Pertahanan.*")

        # TAB 3: SQUAD VALUE EFFICIENCY PLOT
        with v_tab3:
            fig_val = px.scatter(
                df_teams,
                x="Nilai Skuad (£m)",
                y="Total Poin Skuad",
                color="Kategori Tim",
                text="Kode",
                hover_name="Klub",
                hover_data=["Indeks Kekuatan", "Rata-rata Poin Pemain", "Pemain Aktif"],
                trendline="ols",
                title="Analisis Efisiensi Nilai Skuad (£m) vs Total Poin Skuad"
            )
            fig_val.update_traces(textposition='top center')
            fig_val.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                height=520,
                xaxis=dict(gridcolor="#e2e8f0", title="Total Nilai Skuad (£m)"),
                yaxis=dict(gridcolor="#e2e8f0", title="Total Poin Skuad")
            )
            st.plotly_chart(fig_val, use_container_width=True)
            st.caption("💡 *Tim di atas garis tren regresi OLS menghasilkan poin lebih tinggi dari nilai harga skuad mereka (Value for Money tinggi).*")

        # TAB 4: DEEP-DIVE ANALISIS KLUB & TOP ASET FPL
        with v_tab4:
            st.markdown("###### 🔍 Deep-Dive Analisis Profil Klub & Top Aset FPL Utama")
            
            club_names = sorted(df_teams['Klub'].unique().tolist())
            
            # Default selection
            default_club_idx = 0
            for idx, cname in enumerate(club_names):
                if cname in ["Arsenal", "Liverpool", "Manchester City"]:
                    default_club_idx = idx
                    break

            selected_club = st.selectbox(
                "Pilih Klub Premier League yang Ingin Dianalisis Secara Mendalam:",
                options=club_names,
                index=default_club_idx,
                key="team_deepdive_selector"
            )

            c_info = df_teams[df_teams['Klub'] == selected_club].iloc[0]
            t_id = c_info.get('team_id')

            # Display Club Profile Card & Metrics
            st.markdown(f"""
            <div style="padding: 16px 20px; background-color: #f8fafc; border-left: 5px solid #2563eb; border-radius: 10px; margin-bottom: 16px; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <h3 style="margin: 0; color: #0f172a; font-size: 1.35rem; font-weight: 800;">{c_info['Klub']} ({c_info['Kode']})</h3>
                        <p style="margin: 3px 0 0 0; color: #64748b; font-size: 0.9rem;">Kategori: <strong style="color: #1e40af;">{c_info['Kategori Tim']}</strong> · Lawan GW Berikutnya: <strong>{c_info['Lawan Berikutnya']}</strong></p>
                    </div>
                    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 6px 14px; border-radius: 20px; font-weight: 700; color: #1d4ed8; font-size: 0.92rem;">
                        Indeks Kekuatan: {c_info['Indeks Kekuatan']}/100
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Row 1
            dd1, dd2, dd3, dd4 = st.columns(4)
            with dd1:
                st.metric("Indeks Kekuatan", f"{c_info['Indeks Kekuatan']:.1f}", c_info['Kategori Tim'])
            with dd2:
                st.metric("Skor Serangan", f"{c_info['Skor Serangan']:.1f}", f"{c_info['Total Gol']} Gol · {c_info['Total xG']:.1f} xG")
            with dd3:
                st.metric("Skor Pertahanan", f"{c_info['Skor Pertahanan']:.1f}", f"{c_info['Clean Sheet']} CS · {c_info['Total xGC']:.1f} xGC")
            with dd4:
                st.metric("FDR3 (3 Laga)", f"{c_info['FDR3']:.2f}", f"Lawan: {c_info['Lawan Berikutnya']}")

            # Metrics Row 2
            dd5, dd6, dd7, dd8 = st.columns(4)
            with dd5:
                st.metric("Total Poin Skuad", f"{c_info['Total Poin Skuad']} pts", f"{c_info['Pemain Aktif']} Pemain Aktif")
            with dd6:
                st.metric("Rata-rata Poin Pemain", f"{c_info['Rata-rata Poin Pemain']:.2f} pts", f"Skuad: {c_info['Rata-rata Poin Skuad']:.1f} pts")
            with dd7:
                st.metric("Nilai Skuad (£m)", f"£{c_info['Nilai Skuad (£m)']:.1f}m", f"BPS: {c_info['Total BPS']}")
            with dd8:
                st.metric("Total Saves Kiper", f"{c_info['Total Saves']} Saves", f"xGI: {c_info['Total xGI']:.1f}")

            st.markdown("---")

            # TOP 5 FPL ASSETS FOR THIS CLUB
            st.markdown(f"##### ⭐ Top 5 Aset Utama FPL: **{selected_club}**")
            
            # Filter players for this club
            if 'team' in players_df.columns and t_id is not None:
                club_players = players_df[players_df['team'] == t_id].copy()
            else:
                club_players = players_df[players_df['Klub'] == selected_club].copy()

            if not club_players.empty:
                # Rank top 5 players by Total Poin, xPoin, and Form
                top_5_assets = club_players.sort_values(by=['Total Poin', 'xPoin'], ascending=False).head(5)

                top_asset_cols = [
                    'Nama Pemain', 'Posisi', 'Harga (£m)', 'Total Poin', 'xPoin', 'Form',
                    'xG', 'xA', 'Avg Mins (L5M)', '% Ownership', 'Status', 'Peluang Main GW (%)'
                ]

                # Ensure all columns exist
                valid_top_cols = [c for c in top_asset_cols if c in top_5_assets.columns]

                st.dataframe(
                    top_5_assets[valid_top_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                        "Total Poin": st.column_config.NumberColumn(format="%d pts"),
                        "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
                        "Form": st.column_config.NumberColumn(format="%.2f"),
                        "xG": st.column_config.NumberColumn(format="%.2f"),
                        "xA": st.column_config.NumberColumn(format="%.2f"),
                        "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins"),
                        "% Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                        "Peluang Main GW (%)": st.column_config.ProgressColumn(
                            "Peluang Main (%)",
                            min_value=0,
                            max_value=100,
                            format="%d%%"
                        )
                    }
                )
            else:
                st.info(f"Belum ada data pemain untuk klub {selected_club}.")

            # UPCOMING FIXTURES TABLE FOR THIS CLUB
            st.markdown(f"##### 📅 Jadwal Pertandingan Mendatang: **{selected_club}**")
            
            # Extract fixtures from fixtures_data or fpl_data
            raw_fixtures = fixtures_data or []
            if not raw_fixtures and isinstance(fpl_data, dict):
                raw_fixtures = fpl_data.get('fixtures', [])

            club_t_map = teams_dict or {t['id']: t['name'] for t in fpl_data.get('teams', [])} if fpl_data else {}
            
            club_fixtures_list = []
            if raw_fixtures and t_id is not None:
                for fx in raw_fixtures:
                    if not fx.get('finished'):
                        h_id = fx.get('team_h')
                        a_id = fx.get('team_a')
                        if h_id == t_id or a_id == t_id:
                            is_home = (h_id == t_id)
                            opp_id = a_id if is_home else h_id
                            opp_name = club_t_map.get(opp_id, f"Team {opp_id}")
                            diff = fx.get('team_h_difficulty') if is_home else fx.get('team_a_difficulty')
                            kickoff = fx.get('kickoff_time', '')
                            kickoff_str = kickoff[:10] if kickoff else '-'

                            club_fixtures_list.append({
                                'Gameweek': f"GW {fx.get('event', '-')}",
                                'Lawan': opp_name,
                                'Lokasi': '🏠 Kandang (Home)' if is_home else '✈️ Tandang (Away)',
                                'FDR (Tingkat Kesulitan)': diff if diff is not None else 3,
                                'Tanggal Kickoff': kickoff_str
                            })

            if club_fixtures_list:
                df_club_fix = pd.DataFrame(club_fixtures_list).head(6)
                st.dataframe(
                    df_club_fix,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "FDR (Tingkat Kesulitan)": st.column_config.NumberColumn(
                            "FDR (1-5)",
                            help="Skala 1 (Sangat Mudah) hingga 5 (Sangat Sulit)",
                            format="%d"
                        )
                    }
                )
            else:
                st.info(f"Jadwal mendatang untuk {selected_club}: Lawan terdekat adalah {c_info.get('Lawan Berikutnya', '-')}")
    else:
        st.warning("Data tim tidak tersedia.")

