"""
Team Strength Analysis Tab View.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import textwrap
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

            st.markdown("---")

            # STARTING XI FORMATION & TOP HIGHLIGHTS (TOP L5M)
            st.markdown(f"##### 🏟️ Formasi Starting XI & Sorotan Metrik Utama: **{selected_club}**")
            st.caption("Menampilkan 11 pemain dengan **menit bermain tertinggi dalam 5 laga terakhir (Avg Mins L5M)**. Jika menit bermain sama, diurutkan berdasarkan **Total Poin tertinggi** (posisi fleksibel mengikuti rotasi manajer, FWD dapat bernilai 0).")

            if not club_players.empty:
                # 1. Formation Selection & Resolution
                f_ctrl1, f_ctrl2 = st.columns([2, 1])
                with f_ctrl1:
                    formation_mode = st.selectbox(
                        "Pilihan Pola Formasi:",
                        options=[
                            "⚡ Top 11 L5M (Rekomendasi Manajer & Rotasi Nyata)",
                            "4-3-3", "4-4-2", "3-5-2", "3-4-3", "4-5-1", "5-3-2", "5-4-1", "5-2-3"
                        ],
                        index=0,
                        key=f"form_mode_{selected_club}"
                    )

                sort_order_cols = ['Avg Mins (L5M)', 'Total Poin', 'Menit Bermain', 'xPoin']
                
                # Check for available GKPs
                gkps_all = club_players[club_players['Posisi'].isin(['GK', 'GKP'])].sort_values(
                    by=sort_order_cols, ascending=[False, False, False, False]
                )
                top_gkp = gkps_all.head(1) if not gkps_all.empty else pd.DataFrame()
                
                # Outfield players (all except top GKP)
                outfield_all = club_players[~club_players.index.isin(top_gkp.index if not top_gkp.empty else [])].sort_values(
                    by=sort_order_cols, ascending=[False, False, False, False]
                )

                valid_formations = {
                    '4-3-3': (4, 3, 3),
                    '4-4-2': (4, 4, 2),
                    '3-5-2': (3, 5, 2),
                    '3-4-3': (3, 4, 3),
                    '4-5-1': (4, 5, 1),
                    '5-3-2': (5, 3, 2),
                    '5-4-1': (5, 4, 1),
                    '5-2-3': (5, 2, 3),
                }

                chosen_form_name = None
                best_selection = None

                if formation_mode.startswith("⚡"):
                    # Pure top 10 outfield players by L5M, tiebreaker Total Poin
                    top_10_outfield = outfield_all.head(10)
                    sel_defs = top_10_outfield[top_10_outfield['Posisi'] == 'DEF']
                    sel_mids = top_10_outfield[top_10_outfield['Posisi'] == 'MID']
                    sel_fwds = top_10_outfield[top_10_outfield['Posisi'] == 'FWD']

                    chosen_form_name = f"{len(sel_defs)}-{len(sel_mids)}-{len(sel_fwds)}"
                    best_selection = {
                        'GKP': top_gkp,
                        'DEF': sel_defs,
                        'MID': sel_mids,
                        'FWD': sel_fwds
                    }
                else:
                    # Specific formation preset requested
                    n_def, n_mid, n_fwd = valid_formations.get(formation_mode, (4, 4, 2))
                    defs_pool = outfield_all[outfield_all['Posisi'] == 'DEF']
                    mids_pool = outfield_all[outfield_all['Posisi'] == 'MID']
                    fwds_pool = outfield_all[outfield_all['Posisi'] == 'FWD']

                    sel_defs = defs_pool.head(n_def)
                    sel_mids = mids_pool.head(n_mid)
                    sel_fwds = fwds_pool.head(n_fwd)

                    # If some positions don't have enough players (e.g. 0 FWD), backfill from remaining top outfield
                    already_selected_idx = pd.concat([sel_defs, sel_mids, sel_fwds]).index if not (sel_defs.empty and sel_mids.empty and sel_fwds.empty) else []
                    remaining_slots = 10 - len(already_selected_idx)
                    
                    if remaining_slots > 0:
                        backfill = outfield_all[~outfield_all.index.isin(already_selected_idx)].head(remaining_slots)
                        sel_defs = pd.concat([sel_defs, backfill[backfill['Posisi'] == 'DEF']])
                        sel_mids = pd.concat([sel_mids, backfill[backfill['Posisi'] == 'MID']])
                        sel_fwds = pd.concat([sel_fwds, backfill[backfill['Posisi'] == 'FWD']])

                    chosen_form_name = f"{len(sel_defs)}-{len(sel_mids)}-{len(sel_fwds)} ({formation_mode})"
                    best_selection = {
                        'GKP': top_gkp,
                        'DEF': sel_defs,
                        'MID': sel_mids,
                        'FWD': sel_fwds
                    }

                xi_df = pd.concat([
                    best_selection['GKP'],
                    best_selection['DEF'],
                    best_selection['MID'],
                    best_selection['FWD']
                ]).reset_index(drop=True)

                with f_ctrl2:
                    avg_xi_mins = xi_df['Avg Mins (L5M)'].mean() if not xi_df.empty else 0.0
                    tot_xi_xpts = xi_df['xPoin'].sum() if 'xPoin' in xi_df.columns else 0.0
                    st.metric(
                        f"Formasi: {chosen_form_name}",
                        f"⏱️ {avg_xi_mins:.1f} m / pemain",
                        f"Total xPoin: {tot_xi_xpts:.1f} pts"
                    )

                # 2. TOP 6 HIGHLIGHT CARDS FROM THE 11 STARTING PLAYERS
                st.markdown("###### 🌟 Sorotan Metrik Unggulan dari 11 Pemain Starting XI:")

                # Helper to safely extract top player for a metric
                def get_top_stat_player(df_source, sort_by_cols, default_val=0.0):
                    if df_source.empty:
                        return {"Nama": "-", "Nilai": default_val, "Pos": "-", "Sub": "-"}
                    for c in sort_by_cols:
                        if c not in df_source.columns:
                            df_source[c] = default_val
                    sorted_p = df_source.sort_values(by=sort_by_cols, ascending=False).iloc[0]
                    return sorted_p

                top_inf = get_top_stat_player(xi_df, ['Influence', 'Avg Mins (L5M)'])
                top_thr = get_top_stat_player(xi_df, ['Threat', 'Avg Mins (L5M)'])
                top_cre = get_top_stat_player(xi_df, ['Creativity', 'Avg Mins (L5M)'])
                top_def = get_top_stat_player(xi_df, ['Defensive Contribution', 'Tackles', 'Avg Mins (L5M)'])
                top_gol = get_top_stat_player(xi_df, ['Gol', 'xG', 'Avg Mins (L5M)'])
                top_asi = get_top_stat_player(xi_df, ['Asis', 'xA', 'Avg Mins (L5M)'])

                hc1, hc2, hc3, hc4, hc5, hc6 = st.columns(6)
                with hc1:
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #166534; text-transform: uppercase;">⚡ Top Influence</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_inf.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #15803d;">{float(top_inf.get('Influence', 0.0)):.1f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Pos: {top_inf.get('Posisi', '-')} · L5M: {float(top_inf.get('Avg Mins (L5M)', 0)):.0f}m</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                with hc2:
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #991b1b; text-transform: uppercase;">🎯 Top Threat</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_thr.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #b91c1c;">{float(top_thr.get('Threat', 0.0)):.1f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">xG: {float(top_thr.get('xG', 0.0)):.2f} · Pos: {top_thr.get('Posisi', '-')}</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                with hc3:
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #1e40af; text-transform: uppercase;">🪄 Top Creativity</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_cre.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #2563eb;">{float(top_cre.get('Creativity', 0.0)):.1f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">xA: {float(top_cre.get('xA', 0.0)):.2f} · Pos: {top_cre.get('Posisi', '-')}</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                with hc4:
                    tackles_cnt = int(top_def.get('Tackles', 0) or 0)
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #fdf4ff; border: 1px solid #f5d0fe; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #86198f; text-transform: uppercase;">🛡️ Top Def Contrib</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_def.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #a21caf;">{float(top_def.get('Defensive Contribution', 0.0)):.1f} DC</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Tekel: {tackles_cnt} · Pos: {top_def.get('Posisi', '-')}</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                with hc5:
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #92400e; text-transform: uppercase;">⚽ Top Gol (XI)</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_gol.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #b45309;">{int(top_gol.get('Gol', 0))} Gol</div>
                            <div style="font-size: 0.75rem; color: #64748b;">xG: {float(top_gol.get('xG', 0.0)):.2f} · {float(top_gol.get('Avg Mins (L5M)', 0)):.0f}m</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                with hc6:
                    st.markdown(textwrap.dedent(f"""
                        <div style="padding: 12px; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.78rem; font-weight: 700; color: #6b21a8; text-transform: uppercase;">🅰️ Top Asis (XI)</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin: 4px 0 2px 0;">{top_asi.get('Nama Pemain', '-')}</div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #7e22ce;">{int(top_asi.get('Asis', 0))} Asis</div>
                            <div style="font-size: 0.75rem; color: #64748b;">xA: {float(top_asi.get('xA', 0.0)):.2f} · {float(top_asi.get('Avg Mins (L5M)', 0)):.0f}m</div>
                        </div>
                    """).strip(), unsafe_allow_html=True)

                # 3. PITCH / LAPANGAN VISUALIZATION
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("###### ⚽ Visualisasi Lapangan Formasi Starting XI:")

                # Helper to format player card for pitch
                def render_pitch_player_card(p_row):
                    pos = str(p_row.get('Posisi', '-'))
                    name = str(p_row.get('Nama Pemain', '-'))
                    mins = float(p_row.get('Avg Mins (L5M)', 0.0))
                    xpts = float(p_row.get('xPoin', 0.0))
                    pts = int(p_row.get('Total Poin', 0))

                    # Badge color by position
                    badge_bg = "#3b82f6"
                    if pos in ['GK', 'GKP']:
                        badge_bg = "#f59e0b"
                    elif pos == 'MID':
                        badge_bg = "#10b981"
                    elif pos == 'FWD':
                        badge_bg = "#ef4444"

                    return (
                        f'<div class="player-card">'
                        f'<div class="pos-badge" style="background-color: {badge_bg};">{pos}</div>'
                        f'<div class="player-name" title="{name}">{name}</div>'
                        f'<div class="mins-badge">⏱️ {mins:.0f}m (L5M)</div>'
                        f'<div class="stats-row">'
                        f'<span>xP: <strong>{xpts:.1f}</strong></span>'
                        f'<span>Pts: <strong>{pts}</strong></span>'
                        f'</div>'
                        f'</div>'
                    )

                # Pitch Rows Generation
                fwd_cards = "".join([render_pitch_player_card(r) for _, r in best_selection['FWD'].iterrows()]) if not best_selection['FWD'].empty else ""
                mid_cards = "".join([render_pitch_player_card(r) for _, r in best_selection['MID'].iterrows()]) if not best_selection['MID'].empty else ""
                def_cards = "".join([render_pitch_player_card(r) for _, r in best_selection['DEF'].iterrows()]) if not best_selection['DEF'].empty else ""
                gkp_cards = "".join([render_pitch_player_card(r) for _, r in best_selection['GKP'].iterrows()]) if not best_selection['GKP'].empty else ""

                fwd_row_html = f'<div class="row-players">{fwd_cards}</div>' if fwd_cards else ''
                mid_row_html = f'<div class="row-players">{mid_cards}</div>' if mid_cards else ''
                def_row_html = f'<div class="row-players">{def_cards}</div>' if def_cards else ''
                gkp_row_html = f'<div class="row-players">{gkp_cards}</div>' if gkp_cards else ''

                pitch_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
        background: transparent;
        padding: 4px;
        display: flex;
        justify-content: center;
    }}
    .pitch-container {{
        width: 100%;
        max-width: 860px;
        background: linear-gradient(180deg, #15803d 0%, #166534 50%, #14532d 100%);
        border-radius: 16px;
        padding: 20px 14px;
        border: 3px solid #e2e8f0;
        box-shadow: inset 0 0 50px rgba(0,0,0,0.35), 0 8px 20px rgba(0,0,0,0.12);
        position: relative;
        overflow: hidden;
    }}
    /* Pitch markings */
    .pitch-line-top {{
        border-bottom: 2px dashed rgba(255,255,255,0.45);
        margin-bottom: 14px;
        padding-bottom: 4px;
        text-align: center;
    }}
    .pitch-line-mid {{
        position: relative;
        margin: 12px 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .center-line {{
        width: 100%;
        border-top: 1.5px solid rgba(255,255,255,0.4);
        position: absolute;
        top: 50%;
        left: 0;
    }}
    .center-circle {{
        width: 80px;
        height: 80px;
        border: 1.5px solid rgba(255,255,255,0.35);
        border-radius: 50%;
        position: relative;
        z-index: 1;
    }}
    .pitch-line-bot {{
        border-top: 2px solid rgba(255,255,255,0.45);
        margin-top: 14px;
        padding-top: 8px;
        text-align: center;
    }}
    .zone-label {{
        color: rgba(255,255,255,0.75);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .row-players {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        margin: 8px 0;
        position: relative;
        z-index: 2;
    }}
    .player-card {{
        background: rgba(255, 255, 255, 0.96);
        border-radius: 10px;
        padding: 8px 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.22);
        border: 1.5px solid #ffffff;
        text-align: center;
        min-width: 110px;
        max-width: 135px;
        margin: 5px 6px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .player-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }}
    .pos-badge {{
        display: inline-block;
        color: #ffffff;
        font-size: 0.65rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }}
    .player-name {{
        font-weight: 800;
        font-size: 0.86rem;
        color: #0f172a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .mins-badge {{
        background-color: #f1f5f9;
        border-radius: 4px;
        padding: 2px 4px;
        margin: 3px 0;
        font-size: 0.72rem;
        font-weight: 700;
        color: #334155;
    }}
    .stats-row {{
        display: flex;
        justify-content: space-between;
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 2px;
    }}
</style>
</head>
<body>
<div class="pitch-container">
    <div class="pitch-line-top">
        <span class="zone-label">⚔️ Area Serangan Lawan</span>
    </div>

    <!-- FWD ROW -->
    {fwd_row_html}

    <!-- MIDFIELD -->
    {mid_row_html}

    <!-- DEF ROW -->
    {def_row_html}

    <!-- GK ROW -->
    {gkp_row_html}

    <div class="pitch-line-bot">
        <span class="zone-label">🛡️ Gawang & Pertahanan Sendiri</span>
    </div>
</div>
</body>
</html>"""

                components.html(pitch_html, height=560, scrolling=False)

                # 4. Starting XI Detailed Data Table Expander
                with st.expander("📋 Lihat Tabel Rincian Statistik Lengkap 11 Pemain Starting XI", expanded=False):
                    xi_table_cols = [
                        'Nama Pemain', 'Posisi', 'Avg Mins (L5M)', 'Total Poin', 'xPoin', 'Gol', 'Asis', 'xG', 'xA',
                        'Influence', 'Threat', 'Creativity', 'Defensive Contribution', 'Tackles', 'Clearances', 'Recoveries',
                        'Harga (£m)', 'Form', '% Ownership'
                    ]
                    valid_xi_cols = [c for c in xi_table_cols if c in xi_df.columns]
                    st.dataframe(
                        xi_df[valid_xi_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins"),
                            "Total Poin": st.column_config.NumberColumn(format="%d pts"),
                            "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
                            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                            "Form": st.column_config.NumberColumn(format="%.2f"),
                            "xG": st.column_config.NumberColumn(format="%.2f"),
                            "xA": st.column_config.NumberColumn(format="%.2f"),
                            "Influence": st.column_config.NumberColumn(format="%.1f"),
                            "Threat": st.column_config.NumberColumn(format="%.1f"),
                            "Creativity": st.column_config.NumberColumn(format="%.1f"),
                            "Defensive Contribution": st.column_config.NumberColumn(format="%.1f"),
                            "Tackles": st.column_config.NumberColumn(format="%d"),
                            "% Ownership": st.column_config.NumberColumn(format="%.1f%%")
                        }
                    )
            else:
                st.info(f"Belum ada data pemain untuk membentuk formasi {selected_club}.")

            st.markdown("---")

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
                    is_played = bool(fx.get('finished') or fx.get('finished_provisional') or fx.get('started'))
                    if not is_played:
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

