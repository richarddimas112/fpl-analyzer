"""
Interactive Visualizations Tab View (Box Plot, Scatter Plot, Pearson r, and Club Aggregates).
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import pearsonr
from src.views.tab_radar import render_player_comparison_radar_tab
from src.api import fetch_gameweek_live_points, fetch_all_gameweeks_live_points

def render_tab_visualizations(filtered_players, players_df, fpl_data, teams_dict, fdr_summary=None):
    """
    Renders Visualizations Tab: Box Plot (Points by Position Filtered by GW / All GW),
    Interactive Scatter Plot with Pearson Correlation, Club Aggregates Bar Chart, and Player Comparison Radar.
    """
    st.subheader("📈 Visualisasi Interaktif & Radar Komparasi Pemain")
    st.write("Analisis hubungan antar variabel statistik pemain, perbandingan sebaran poin antar posisi (Box Plot), agregat klub, serta komparasi head-to-head 2 pemain dengan grafik radar interaktif.")

    chart_subtab_box, chart_subtab1, chart_subtab2, chart_subtab3 = st.tabs([
        "📦 Box Plot Distribusi Poin per Posisi",
        "🔵 Scatter Plot & Pearson Correlation", 
        "📊 Bar Chart Agregat Klub",
        "⚔️ Komparasi 2 Pemain (Radar Chart)"
    ])

    # SECTION: BOX PLOT DISTRIBUSI POIN PER POSISI (FILTER GW / ALL GW)
    with chart_subtab_box:
        st.markdown("##### 📦 Box Plot: Komparasi Distribusi Poin Antar Posisi (GKP, DEF, MID, FWD)")
        st.write(
            "Gunakan visualisasi Box Plot (Diagram Kotak Garis) ini untuk membandingkan sebaran nilai, median, rata-rata, "
            "kuartil (IQR), dan potensi ledakan poin (outliers/haulers) antar posisi pemain. "
            "Pilihan **Semua Gameweek** menggabungkan seluruh penampilan per pertandingan (GW1, GW2, dst.) "
            "sehingga distribusi mencerminkan poin per Gameweek, bukan akumulasi musim."
        )

        # 1. Pilihan Gameweek Filter
        events = fpl_data.get('events', []) if fpl_data else []
        played_or_active_events = [e for e in events if e.get('finished') or e.get('is_current')]
        played_gw_ids = [e['id'] for e in played_or_active_events]
        
        # Build options dictionary: label -> event_id or 'ALL'
        gw_options_dict = {"Semua Gameweek (Gabungan GW1, GW2, dst. - Poin per Match)": "ALL"}
        for ev in played_or_active_events:
            ev_id = ev['id']
            if ev.get('is_current') and not ev.get('finished'):
                gw_options_dict[f"Gameweek {ev_id} ⚡ (Sedang Berjalan)"] = ev_id
            elif ev.get('finished'):
                gw_options_dict[f"Gameweek {ev_id} ✅ (Selesai)"] = ev_id
            else:
                gw_options_dict[f"Gameweek {ev_id}"] = ev_id

        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([3, 2, 2])
        with ctrl_c1:
            selected_gw_label = st.selectbox(
                "🗓️ Pilih Ruang Lingkup Gameweek:",
                options=list(gw_options_dict.keys()),
                index=0,
                key="box_plot_gw_sel"
            )
            selected_gw_val = gw_options_dict[selected_gw_label]

        with ctrl_c2:
            show_points_mode = st.selectbox(
                "📍 Tampilan Titik Pemain:",
                options=["Semua Titik (Jitter)", "Hanya Outlier", "Tanpa Titik"],
                index=0,
                key="box_plot_points_mode"
            )

        with ctrl_c3:
            club_filter = st.selectbox(
                "🏟️ Filter Klub:",
                options=["Semua Klub"] + sorted(list(teams_dict.values())),
                index=0,
                key="box_plot_club_sel"
            )

        filter_row1, filter_row2 = st.columns([3, 3])
        with filter_row1:
            filter_min_minutes = st.checkbox(
                "⏱️ Hanya sertakan pemain yang bermain (Menit > 0)",
                value=True,
                key="box_plot_min_mins",
                help="Sangat disarankan: Menyaring penampilan pemain cadangan/tidak bertanding (0 menit) agar distribusi poin pemain yang aktif di lapangan tidak tertekan ke angka 0."
            )

        with filter_row2:
            min_price = float(players_df['Harga (£m)'].min()) if not players_df.empty else 4.0
            max_price = float(players_df['Harga (£m)'].max()) if not players_df.empty else 15.0
            price_range_sel = st.slider(
                "💰 Rentang Harga Pemain (£m):",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=0.5,
                key="box_plot_price_slider"
            )

        # 2. Build Dataset (Always per-match / per-Gameweek points)
        if selected_gw_val == "ALL":
            all_gw_map = fetch_all_gameweeks_live_points(tuple(played_gw_ids))
            live_rows = []
            for gw_id in played_gw_ids:
                gw_data = all_gw_map.get(gw_id, {})
                for _, p in players_df.iterrows():
                    p_id = int(p['id'])
                    st_data = gw_data.get(p_id, {})
                    pts = float(st_data.get('total_points', 0))
                    mins = int(st_data.get('minutes', 0))
                    goals = int(st_data.get('goals_scored', 0))
                    assists = int(st_data.get('assists', 0))
                    bonus = int(st_data.get('bonus', 0))
                    bps = int(st_data.get('bps', 0))
                    clean_sheet = int(st_data.get('clean_sheets', 0))
                    pos = 'GKP' if p['Posisi'] == 'GK' else p['Posisi']
                    live_rows.append({
                        'id': p_id,
                        'Nama Pemain': f"{p['Nama Pemain']} (GW{gw_id})",
                        'Nama Asli': p['Nama Pemain'],
                        'Gameweek': f"GW{gw_id}",
                        'Klub': p['Klub'],
                        'Posisi_Display': pos,
                        'Harga (£m)': float(p['Harga (£m)']),
                        'Poin': pts,
                        'Menit': mins,
                        'Gol_Stat': goals,
                        'Asis_Stat': assists,
                        'Bonus_Stat': bonus,
                        'CS_Stat': clean_sheet,
                        'BPS': bps
                    })
            raw_box = pd.DataFrame(live_rows)
            scope_title = f"Gabungan Gameweek ({', '.join(['GW' + str(i) for i in played_gw_ids])} - Poin per Match)"
        else:
            live_points_map = fetch_gameweek_live_points(selected_gw_val)
            live_rows = []
            for _, p in players_df.iterrows():
                p_id = int(p['id'])
                st_data = live_points_map.get(p_id, {})
                pts = float(st_data.get('total_points', 0))
                mins = int(st_data.get('minutes', 0))
                goals = int(st_data.get('goals_scored', 0))
                assists = int(st_data.get('assists', 0))
                bonus = int(st_data.get('bonus', 0))
                bps = int(st_data.get('bps', 0))
                clean_sheet = int(st_data.get('clean_sheets', 0))
                pos = 'GKP' if p['Posisi'] == 'GK' else p['Posisi']
                live_rows.append({
                    'id': p_id,
                    'Nama Pemain': p['Nama Pemain'],
                    'Nama Asli': p['Nama Pemain'],
                    'Gameweek': f"GW{selected_gw_val}",
                    'Klub': p['Klub'],
                    'Posisi_Display': pos,
                    'Harga (£m)': float(p['Harga (£m)']),
                    'Poin': pts,
                    'Menit': mins,
                    'Gol_Stat': goals,
                    'Asis_Stat': assists,
                    'Bonus_Stat': bonus,
                    'CS_Stat': clean_sheet,
                    'BPS': bps
                })
            raw_box = pd.DataFrame(live_rows)
            scope_title = f"Gameweek {selected_gw_val}"

        # Apply Filters
        if filter_min_minutes:
            raw_box = raw_box[raw_box['Menit'] > 0]

        if club_filter != "Semua Klub":
            raw_box = raw_box[raw_box['Klub'] == club_filter]

        raw_box = raw_box[(raw_box['Harga (£m)'] >= price_range_sel[0]) & (raw_box['Harga (£m)'] <= price_range_sel[1])]

        if raw_box.empty:
            st.warning("Tidak ada data pemain yang cocok dengan kombinasi filter yang dipilih.")
        else:
            # 3. Position Summary Metric Cards
            pos_keys = ["GKP", "DEF", "MID", "FWD"]
            pos_labels = {"GKP": "🧤 Kiper (GKP)", "DEF": "🛡️ Bek (DEF)", "MID": "🎯 Gelandang (MID)", "FWD": "⚡ Penyerang (FWD)"}
            pos_colors = {"GKP": "#f59e0b", "DEF": "#2563eb", "MID": "#10b981", "FWD": "#ef4444"}
            
            m_cols = st.columns(4)
            pos_stats = {}
            for idx, pos in enumerate(pos_keys):
                sub = raw_box[raw_box['Posisi_Display'] == pos]
                with m_cols[idx]:
                    if not sub.empty:
                        m_mean = sub['Poin'].mean()
                        m_median = sub['Poin'].median()
                        m_max = sub['Poin'].max()
                        top_player_row = sub.sort_values(by='Poin', ascending=False).iloc[0]
                        top_name = f"{top_player_row['Nama Pemain']} ({int(m_max)} pts)"
                        n_players = len(sub)
                        pos_stats[pos] = {
                            "mean": m_mean, "median": m_median, "max": m_max,
                            "top_name": top_player_row['Nama Pemain'], "count": n_players
                        }
                        
                        sample_txt = f"{n_players} penampilan" if selected_gw_val == "ALL" else f"{n_players} pemain"

                        st.markdown(
                            f"""
                            <div style="border-top: 4px solid {pos_colors[pos]}; background-color: #ffffff; padding: 10px 12px; border-radius: 8px; border-left: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 4px;">{pos_labels[pos]}</div>
                                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                    <span style="font-size: 0.8rem; color: #64748b;">Rerata:</span>
                                    <span style="font-size: 1.1rem; font-weight: 800; color: {pos_colors[pos]};">{m_mean:.2f} pts</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #475569; margin-top: 2px;">
                                    <span>Median: <b>{m_median:.1f}</b></span>
                                    <span>Tertinggi: <b>{int(m_max)}</b></span>
                                </div>
                                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    ⭐ {top_name}
                                </div>
                                <div style="font-size: 0.7rem; color: #94a3b8; text-align: right; margin-top: 2px;">
                                    n={sample_txt}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="border-top: 4px solid {pos_colors[pos]}; background-color: #ffffff; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                                <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b;">{pos_labels[pos]}</div>
                                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">Tidak ada pemain</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # 4. Render Box Plot with Plotly
            pts_param = "all" if show_points_mode == "Semua Titik (Jitter)" else ("outliers" if show_points_mode == "Hanya Outlier" else False)
            
            fig_box = px.box(
                raw_box,
                x="Posisi_Display",
                y="Poin",
                color="Posisi_Display",
                color_discrete_map=pos_colors,
                category_orders={"Posisi_Display": pos_keys},
                points=pts_param,
                hover_name="Nama Pemain",
                hover_data={
                    "Posisi_Display": False,
                    "Gameweek": True,
                    "Klub": True,
                    "Harga (£m)": ":.1f",
                    "Poin": True,
                    "Menit": True,
                    "Gol_Stat": True,
                    "Asis_Stat": True,
                    "Bonus_Stat": True,
                    "CS_Stat": True
                },
                title=f"Distribusi Poin Pemain per Posisi ({scope_title})"
            )

            fig_box.update_traces(
                boxmean=True,  # Menampilkan garis rata-rata (dashed line)
                jitter=0.35,
                pointpos=-1.8 if show_points_mode == "Semua Titik (Jitter)" else 0
            )

            fig_box.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=520,
                xaxis=dict(
                    title="Posisi Pemain",
                    gridcolor="#e2e8f0",
                    categoryorder="array",
                    categoryarray=pos_keys
                ),
                yaxis=dict(
                    title="Poin per Match (GW)",
                    gridcolor="#e2e8f0",
                    zeroline=True,
                    zerolinecolor="#cbd5e1"
                ),
                showlegend=False
            )

            st.plotly_chart(fig_box, use_container_width=True)
            st.caption(
                "💡 *Garis horizontal tebal di dalam kotak menunjukkan **Median**. Garis putus-putus menunjukkan **Rerata (Mean)**. "
                "Kotak mencakup rentang Interkuartil (IQR: Q1 - Q3). Titik-titik di luar garis adalah **Outliers** (pencetak poin tinggi/hauler).* "
                "Arahkan kursor ke titik atau kotak untuk melihat rincian pemain."
            )

            # 5. Key Insights Callout
            if pos_stats:
                best_mean_pos = max(pos_stats.items(), key=lambda x: x[1]['mean'])
                best_haul_pos = max(pos_stats.items(), key=lambda x: x[1]['max'])
                
                st.markdown(
                    f"""
                    <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
                        <span style="font-weight: 700; color: #1e293b;">💡 Temuan Utama Distribusi Poin ({scope_title}):</span>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px; font-size: 0.88rem; color: #334155; line-height: 1.6;">
                            <li>Posisi dengan <b>rerata poin per match tertinggi</b> adalah <b>{pos_labels[best_mean_pos[0]]}</b> dengan rata-rata <b>{best_mean_pos[1]['mean']:.2f} poin</b>.</li>
                            <li>Poin ledakan tertinggi (<b>haul rekor</b>) dicetak oleh posisi <b>{pos_labels[best_haul_pos[0]]}</b> yaitu <b>{best_haul_pos[1]['top_name']}</b> dengan <b>{int(best_haul_pos[1]['max'])} poin</b>.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 6. Top Performers Breakdown Table
            with st.expander("🏆 **Lihat Tabel Penampilan Terbaik (Top Performers) per Posisi**", expanded=False):
                top_perf_cols = st.columns(4)
                for idx, pos in enumerate(pos_keys):
                    sub = raw_box[raw_box['Posisi_Display'] == pos].sort_values(by='Poin', ascending=False).head(5)
                    with top_perf_cols[idx]:
                        st.markdown(f"**Top 5 Penampilan {pos}**")
                        if not sub.empty:
                            disp = sub[['Nama Pemain', 'Klub', 'Harga (£m)', 'Poin', 'Menit']].copy()
                            disp['Harga'] = disp['Harga (£m)'].apply(lambda x: f"£{x:.1f}m")
                            disp = disp[['Nama Pemain', 'Klub', 'Harga', 'Poin', 'Menit']]
                            st.dataframe(disp, hide_index=True, use_container_width=True)
                        else:
                            st.caption("Tidak ada data.")

    # SECTION 2: SCATTER PLOT & PEARSON R
    with chart_subtab1:
        st.markdown("##### 🔵 Scatter Plot Multi-Metrik dengan Garis Tren Regresi")

        num_cols = [
            'xPoin', 'Total Poin', 'Harga (£m)', 'xG', 'xA', 'xGI',
            'Influence', 'Creativity', 'Threat', 'ICT Index',
            'Tackles', 'Defensive Contribution', 'Defensive Contribution per 90',
            'Form', 'BPS', '% Ownership', 'xG per 90', 'xA per 90',
            'FDR1', 'FDR3', 'FDR5', 'Menit Bermain', 'Gol', 'Asis', 'Clean Sheet', 'Saves'
        ]

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            x_var = st.selectbox("Metrik Sumbu X", options=num_cols, index=3, key="px_x") # xG
        with p_col2:
            y_var = st.selectbox("Metrik Sumbu Y", options=num_cols, index=1, key="px_y") # Total Poin
        with p_col3:
            color_var = st.selectbox("Warna Kelompok", options=['Posisi', 'Klub'], index=0, key="px_color")

        if not filtered_players.empty and len(filtered_players) > 2:
            # Calculate Pearson Correlation
            x_vals = filtered_players[x_var].astype(float)
            y_vals = filtered_players[y_var].astype(float)
            
            # Filter out NaNs if any
            valid_mask = ~(x_vals.isna() | y_vals.isna())
            r_coef, p_val = pearsonr(x_vals[valid_mask], y_vals[valid_mask])

            # Determine correlation strength category
            abs_r = abs(r_coef)
            if abs_r >= 0.8:
                strength = "Sangat Kuat 🚀"
            elif abs_r >= 0.6:
                strength = "Kuat 💪"
            elif abs_r >= 0.4:
                strength = "Sedang ⚖️"
            elif abs_r >= 0.2:
                strength = "Lemah 📉"
            else:
                strength = "Sangat Lemah / Tidak Ada Korelasi 🔴"

            direction = "Positif (+)" if r_coef > 0 else "Negatif (-)"

            # Display Correlation Card
            st.markdown(f"""
            <div class="corr-card">
                <div class="corr-title">Hasil Analisis Korelasi Pearson ({x_var} vs {y_var})</div>
                <div class="corr-value">r = {r_coef:.4f}</div>
                <div class="corr-desc">
                    Hubungan <strong>{direction}</strong> dengan tingkat korelasi <strong>{strength}</strong> (p-value = {p_val:.4e}).
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Generate Plotly Scatter Chart
            fig = px.scatter(
                filtered_players,
                x=x_var,
                y=y_var,
                color=color_var,
                hover_name='Nama Pemain',
                hover_data=['Klub', 'Posisi', 'Harga (£m)', 'xPoin', 'Total Poin', 'xG', 'xA', 'Form'],
                trendline="ols",
                trendline_color_override="#1e293b",
                title=f"Hubungan {x_var} vs {y_var} (Trendline OLS)"
            )

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=520,
                xaxis=dict(gridcolor="#e2e8f0", title=x_var),
                yaxis=dict(gridcolor="#e2e8f0", title=y_var),
                legend=dict(bordercolor="#e2e8f0", borderwidth=1)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 *Garis hitam putus-putus menunjukkan garis tren regresi linear (OLS). Arahkan kursor ke titik untuk detail pemain.*")
        else:
            st.warning("Data pemain terlalu sedikit untuk menghitung korelasi dan membuat scatter plot.")

    # SECTION 2: CLUB AGGREGATE BAR CHART
    with chart_subtab2:
        st.markdown("##### 📊 Perbandingan Rata-rata & Total Metrik per Klub")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            bar_metric = st.selectbox(
                "Pilih Metrik Klub:",
                options=[
                    'Total Poin', 'xPoin', 'xG', 'xA', 
                    'Influence', 'Creativity', 'Threat', 'ICT Index',
                    'Tackles', 'Defensive Contribution', 'Clean Sheet', 'Saves',
                    'Form', 'Harga (£m)', 'Gol', 'Asis', 'BPS'
                ],
                index=0,
                key="bar_metric"
            )
        with b_col2:
            agg_type = st.selectbox(
                "Tipe Agregasi:",
                options=["Rata-rata per Pemain", "Total Akumulasi Seluruh Skuad"],
                key="bar_agg"
            )

        if not filtered_players.empty:
            if agg_type == "Rata-rata per Pemain":
                club_data = filtered_players.groupby('Klub')[bar_metric].mean().reset_index()
                chart_title = f"Rata-rata {bar_metric} per Pemain Berdasarkan Klub"
            else:
                club_data = filtered_players.groupby('Klub')[bar_metric].sum().reset_index()
                chart_title = f"Total Akumulasi {bar_metric} Skuad Berdasarkan Klub"

            club_data = club_data.sort_values(by=bar_metric, ascending=True)

            fig_bar = px.bar(
                club_data,
                x=bar_metric,
                y='Klub',
                orientation='h',
                color=bar_metric,
                color_continuous_scale="Blues",
                title=chart_title
            )

            fig_bar.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=520,
                xaxis=dict(gridcolor="#e2e8f0", title=bar_metric),
                yaxis=dict(gridcolor="#e2e8f0", title="Klub")
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Data tidak tersedia untuk visualisasi klub.")

    # SECTION 3: PLAYER RADAR COMPARISON
    with chart_subtab3:
        render_player_comparison_radar_tab(players_df, fpl_data, teams_dict, fdr_summary=fdr_summary)
