"""
Player Stats & Progression View Tab (Tab 1).
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.api import fetch_player_element_summary

def render_tab_player_stats(filtered_players, players_df, models_dict, fpl_data, teams_dict):
    """
    Renders Tab 1: Player Stats Table, Regression Model Explanations, Classical Diagnostics, and Match Progression charts.
    """
    st.subheader("📋 Tabel Statistik & Prediksi xPoin Pemain")

    # Summary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pemain Terfilter", len(filtered_players))
    with col2:
        top_xpoin = filtered_players.sort_values(by="xPoin", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Prediksi xPoin Teratas", f"{top_xpoin['Nama Pemain']} ({top_xpoin['xPoin']} pts)" if top_xpoin is not None else "-")
    with col3:
        top_pts = filtered_players.sort_values(by="Total Poin", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Poin Tertinggi", f"{top_pts['Nama Pemain']} ({top_pts['Total Poin']} pts)" if top_pts is not None else "-")
    with col4:
        top_form = filtered_players.sort_values(by="Form", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Form Terbaik", f"{top_form['Nama Pemain']} ({top_form['Form']})" if top_form is not None else "-")

    # Expander for Custom Column Selection
    default_cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin', 'Avg Mins (L5M)', 'Total Poin',
        'FDR1', 'FDR3', 'FDR5', 'Form', '% Ownership', 'xG', 'xA', 'Status', 'Peluang Main GW (%)'
    ]
    
    with st.expander("⚙️ Pilih Kolom yang Ingin Ditampilkan pada Tabel", expanded=False):
        available_table_cols = [c for c in players_df.columns if c not in ['id', 'team']]
        selected_cols = st.multiselect(
            "Centang/Pilih kolom data FPL yang ingin dimunculkan di tabel:",
            options=available_table_cols,
            default=default_cols,
            key="table_cols_picker"
        )
    
    if not selected_cols:
        selected_cols = default_cols

    # Expander for Positional Regression Model Details
    with st.expander("🤖 Detail 4 Model Regression xPoin Berdasarkan Posisi (FWD, MID, DEF, GK)", expanded=False):
        pos_tab1, pos_tab2, pos_tab3, pos_tab4 = st.tabs(["⚽ FWD (Penyerang)", "🎯 MID (Gelandang)", "🛡️ DEF (Bek)", "🧤 GK (Kiper)"])
        
        tab_mapping = [
            (pos_tab1, "FWD"),
            (pos_tab2, "MID"),
            (pos_tab3, "DEF"),
            (pos_tab4, "GK")
        ]
        
        for tab_obj, pos_key in tab_mapping:
            with tab_obj:
                pos_info = models_dict.get(pos_key, {})
                if pos_info:
                    st.markdown(f"#### 📐 Model Regresi Linier Posisi **{pos_key}**")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("R² Score (Akurasi Fitting)", f"{pos_info['r2']:.4f}")
                    with m2:
                        st.metric("Intercept (Konstanta β₀)", f"{pos_info['intercept']:.4f}")
                    with m3:
                        src_label = 'Histori Pertandingan Aktual API' if pos_info['is_real_history'] else 'Simulasi Regresi Performa FPL'
                        st.metric("Sumber Data Latihan", src_label)

                    st.markdown("##### Bobot / Koefisien Variabel Input (β):")
                    st.dataframe(pos_info['coef_df'], use_container_width=True)
                    st.caption(f"💡 *Model {pos_key} menghitung prediksi xPoin laga mendatang menggunakan variabel X spesifik posisi {pos_key}.*")

    # Expander for Classical Assumption Diagnostics (Per Posisi)
    with st.expander("🧪 Diagnostik Uji Asumsi Klasik Model Regresi", expanded=False):
        diag_pos_choice = st.selectbox(
            "Pilih Posisi Pemain untuk Melihat Uji Asumsi Klasik (FWD | MID | DEF | GK):",
            options=["FWD", "MID", "DEF", "GK"],
            key="pos_diag_selector"
        )
        
        diag_pos_info = models_dict.get(diag_pos_choice, {})
        diag_results = diag_pos_info.get('diag_results') if diag_pos_info else None
        
        if diag_results is not None:
            st.markdown(f"##### 📌 Ringkasan Hasil Uji Asumsi Klasik (Model {diag_pos_choice})")
            
            # Metrics Row
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                v_pass = diag_results['all_vif_ok']
                st.metric("1. Multikolinearitas", "🟢 Lolos" if v_pass else "🔴 Terindikasi", "VIF < 10" if v_pass else "VIF >= 10")
            with d2:
                l_pass = diag_results['linearity']['passed']
                st.metric("2. Linearitas", "🟢 Linear" if l_pass else "🔴 Non-Linear", f"p = {diag_results['linearity']['p_value']}")
            with d3:
                n_pass = diag_results['normality']['passed']
                st.metric("3. Normalitas Residual", "🟢 Normal" if n_pass else "🔴 Tidak Normal", f"p = {diag_results['normality']['p_value']}")
            with d4:
                h_pass = diag_results['homoscedasticity']['passed']
                st.metric("4. Homoskedastisitas", "🟢 Varian Konstan" if h_pass else "🔴 Heteroskedastisitas", f"p = {diag_results['homoscedasticity']['p_value']}")

            st.divider()

            # 1. Uji Multikolinearitas
            st.markdown(f"##### 1. Uji Multikolinearitas (Variance Inflation Factor & Tolerance) - {diag_pos_choice}")
            st.dataframe(diag_results['vif_df'], use_container_width=True)
            st.caption("💡 *Kriteria: Nilai VIF < 10.0 dan Tolerance > 0.10 menandakan variabel independen bebas dari multikolinearitas.*")

            # 2. Uji Linearitas
            st.markdown(f"##### 2. Uji Linearitas (Rainbow Test) - {diag_pos_choice}")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                st.markdown(f"**Statistic (Rainbow Test):** `{diag_results['linearity']['stat']}`")
            with lc2:
                st.markdown(f"**p-value:** `{diag_results['linearity']['p_value']}`")
            with lc3:
                l_status = "🟢 Linear (p > 0.05)" if diag_results['linearity']['passed'] else "🔴 Non-Linear (p <= 0.05)"
                st.markdown(f"**Status:** {l_status}")
            st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan hubungan antara variabel independen dan dependen bersifat linear.*")

            # 3. Uji Normalitas Residual
            st.markdown(f"##### 3. Uji Normalitas Residual (Shapiro-Wilk Test) - {diag_pos_choice}")
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                st.markdown(f"**Statistic (W):** `{diag_results['normality']['stat']}`")
            with nc2:
                st.markdown(f"**p-value:** `{diag_results['normality']['p_value']}`")
            with nc3:
                n_status = "🟢 Residual Normal (p > 0.05)" if diag_results['normality']['passed'] else "🔴 Tidak Normal (p <= 0.05)"
                st.markdown(f"**Status:** {n_status}")
            st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan residual e = (Y_aktual - Y_prediksi) terdistribusi secara normal.*")

            # 4. Uji Homoskedastisitas
            st.markdown(f"##### 4. Uji Homoskedastisitas (Breusch-Pagan Test) - {diag_pos_choice}")
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                st.markdown(f"**LM Statistic:** `{diag_results['homoscedasticity']['stat']}`")
            with hc2:
                st.markdown(f"**p-value (LM-Test):** `{diag_results['homoscedasticity']['p_value']}`")
            with hc3:
                h_status = "🟢 Homoskedastisitas (Varian Konstan)" if diag_results['homoscedasticity']['passed'] else "🔴 Heteroskedastisitas (p <= 0.05)"
                st.markdown(f"**Status:** {h_status}")
            st.caption("💡 *Kriteria: Nilai p-value > 0.05 menunjukkan varian dari residual bersifat konstan (homoskedastisitas).*")
        else:
            st.warning(f"Diagnostik Uji Asumsi Klasik untuk model {diag_pos_choice} tidak dapat dihitung.")

    sorted_players = filtered_players.sort_values(by="xPoin", ascending=False)

    st.dataframe(
        sorted_players[selected_cols],
        use_container_width=True,
        height=520,
        column_config={
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
            "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins"),
            "Total Poin": st.column_config.NumberColumn(format="%d pts"),
            "FDR1": st.column_config.NumberColumn(format="%.1f"),
            "FDR3": st.column_config.NumberColumn(format="%.2f"),
            "FDR5": st.column_config.NumberColumn(format="%.2f"),
            "Menit Bermain": st.column_config.NumberColumn(format="%d mins"),
            "Form": st.column_config.NumberColumn(format="%.2f"),
            "% Ownership": st.column_config.NumberColumn(format="%.1f%%"),
            "Net Transfers GW": st.column_config.NumberColumn(format="%+d"),
            "Transfers In GW": st.column_config.NumberColumn(format="%d"),
            "Transfers Out GW": st.column_config.NumberColumn(format="%d"),
            "xG": st.column_config.NumberColumn(format="%.2f"),
            "xA": st.column_config.NumberColumn(format="%.2f"),
            "xGI": st.column_config.NumberColumn(format="%.2f"),
            "xG per 90": st.column_config.NumberColumn(format="%.2f"),
            "xA per 90": st.column_config.NumberColumn(format="%.2f"),
            "xGI per 90": st.column_config.NumberColumn(format="%.2f"),
            "ICT Index": st.column_config.NumberColumn(format="%.1f"),
            "BPS": st.column_config.NumberColumn(format="%d"),
            "Bonus Poin": st.column_config.NumberColumn(format="%d"),
            "Kartu Kuning": st.column_config.NumberColumn(format="%d"),
            "Kartu Merah": st.column_config.NumberColumn(format="%d"),
            "Saves": st.column_config.NumberColumn(format="%d"),
            "Peluang Main GW (%)": st.column_config.ProgressColumn(
                "Peluang Main GW (%)",
                min_value=0,
                max_value=100,
                format="%d%%"
            )
        }
    )

    # ---------------------------------------------------------------------
    # ENHANCED SECTION: PLAYER PERFORMANCE & PROGRESSION OVER TIME
    # ---------------------------------------------------------------------
    st.divider()
    st.markdown("### 📈 Visualisasi Tren & Progresi Performa Pemain per Gameweek")
    st.write("Visualisasikan performa pertandingan demi pertandingan pemain sepanjang musim berjalan (total poin, gol, asis, xG, xA, dan menit bermain) menggunakan data real-time API FPL.")

    # Player Selector
    player_pool = filtered_players if not filtered_players.empty else players_df
    player_choices = player_pool.to_dict('records')
    
    default_idx = 0
    for i, p in enumerate(player_choices):
        if p.get('Nama Pemain') in ['Haaland', 'M.Salah', 'Saka', 'Palmer']:
            default_idx = i
            break

    def format_player_option(p):
        if not p or not isinstance(p, dict):
            return ""
        p_name = p.get('Nama Pemain', '')
        p_klub = p.get('Klub', '')
        p_pos = p.get('Posisi', '')
        p_pts = p.get('Total Poin', 0)
        try:
            price_str = f"£{float(p.get('Harga (£m)', 0.0)):.1f}m"
        except (ValueError, TypeError):
            price_str = "-"
        try:
            xpts_str = f"{float(p.get('xPoin', 0.0)):.2f}"
        except (ValueError, TypeError):
            xpts_str = "-"
        return f"{p_name} ({p_klub} - {p_pos}) · {price_str} · Total Poin: {p_pts} pts · xPoin: {xpts_str}"

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        selected_player = st.selectbox(
            "Pilih Pemain untuk Melihat Tren Performa Gameweek:",
            options=player_choices,
            index=default_idx if 0 <= default_idx < len(player_choices) else 0,
            format_func=format_player_option,
            key="player_trend_selector"
        )
    with sel_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        view_mode = st.radio(
            "Mode Tampilan Nilai:",
            options=["Per Gameweek (Match)", "Akumulatif (Kumulatif)"],
            horizontal=True,
            key="player_prog_mode"
        )

    if selected_player:
        sel_pid = selected_player.get('id')
        if not sel_pid:
            for el in fpl_data.get('elements', []):
                if el.get('web_name') == selected_player.get('Nama Pemain'):
                    sel_pid = el.get('id')
                    break
        sel_pname = selected_player.get('Nama Pemain', 'Pemain')
        sel_club = selected_player.get('Klub', '-')
        sel_pos = selected_player.get('Posisi', '-')
        sel_cost = float(selected_player.get('Harga (£m)', 0.0) or 0.0)
        sel_pts = int(selected_player.get('Total Poin', 0) or 0)
        sel_xg = float(selected_player.get('xG', 0.0) or 0.0)
        sel_xa = float(selected_player.get('xA', 0.0) or 0.0)
        sel_goals = int(selected_player.get('Gol', 0) or 0)
        sel_assists = int(selected_player.get('Asis', 0) or 0)
        sel_form = float(selected_player.get('Form', 0.0) or 0.0)
        sel_status = selected_player.get('Status', 'Tersedia')
        sel_chance = selected_player.get('Peluang Main GW (%)', 100)
        sel_next_opp = selected_player.get('Lawan GW Berikutnya', '-')
        sel_fdr = selected_player.get('FDR1', 3.0)

        # Quick Player Bio & KPI Metrics Card Row
        kp1, kp2, kp3, kp4, kp5, kp6 = st.columns(6)
        with kp1:
            st.metric("Total Poin", f"{sel_pts} pts", f"Form: {sel_form}")
        with kp2:
            st.metric("Total Gol / xG", f"{sel_goals} / {sel_xg:.2f}", f"xG: {sel_xg:.2f}")
        with kp3:
            st.metric("Total Asis / xA", f"{sel_assists} / {sel_xa:.2f}", f"xA: {sel_xa:.2f}")
        with kp4:
            st.metric("Harga (£m)", f"£{sel_cost:.1f}m", f"{sel_pos} · {sel_club}")
        with kp5:
            st.metric("Status Kebugaran", sel_status, f"Peluang: {sel_chance}%")
        with kp6:
            st.metric("Lawan Berikutnya", sel_next_opp, f"FDR: {sel_fdr}")

        # Fetch player element summary
        with st.spinner(f"Mengambil data histori pertandingan {sel_pname} dari API FPL..."):
            p_summary = fetch_player_element_summary(sel_pid)

        p_history = p_summary.get('history', [])
        p_past = p_summary.get('history_past', [])
        p_fixtures = p_summary.get('fixtures', [])

        if p_history:
            # Build match-by-match dataframe
            hist_rows = []
            for h in p_history:
                opp_id = h.get('opponent_team')
                opp_name = teams_dict.get(opp_id, f"Team {opp_id}")
                loc = "H" if h.get('was_home') else "A"
                gw_round = h.get('round', 1)
                gw_label = f"GW{gw_round} vs {opp_name} ({loc})"
                
                hist_rows.append({
                    'Gameweek': f"GW{gw_round}",
                    'Label Pertandingan': gw_label,
                    'GW_Num': gw_round,
                    'Lawan': f"{opp_name} ({loc})",
                    'Total Poin': int(h.get('total_points', 0)),
                    'Gol': int(h.get('goals_scored', 0)),
                    'Asis': int(h.get('assists', 0)),
                    'xG': round(float(h.get('expected_goals', 0.0)), 2),
                    'xA': round(float(h.get('expected_assists', 0.0)), 2),
                    'xGI': round(float(h.get('expected_goal_involvements', 0.0)), 2),
                    'xGC': round(float(h.get('expected_goals_conceded', 0.0)), 2),
                    'Menit Bermain': int(h.get('minutes', 0)),
                    'BPS': int(h.get('bps', 0)),
                    'Bonus Poin': int(h.get('bonus', 0)),
                    'Clean Sheet': int(h.get('clean_sheets', 0)),
                    'Saves': int(h.get('saves', 0)),
                    'Harga (£m)': h.get('value', 0) / 10.0,
                })

            df_phist = pd.DataFrame(hist_rows).sort_values(by='GW_Num')

            # Calculate cumulative metrics
            df_phist['Kumulatif Total Poin'] = df_phist['Total Poin'].cumsum()
            df_phist['Kumulatif Gol'] = df_phist['Gol'].cumsum()
            df_phist['Kumulatif Asis'] = df_phist['Asis'].cumsum()
            df_phist['Kumulatif xG'] = df_phist['xG'].cumsum().round(2)
            df_phist['Kumulatif xA'] = df_phist['xA'].cumsum().round(2)
            df_phist['Kumulatif xGI'] = df_phist['xGI'].cumsum().round(2)
            df_phist['Kumulatif Menit Bermain'] = df_phist['Menit Bermain'].cumsum()

            # Metric multiselect
            available_metrics = ['Total Poin', 'xG', 'xA', 'Gol', 'Asis', 'xGI', 'Menit Bermain', 'BPS', 'Bonus Poin']
            default_metrics = ['Total Poin', 'xG', 'xA', 'Gol', 'Asis']
            
            m_col1, m_col2 = st.columns([3, 1])
            with m_col1:
                chosen_metrics = st.multiselect(
                    "Pilih Metrik untuk Ditampilkan pada Grafik:",
                    options=available_metrics,
                    default=default_metrics,
                    key="p_metrics_multiselect"
                )
            with m_col2:
                chart_engine = st.selectbox(
                    "Tipe Grafik:",
                    options=["📈 Streamlit Line Chart", "📶 Streamlit Bar Chart", "✨ Dual-Axis Plotly Combo", "🌊 Streamlit Area Chart"],
                    index=2,
                    key="p_chart_type"
                )

            if not chosen_metrics:
                chosen_metrics = default_metrics

            prefix = "Kumulatif " if view_mode == "Akumulatif (Kumulatif)" else ""
            plot_cols = [(prefix + m) if (prefix + m) in df_phist.columns else m for m in chosen_metrics]

            # Render Charts based on selection
            if chart_engine == "✨ Dual-Axis Plotly Combo":
                fig = go.Figure()
                
                # Primary axis: Total Poin / Menit
                for m in chosen_metrics:
                    col_name = (prefix + m) if (prefix + m) in df_phist.columns else m
                    if col_name in df_phist.columns:
                        if m in ['Total Poin', 'Menit Bermain', 'BPS']:
                            fig.add_trace(go.Bar(
                                x=df_phist['Label Pertandingan'],
                                y=df_phist[col_name],
                                name=f"{col_name}",
                                yaxis='y1',
                                opacity=0.75
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=df_phist['Label Pertandingan'],
                                y=df_phist[col_name],
                                name=f"{col_name}",
                                mode='lines+markers',
                                yaxis='y2',
                                line=dict(width=3)
                            ))

                fig.update_layout(
                    title=f"Progresi Performa Match-by-Match: {sel_pname} ({view_mode})",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
                    height=480,
                    xaxis=dict(gridcolor="#e2e8f0", title="Gameweek & Pertandingan", tickangle=-45),
                    yaxis=dict(title="Poin / Menit / BPS", gridcolor="#e2e8f0"),
                    yaxis2=dict(
                        title="xG / xA / Gol / Asis",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            elif chart_engine == "📈 Streamlit Line Chart":
                chart_data = df_phist.set_index('Label Pertandingan')[plot_cols]
                st.line_chart(chart_data, height=420)

            elif chart_engine == "📶 Streamlit Bar Chart":
                chart_data = df_phist.set_index('Label Pertandingan')[plot_cols]
                st.bar_chart(chart_data, height=420)

            elif chart_engine == "🌊 Streamlit Area Chart":
                chart_data = df_phist.set_index('Label Pertandingan')[plot_cols]
                st.area_chart(chart_data, height=420)

            # Match-by-Match Details Table with custom styling
            with st.expander(f"📋 Rincian Lengkap Seluruh Pertandingan Musim Ini ({sel_pname})", expanded=False):
                st.dataframe(
                    df_phist[[
                        'Gameweek', 'Lawan', 'Total Poin', 'Gol', 'Asis', 'xG', 'xA', 'xGI',
                        'Menit Bermain', 'BPS', 'Bonus Poin', 'Clean Sheet', 'Saves', 'Harga (£m)'
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

            # Historical Past Seasons (If available)
            if p_past:
                with st.expander(f"📜 Rekam Jejak Musim-Musim Sebelumnya ({sel_pname})", expanded=False):
                    past_rows = []
                    for past in p_past:
                        past_rows.append({
                            'Musim': past.get('season_name'),
                            'Total Poin': past.get('total_points'),
                            'Menit Bermain': past.get('minutes'),
                            'Gol': past.get('goals_scored'),
                            'Asis': past.get('assists'),
                            'Clean Sheet': past.get('clean_sheets'),
                            'xG': round(float(past.get('expected_goals', 0.0)), 2),
                            'xA': round(float(past.get('expected_assists', 0.0)), 2),
                            'BPS': past.get('bps'),
                            'Bonus Poin': past.get('bonus'),
                            'Harga Awal (£m)': past.get('start_cost', 0) / 10.0,
                            'Harga Akhir (£m)': past.get('end_cost', 0) / 10.0
                        })
                    st.dataframe(pd.DataFrame(past_rows), use_container_width=True, hide_index=True)

            # Upcoming Fixtures for this player
            if p_fixtures:
                unplayed_p_fixtures = [
                    fx for fx in p_fixtures 
                    if not fx.get('finished') and not fx.get('finished_provisional') and not fx.get('started')
                ]
                if unplayed_p_fixtures:
                    with st.expander(f"🗓️ Jadwal Pertandingan Mendatang ({sel_pname})", expanded=False):
                        fix_rows = []
                        for fx in unplayed_p_fixtures[:5]:
                            is_h = fx.get('is_home')
                            opp_team_id = fx.get('team_a') if is_h else fx.get('team_h')
                            opp_team_name = teams_dict.get(opp_team_id, f"Team {opp_team_id}")
                            fix_rows.append({
                                'Gameweek': f"GW{fx.get('event')}",
                                'Lawan': f"{opp_team_name} ({'Kandang (H)' if is_h else 'Tandang (A)'})",
                                'Tingkat Kesulitan (FDR)': fx.get('difficulty', 3),
                                'Kickoff': fx.get('kickoff_time', '')[:10] if fx.get('kickoff_time') else '-'
                            })
                        st.dataframe(pd.DataFrame(fix_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"Pemain {sel_pname} belum memiliki riwayat pertandingan yang tercatat di musim berjalan ini.")
