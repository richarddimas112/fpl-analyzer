"""
Interactive Visualizations Tab View (Scatter Plot, Pearson r, and Club Aggregates).
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import pearsonr
from src.views.tab_radar import render_player_comparison_radar_tab

def render_tab_visualizations(filtered_players, players_df, fpl_data, teams_dict):
    """
    Renders Tab 2: Interactive Scatter Plot with Pearson Correlation, Club Aggregates Bar Chart, and Player Comparison Radar.
    """
    st.subheader("📈 Visualisasi Interaktif & Radar Komparasi Pemain")
    st.write("Analisis hubungan antar variabel statistik pemain, perbandingan agregat klub, serta komparasi head-to-head 2 pemain dengan grafik radar interaktif.")

    chart_subtab1, chart_subtab2, chart_subtab3 = st.tabs([
        "🔵 Scatter Plot & Pearson Correlation", 
        "📊 Bar Chart Agregat Klub",
        "⚔️ Komparasi 2 Pemain (Radar Chart)"
    ])

    # SECTION 1: SCATTER PLOT & PEARSON R
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
        render_player_comparison_radar_tab(players_df, fpl_data, teams_dict)
