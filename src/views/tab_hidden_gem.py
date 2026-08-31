"""
Hidden Gem & Haul Predictor View Tab.
Integrates XGBoost Classifier (>10 Pts Prediction), Underperformance Index (xGI - GI), Volatility Analysis, and Seaborn Visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.models_hidden_gem import (
    build_hidden_gem_and_haul_dataset,
    train_xgboost_haul_model,
    render_seaborn_underperformance_plot
)

def render_tab_hidden_gem(fpl_data, fdr_summary, current_gw, filtered_player_ids=None, teams_dict=None):
    """
    Renders Tab 7: Hidden Gem Detector & XGBoost Haul (>10 Pts) Predictor.
    Integrates with global sidebar filters ('Filter Pemain FPL') to eliminate filter redundancy.
    """
    st.subheader("💎 Hidden Gem Detector & Prediksi Haul (>10 Poin) dengan XGBoost")
    st.write(
        "Modul prediktif cerdas untuk mengidentifikasi **aset diferensial potensial** dan memproyeksikan probabilitas terjadinya "
        "**'Haul' (Perolehan Skor ≥ 10 Poin)** di Gameweek berikutnya melalui 3 pilar: Deteksi Underperformance (xGI vs GI), "
        "Analisis Volatilitas/Ceiling Poin, dan Model Klasifikasi XGBoost Berbobot."
    )

    with st.spinner("Mengolah histori 5 match terakhir seluruh pemain, menghitung Underperformance Index, dan melatih XGBoost Classifier..."):
        df_train, df_upcoming = build_hidden_gem_and_haul_dataset(fpl_data, fdr_summary, current_gw)
        df_pred, feat_df, metrics = train_xgboost_haul_model(df_train, df_upcoming)

    if df_pred.empty:
        st.warning("Data pemain belum mencukupi untuk menjalankan analisis Hidden Gem.")
        return

    # Filter df_pred using global sidebar filters if available
    if filtered_player_ids is not None:
        # Keep global filtered players
        df_display_pool = df_pred[df_pred['id'].isin(filtered_player_ids)].copy()
    else:
        df_display_pool = df_pred.copy()

    # --- KPI SUMMARY CARDS ---
    top_gem = df_display_pool[df_display_pool['% Ownership'] <= 12.0].sort_values(by='Probability of Haul (%)', ascending=False).iloc[0] if not df_display_pool[df_display_pool['% Ownership'] <= 12.0].empty else (df_display_pool.iloc[0] if not df_display_pool.empty else df_pred.iloc[0])
    top_unlucky = df_display_pool.sort_values(by='underperformance_index', ascending=False).iloc[0] if not df_display_pool.empty else df_pred.iloc[0]
    top_ceiling = df_display_pool.sort_values(by='max_points', ascending=False).iloc[0] if not df_display_pool.empty else df_pred.iloc[0]
    highest_haul_prob = df_display_pool.iloc[0] if not df_display_pool.empty else df_pred.iloc[0]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div style="padding: 14px; background-color: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 8px;">
            <div style="font-size: 0.8rem; color: #166534; font-weight: 700; text-transform: uppercase;">💎 Top Hidden Gem</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{}</div>
            <div style="font-size: 0.85rem; color: #15803d; font-weight: 600;">Prob: {}% · Own: {}% · £{}m</div>
        </div>
        """.format(
            top_gem['Nama Pemain'],
            top_gem['Probability of Haul (%)'],
            top_gem['% Ownership'],
            top_gem['Harga (£m)']
        ), unsafe_allow_html=True)

    with kpi2:
        st.markdown("""
        <div style="padding: 14px; background-color: #fef2f2; border-left: 4px solid #dc2626; border-radius: 8px;">
            <div style="font-size: 0.8rem; color: #991b1b; font-weight: 700; text-transform: uppercase;">💣 Most Unlucky (Bom Waktu)</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{}</div>
            <div style="font-size: 0.85rem; color: #b91c1c; font-weight: 600;">Index: +{} (xGI {} vs GI {})</div>
        </div>
        """.format(
            top_unlucky['Nama Pemain'],
            top_unlucky['underperformance_index'],
            top_unlucky['roll_xGI'],
            top_unlucky['roll_actual_GI']
        ), unsafe_allow_html=True)

    with kpi3:
        st.markdown("""
        <div style="padding: 14px; background-color: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px;">
            <div style="font-size: 0.8rem; color: #1e40af; font-weight: 700; text-transform: uppercase;">🔥 Probabilitas Haul Tertinggi</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{}</div>
            <div style="font-size: 0.85rem; color: #2563eb; font-weight: 600;">Prob: {}% · Lawan: {}</div>
        </div>
        """.format(
            highest_haul_prob['Nama Pemain'],
            highest_haul_prob['Probability of Haul (%)'],
            highest_haul_prob['Lawan GW Berikutnya']
        ), unsafe_allow_html=True)

    with kpi4:
        st.markdown("""
        <div style="padding: 14px; background-color: #faf5ff; border-left: 4px solid #8b5cf6; border-radius: 8px;">
            <div style="font-size: 0.8rem; color: #5b21b6; font-weight: 700; text-transform: uppercase;">🎯 Model XGBoost ROC-AUC</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{:.3f}</div>
            <div style="font-size: 0.85rem; color: #6d28d9; font-weight: 600;">{} Match Samples · Pos Weight: {}</div>
        </div>
        """.format(
            metrics.get('auc', 0.85),
            metrics.get('total_samples', 0),
            metrics.get('scale_pos_weight', 1.0)
        ), unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 1: SEABORN 2D SCATTER PLOT ---
    st.markdown("### 1. 📊 Feature Engineering & Visualisasi Deteksi Underperformance (Seaborn)")
    st.markdown(
        "Scatter plot 2 dimensi membandingkan **`roll_xGI` (Expected Goal Involvement 5 laga terakhir)** vs "
        "**`roll_actual_GI` (Gol + Asis aktual 5 laga terakhir)**. Pemain di **Kuadran Merah (Kanan Bawah)** adalah aset dengan "
        "ancaman peluang tinggi namun belum terkonversi menjadi poin aktual (*Underperforming / Unlucky*), menjadikannya **kandidat ledakan poin (Haul) potensial**."
    )

    render_seaborn_underperformance_plot(df_pred)

    st.markdown("---")

    # --- SECTION 2: XGBOOST MODEL EVALUATION & FEATURE IMPORTANCE ---
    st.markdown("### 2. 🌲 Model XGBoost Classifier: Evaluasi & Feature Importance")
    col_feat1, col_feat2 = st.columns([1, 1])

    with col_feat1:
        st.markdown("##### 📈 Tingkat Kepentingan Fitur (Feature Importance)")
        fig_feat = px.bar(
            feat_df,
            x='Tingkat Kepentingan (%)',
            y='Nama Fitur',
            orientation='h',
            text='Tingkat Kepentingan (%)',
            color='Tingkat Kepentingan (%)',
            color_continuous_scale='Viridis',
            title="Kontribusi Fitur dalam Memprediksi Haul (≥10 Pts)"
        )
        fig_feat.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Tingkat Kepentingan (%)",
            yaxis_title="",
            margin=dict(l=10, r=20, t=40, b=20),
            height=340,
            showlegend=False
        )
        fig_feat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_feat2:
        st.markdown("##### ⚙️ Metrik Diagnostik & Penanganan Class Imbalance")
        st.markdown("""
        Model dikonfigurasi untuk menangani kelangkaan event *Haul* (hanya ~5-8% dari total penampilan pemain) dengan parameter **`scale_pos_weight`**:
        """)
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("ROC-AUC Score", f"{metrics.get('auc', 0):.4f}", help="Kemampuan model membedakan pemain yang akan haul vs tidak")
        m_c2.metric("Precision (Haul)", f"{metrics.get('precision', 0):.4f}", help="Akurasi saat model memprediksi terjadinya haul")
        m_c1.metric("Recall (Haul)", f"{metrics.get('recall', 0):.4f}", help="Persentase haul aktual yang berhasil tertangkap model")
        m_c2.metric("Class Weight Ratio", f"{metrics.get('scale_pos_weight', 1.0)}x", help="Pembobotan ekstra untuk kelas positif haul")

        st.info(
            f"💡 **Formula Feature Engineering:**\n"
            f"- `actual_GI = goals_scored + assists` (5 match terakhir)\n"
            f"- `underperformance_index = roll_xGI - roll_actual_GI`\n"
            f"- `std_points` & `max_points` mengukur volatilitas dan batas ceiling ledakan skor pemain."
        )

    st.markdown("---")

    # --- SECTION 3: TABEL KLASIFIKASI PROBABILITAS HAUL LENGKAP ---
    st.markdown("### 3. 🎯 Tabel Klasifikasi Probabilitas Haul GW Mendatang (Diurutkan Tertinggi)")
    st.caption("ℹ️ *Tabel ini otomatis tersinkronisasi dengan kontrol filter global di Sidebar (Klub, Posisi, Rentang Kepemilikan/Ownership, Harga, Menit Bermain, & Pencarian Nama Pemain).*")
    
    # Filter Spesifik Tab (Hanya Kategori Gem & Minimal Probabilitas Haul untuk menghindari redundansi)
    f_col1, f_col2 = st.columns([2, 1])
    
    all_categories = sorted(df_pred['Kategori Gem'].dropna().unique().tolist())

    with f_col1:
        filter_categories = st.multiselect(
            "Filter Kategori Gem:",
            options=all_categories,
            default=all_categories,
            key="hg_filter_categories",
            help="Pilih klasifikasi aset spesifik (misal: Ultimate Hidden Gem, Elite Captaincy, Bom Waktu, dll)"
        )
    with f_col2:
        min_prob = st.slider(
            "Minimal Probability of Haul (%):",
            min_value=0.0,
            max_value=60.0,
            value=10.0,
            step=2.0,
            key="hg_min_prob",
            help="Tampilkan hanya pemain dengan probabilitas haul di atas batas ini"
        )

    # Filter dari pool yang sudah disaring oleh Sidebar Global Filters
    display_df = df_display_pool[
        (df_display_pool['Kategori Gem'].isin(filter_categories if filter_categories else all_categories)) &
        (df_display_pool['Probability of Haul (%)'] >= min_prob)
    ].copy()

    table_cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', '% Ownership',
        'Probability of Haul (%)', 'Kategori Gem', 'Lawan GW Berikutnya', 'FDR',
        'underperformance_index', 'roll_xGI', 'roll_actual_GI',
        'player_form', 'minutes', 'std_points', 'max_points'
    ]

    st.dataframe(
        display_df[table_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability of Haul (%)": st.column_config.ProgressColumn(
                "Prob. Haul (%)",
                help="Probabilitas model XGBoost memprediksi pemain meraih ≥10 poin",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0
            ),
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "% Ownership": st.column_config.NumberColumn(format="%.1f%%"),
            "underperformance_index": st.column_config.NumberColumn(
                "Underperformance (Δ)",
                help="roll_xGI - roll_actual_GI. Nilai positif tinggi menandakan pemain sangat apes dan berpeluang balik modal.",
                format="+%.2f"
            ),
            "roll_xGI": st.column_config.NumberColumn("Roll xGI (5M)", format="%.2f"),
            "roll_actual_GI": st.column_config.NumberColumn("Roll GI (5M)", format="%.1f"),
            "player_form": st.column_config.NumberColumn("Form (L5M)", format="%.2f"),
            "minutes": st.column_config.NumberColumn("Avg Mins (L5M)", format="%.1f"),
            "std_points": st.column_config.NumberColumn("Volatilitas (Std)", format="%.2f"),
            "max_points": st.column_config.NumberColumn("Ceiling (Max Pts)", format="%d pts"),
            "FDR": st.column_config.NumberColumn("FDR", format="%.1f")
        }
    )

    # Unduh CSV
    csv_data = display_df[table_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Prediksi Haul & Hidden Gem (CSV)",
        data=csv_data,
        file_name=f"fpl_hidden_gem_haul_predictions_gw{current_gw}.csv",
        mime="text/csv"
    )
