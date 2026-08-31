"""
Option B: Bottom-Up Component Model Tab View.
"""

import pandas as pd
import streamlit as st

def render_tab_option_b(filtered_players, stats_xg, stats_xa):
    """
    Renders Tab 5: Option B Component Model Breakdown and Diagnostics.
    """
    st.subheader("🧮 Option B: Bottom-Up Component Model xPoin")
    st.write("Model perhitungan xPoin berdasarkan breakdown komponen individual: estimasi menit, xG/xA match regresi, kontribusi defensif, clean sheet, saves, dan bonus points.")

    # Summary Metrics Row for Option B
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric("Pemain Terfilter", len(filtered_players))
    with b_col2:
        top_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Top xPoin (Option B)", f"{top_optb['Nama Pemain']} ({top_optb['xPoin (Option B)']:.2f} pts)" if top_optb is not None else "-")
    with b_col3:
        top_xg_match = filtered_players.sort_values(by="xG Pred (Match)", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Top xG Pred (Match)", f"{top_xg_match['Nama Pemain']} ({top_xg_match['xG Pred (Match)']:.2f})" if top_xg_match is not None else "-")
    with b_col4:
        top_xa_match = filtered_players.sort_values(by="xA Pred (Match)", ascending=False).iloc[0] if not filtered_players.empty else None
        st.metric("Top xA Pred (Match)", f"{top_xa_match['Nama Pemain']} ({top_xa_match['xA Pred (Match)']:.2f})" if top_xa_match is not None else "-")

    with st.expander("🤖 Detail Model Regresi xG & xA (Match-Level Prediction)", expanded=False):
        m_tab1, m_tab2 = st.tabs(["⚽ Model Prediksi xG", "🎯 Model Prediksi xA"])
        with m_tab1:
            st.markdown("#### 📐 Regresi Linier Ekspektasi Gol (xG)")
            pos_sel_xg = st.selectbox("Pilih Posisi (xG):", ["FWD", "MID", "DEF"], key="optb_pos_xg_sel")
            p_stats_xg = stats_xg.get(pos_sel_xg, stats_xg)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xg.get('r2', 0.0):.4f}")
            with mc2:
                st.metric("Mean Absolute Error (MAE)", f"{p_stats_xg.get('mae', 0.0):.4f}")
            with mc3:
                st.metric("Intercept (Konstanta β₀)", f"{p_stats_xg.get('intercept', 0.0):.4f}")
            st.dataframe(p_stats_xg.get('coef_df', pd.DataFrame()), use_container_width=True)
            st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
            st.dataframe(p_stats_xg.get('eval_df', pd.DataFrame()), use_container_width=True)

        with m_tab2:
            st.markdown("#### 📐 Regresi Linier Ekspektasi Asis (xA)")
            pos_sel_xa = st.selectbox("Pilih Posisi (xA):", ["FWD", "MID", "DEF"], key="optb_pos_xa_sel")
            p_stats_xa = stats_xa.get(pos_sel_xa, stats_xa)
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xa.get('r2', 0.0):.4f}")
            with ac2:
                st.metric("Mean Absolute Error (MAE)", f"{p_stats_xa.get('mae', 0.0):.4f}")
            with ac3:
                st.metric("Intercept (Konstanta β₀)", f"{p_stats_xa.get('intercept', 0.0):.4f}")
            st.dataframe(p_stats_xa.get('coef_df', pd.DataFrame()), use_container_width=True)
            st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
            st.dataframe(p_stats_xa.get('eval_df', pd.DataFrame()), use_container_width=True)

    sorted_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False)
    optb_cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin (Option B)', 'xPoin',
        'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
        'xG Pred (Match)', 'xA Pred (Match)', 'Avg Mins (L5M)', 'Lawan GW Berikutnya', 'Status'
    ]

    st.dataframe(
        sorted_optb[optb_cols],
        use_container_width=True,
        height=520,
        column_config={
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "xPoin (Option B)": st.column_config.NumberColumn(format="%.2f pts"),
            "xPoin": st.column_config.NumberColumn(format="%.2f pts"),
            "xMins Pts": st.column_config.NumberColumn(format="%.2f"),
            "xG Pts": st.column_config.NumberColumn(format="%.2f"),
            "xA Pts": st.column_config.NumberColumn(format="%.2f"),
            "xSaves Pts": st.column_config.NumberColumn(format="%.2f"),
            "xDC Pts": st.column_config.NumberColumn(format="%.2f"),
            "xCS Pts": st.column_config.NumberColumn(format="%.2f"),
            "xBP": st.column_config.NumberColumn(format="%.2f"),
            "xG Pred (Match)": st.column_config.NumberColumn(format="%.2f"),
            "xA Pred (Match)": st.column_config.NumberColumn(format="%.2f"),
            "Avg Mins (L5M)": st.column_config.NumberColumn(format="%.1f mins")
        }
    )
    st.caption("💡 *Option B Formula: xPoin = xMins_Pts + xG_Pts + xA_Pts + xSaves_Pts + xDC_Pts + xCS_Pts + xBP.*")
