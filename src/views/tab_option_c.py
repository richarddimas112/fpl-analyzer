"""
Option C: Current Season Machine Learning Model Tab View.
"""

import streamlit as st
from src.models import build_option_c_model_and_view

def render_tab_option_c(fpl_data, fdr_summary, current_gw, filtered_players=None, price_range=None, df_option_c=None):
    """
    Renders Tab 6: Current Season Machine Learning Models (Ridge, Linear Regression, Gradient Boosting).
    Synchronized with main sidebar filters (price range, clubs, positions, etc.).
    """
    st.subheader("🔮 Option C: Prediksi Machine Learning Berdasarkan Musim Berjalan (Current Season Only)")
    st.write("Model prediksi prediktif xPoin yang dilatih **khusus menggunakan histori pertandingan musim berjalan**. Mengekstrak tren performa terkini (Rolling 3 Match Form, Rolling 5 Match Minutes, Rolling xG/xA/xGC, Home/Away advantage, dan FDR lawan) menggunakan 3 algoritma Machine Learning.")

    if df_option_c is not None and not df_option_c.empty:
        df_view_c = df_option_c
        _, models_c = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw)
    else:
        with st.spinner("Melatih Model Machine Learning Option C (Linear Regression, Ridge, Gradient Boosting)..."):
            df_view_c, models_c = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw)

    if not df_view_c.empty and models_c:
        # Comparison of 3 Models
        st.markdown("#### 📊 Evaluasi Performa 3 Algoritma Machine Learning (Current Season Data)")
        eval_c1, eval_c2, eval_c3 = st.columns(3)

        with eval_c1:
            lr_res = models_c.get('Linear Regression', {})
            st.markdown("""
            <div style="padding: 12px; background-color: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 6px; margin-bottom: 8px;">
                <strong style="color: #1e40af; font-size: 1rem;">1. Multiple Linear Regression</strong>
            </div>
            """, unsafe_allow_html=True)
            st.metric("R² Score", f"{lr_res.get('r2', 0):.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{lr_res.get('mae', 0):.4f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{lr_res.get('rmse', 0):.4f}")

        with eval_c2:
            rd_res = models_c.get('Ridge Regression', {})
            st.markdown("""
            <div style="padding: 12px; background-color: #f8fafc; border-left: 4px solid #10b981; border-radius: 6px; margin-bottom: 8px;">
                <strong style="color: #065f46; font-size: 1rem;">2. Ridge Regression (L2 Penalty)</strong>
            </div>
            """, unsafe_allow_html=True)
            st.metric("R² Score", f"{rd_res.get('r2', 0):.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{rd_res.get('mae', 0):.4f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{rd_res.get('rmse', 0):.4f}")

        with eval_c3:
            gb_res = models_c.get('Gradient Boosting', {})
            st.markdown("""
            <div style="padding: 12px; background-color: #f8fafc; border-left: 4px solid #8b5cf6; border-radius: 6px; margin-bottom: 8px;">
                <strong style="color: #5b21b6; font-size: 1rem;">3. Gradient Boosting Regressor</strong>
            </div>
            """, unsafe_allow_html=True)
            st.metric("R² Score", f"{gb_res.get('r2', 0):.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{gb_res.get('mae', 0):.4f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{gb_res.get('rmse', 0):.4f}")

        # Model Interpretation / Feature Importance Expander
        with st.expander("🔍 Interpretasi Bobot Fitur & Tingkat Kepentingan (Feature Importance)", expanded=False):
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown("##### 🌲 Tingkat Kepentingan Fitur - Gradient Boosting (%)")
                st.dataframe(models_c.get('Gradient Boosting', {}).get('importance_df'), use_container_width=True)
            with exp_c2:
                st.markdown("##### 📐 Koefisien Regresi Linier & Ridge (β)")
                st.dataframe(models_c.get('Ridge Regression', {}).get('coef_df'), use_container_width=True)

        # Apply Main Sidebar Filters to Option C Prediction Table
        df_filtered_c = df_view_c.copy()

        # 1. Apply global player filter if available (matches position, clubs, ownership, search, etc.)
        if filtered_players is not None:
            if not filtered_players.empty and 'id' in filtered_players.columns:
                valid_ids = set(filtered_players['id'].dropna().astype(int).tolist())
                df_filtered_c = df_filtered_c[df_filtered_c['id'].isin(valid_ids)]
            else:
                df_filtered_c = df_filtered_c.iloc[0:0]

        # 2. Explicitly apply price range filter
        if price_range is not None:
            p_min, p_max = float(price_range[0]), float(price_range[1])
            df_filtered_c = df_filtered_c[
                (df_filtered_c['Harga (£m)'] >= p_min) & (df_filtered_c['Harga (£m)'] <= p_max)
            ]

        st.markdown("#### 📋 Tabel Rangkuman Prediksi xPoin Option C (Gameweek Mendatang)")

        # Filter summary metrics row
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1:
            st.metric("Pemain Terfilter", f"{len(df_filtered_c)} / {len(df_view_c)}")
        with c_kpi2:
            if price_range is not None:
                st.metric("Filter Rentang Harga", f"£{price_range[0]:.1f}m - £{price_range[1]:.1f}m")
            else:
                st.metric("Filter Rentang Harga", "Semua Harga")
        with c_kpi3:
            if not df_filtered_c.empty:
                top_player = df_filtered_c.sort_values(by="xPoin (Option C Ensemble)", ascending=False).iloc[0]
                st.metric("Top xPoin Terfilter", f"{top_player['Nama Pemain']} ({top_player['xPoin (Option C Ensemble)']:.2f} pts)")
            else:
                st.metric("Top xPoin Terfilter", "-")

        if df_filtered_c.empty:
            st.info("ℹ️ Tidak ada pemain yang memenuhi kriteria filter saat ini. Coba sesuaikan rentang harga, posisi, atau klub di sidebar filter.")
        else:
            # Sort and Filter View
            sort_opt_c = st.selectbox(
                "Urutkan Tabel Berdasarkan Prediksi Model:",
                options=["xPoin (Option C Ensemble)", "xPoin (Gradient Boosting)", "xPoin (Ridge Reg)", "xPoin (Linear Reg)"],
                index=0,
                key="sort_opt_c_sel"
            )
            
            sorted_df_c = df_filtered_c.sort_values(by=sort_opt_c, ascending=False)
            
            display_c_cols = [
                'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)',
                'xPoin (Option C Ensemble)', 'xPoin (Gradient Boosting)', 'xPoin (Ridge Reg)', 'xPoin (Linear Reg)',
                'roll_mins_5', 'roll_pts_3', 'roll_xg_3', 'roll_xa_3', 'Lawan GW Berikutnya', 'FDR1', 'Peluang Main GW (%)'
            ]
            existing_c_cols = [c for c in display_c_cols if c in sorted_df_c.columns]

            st.dataframe(
                sorted_df_c[existing_c_cols],
                use_container_width=True,
                height=520,
                column_config={
                    "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
                    "xPoin (Option C Ensemble)": st.column_config.NumberColumn(format="%.2f pts"),
                    "xPoin (Gradient Boosting)": st.column_config.NumberColumn(format="%.2f pts"),
                    "xPoin (Ridge Reg)": st.column_config.NumberColumn(format="%.2f pts"),
                    "xPoin (Linear Reg)": st.column_config.NumberColumn(format="%.2f pts"),
                    "roll_mins_5": st.column_config.NumberColumn(format="%.1f m", help="Rata-rata menit 5 laga terakhir"),
                    "roll_pts_3": st.column_config.NumberColumn(format="%.2f", help="Rata-rata poin 3 laga terakhir"),
                    "roll_xg_3": st.column_config.NumberColumn(format="%.2f", help="Rata-rata xG 3 laga terakhir"),
                    "roll_xa_3": st.column_config.NumberColumn(format="%.2f", help="Rata-rata xA 3 laga terakhir"),
                    "FDR1": st.column_config.NumberColumn(format="%.2f"),
                    "Peluang Main GW (%)": st.column_config.ProgressColumn(
                        "Peluang Main (%)",
                        min_value=0,
                        max_value=100,
                        format="%d%%"
                    )
                }
            )
            
            price_caption = f" (Rentang Harga: £{price_range[0]:.1f}m - £{price_range[1]:.1f}m)" if price_range is not None else ""
            st.caption(f"💡 *Tabel rangkuman prediksi menampilkan {len(sorted_df_c)} pemain berdasarkan filter aktif{price_caption}. Menggunakan data murni histori musim berjalan.*")
    else:
        st.warning("Belum ada data history musim berjalan yang cukup untuk melatih model Option C.")
