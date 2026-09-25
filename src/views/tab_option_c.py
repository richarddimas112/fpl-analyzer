"""
Option C: Current Season Machine Learning Model Tab View.
Enriched with Positional Dedicated Models (FWD, MID, DEF, GK) and Purged Walk-Forward Time-Series Cross Validation.
"""

import pandas as pd
import streamlit as st
from src.models import build_option_c_model_and_view

def render_tab_option_c(fpl_data, fdr_summary, current_gw, filtered_players=None, price_range=None, df_option_c=None, fixtures_data=None):
    """
    Renders Tab 6: Current Season Machine Learning Models (Ridge, Linear Regression, Gradient Boosting).
    Now features dedicated positional models and Purged Walk-Forward Time-Series Cross Validation.
    """
    st.subheader("🔮 Option C: Machine Learning Musim Berjalan (Model Posisi & Purged Walk-Forward CV)")
    st.write(
        "Model prediksi prediktif xPoin yang dilatih **terpisah per posisi (FWD, MID, DEF, GK)** menggunakan histori "
        "pertandingan murni musim berjalan. Evaluasi performa divalidasi secara ketat menggunakan **Purged Walk-Forward Time-Series Cross Validation** "
        "(pengujian *out-of-sample* maju secara kronologis tanpa *lookahead data leakage*)."
    )

    if df_option_c is not None and not df_option_c.empty:
        df_view_c = df_option_c
        _, models_c = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw, fixtures_data=fixtures_data)
    else:
        with st.spinner("Melatih Model Machine Learning Option C terpisah per posisi & Walk-Forward CV..."):
            df_view_c, models_c = build_option_c_model_and_view(fpl_data, fdr_summary, current_gw, fixtures_data=fixtures_data)

    if not df_view_c.empty and models_c:
        # Pilihan Posisi untuk Menampilkan Evaluasi Model
        st.markdown("#### 🧭 Evaluasi Model & Validasi Time-Series per Posisi")
        
        pos_tabs = st.tabs(["🌐 Seluruh Posisi (Overall)", "⚽ FWD (Penyerang)", "🎯 MID (Gelandang)", "🛡️ DEF (Bek)", "🧤 GK (Kiper)"])
        pos_keys = ["ALL", "FWD", "MID", "DEF", "GK"]

        for tab_idx, tab_obj in enumerate(pos_tabs):
            pos_key = pos_keys[tab_idx]
            with tab_obj:
                pos_info = models_c.get(pos_key, models_c.get('ALL', {}))
                cv_mets = pos_info.get('cv_metrics', {})
                in_sample = pos_info.get('in_sample', {})
                n_samples = pos_info.get('n_samples', 0)
                n_folds = pos_info.get('folds_evaluated', 0)
                has_wf = pos_info.get('has_walk_forward_cv', False)

                # Status Banner
                st.markdown(
                    f"<div style='background: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;'>"
                    f"<strong style='color: #166534;'>🏆 Evaluasi Model Posisi: {pos_key} ({n_samples} Baris Laga)</strong><br>"
                    f"<span style='color: #15803d; font-size: 0.88rem;'>Metode Validasi: <strong>Purged Walk-Forward Time-Series CV</strong> ({n_folds} Fold Temporal Gameweek) • Generalisasi Out-of-Sample Murni Bebas Data Leakage</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # 4 Columns of Metrics (LR, Ridge, GBR, Ensemble)
                col_lr, col_rd, col_gb, col_ens = st.columns(4)

                with col_lr:
                    lr_m = cv_mets.get('Linear Regression', {})
                    lr_in = in_sample.get('Linear Regression', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #f8fafc; border-left: 3px solid #3b82f6; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #1e40af; font-size: 0.95rem;">1. Multiple Linear Reg</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{lr_m.get('mae', 0):.4f}", f"In-Sample: {lr_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{lr_m.get('rmse', 0):.4f}")
                    st.metric("CV R² Score", f"{lr_m.get('r2', 0):.4f}")

                with col_rd:
                    rd_m = cv_mets.get('Ridge Regression', {})
                    rd_in = in_sample.get('Ridge Regression', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #f8fafc; border-left: 3px solid #10b981; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #065f46; font-size: 0.95rem;">2. Ridge Reg (L2)</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{rd_m.get('mae', 0):.4f}", f"In-Sample: {rd_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{rd_m.get('rmse', 0):.4f}")
                    st.metric("CV R² Score", f"{rd_m.get('r2', 0):.4f}")

                with col_gb:
                    gb_m = cv_mets.get('Gradient Boosting', {})
                    gb_in = in_sample.get('Gradient Boosting', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #f8fafc; border-left: 3px solid #8b5cf6; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #5b21b6; font-size: 0.95rem;">3. Gradient Boosting</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{gb_m.get('mae', 0):.4f}", f"In-Sample: {gb_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{gb_m.get('rmse', 0):.4f}")
                    st.metric("CV R² Score", f"{gb_m.get('r2', 0):.4f}")

                with col_ens:
                    ens_m = cv_mets.get('Ensemble', {})
                    ens_in = in_sample.get('Ensemble', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #fefce8; border-left: 3px solid #eab308; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #854d0e; font-size: 0.95rem;">⭐ Ensemble Option C</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{ens_m.get('mae', 0):.4f}", f"In-Sample: {ens_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{ens_m.get('rmse', 0):.4f}")
                    st.metric("CV R² Score", f"{ens_m.get('r2', 0):.4f}")

                # Expander for Walk-Forward Fold Details
                fold_details = pos_info.get('fold_details', [])
                if fold_details:
                    with st.expander(f"📅 Rincian Hasil Purged Walk-Forward Fold per Gameweek ({pos_key})", expanded=False):
                        df_folds = pd.DataFrame(fold_details)
                        st.dataframe(df_folds, use_container_width=True)
                        st.caption("💡 *Setiap fold menguji model hanya pada gameweek berikutnya secara murni ke depan (Walk-Forward), mencegah bias masa depan.*")

                # Expander for Feature Importance / Weights
                with st.expander(f"🔍 Interpretasi Bobot Fitur Posisi {pos_key} (Feature Importance & Koefisien)", expanded=False):
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"##### 🌲 Feature Importance - Gradient Boosting ({pos_key})")
                        if 'importance_df' in pos_info and not pos_info['importance_df'].empty:
                            st.dataframe(
                                pos_info['importance_df'],
                                use_container_width=True,
                                column_config={
                                    "Tingkat Kepentingan (%)": st.column_config.ProgressColumn(
                                        "Tingkat Kepentingan (%)",
                                        format="%.1f%%",
                                        min_value=0.0,
                                        max_value=100.0
                                    )
                                }
                            )
                    with ic2:
                        st.markdown(f"##### 📐 Koefisien Regresi Linier & Ridge ({pos_key})")
                        if 'coef_df' in pos_info and not pos_info['coef_df'].empty:
                            st.dataframe(pos_info['coef_df'], use_container_width=True)

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
            st.caption(f"💡 *Tabel rangkuman prediksi menampilkan {len(sorted_df_c)} pemain berdasarkan filter aktif{price_caption}. Dilatih dengan model spesifik per posisi dan divalidasi Walk-Forward CV.*")
    else:
        st.warning("Belum ada data history musim berjalan yang cukup untuk melatih model Option C.")

