"""
Option C: Current Season Machine Learning Model Tab View.
Enriched with positional models (FWD, MID, DEF, GK) and Walk-Forward Time-Series Cross Validation.
"""

import pandas as pd
import streamlit as st
from src.models import build_option_c_model_and_view

def render_tab_option_c(fpl_data, fdr_summary, current_gw, filtered_players=None, price_range=None, df_option_c=None, fixtures_data=None):
    """
    Renders Tab 6: Current Season Machine Learning Models (Ridge, Linear Regression, Gradient Boosting).
    Shows positional model predictions and Walk-Forward evaluation metrics.
    """
    st.subheader("🔮 Option C: Machine Learning Musim Berjalan (Model Posisi & Walk-Forward CV)")
    st.write(
        "Model prediksi prediktif xPoin yang dilatih **terpisah per posisi (FWD, MID, DEF, GK)** menggunakan histori "
        "pertandingan murni musim berjalan. Validasi **Walk-Forward Time-Series** ditampilkan bila jumlah gameweek dan sampel mencukupi; "
        "jika belum, metrik out-of-sample tidak ditampilkan sebagai hasil validasi."
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
        
        pos_tabs = st.tabs(["🌐 All Positions", "⚽ Forwards", "🎯 Midfielders", "🛡️ Defenders", "🧤 Goalkeepers"])
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
                used_global_fallback = pos_info.get('used_global_position_fallback', False)

                # Do not label in-sample metrics as out-of-sample when no temporal folds exist.
                if has_wf:
                    st.markdown(
                        f"<div style='background: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;'>"
                        f"<strong style='color: #166534;'>Evaluasi Model {pos_key} ({n_samples} baris laga)</strong><br>"
                        f"<span style='color: #15803d; font-size: 0.88rem;'>Walk-Forward Time-Series CV: {n_folds} fold gameweek, dievaluasi out-of-sample secara kronologis. "
                        f"{'Data gabungan lintas posisi dipakai karena sampel posisi ini sedikit.' if used_global_fallback else 'Model memakai data posisi ini.'}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(
                        f"Data belum cukup untuk validasi out-of-sample pada {pos_key}. "
                        "Metrik CV disembunyikan; metrik in-sample di bawah hanya diagnostik dan bukan bukti kemampuan prediksi."
                    )
                    if used_global_fallback:
                        st.caption("Prediksi posisi ini memakai fallback data gabungan lintas posisi karena sampel historis khusus posisi belum cukup.")

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
                    st.metric("CV MAE (Out-of-Sample)", f"{lr_m['mae']:.4f}" if has_wf and 'mae' in lr_m else "—", f"In-Sample: {lr_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{lr_m['rmse']:.4f}" if has_wf and 'rmse' in lr_m else "—")
                    st.metric("CV R² Score", f"{lr_m['r2']:.4f}" if has_wf and 'r2' in lr_m else "—")

                with col_rd:
                    rd_m = cv_mets.get('Ridge Regression', {})
                    rd_in = in_sample.get('Ridge Regression', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #f8fafc; border-left: 3px solid #10b981; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #065f46; font-size: 0.95rem;">2. Ridge Reg (L2)</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{rd_m['mae']:.4f}" if has_wf and 'mae' in rd_m else "—", f"In-Sample: {rd_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{rd_m['rmse']:.4f}" if has_wf and 'rmse' in rd_m else "—")
                    st.metric("CV R² Score", f"{rd_m['r2']:.4f}" if has_wf and 'r2' in rd_m else "—")

                with col_gb:
                    gb_m = cv_mets.get('Gradient Boosting', {})
                    gb_in = in_sample.get('Gradient Boosting', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #f8fafc; border-left: 3px solid #8b5cf6; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #5b21b6; font-size: 0.95rem;">3. Gradient Boosting</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{gb_m['mae']:.4f}" if has_wf and 'mae' in gb_m else "—", f"In-Sample: {gb_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{gb_m['rmse']:.4f}" if has_wf and 'rmse' in gb_m else "—")
                    st.metric("CV R² Score", f"{gb_m['r2']:.4f}" if has_wf and 'r2' in gb_m else "—")

                with col_ens:
                    ens_m = cv_mets.get('Ensemble', {})
                    ens_in = in_sample.get('Ensemble', {})
                    st.markdown("""
                    <div style="padding: 10px; background-color: #fefce8; border-left: 3px solid #eab308; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #854d0e; font-size: 0.95rem;">⭐ Ensemble Option C</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("CV MAE (Out-of-Sample)", f"{ens_m['mae']:.4f}" if has_wf and 'mae' in ens_m else "—", f"In-Sample: {ens_in.get('mae', 0):.3f}")
                    st.metric("CV RMSE", f"{ens_m['rmse']:.4f}" if has_wf and 'rmse' in ens_m else "—")
                    st.metric("CV R² Score", f"{ens_m['r2']:.4f}" if has_wf and 'r2' in ens_m else "—")

                # Expander for Walk-Forward Fold Details
                fold_details = pos_info.get('fold_details', [])
                if fold_details:
                    with st.expander(f"📅 Rincian Hasil Walk-Forward Fold per Gameweek ({pos_key})", expanded=False):
                        df_folds = pd.DataFrame(fold_details)
                        st.dataframe(df_folds, use_container_width=True)
                        st.caption("💡 *Setiap fold melatih data hingga satu gameweek, lalu menguji pada gameweek berikutnya yang memiliki sampel.*")

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

            # -----------------------------------------------------------------
            # INTERACTIVE PLAYER INSPECTION DRAWER (Option C - Sports Broadcast)
            # -----------------------------------------------------------------
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Kartu Profil & Komparasi Model Option C (Player Inspection Card)")

            c_options = sorted_df_c['Nama Pemain'].tolist()
            if c_options:
                c_sel1, c_sel2 = st.columns([2, 4])
                with c_sel1:
                    inspect_c_name = st.selectbox(
                        "Pilih Pemain untuk Diinspeksi (Option C):",
                        options=c_options,
                        index=0,
                        key="optc_inspect_player_sb"
                    )

                p_c = sorted_df_c[sorted_df_c['Nama Pemain'] == inspect_c_name].iloc[0]

                pos_badge_color = {
                    'FWD': '#f43f5e',
                    'MID': '#f59e0b',
                    'DEF': '#06b6d4',
                    'GK': '#10b981'
                }.get(p_c['Posisi'], '#64748b')

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #111a2e 0%, #0d1527 100%); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 14px; margin-bottom: 16px;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">{p_c['Nama Pemain']}</span>
                                <span style="color: {pos_badge_color}; font-weight: 800; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1);">{p_c['Posisi']}</span>
                                <span style="color: #94a3b8; font-size: 0.88rem; font-weight: 600;">{p_c['Klub']}</span>
                            </div>
                            <div style="color: #64748b; font-size: 0.8rem; margin-top: 3px; font-family: 'JetBrains Mono', monospace;">
                                Lawan: <b style="color: #e2e8f0;">{p_c.get('Lawan GW Berikutnya', '-')}</b> &bull; FDR: <b style="color: #06b6d4;">{p_c.get('FDR1', 3.0):.1f}</b> &bull; Harga: <b style="color: #38ef7d;">£{p_c['Harga (£m)']:.1f}m</b> &bull; Peluang Main: <b>{p_c.get('Peluang Main GW (%)', 100):.0f}%</b>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace;">Option C Ensemble xPoin</div>
                            <div style="font-size: 2.1rem; font-weight: 900; color: #a855f7; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                                {p_c['xPoin (Option C Ensemble)']:.2f} <span style="font-size: 0.9rem; font-weight: 600; color: #64748b;">pts</span>
                            </div>
                        </div>
                    </div>
                    
                </div>
                """, unsafe_allow_html=True)

                st.caption("🤖 Komparasi Multi-Model ML & Tren Musim Berjalan")
                cm1, cm2, cm3, cm4, cm5 = st.columns(5)
                with cm1:
                    st.metric("Gradient Boosting", f"{p_c.get('xPoin (Gradient Boosting)', 0.0):.2f} pts")
                with cm2:
                    st.metric("Ridge Regression", f"{p_c.get('xPoin (Ridge Reg)', 0.0):.2f} pts")
                with cm3:
                    st.metric("Linear Regression", f"{p_c.get('xPoin (Linear Reg)', 0.0):.2f} pts")
                with cm4:
                    st.metric("Avg Poin (L3M)", f"{p_c.get('roll_pts_3', 0.0):.2f} pts")
                with cm5:
                    st.metric("Avg xG (L3M)", f"{p_c.get('roll_xg_3', 0.0):.2f}")
    else:
        st.warning("Belum ada data history musim berjalan yang cukup untuk melatih model Option C.")

