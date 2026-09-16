"""
Option B: Bottom-Up Component Model Tab View.
"""

import pandas as pd
import streamlit as st
from src.firebase_client import (
    load_firebase_config,
    fetch_gw_differentials_from_firestore,
    is_gw4_finished,
    sync_historical_gw1_to_gw4
)

def render_tab_option_b(filtered_players, stats_xg, stats_xa, df_teams=None, fixtures=None, teams_dict=None, current_gw=4):
    """
    Renders Tab 5: Option B Component Model Breakdown and Diagnostics.
    Now enriched with 'Diff Attack Team' and 'Diff Defense Team' from Firebase Firestore.
    """
    st.subheader("🧮 Option B: Bottom-Up Component Model xPoin")
    st.write(
        "Model perhitungan xPoin individual berbasis dekonstruksi komponen: estimasi menit bermain, "
        "modulasi **Diff Attack Team** & **Diff Defense Team** (Skor Tim vs Lawan), kontribusi defensif (xDC), "
        "potensi clean sheet (xCS), saves kiper, dan bonus points (xBP)."
    )

    # -------------------------------------------------------------------------
    # FIREBASE FIRESTORE SYNC & STATUS EXPANDER
    # -------------------------------------------------------------------------
    with st.expander("☁️ Status Database Firebase & Sinkronisasi Variabel Differential (GW1 – GW4)", expanded=False):
        fb_cfg = load_firebase_config()
        c_status1, c_status2, c_status3 = st.columns([1.5, 1.5, 1.2])
        
        with c_status1:
            if fb_cfg:
                st.success(f"🟢 **Firestore Terhubung**: `{fb_cfg.get('projectId', 'GCP')}`")
                st.caption(f"Database ID: `{fb_cfg.get('firestoreDatabaseId', '(default)')}` (Mode Free Tier Hemat Kuota)")
            else:
                st.warning("⚠️ Konfigurasi Firebase belum terdeteksi.")
                
        with c_status2:
            if fixtures:
                gw4_fin, gw4_msg = is_gw4_finished(fixtures)
                if gw4_fin:
                    st.success(f"🏁 **Status GW4**: {gw4_msg}")
                else:
                    st.info(f"⏳ **Status GW4**: {gw4_msg}")
            else:
                st.write("Status fixture belum dimuat.")
                
        with c_status3:
            if st.button("⚡ Sinkronisasi Sekarang ke Firebase", help="Simpan kalkulasi differential GW1-GW4 ke Firestore"):
                if fixtures is not None and df_teams is not None and teams_dict is not None:
                    with st.spinner("Menyimpan kalkulasi ke Firestore..."):
                        ok, msg = sync_historical_gw1_to_gw4(fixtures, df_teams, teams_dict, force=True)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                else:
                    st.warning("Data fixture dan tim sedang dipersiapkan, silakan coba beberapa saat lagi.")

        st.markdown(
            """
            **Bagaimana Variabel Differential Bekerja di Option B:**
            1. **`Diff Attack Team` (Home/Away)**: Selisih Skor Serangan tim terhadap Skor Pertahanan lawan.
               - Jika positif: Tim unggul ofensif $\\rightarrow$ meningkatkan **xG Pred (Match)** dan **xA Pred (Match)**.
               - *Contoh GW4*: **Brentford** (+26.2) memiliki ekspektasi serang tinggi vs Bournemouth; sebaliknya **Bournemouth** (-9.4).
            2. **`Diff Defense Team` (Home/Away)**: Selisih Skor Pertahanan tim terhadap Skor Serangan lawan.
               - Jika positif: Pertahanan tim lebih rapat dari gempuran lawan $\\rightarrow$ memperbesar peluang **xCS Pts**.
               - Jika negatif: Pertahanan tim berada di bawah tekanan $\\rightarrow$ intensitas duel dan aksi bertahan naik (**xDC Pts** naik).
            3. **Keamanan & Efisiensi Firestore**:
               - Menggunakan caching memory Streamlit (1 jam) sehingga read dibatasi maksimal 1 kali per jam.
               - Query dan update hanya menyentuh dokumen ringkasan per Gameweek (`gw_1` s/d `gw_38`), 100% aman dalam kuota gratis (Free Tier).
            """
        )

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
            st.markdown("#### 📐 Model Prediksi xG (Regresi Linier)")
            st.info(
                "**Variabel Fitur Model xG:**\n"
                "1. `Opponent_xGC_per_90`: Ekspektasi kebobolan lawan per 90 menit\n"
                "2. `was_home`: Status laga kandang (1 = Home, 0 = Away)\n"
                "3. `form`: Tren performa pemain FPL\n"
                "4. `thread_per_90`: Ancaman gol (threat) per 90 menit\n"
                "5. `FDR`: Fixture Difficulty Rating lawan\n"
                "6. `Diff Attack Team`: Differential Serang Tim vs Pertahanan Lawan (Home/Away)"
            )
            pos_sel_xg = st.selectbox("Pilih Posisi (xG):", ["FWD", "MID", "DEF"], key="optb_pos_xg_sel")
            p_stats_xg = stats_xg.get(pos_sel_xg, stats_xg)
            
            # Highlight variabel dengan kontribusi terbesar
            top_var_xg = p_stats_xg.get('top_feature', '-')
            top_pct_xg = p_stats_xg.get('top_pct', 0.0)
            st.success(
                f"🌟 **Variabel dengan Kontribusi Terbesar (xG - {pos_sel_xg}):** `{top_var_xg}` "
                f"dengan pengaruh relatif **{top_pct_xg:.1f}%** berdasarkan koefisien regresi terstandarisasi (*Standardized Beta*)."
            )

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xg.get('r2', 0.0):.4f}")
            with mc2:
                st.metric("Mean Absolute Error (MAE)", f"{p_stats_xg.get('mae', 0.0):.4f}")
            with mc3:
                st.metric("Intercept (Konstanta β₀)", f"{p_stats_xg.get('intercept', 0.0):.4f}")
            with mc4:
                st.metric("Variabel Terpenting", f"{top_var_xg}", f"{top_pct_xg:.1f}%")

            st.markdown("##### 📊 Peringkat Kontribusi Variabel Fitur (Feature Importance)")
            st.dataframe(
                p_stats_xg.get('coef_df', pd.DataFrame()),
                use_container_width=True,
                column_config={
                    "Kontribusi Relatif (%)": st.column_config.ProgressColumn(
                        "Kontribusi Relatif (%)",
                        help="Persentase kontribusi absolut dari Standardized Beta terhadap total pengaruh model",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    )
                }
            )
            st.caption("💡 *Catatan: 'Beta Standar' menstandarkan skala unit masing-masing fitur sehingga kontribusi tiap variabel dapat diperbandingkan secara adil dan objektif.*")
            st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
            st.dataframe(p_stats_xg.get('eval_df', pd.DataFrame()), use_container_width=True)

        with m_tab2:
            st.markdown("#### 📐 Model Prediksi xA (Regresi Linier)")
            st.info(
                "**Variabel Fitur Model xA:**\n"
                "1. `Opponent_xGC_per_90`: Ekspektasi kebobolan lawan per 90 menit\n"
                "2. `was_home`: Status laga kandang (1 = Home, 0 = Away)\n"
                "3. `is_setpiece_taker`: Eksekutor bola mati (corner / freekick)\n"
                "4. `form`: Tren performa pemain FPL\n"
                "5. `creativity_per_90`: Kreativitas peluang per 90 menit\n"
                "6. `FDR`: Fixture Difficulty Rating lawan\n"
                "7. `Diff Attack Team`: Differential Serang Tim vs Pertahanan Lawan (Home/Away)"
            )
            pos_sel_xa = st.selectbox("Pilih Posisi (xA):", ["FWD", "MID", "DEF"], key="optb_pos_xa_sel")
            p_stats_xa = stats_xa.get(pos_sel_xa, stats_xa)

            # Highlight variabel dengan kontribusi terbesar
            top_var_xa = p_stats_xa.get('top_feature', '-')
            top_pct_xa = p_stats_xa.get('top_pct', 0.0)
            st.success(
                f"🌟 **Variabel dengan Kontribusi Terbesar (xA - {pos_sel_xa}):** `{top_var_xa}` "
                f"dengan pengaruh relatif **{top_pct_xa:.1f}%** berdasarkan koefisien regresi terstandarisasi (*Standardized Beta*)."
            )

            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                st.metric("R² Score (Akurasi Fitting)", f"{p_stats_xa.get('r2', 0.0):.4f}")
            with ac2:
                st.metric("Mean Absolute Error (MAE)", f"{p_stats_xa.get('mae', 0.0):.4f}")
            with ac3:
                st.metric("Intercept (Konstanta β₀)", f"{p_stats_xa.get('intercept', 0.0):.4f}")
            with ac4:
                st.metric("Variabel Terpenting", f"{top_var_xa}", f"{top_pct_xa:.1f}%")

            st.markdown("##### 📊 Peringkat Kontribusi Variabel Fitur (Feature Importance)")
            st.dataframe(
                p_stats_xa.get('coef_df', pd.DataFrame()),
                use_container_width=True,
                column_config={
                    "Kontribusi Relatif (%)": st.column_config.ProgressColumn(
                        "Kontribusi Relatif (%)",
                        help="Persentase kontribusi absolut dari Standardized Beta terhadap total pengaruh model",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    )
                }
            )
            st.caption("💡 *Catatan: 'Beta Standar' menstandarkan skala unit masing-masing fitur sehingga kontribusi tiap variabel dapat diperbandingkan secara adil dan objektif.*")
            st.markdown("##### 🔍 Top 10 Komparasi Data Training (Aktual vs Prediksi)")
            st.dataframe(p_stats_xa.get('eval_df', pd.DataFrame()), use_container_width=True)

    sorted_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False)
    optb_cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin (Option B)',
        'Diff Attack Team', 'Diff Defense Team',
        'xPoin', 'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
        'xG Pred (Match)', 'xA Pred (Match)', 'Avg Mins (L5M)', 'Lawan GW Berikutnya', 'Status'
    ]

    # Pastikan seluruh kolom tersedia
    for col in optb_cols:
        if col not in sorted_optb.columns:
            sorted_optb[col] = 0.0

    st.dataframe(
        sorted_optb[optb_cols],
        use_container_width=True,
        height=540,
        column_config={
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "xPoin (Option B)": st.column_config.NumberColumn(format="%.2f pts"),
            "Diff Attack Team": st.column_config.NumberColumn(
                "Diff Attack Team",
                format="%+.1f",
                help="Differential Serangan Tim vs Pertahanan Lawan (Skor Serangan - Skor Pertahanan Lawan)"
            ),
            "Diff Defense Team": st.column_config.NumberColumn(
                "Diff Defense Team",
                format="%+.1f",
                help="Differential Pertahanan Tim vs Serangan Lawan (Skor Pertahanan - Skor Serangan Lawan)"
            ),
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
    st.caption("💡 *Option B Formula: xPoin = xMins_Pts + xG_Pts(diff_att) + xA_Pts(diff_att) + xSaves_Pts + xDC_Pts(diff_def) + xCS_Pts(diff_def) + xBP.*")

