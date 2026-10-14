"""
Option B: Bottom-Up Component Model Tab View.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

    # -------------------------------------------------------------------------
    # VISUALISASI KOMPARASI xG vs G & xA vs A (GROUP BY POSISI PEMAIN)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Analisis Efisiensi: Komparasi xG vs Gol (G) & xA vs Asis (A) per Posisi")
    st.write(
        "Visualisasi di bawah membandingkan output aktual (*Goals Scored* & *Assists*) terhadap ekspektasi statistik "
        "(*Expected Goals* & *Expected Assists*) yang dikelompokkan berdasarkan posisi pemain (**FWD, MID, DEF, GK**). "
        "Insight ini menjadi dasar kalibrasi **Weighted Multiplier** posisi di Model B."
    )

    if not filtered_players.empty:
        # Menyiapkan kolom agregasi
        viz_df = filtered_players.copy()
        
        # Ekstraksi kolom numerik
        col_g = 'Gol' if 'Gol' in viz_df.columns else ('goals_scored' if 'goals_scored' in viz_df.columns else None)
        col_xg = 'xG' if 'xG' in viz_df.columns else ('expected_goals' if 'expected_goals' in viz_df.columns else None)
        col_a = 'Asis' if 'Asis' in viz_df.columns else ('assists' if 'assists' in viz_df.columns else None)
        col_xa = 'xA' if 'xA' in viz_df.columns else ('expected_assists' if 'expected_assists' in viz_df.columns else None)

        if col_g and col_xg and col_a and col_xa:
            viz_df['g_num'] = pd.to_numeric(viz_df[col_g], errors='coerce').fillna(0.0)
            viz_df['xg_num'] = pd.to_numeric(viz_df[col_xg], errors='coerce').fillna(0.0)
            viz_df['a_num'] = pd.to_numeric(viz_df[col_a], errors='coerce').fillna(0.0)
            viz_df['xa_num'] = pd.to_numeric(viz_df[col_xa], errors='coerce').fillna(0.0)

            # Agregasi Group By Posisi
            pos_order = ['FWD', 'MID', 'DEF', 'GK']
            agg_pos = viz_df.groupby('Posisi')[['g_num', 'xg_num', 'a_num', 'xa_num']].sum().reindex(pos_order).fillna(0.0)

            # Delta & Rasio Konversi
            agg_pos['Delta_G_xG'] = agg_pos['g_num'] - agg_pos['xg_num']
            agg_pos['Delta_A_xA'] = agg_pos['a_num'] - agg_pos['xa_num']
            agg_pos['Ratio_G_xG'] = (agg_pos['g_num'] / agg_pos['xg_num'].replace(0, float('nan'))).round(2)
            agg_pos['Ratio_A_xA'] = (agg_pos['a_num'] / agg_pos['xa_num'].replace(0, float('nan'))).round(2)

            # Kartu Metrik Ringkasan per Posisi
            col_card1, col_card2, col_card3, col_card4 = st.columns(4)
            card_cols = [col_card1, col_card2, col_card3, col_card4]

            for idx, p_key in enumerate(pos_order):
                if p_key in agg_pos.index:
                    row_p = agg_pos.loc[p_key]
                    with card_cols[idx]:
                        delta_g_sign = "+" if row_p['Delta_G_xG'] >= 0 else ""
                        delta_a_sign = "+" if row_p['Delta_A_xA'] >= 0 else ""
                        icon = "⚽" if p_key == 'FWD' else ("🎯" if p_key == 'MID' else ("🛡️" if p_key == 'DEF' else "🧤"))
                        st.markdown(
                            f"""
                            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                                <div style="font-weight: 700; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">{icon} {p_key} Overview</div>
                                <div style="font-size: 0.82rem; color: #475569;">
                                    <b>Gol vs xG:</b> <span style="font-weight: 700; color: #0f172a;">{row_p['g_num']:.0f}</span> / {row_p['xg_num']:.1f} 
                                    <span style="color: {'#16a34a' if row_p['Delta_G_xG'] >= 0 else '#dc2626'}; font-weight: 600;">({delta_g_sign}{row_p['Delta_G_xG']:.1f})</span>
                                </div>
                                <div style="font-size: 0.82rem; color: #475569; margin-top: 2px;">
                                    <b>Asis vs xA:</b> <span style="font-weight: 700; color: #0f172a;">{row_p['a_num']:.0f}</span> / {row_p['xa_num']:.1f}
                                    <span style="color: {'#16a34a' if row_p['Delta_A_xA'] >= 0 else '#dc2626'}; font-weight: 600;">({delta_a_sign}{row_p['Delta_A_xA']:.1f})</span>
                                </div>
                                <div style="margin-top: 6px; font-size: 0.75rem; color: #64748b;">
                                    Rasio: G/xG = <b>{row_p['Ratio_G_xG'] if pd.notna(row_p['Ratio_G_xG']) else '0.00'}x</b> | A/xA = <b>{row_p['Ratio_A_xA'] if pd.notna(row_p['Ratio_A_xA']) else '0.00'}x</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Dua Grafik Berdampingan: xG vs G dan xA vs A
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Grouped Bar Chart: xG vs G per Posisi
                fig_g = go.Figure()
                fig_g.add_trace(go.Bar(
                    x=pos_order,
                    y=[agg_pos.loc[p, 'xg_num'] for p in pos_order],
                    name='Expected Goals (xG)',
                    marker_color='#93c5fd',
                    text=[f"{agg_pos.loc[p, 'xg_num']:.1f}" for p in pos_order],
                    textposition='outside'
                ))
                fig_g.add_trace(go.Bar(
                    x=pos_order,
                    y=[agg_pos.loc[p, 'g_num'] for p in pos_order],
                    name='Aktual Gol (G)',
                    marker_color='#2563eb',
                    text=[f"{agg_pos.loc[p, 'g_num']:.0f}" for p in pos_order],
                    textposition='outside'
                ))
                fig_g.update_layout(
                    title="⚽ Komparasi Expected Goals (xG) vs Aktual Gol (G)",
                    barmode='group',
                    xaxis_title="Posisi Pemain",
                    yaxis_title="Total Gol / xG",
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=30),
                    plot_bgcolor='#fafafa',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_g, use_container_width=True)

            with chart_col2:
                # Grouped Bar Chart: xA vs A per Posisi
                fig_a = go.Figure()
                fig_a.add_trace(go.Bar(
                    x=pos_order,
                    y=[agg_pos.loc[p, 'xa_num'] for p in pos_order],
                    name='Expected Assists (xA)',
                    marker_color='#fde68a',
                    text=[f"{agg_pos.loc[p, 'xa_num']:.1f}" for p in pos_order],
                    textposition='outside'
                ))
                fig_a.add_trace(go.Bar(
                    x=pos_order,
                    y=[agg_pos.loc[p, 'a_num'] for p in pos_order],
                    name='Aktual Asis (A)',
                    marker_color='#d97706',
                    text=[f"{agg_pos.loc[p, 'a_num']:.0f}" for p in pos_order],
                    textposition='outside'
                ))
                fig_a.update_layout(
                    title="🎯 Komparasi Expected Assists (xA) vs Aktual Asis (A)",
                    barmode='group',
                    xaxis_title="Posisi Pemain",
                    yaxis_title="Total Asis / xA",
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=30),
                    plot_bgcolor='#fafafa',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_a, use_container_width=True)

            # Scatter Plot: Distribusi Pemain xG vs G & xA vs A dengan Garis Paritas (x=y)
            with st.expander("📈 Scatter Plot Distribusi Pemain: Aktual vs Ekspektasi per Posisi", expanded=False):
                sc1, sc2 = st.columns(2)
                
                with sc1:
                    fig_sc_g = px.scatter(
                        viz_df[viz_df['g_num'] + viz_df['xg_num'] > 0],
                        x='xg_num',
                        y='g_num',
                        color='Posisi',
                        hover_name='Nama Pemain',
                        hover_data={'Klub': True, 'Harga (£m)': ':.1f', 'xg_num': ':.2f', 'g_num': ':.0f'},
                        labels={'xg_num': 'Expected Goals (xG)', 'g_num': 'Aktual Gol (G)'},
                        title="Distribusi Finisher: Gol vs xG (Titik di atas garis = Overperformer)",
                        color_discrete_map={'FWD': '#ef4444', 'MID': '#f59e0b', 'DEF': '#3b82f6', 'GK': '#10b981'}
                    )
                    max_val_g = max(viz_df['xg_num'].max(), viz_df['g_num'].max(), 1.0)
                    fig_sc_g.add_shape(
                        type='line', line=dict(dash='dash', color='gray', width=1),
                        x0=0, y0=0, x1=max_val_g, y1=max_val_g
                    )
                    fig_sc_g.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=30))
                    st.plotly_chart(fig_sc_g, use_container_width=True)

                with sc2:
                    fig_sc_a = px.scatter(
                        viz_df[viz_df['a_num'] + viz_df['xa_num'] > 0],
                        x='xa_num',
                        y='a_num',
                        color='Posisi',
                        hover_name='Nama Pemain',
                        hover_data={'Klub': True, 'Harga (£m)': ':.1f', 'xa_num': ':.2f', 'a_num': ':.0f'},
                        labels={'xa_num': 'Expected Assists (xA)', 'a_num': 'Aktual Asis (A)'},
                        title="Distribusi Playmaker: Asis vs xA (Titik di atas garis = Overperformer)",
                        color_discrete_map={'FWD': '#ef4444', 'MID': '#f59e0b', 'DEF': '#3b82f6', 'GK': '#10b981'}
                    )
                    max_val_a = max(viz_df['xa_num'].max(), viz_df['a_num'].max(), 1.0)
                    fig_sc_a.add_shape(
                        type='line', line=dict(dash='dash', color='gray', width=1),
                        x0=0, y0=0, x1=max_val_a, y1=max_val_a
                    )
                    fig_sc_a.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=30))
                    st.plotly_chart(fig_sc_a, use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    sorted_optb = filtered_players.sort_values(by="xPoin (Option B)", ascending=False)
    optb_cols = [
        'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)', 'xPoin (Option B)',
        'Diff Attack Team', 'Diff Defense Team', 'Peluang CS (%)',
        'xG Mult', 'xA Mult',
        'xPoin', 'xMins Pts', 'xG Pts', 'xA Pts', 'xSaves Pts', 'xDC Pts', 'xCS Pts', 'xBP',
        'xG Pred (Match)', 'xA Pred (Match)', 'Avg Mins (L5M)', 'Lawan GW Berikutnya', 'Status'
    ]

    # Pastikan seluruh kolom tersedia
    for col in optb_cols:
        if col not in sorted_optb.columns:
            sorted_optb[col] = 0.0

    st.markdown("#### 📋 Tabel Rangkuman Dekonstruksi Komponen xPoin (Option B)")
    
    # -------------------------------------------------------------------------
    # INTERACTIVE TOGGLE & SLIDERS FOR POSITION WEIGHTED MULTIPLIERS
    # -------------------------------------------------------------------------
    with st.expander("⚙️ Kalibrasi Weighted Multiplier Posisi (Goal > xG & Assist > xA Tuning)", expanded=False):
        c_tog1, c_tog2 = st.columns([2, 4])
        with c_tog1:
            use_weighted = st.toggle(
                "Aktifkan Weighted Multiplier Posisi",
                value=True,
                help="Bila aktif, menerapkan multiplier terbobot posisi (FWD/MID vs DEF/GK) berdasarkan tren konversi historis.",
                key="toggle_pos_weighted_mult"
            )
        with c_tog2:
            st.caption("💡 *Sesuaikan slider di bawah untuk menguji skenario efisiensi penyelesaian (finishing) & kreasi peluang per posisi secara realtime.*")

        if use_weighted:
            c_fwd, c_mid, c_def, c_gk = st.columns(4)
            with c_fwd:
                st.markdown("**⚽ FWD (Penyerang)**")
                fwd_xg_m = st.slider("xG Multiplier (FWD)", 0.70, 1.40, 1.08, 0.01, key="slider_fwd_xg")
                fwd_xa_m = st.slider("xA Multiplier (FWD)", 0.70, 1.40, 1.05, 0.01, key="slider_fwd_xa")
            with c_mid:
                st.markdown("**🎯 MID (Gelandang)**")
                mid_xg_m = st.slider("xG Multiplier (MID)", 0.70, 1.40, 1.04, 0.01, key="slider_mid_xg")
                mid_xa_m = st.slider("xA Multiplier (MID)", 0.70, 1.40, 1.06, 0.01, key="slider_mid_xa")
            with c_def:
                st.markdown("**🛡️ DEF (Bek)**")
                def_xg_m = st.slider("xG Multiplier (DEF)", 0.50, 1.30, 0.88, 0.01, key="slider_def_xg")
                def_xa_m = st.slider("xA Multiplier (DEF)", 0.50, 1.30, 0.92, 0.01, key="slider_def_xa")
            with c_gk:
                st.markdown("**🧤 GK (Kiper)**")
                st.write("xG Mult: `0.00x` *(Proteksi Kiper)*")
                gk_xa_m = st.slider("xA Multiplier (GK)", 0.00, 1.00, 0.50, 0.05, key="slider_gk_xa")
                gk_xg_m = 0.00

            active_xg_mult = {'FWD': fwd_xg_m, 'MID': mid_xg_m, 'DEF': def_xg_m, 'GK': gk_xg_m}
            active_xa_mult = {'FWD': fwd_xa_m, 'MID': mid_xa_m, 'DEF': def_xa_m, 'GK': gk_xa_m}
        else:
            active_xg_mult = {'FWD': 1.0, 'MID': 1.0, 'DEF': 1.0, 'GK': 0.0}
            active_xa_mult = {'FWD': 1.0, 'MID': 1.0, 'DEF': 1.0, 'GK': 1.0}
            st.info("ℹ️ Mode mentah (tanpa weighted multiplier): Seluruh pengali posisi diset ke 1.00x.")

    # Rekalkulasi xG Pts, xA Pts, dan xPoin (Option B) secara dinamis sesuai multiplier aktif
    poin_gol_map = {'GK': 10.0, 'DEF': 6.0, 'MID': 5.0, 'FWD': 4.0}
    poin_gol = sorted_optb['Posisi'].map(poin_gol_map).fillna(4.0)
    xg_mult_series = sorted_optb['Posisi'].map(active_xg_mult).fillna(1.0)
    xa_mult_series = sorted_optb['Posisi'].map(active_xa_mult).fillna(1.0)

    sorted_optb['xG Mult'] = xg_mult_series.round(2)
    sorted_optb['xA Mult'] = xa_mult_series.round(2)
    sorted_optb['xG Pts'] = (sorted_optb['xG Pred (Match)'] * poin_gol * xg_mult_series).round(2)
    sorted_optb['xA Pts'] = (sorted_optb['xA Pred (Match)'] * 3.0 * xa_mult_series).round(2)
    sorted_optb['xPoin (Option B)'] = (
        sorted_optb['xMins Pts'] + sorted_optb['xG Pts'] + sorted_optb['xA Pts'] +
        sorted_optb['xSaves Pts'] + sorted_optb['xDC Pts'] + sorted_optb['xCS Pts'] + sorted_optb['xBP']
    ).round(2)

    # Re-sort descending based on recalibrated Option B xPoin
    sorted_optb = sorted_optb.sort_values(by="xPoin (Option B)", ascending=False)

    st.info(
        "⚖️ **Positional Weighted Multipliers & Normalized Points Formula**: Komponen `xG Pts` dan `xA Pts` kini dikalibrasi menggunakan **multiplier terbobot per posisi** "
        "(FWD: 1.08x xG / 1.05x xA, MID: 1.04x xG / 1.06x xA, DEF: 0.88x xG / 0.92x xA, GK: 0.00x xG / 0.50x xA) "
        "untuk merefleksikan tren efisiensi konversi historis *'Goal > xG'* dan menggantikan nilai mentah xG/xA dengan kontribusi poin ternormalisasi: "
        "`xG Pts = xG Pred * Poin Gol Posisi * Pos_xG_Mult` dan `xA Pts = xA Pred * 3.0 * Pos_xA_Mult`."
    )

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
            "Peluang CS (%)": st.column_config.ProgressColumn(
                "Peluang CS (%)",
                help="Probabilitas Clean Sheet tim dihitung menggunakan Model Bivariat Dixon-Coles Poisson",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0
            ),
            "xG Mult": st.column_config.NumberColumn(
                "xG Mult",
                format="%.2fx",
                help="Weighted Multiplier Posisi untuk xG berbasis tren historis 'Goal > xG' (FWD: 1.08x, MID: 1.04x, DEF: 0.88x)"
            ),
            "xA Mult": st.column_config.NumberColumn(
                "xA Mult",
                format="%.2fx",
                help="Weighted Multiplier Posisi untuk xA berbasis tren historis 'Assist > xA' (MID: 1.06x, FWD: 1.05x, DEF: 0.92x)"
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
    st.caption("💡 *Option B Formula: xPoin = xMins_Pts + xG_Pts(diff_att * Pos_xG_Mult) + xA_Pts(diff_att * Pos_xA_Mult) + xSaves_Pts + xDC_Pts(diff_def) + xCS_Pts(Dixon-Coles CS%) + xBP.*")

