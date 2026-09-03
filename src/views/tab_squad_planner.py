"""
Tab View: 15-Player Squad Planner, Multi-Option xPoints Comparison & 10-Match FDR Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

SLOT_DEFINITIONS = [
    ("GKP 1", "GK"),
    ("GKP 2", "GK"),
    ("DEF 1", "DEF"),
    ("DEF 2", "DEF"),
    ("DEF 3", "DEF"),
    ("DEF 4", "DEF"),
    ("DEF 5", "DEF"),
    ("MID 1", "MID"),
    ("MID 2", "MID"),
    ("MID 3", "MID"),
    ("MID 4", "MID"),
    ("MID 5", "MID"),
    ("FWD 1", "FWD"),
    ("FWD 2", "FWD"),
    ("FWD 3", "FWD"),
]

def get_default_squad_ids(players_df):
    """Generate a high-quality default 15-player squad within budget and position constraints."""
    squad_ids = {}
    used_ids = set()
    club_counts = {}

    def pick_players(pos, count):
        pos_df = players_df[players_df['Posisi'] == pos].sort_values(
            by=['xPoin', 'Total Poin', 'Avg Mins (L5M)'], ascending=False
        )
        picked = []
        for _, row in pos_df.iterrows():
            pid = int(row['id'])
            club = row['Klub']
            if pid not in used_ids and club_counts.get(club, 0) < 3:
                picked.append(pid)
                used_ids.add(pid)
                club_counts[club] = club_counts.get(club, 0) + 1
                if len(picked) == count:
                    break
        return picked

    gks = pick_players('GK', 2)
    defs = pick_players('DEF', 5)
    mids = pick_players('MID', 5)
    fwds = pick_players('FWD', 3)

    for i, pid in enumerate(gks):
        squad_ids[f"GKP {i+1}"] = pid
    for i, pid in enumerate(defs):
        squad_ids[f"DEF {i+1}"] = pid
    for i, pid in enumerate(mids):
        squad_ids[f"MID {i+1}"] = pid
    for i, pid in enumerate(fwds):
        squad_ids[f"FWD {i+1}"] = pid

    return squad_ids

def render_tab_squad_planner(players_df, fpl_data, fdr_summary, current_gw, df_option_c=None):
    """
    Renders Tab: 15-Player Squad Planner, Multi-Option xPoints Comparison & 10-Match FDR.
    """
    st.subheader("👥 Perencana Skuad 15 Pemain, Komparasi Multi-Option xPoin & FDR 10 Match")
    st.write(
        "Pilih, ganti, dan kelola 15 pemain FPL pilihan Anda (2 Kiper, 5 Bek, 5 Gelandang, 3 Penyerang). "
        "Bandingkan estimasi **xPoin dari seluruh model prediksi** (Default Model, Option B Component Model, dan Option C Current Season Machine Learning Ensemble), "
        "serta analisis tingkat kemudahan jadwal **FDR untuk 10 pertandingan mendatang**."
    )

    if players_df.empty:
        st.warning("Data pemain tidak tersedia.")
        return

    # 1. Initialize or maintain session state for 15 squad slots
    if "my_15_squad_slots" not in st.session_state:
        st.session_state["my_15_squad_slots"] = get_default_squad_ids(players_df)

    squad_slots = st.session_state["my_15_squad_slots"]

    # Quick action buttons row
    action_c1, action_c2, action_c3 = st.columns([2, 2, 4])
    with action_c1:
        if st.button("🔄 Reset ke Skuad Rekomendasi (Top xPoin)", use_container_width=True):
            st.session_state["my_15_squad_slots"] = get_default_squad_ids(players_df)
            st.rerun()
    with action_c2:
        if st.button("💰 Rekomendasi Budget Squad (<£95m)", use_container_width=True):
            # Budget friendly squad (pick high value pts per cost)
            cheap_df = players_df[players_df['Harga (£m)'] <= 8.5].sort_values(
                by=['xPoin per £m', 'xPoin'], ascending=False
            )
            cheap_slots = get_default_squad_ids(cheap_df)
            st.session_state["my_15_squad_slots"] = cheap_slots
            st.rerun()

    # Enrich players_df with Option C predictions if available
    df_merged = players_df.copy()
    if df_option_c is not None and not df_option_c.empty:
        opt_c_cols = ['id', 'xPoin (Option C Ensemble)', 'xPoin (Gradient Boosting)', 'xPoin (Ridge Reg)', 'xPoin (Linear Reg)']
        existing_c = [c for c in opt_c_cols if c in df_option_c.columns]
        if 'id' in existing_c and len(existing_c) > 1:
            df_merged = df_merged.merge(df_option_c[existing_c], on='id', how='left')
    
    # Fill any missing Option C columns safely
    for c_name in ['xPoin (Option C Ensemble)', 'xPoin (Gradient Boosting)', 'xPoin (Ridge Reg)', 'xPoin (Linear Reg)']:
        if c_name not in df_merged.columns:
            df_merged[c_name] = df_merged['xPoin']
        else:
            df_merged[c_name] = df_merged[c_name].fillna(df_merged['xPoin'])

    # Build Current Squad DataFrame
    squad_rows = []
    for slot_name, expected_pos in SLOT_DEFINITIONS:
        pid = squad_slots.get(slot_name)
        p_row = df_merged[df_merged['id'] == pid]
        if not p_row.empty:
            r = p_row.iloc[0].to_dict()
            r['Slot'] = slot_name
            r['Slot_Pos'] = expected_pos
            squad_rows.append(r)
        else:
            # Fallback if player not found
            fallback = df_merged[df_merged['Posisi'] == expected_pos].iloc[0].to_dict()
            fallback['Slot'] = slot_name
            fallback['Slot_Pos'] = expected_pos
            squad_rows.append(fallback)
            st.session_state["my_15_squad_slots"][slot_name] = int(fallback['id'])

    squad_df = pd.DataFrame(squad_rows)

    # Consensus xPoin calculation (average of Default, Option B, and Option C Ensemble)
    squad_df['Konsensus xPoin'] = (
        (squad_df['xPoin'] + squad_df['xPoin (Option B)'] + squad_df['xPoin (Option C Ensemble)']) / 3.0
    ).round(2)

    # 2. SQUAD SUMMARY & VALIDATION BANNER
    total_cost = squad_df['Harga (£m)'].sum()
    budget_rem = 100.0 - total_cost
    tot_xp_default = squad_df['xPoin'].sum()
    tot_xp_opt_b = squad_df['xPoin (Option B)'].sum()
    tot_xp_opt_c = squad_df['xPoin (Option C Ensemble)'].sum()
    tot_xp_consensus = squad_df['Konsensus xPoin'].sum()
    avg_fdr10 = squad_df['FDR10'].mean() if 'FDR10' in squad_df.columns else 3.0

    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        cost_color = "normal" if total_cost <= 100.0 else "inverse"
        st.metric("Total Biaya Skuad", f"£{total_cost:.1f}m", f"Sisa: £{budget_rem:.1f}m", delta_color=cost_color)
    with m2:
        st.metric("Total xPoin (Default)", f"{tot_xp_default:.2f} pts")
    with m3:
        st.metric("Total xPoin (Option B)", f"{tot_xp_opt_b:.2f} pts")
    with m4:
        st.metric("Total xPoin (Option C)", f"{tot_xp_opt_c:.2f} pts")
    with m5:
        st.metric("Konsensus xPoin", f"{tot_xp_consensus:.2f} pts")
    with m6:
        st.metric("Rata-rata FDR10", f"{avg_fdr10:.2f}", help="Rata-rata FDR 10 pertandingan mendatang seluruh 15 pemain")

    # FPL Rules Validation
    club_counts = squad_df['Klub'].value_counts()
    over_limit_clubs = club_counts[club_counts > 3]
    if not over_limit_clubs.empty:
        st.error(f"⚠️ **Peringatan Batas Klub FPL**: Anda memilih lebih dari 3 pemain dari klub: {', '.join([f'{k} ({v})' for k, v in over_limit_clubs.items()])}. Aturan resmi FPL membatasi maksimal 3 pemain per klub.")
    if total_cost > 100.0:
        st.warning(f"⚠️ **Melebihi Anggaran**: Total biaya skuad £{total_cost:.1f}m melebihi pagu standar £100.0m sebesar £{abs(budget_rem):.1f}m.")

    # 3. INTERACTIVE SECTION: MEMILIH & MENGGANTI 15 PEMAIN
    with st.expander("🛠️ **Panel Penggantian Pemain (Ganti Pemain di Setiap Slot)**", expanded=True):
        st.write("Ubah pemain pada salah satu dari 15 slot di bawah. Daftar pilihan otomatis disaring sesuai posisi slot.")
        
        pos_tabs = st.tabs(["🧤 Kiper (2 GKP)", "🛡️ Bek (5 DEF)", "🎯 Gelandang (5 MID)", "⚡ Penyerang (3 FWD)", "🔁 Tukar Cepat (Swap Tool)"])
        
        # Helper to format player selectbox option
        player_dict_by_id = {int(r['id']): r for _, r in df_merged.iterrows()}
        
        def make_player_label(p_row):
            return f"{p_row['Klub']} | {p_row['Nama Pemain']} (£{p_row['Harga (£m)']:.1f}m) - xPoin: {p_row['xPoin']:.2f} | FDR1: {p_row['FDR1']:.1f}"

        # Tab GKP
        with pos_tabs[0]:
            gk_pool = df_merged[df_merged['Posisi'] == 'GK'].sort_values(by=['xPoin', 'Total Poin'], ascending=False)
            gk_options = gk_pool['id'].tolist()
            col_gk1, col_gk2 = st.columns(2)
            
            for idx, col in enumerate([col_gk1, col_gk2]):
                slot_key = f"GKP {idx+1}"
                curr_pid = squad_slots.get(slot_key)
                with col:
                    st.markdown(f"**Slot {slot_key}**")
                    curr_idx = gk_options.index(curr_pid) if curr_pid in gk_options else 0
                    sel_id = st.selectbox(
                        f"Pilih Pemain {slot_key}",
                        options=gk_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=f"sel_slot_{slot_key}"
                    )
                    if sel_id != curr_pid:
                        st.session_state["my_15_squad_slots"][slot_key] = sel_id
                        st.rerun()

        # Tab DEF
        with pos_tabs[1]:
            def_pool = df_merged[df_merged['Posisi'] == 'DEF'].sort_values(by=['xPoin', 'Total Poin'], ascending=False)
            def_options = def_pool['id'].tolist()
            def_cols = st.columns(5)
            
            for idx, col in enumerate(def_cols):
                slot_key = f"DEF {idx+1}"
                curr_pid = squad_slots.get(slot_key)
                with col:
                    st.markdown(f"**Slot {slot_key}**")
                    curr_idx = def_options.index(curr_pid) if curr_pid in def_options else 0
                    sel_id = st.selectbox(
                        f"Pilih {slot_key}",
                        options=def_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=f"sel_slot_{slot_key}"
                    )
                    if sel_id != curr_pid:
                        st.session_state["my_15_squad_slots"][slot_key] = sel_id
                        st.rerun()

        # Tab MID
        with pos_tabs[2]:
            mid_pool = df_merged[df_merged['Posisi'] == 'MID'].sort_values(by=['xPoin', 'Total Poin'], ascending=False)
            mid_options = mid_pool['id'].tolist()
            mid_cols = st.columns(5)
            
            for idx, col in enumerate(mid_cols):
                slot_key = f"MID {idx+1}"
                curr_pid = squad_slots.get(slot_key)
                with col:
                    st.markdown(f"**Slot {slot_key}**")
                    curr_idx = mid_options.index(curr_pid) if curr_pid in mid_options else 0
                    sel_id = st.selectbox(
                        f"Pilih {slot_key}",
                        options=mid_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=f"sel_slot_{slot_key}"
                    )
                    if sel_id != curr_pid:
                        st.session_state["my_15_squad_slots"][slot_key] = sel_id
                        st.rerun()

        # Tab FWD
        with pos_tabs[3]:
            fwd_pool = df_merged[df_merged['Posisi'] == 'FWD'].sort_values(by=['xPoin', 'Total Poin'], ascending=False)
            fwd_options = fwd_pool['id'].tolist()
            fwd_cols = st.columns(3)
            
            for idx, col in enumerate(fwd_cols):
                slot_key = f"FWD {idx+1}"
                curr_pid = squad_slots.get(slot_key)
                with col:
                    st.markdown(f"**Slot {slot_key}**")
                    curr_idx = fwd_options.index(curr_pid) if curr_pid in fwd_options else 0
                    sel_id = st.selectbox(
                        f"Pilih {slot_key}",
                        options=fwd_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=f"sel_slot_{slot_key}"
                    )
                    if sel_id != curr_pid:
                        st.session_state["my_15_squad_slots"][slot_key] = sel_id
                        st.rerun()

        # Tab Quick Swap Tool
        with pos_tabs[4]:
            st.markdown("##### 🔁 Fasilitas Tukar Cepat Satu Pemain")
            sw_c1, sw_c2, sw_c3 = st.columns([3, 4, 2])
            with sw_c1:
                swap_slot_choice = st.selectbox(
                    "Pilih Slot yang Ingin Diganti:",
                    options=[s[0] for s in SLOT_DEFINITIONS],
                    key="quick_swap_slot"
                )
                current_in_slot = squad_df[squad_df['Slot'] == swap_slot_choice].iloc[0]
                target_pos = current_in_slot['Slot_Pos']
                st.caption(f"Pemain saat ini: **{current_in_slot['Nama Pemain']}** ({current_in_slot['Klub']} - £{current_in_slot['Harga (£m)']}m)")
            with sw_c2:
                candidates = df_merged[df_merged['Posisi'] == target_pos].sort_values(by='xPoin', ascending=False)
                cand_options = candidates['id'].tolist()
                replacement_choice = st.selectbox(
                    f"Pilih Pemain Pengganti ({target_pos}):",
                    options=cand_options,
                    format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                    key="quick_swap_replacement"
                )
            with sw_c3:
                st.write("")
                st.write("")
                if st.button("🚀 Konfirmasi Ganti", use_container_width=True):
                    st.session_state["my_15_squad_slots"][swap_slot_choice] = replacement_choice
                    st.success(f"Berhasil mengganti pemain di slot {swap_slot_choice}!")
                    st.rerun()

    st.divider()

    # 4. TABEL MULTI-OPTION XPOIN (REQUIREMENT 2)
    st.markdown("### 📊 Tabel Analisis Prediksi xPoin dari Seluruh Opsi (15 Pemain Pilihan)")
    st.write(
        "Tabel ini membandingkan proyeksi xPoin dari ketiga opsi model: "
        "**Option A (Default Model Klasik)**, **Option B (Component Model FPL)**, dan **Option C (Current Season Machine Learning Ensemble)**."
    )

    # Captain and Vice Captain recommendations
    top_captain = squad_df.sort_values(by='Konsensus xPoin', ascending=False).iloc[0]
    top_vc = squad_df.sort_values(by='Konsensus xPoin', ascending=False).iloc[1]

    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.info(f"👑 **Rekomendasi Kapten (©)**: **{top_captain['Nama Pemain']}** ({top_captain['Klub']}) - Konsensus xPoin: **{top_captain['Konsensus xPoin']:.2f} pts** (Default: {top_captain['xPoin']:.2f} | Opt B: {top_captain['xPoin (Option B)']:.2f} | Opt C: {top_captain['xPoin (Option C Ensemble)']:.2f})")
    with cap_col2:
        st.info(f"🥈 **Rekomendasi Wakil Kapten (Ⓥ)**: **{top_vc['Nama Pemain']}** ({top_vc['Klub']}) - Konsensus xPoin: **{top_vc['Konsensus xPoin']:.2f} pts** (Default: {top_vc['xPoin']:.2f} | Opt B: {top_vc['xPoin (Option B)']:.2f} | Opt C: {top_vc['xPoin (Option C Ensemble)']:.2f})")

    # View options
    sort_squad_by = st.selectbox(
        "Urutkan Tabel Pemain Skuad Berdasarkan:",
        options=[
            "Konsensus xPoin",
            "xPoin (Default Model)",
            "xPoin (Option B)",
            "xPoin (Option C Ensemble)",
            "Harga (£m)",
            "FDR10 (Rata-rata 10 Laga)",
            "Slot Asli Skuad"
        ],
        index=0,
        key="sort_squad_table_sel"
    )

    squad_display_df = squad_df.copy()
    if sort_squad_by == "xPoin (Default Model)":
        squad_display_df = squad_display_df.sort_values(by="xPoin", ascending=False)
    elif sort_squad_by == "xPoin (Option B)":
        squad_display_df = squad_display_df.sort_values(by="xPoin (Option B)", ascending=False)
    elif sort_squad_by == "xPoin (Option C Ensemble)":
        squad_display_df = squad_display_df.sort_values(by="xPoin (Option C Ensemble)", ascending=False)
    elif sort_squad_by == "Konsensus xPoin":
        squad_display_df = squad_display_df.sort_values(by="Konsensus xPoin", ascending=False)
    elif sort_squad_by == "Harga (£m)":
        squad_display_df = squad_display_df.sort_values(by="Harga (£m)", ascending=False)
    elif sort_squad_by == "FDR10 (Rata-rata 10 Laga)":
        squad_display_df = squad_display_df.sort_values(by="FDR10", ascending=True)

    display_cols_xpoin = [
        'Slot', 'Nama Pemain', 'Klub', 'Posisi', 'Harga (£m)',
        'Lawan GW Berikutnya', 'FDR1',
        'xPoin', 'xPoin (Option B)', 'xPoin (Option C Ensemble)',
        'xPoin (Gradient Boosting)', 'xPoin (Ridge Reg)', 'xPoin (Linear Reg)',
        'Konsensus xPoin', 'Peluang Main GW (%)'
    ]

    st.dataframe(
        squad_display_df[display_cols_xpoin],
        use_container_width=True,
        column_config={
            "Slot": st.column_config.TextColumn("Slot"),
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "FDR1": st.column_config.NumberColumn(format="%.1f"),
            "xPoin": st.column_config.NumberColumn("xPoin (Default)", format="%.2f pts", help="Model Klasik ML Multi-Linear Regression"),
            "xPoin (Option B)": st.column_config.NumberColumn("xPoin (Option B)", format="%.2f pts", help="Model Komponen FPL (xMins + xG + xA + xCS + xSaves + xDC + xBP)"),
            "xPoin (Option C Ensemble)": st.column_config.NumberColumn("xPoin (Option C)", format="%.2f pts", help="Ensemble Model Musim Berjalan (Gradient Boosting + Ridge + Linear)"),
            "xPoin (Gradient Boosting)": st.column_config.NumberColumn("Opt C (GradBoost)", format="%.2f pts"),
            "xPoin (Ridge Reg)": st.column_config.NumberColumn("Opt C (Ridge)", format="%.2f pts"),
            "xPoin (Linear Reg)": st.column_config.NumberColumn("Opt C (Linear)", format="%.2f pts"),
            "Konsensus xPoin": st.column_config.NumberColumn("Konsensus xPoin", format="%.2f pts", help="Rata-rata seluruh opsi"),
            "Peluang Main GW (%)": st.column_config.ProgressColumn(
                "Peluang Main (%)",
                min_value=0,
                max_value=100,
                format="%d%%"
            )
        }
    )

    # Option B Breakdown Expander
    with st.expander("🔍 Rincian Poin Komponen Option B untuk 15 Pemain Pilihan", expanded=False):
        st.write("Detail kontribusi komponen poin FPL (xMins, xG Poin, xA Poin, Clean Sheet, Saves, Defensive Contribution, dan Bonus Poin):")
        comp_cols = [
            'Slot', 'Nama Pemain', 'Klub', 'Posisi',
            'xMins Pts', 'xG Pts', 'xA Pts', 'xCS Pts', 'xSaves Pts', 'xDC Pts', 'xBP', 'xPoin (Option B)'
        ]
        st.dataframe(
            squad_display_df[comp_cols],
            use_container_width=True,
            column_config={
                "xMins Pts": st.column_config.NumberColumn(format="%.2f"),
                "xG Pts": st.column_config.NumberColumn(format="%.2f"),
                "xA Pts": st.column_config.NumberColumn(format="%.2f"),
                "xCS Pts": st.column_config.NumberColumn(format="%.2f"),
                "xSaves Pts": st.column_config.NumberColumn(format="%.2f"),
                "xDC Pts": st.column_config.NumberColumn(format="%.2f"),
                "xBP": st.column_config.NumberColumn(format="%.2f"),
                "xPoin (Option B)": st.column_config.NumberColumn(format="%.2f pts")
            }
        )

    st.divider()

    # 5. TABEL FDR UNTUK 10 MATCH MENDATANG (REQUIREMENT 3)
    st.markdown("### 🗓️ Analisis Jadwal & FDR untuk 10 Match Mendatang (15 Pemain Pilihan)")
    st.write(
        "Tabel jadwal rinci lawan dan tingkat kesulitan (FDR) untuk 10 pertandingan mendatang masing-masing pemain. "
        "Membantu Anda merencanakan transfer jangka panjang, pemilihan rotasi bek, dan persiapan double gameweek/blank gameweek."
    )

    # Build 10-match fixture matrix for the selected 15 players
    fdr10_rows = []
    for _, row in squad_df.iterrows():
        t_id = row.get('team')
        f_info = fdr_summary.get(t_id, {})
        up10 = f_info.get('upcoming_10', [])
        
        p_row = {
            'Slot': row['Slot'],
            'Pemain': row['Nama Pemain'],
            'Klub': row['Klub'],
            'Posisi': row['Posisi'],
            'Harga (£m)': row['Harga (£m)'],
            'FDR10 Rata-rata': f_info.get('FDR10', 3.0),
            'FDR3': f_info.get('FDR3', 3.0),
            'FDR5': f_info.get('FDR5', 3.0),
        }
        
        for idx in range(10):
            col_key = f"Match +{idx+1}"
            if idx < len(up10):
                m = up10[idx]
                opp = m.get('opp_name', 'TBD')
                ha = "H" if m.get('is_home') == 1 else "A"
                f_val = m.get('fdr', 3)
                p_row[col_key] = f"{opp} ({ha}) [{f_val}]"
            else:
                p_row[col_key] = "-"
        
        fdr10_rows.append(p_row)

    df_fdr10_squad = pd.DataFrame(fdr10_rows)

    # Sort option for 10-match FDR
    sort_fdr10 = st.radio(
        "Urutkan Tabel Jadwal Berdasarkan:",
        options=["Jadwal 10 Match Paling Menguntungkan (FDR10 Terendah)", "Jadwal 10 Match Paling Berat (FDR10 Tertinggi)", "Urutan Slot Skuad"],
        horizontal=True,
        key="sort_fdr10_radio"
    )

    if sort_fdr10 == "Jadwal 10 Match Paling Menguntungkan (FDR10 Terendah)":
        df_fdr10_squad = df_fdr10_squad.sort_values(by="FDR10 Rata-rata", ascending=True)
    elif sort_fdr10 == "Jadwal 10 Match Paling Berat (FDR10 Tertinggi)":
        df_fdr10_squad = df_fdr10_squad.sort_values(by="FDR10 Rata-rata", ascending=False)

    st.dataframe(
        df_fdr10_squad,
        use_container_width=True,
        column_config={
            "Harga (£m)": st.column_config.NumberColumn(format="£%.1fm"),
            "FDR10 Rata-rata": st.column_config.NumberColumn("FDR10 Avg", format="%.2f", help="Rata-rata FDR 10 pertandingan mendatang"),
            "FDR3": st.column_config.NumberColumn("FDR3 Avg", format="%.2f"),
            "FDR5": st.column_config.NumberColumn("FDR5 Avg", format="%.2f"),
        }
    )

    # Quick Insight Cards for Fixtures
    best_fixtures_p = df_fdr10_squad.sort_values(by="FDR10 Rata-rata", ascending=True).iloc[0]
    worst_fixtures_p = df_fdr10_squad.sort_values(by="FDR10 Rata-rata", ascending=False).iloc[0]

    ins_c1, ins_c2 = st.columns(2)
    with ins_c1:
        st.success(f"🟢 **Jadwal 10 Match Paling Mudah**: **{best_fixtures_p['Pemain']}** ({best_fixtures_p['Klub']}) memiliki rata-rata FDR10 **{best_fixtures_p['FDR10 Rata-rata']:.2f}**.")
    with ins_c2:
        st.warning(f"🔴 **Jadwal 10 Match Paling Menantang**: **{worst_fixtures_p['Pemain']}** ({worst_fixtures_p['Klub']}) menghadapi rata-rata FDR10 **{worst_fixtures_p['FDR10 Rata-rata']:.2f}**.")

    st.caption("💡 *Keterangan Format Jadwal: `Lawan (H/A) [FDR]`. H = Home (Kandang), A = Away (Tandang), [2] = Mudah, [3] = Netral, [4-5] = Sulit.*")
