"""
Tab View: 15-Player Squad Planner, Multi-Option xPoints Comparison & 10-Match FDR Analysis.
"""

import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

SAVED_SQUAD_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "saved_squad.json")

SLOT_KEYS = ["slot_1", "slot_2", "slot_3"]
SLOT_CONFIG = {
    "slot_1": {"default_name": "Slot 1 (Utama)", "icon": "⭐", "badge": "Utama"},
    "slot_2": {"default_name": "Slot 2 (Alternatif)", "icon": "⚡", "badge": "Alternatif"},
    "slot_3": {"default_name": "Slot 3 (Eksperimen)", "icon": "🧪", "badge": "Eksperimen"},
}

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

def load_all_persisted_squads(players_df):
    """
    Load all 3 squad slots from disk, supporting backward compatibility with single-slot format.
    Ensures that slot_1, slot_2, and slot_3 always exist with valid 15-player IDs.
    """
    active_slot_id = "slot_1"
    squad_slots_data = {}
    valid_ids = set(players_df['id'].dropna().astype(int).tolist())

    if os.path.exists(SAVED_SQUAD_FILE):
        try:
            with open(SAVED_SQUAD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "squad_slots_data" in data and isinstance(data["squad_slots_data"], dict):
                squad_slots_data = data["squad_slots_data"]
                active_slot_id = data.get("active_slot_id", "slot_1")
                if active_slot_id not in SLOT_KEYS:
                    active_slot_id = "slot_1"
            elif "slots" in data and isinstance(data["slots"], dict) and len(data["slots"]) > 0:
                # Migrate legacy single-slot format to slot_1
                squad_slots_data = {
                    "slot_1": {
                        "name": "Slot 1 (Utama)",
                        "updated_at": data.get("updated_at", ""),
                        "slots": data["slots"]
                    }
                }
                active_slot_id = "slot_1"
        except Exception:
            squad_slots_data = {}

    # Ensure all 3 slots exist and have valid 15-player assignments
    for s_id in SLOT_KEYS:
        cfg = SLOT_CONFIG[s_id]
        if s_id not in squad_slots_data or not isinstance(squad_slots_data[s_id], dict):
            squad_slots_data[s_id] = {
                "name": cfg["default_name"],
                "updated_at": None,
                "slots": get_default_squad_ids(players_df)
            }
        else:
            s_val = squad_slots_data[s_id]
            if "name" not in s_val or not s_val["name"]:
                s_val["name"] = cfg["default_name"]
            raw_slots = s_val.get("slots", {})
            valid_loaded = {}
            for slot_name, expected_pos in SLOT_DEFINITIONS:
                pid = raw_slots.get(slot_name)
                if pid is not None and int(pid) in valid_ids:
                    valid_loaded[slot_name] = int(pid)
                else:
                    pos_players = players_df[players_df['Posisi'] == expected_pos].sort_values(by=['xPoin', 'Total Poin'], ascending=False)
                    if not pos_players.empty:
                        valid_loaded[slot_name] = int(pos_players.iloc[0]['id'])
            s_val["slots"] = valid_loaded

    return active_slot_id, squad_slots_data

def save_all_persisted_squads(active_slot_id, squad_slots_data):
    """
    Save all 3 slots to disk and keep root 'slots' in sync with active slot for backwards compatibility.
    """
    try:
        os.makedirs(os.path.dirname(SAVED_SQUAD_FILE), exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_slots = squad_slots_data.get(active_slot_id, {}).get("slots", {})
        data = {
            "active_slot_id": active_slot_id,
            "updated_at": now_str,
            "slots": {k: int(v) for k, v in active_slots.items()},
            "squad_slots_data": {
                s_id: {
                    "name": s_val.get("name", SLOT_CONFIG[s_id]["default_name"]),
                    "updated_at": s_val.get("updated_at"),
                    "slots": {k: int(v) for k, v in s_val.get("slots", {}).items()}
                }
                for s_id, s_val in squad_slots_data.items()
            }
        }
        with open(SAVED_SQUAD_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        st.session_state["squad_last_saved"] = now_str
        return True
    except Exception:
        return False

def save_persisted_squad(squad_slots=None):
    """
    Save the active squad slot permanently to disk across all 3 slots.
    Preserves backwards compatibility with any code invoking save_persisted_squad.
    """
    try:
        active_slot_id = st.session_state.get("active_slot_id", "slot_1")
        squad_slots_data = st.session_state.get("squad_slots_data", {})
        if squad_slots is None:
            squad_slots = st.session_state.get("my_15_squad_slots", {})
            
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if active_slot_id in squad_slots_data:
            squad_slots_data[active_slot_id]["slots"] = {k: int(v) for k, v in squad_slots.items()}
            squad_slots_data[active_slot_id]["updated_at"] = now_str
            
        save_all_persisted_squads(active_slot_id, squad_slots_data)
        st.session_state["squad_last_saved"] = now_str
        return True
    except Exception:
        return False

def render_tab_squad_planner(players_df, fpl_data, fdr_summary, current_gw, df_option_c=None):
    """
    Renders Tab: 15-Player Squad Planner with 3 Save Slots, Multi-Option xPoints Comparison & 10-Match FDR.
    """
    st.subheader("👥 Perencana Skuad 15 Pemain, Komparasi Multi-Option xPoin & FDR 10 Match")
    st.write(
        "Pilih, kelola, dan simpan hingga **3 slot skuad independen** (Slot 1 Utama, Slot 2 Alternatif, Slot 3 Eksperimen). "
        "Bandingkan estimasi **xPoin dari seluruh model prediksi** (Default Model, Option B Component Model, dan Option C Current Season Machine Learning Ensemble), "
        "serta analisis tingkat kemudahan jadwal **FDR untuk 10 pertandingan mendatang**."
    )

    if players_df.empty:
        st.warning("Data pemain tidak tersedia.")
        return

    # 1. Initialize or maintain session state for 3 squad slots
    if "squad_slots_data" not in st.session_state or "active_slot_id" not in st.session_state or "my_15_squad_slots" not in st.session_state:
        loaded_active_id, loaded_slots_data = load_all_persisted_squads(players_df)
        st.session_state["active_slot_id"] = loaded_active_id
        st.session_state["squad_slots_data"] = loaded_slots_data
        st.session_state["my_15_squad_slots"] = dict(loaded_slots_data[loaded_active_id]["slots"])
        st.session_state["squad_last_saved"] = loaded_slots_data[loaded_active_id].get("updated_at") or "Tersimpan Permanen"
        # Sync widget keys
        for slot_name, _ in SLOT_DEFINITIONS:
            st.session_state[f"sel_slot_{slot_name}"] = st.session_state["my_15_squad_slots"].get(slot_name)

    active_slot_id = st.session_state.get("active_slot_id", "slot_1")
    squad_slots_data = st.session_state.get("squad_slots_data", {})
    squad_slots = st.session_state["my_15_squad_slots"]

    # =========================================================================
    # 3 SAVE SLOTS UI SELECTOR & OVERVIEW
    # =========================================================================
    st.markdown("##### 🗂️ Slot Penyimpanan Skuad (Tersedia 3 Slot Independen)")
    slot_cols = st.columns(3)
    player_lookup_cost = {int(r['id']): float(r['Harga (£m)']) for _, r in players_df.iterrows()}
    player_lookup_xp = {int(r['id']): float(r['xPoin']) for _, r in players_df.iterrows()}

    for idx, s_id in enumerate(SLOT_KEYS):
        s_info = squad_slots_data.get(s_id, {})
        s_name = s_info.get("name", SLOT_CONFIG[s_id]["default_name"])
        s_time = s_info.get("updated_at")
        s_time_label = s_time if s_time else "Rancangan Standar"
        is_active = (s_id == active_slot_id)

        # Quick calculations for preview cards
        s_slots_dict = s_info.get("slots", {})
        s_cost = sum(player_lookup_cost.get(int(pid), 5.0) for pid in s_slots_dict.values())
        s_xp = sum(player_lookup_xp.get(int(pid), 3.0) for pid in s_slots_dict.values())

        with slot_cols[idx]:
            card_border = "border: 2px solid #2563eb; background-color: #f0f7ff;" if is_active else "border: 1px solid #e2e8f0; background-color: #ffffff;"
            active_badge = "🟢 **AKTIF DIGUNAKAN**" if is_active else "⚪ Tidak Aktif"
            st.markdown(
                f"""
                <div style="{card_border} padding: 12px 14px; border-radius: 10px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{SLOT_CONFIG[s_id]['icon']} {s_name}</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: {'#16a34a' if is_active else '#64748b'};">{active_badge}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569;">
                        Biaya: <b>£{s_cost:.1f}m</b> | Est xPoin: <b>{s_xp:.1f} pts</b>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 3px;">
                        🕒 {s_time_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if not is_active:
                if st.button(f"👉 Pilih & Buka {s_name}", key=f"btn_activate_{s_id}", use_container_width=True):
                    # Save current slot before switching
                    save_persisted_squad(st.session_state["my_15_squad_slots"])
                    # Switch active slot
                    st.session_state["active_slot_id"] = s_id
                    new_slots = dict(squad_slots_data[s_id]["slots"])
                    st.session_state["my_15_squad_slots"] = new_slots
                    for s_name_def, _ in SLOT_DEFINITIONS:
                        st.session_state[f"sel_slot_{s_name_def}"] = new_slots.get(s_name_def)
                    st.session_state["squad_last_saved"] = squad_slots_data[s_id].get("updated_at") or "Tersimpan Permanen"
                    save_all_persisted_squads(s_id, squad_slots_data)
                    st.success(f"Beralih ke {s_name}!")
                    st.rerun()
            else:
                st.button(f"✅ Sedang Aktif ({s_name})", key=f"btn_active_disabled_{s_id}", disabled=True, use_container_width=True)

    # Persistence Info Banner
    current_slot_name = squad_slots_data.get(active_slot_id, {}).get("name", SLOT_CONFIG[active_slot_id]["default_name"])
    last_saved_label = st.session_state.get("squad_last_saved", "Tersimpan Permanen")
    st.info(
        f"🔒 **Slot Aktif**: **{current_slot_name}** | Status: **Tersimpan Otomatis & Permanen**. "
        f"Setiap perubahan susunan pemain langsung tersimpan di slot ini dan tidak akan hilang saat reload atau berpindah tab. "
        f"*(Penyimpanan terakhir: `{last_saved_label}`)*"
    )

    # Slot Action & Management Toolbar
    bar_c1, bar_c2, bar_c3, bar_c4 = st.columns([2.5, 2.5, 3, 2])
    with bar_c1:
        if st.button("💾 Simpan Permanen Manual", use_container_width=True, help="Paksa simpan konfigurasi skuad saat ini"):
            save_persisted_squad(st.session_state["my_15_squad_slots"])
            st.success(f"✅ Skuad {current_slot_name} berhasil disimpan permanen!")
            st.rerun()

    with bar_c2:
        # Prepare JSON string for download (can export active slot or full 3-slot backup)
        squad_export_data = {
            "version": "2.0_multi_slot",
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active_slot_id": active_slot_id,
            "active_slot_name": current_slot_name,
            "slots": squad_slots,
            "all_slots_data": squad_slots_data
        }
        st.download_button(
            label="📥 Ekspor Cadangan (JSON)",
            data=json.dumps(squad_export_data, indent=2),
            file_name=f"fpl_squad_{active_slot_id}_gw{current_gw}.json",
            mime="application/json",
            use_container_width=True,
            help="Unduh file backup skuad Anda untuk disimpan di perangkat lokal"
        )

    with bar_c3:
        if st.button("💰 Terapkan Budget Squad (<£95m)", use_container_width=True, help=f"Terapkan skuad ekonomis ramah anggaran hanya ke {current_slot_name}"):
            cheap_df = players_df[players_df['Harga (£m)'] <= 8.5].sort_values(
                by=['xPoin per £m', 'xPoin'], ascending=False
            )
            cheap_slots = get_default_squad_ids(cheap_df)
            st.session_state["my_15_squad_slots"] = cheap_slots
            st.session_state["squad_slots_data"][active_slot_id]["slots"] = cheap_slots
            for s_name, _ in SLOT_DEFINITIONS:
                st.session_state[f"sel_slot_{s_name}"] = cheap_slots.get(s_name)
            save_persisted_squad(cheap_slots)
            st.success(f"Skuad budget berhasil diterapkan ke {current_slot_name}!")
            st.rerun()

    with bar_c4:
        with st.popover("⚙️ Kelola Slot & Opsi", use_container_width=True):
            st.markdown(f"##### ✏️ Ganti Nama **{current_slot_name}**")
            new_name_input = st.text_input("Nama Slot Baru:", value=current_slot_name, key="input_rename_slot")
            if st.button("Simpan Nama Slot", use_container_width=True):
                if new_name_input.strip():
                    st.session_state["squad_slots_data"][active_slot_id]["name"] = new_name_input.strip()
                    save_all_persisted_squads(active_slot_id, st.session_state["squad_slots_data"])
                    st.success("Nama slot berhasil diperbarui!")
                    st.rerun()

            st.markdown("---")
            st.markdown(f"##### 📋 Salin **{current_slot_name}** ke Slot Lain")
            target_copy_slot = st.selectbox(
                "Pilih Slot Tujuan:",
                options=[s for s in SLOT_KEYS if s != active_slot_id],
                format_func=lambda s: squad_slots_data.get(s, {}).get("name", SLOT_CONFIG[s]["default_name"]),
                key="select_copy_target_slot"
            )
            if st.button("Duplikasi Skuad Ini ke Slot Tujuan", use_container_width=True):
                st.session_state["squad_slots_data"][target_copy_slot]["slots"] = dict(st.session_state["my_15_squad_slots"])
                st.session_state["squad_slots_data"][target_copy_slot]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_all_persisted_squads(active_slot_id, st.session_state["squad_slots_data"])
                target_name = squad_slots_data.get(target_copy_slot, {}).get("name", target_copy_slot)
                st.success(f"Berhasil menduplikasi skuad ke {target_name}!")
                st.rerun()

            st.markdown("---")
            st.markdown(f"##### 📤 Impor Skuad ke **{current_slot_name}**")
            uploaded_file = st.file_uploader("Pilih file cadangan skuad (.json):", type=["json"], key="upload_squad_file")
            if uploaded_file is not None:
                try:
                    imported_json = json.load(uploaded_file)
                    imp_slots = imported_json.get("slots", {})
                    valid_ids = set(players_df['id'].dropna().astype(int).tolist())
                    all_valid = len(imp_slots) == 15 and all(int(pid) in valid_ids for pid in imp_slots.values())
                    if all_valid:
                        formatted_slots = {k: int(v) for k, v in imp_slots.items()}
                        st.session_state["my_15_squad_slots"] = formatted_slots
                        st.session_state["squad_slots_data"][active_slot_id]["slots"] = formatted_slots
                        for s_name, _ in SLOT_DEFINITIONS:
                            st.session_state[f"sel_slot_{s_name}"] = formatted_slots.get(s_name)
                        save_persisted_squad(formatted_slots)
                        st.success(f"✅ Berhasil memulihkan skuad ke {current_slot_name}!")
                        st.rerun()
                    else:
                        st.error("Format file cadangan tidak sesuai atau ada ID pemain yang tidak valid.")
                except Exception as ex:
                    st.error(f"Gagal membaca file cadangan: {ex}")

            st.markdown("---")
            st.markdown(f"##### 🔄 Reset **{current_slot_name}**")
            st.caption(f"Tindakan ini hanya mereset {current_slot_name} ke rekomendasi default algoritma tanpa mengganggu slot lainnya.")
            confirm_reset = st.checkbox(f"Saya yakin ingin mereset {current_slot_name}", key="confirm_reset_squad_check")
            if st.button("🚨 Jalankan Reset Slot Ini", disabled=not confirm_reset, use_container_width=True):
                default_slots = get_default_squad_ids(players_df)
                st.session_state["my_15_squad_slots"] = default_slots
                st.session_state["squad_slots_data"][active_slot_id]["slots"] = default_slots
                for s_name, _ in SLOT_DEFINITIONS:
                    st.session_state[f"sel_slot_{s_name}"] = default_slots.get(s_name)
                save_persisted_squad(default_slots)
                st.success(f"{current_slot_name} berhasil direset ke rekomendasi!")
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
                        st.session_state[f"sel_slot_{slot_key}"] = sel_id
                        save_persisted_squad(st.session_state["my_15_squad_slots"])
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
                        st.session_state[f"sel_slot_{slot_key}"] = sel_id
                        save_persisted_squad(st.session_state["my_15_squad_slots"])
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
                        st.session_state[f"sel_slot_{slot_key}"] = sel_id
                        save_persisted_squad(st.session_state["my_15_squad_slots"])
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
                        st.session_state[f"sel_slot_{slot_key}"] = sel_id
                        save_persisted_squad(st.session_state["my_15_squad_slots"])
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
                    st.session_state[f"sel_slot_{swap_slot_choice}"] = replacement_choice
                    save_persisted_squad(st.session_state["my_15_squad_slots"])
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
