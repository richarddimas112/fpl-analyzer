"""
Tab View: 15-Player Squad Planner, Multi-Option xPoints Comparison & 10-Match FDR Analysis.
"""

import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from src.auth import login_user, register_user, reset_password, logout_user
from src.firebase_client import get_firestore_url
from src.fpl_sync import (
    parse_fpl_team_id,
    fetch_fpl_manager_profile,
    fetch_fpl_realtime_squad,
    map_fpl_picks_to_squad_slots,
    calculate_squad_transfer_delta,
    parse_and_map_squad_from_text
)
import requests

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



def fetch_squad_from_firestore(uid, token):
    url, _ = get_firestore_url("squads", uid)
    if not url: return None
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            doc = r.json()
            if "fields" in doc and "squad_json" in doc["fields"]:
                squad_json_str = doc["fields"]["squad_json"]["stringValue"]
                return json.loads(squad_json_str)
    except Exception:
        pass
    return None

def save_squad_to_firestore(uid, token, active_slot_id, squad_slots_data, fpl_profile=None):
    url, _ = get_firestore_url("squads", uid)
    if not url: return False
    headers = {"Authorization": f"Bearer {token}"}
    payload_data = {
        "active_slot_id": active_slot_id,
        "squad_slots_data": squad_slots_data
    }
    if fpl_profile:
        payload_data["fpl_profile"] = fpl_profile
    payload = {
        "fields": {
            "uid": {"stringValue": uid},
            "squad_json": {"stringValue": json.dumps(payload_data)},
            "updated_at": {"stringValue": datetime.utcnow().isoformat() + "Z"}
        }
    }
    try:
        r = requests.patch(url, json=payload, headers=headers, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def load_all_persisted_squads(players_df):
    active_slot_id = "slot_1"
    squad_slots_data = {}
    fpl_profile = None
    valid_ids = set(players_df['id'].dropna().astype(int).tolist())

    if "user_token" in st.session_state and "user_uid" in st.session_state:
        cloud_data = fetch_squad_from_firestore(st.session_state["user_uid"], st.session_state["user_token"])
        if cloud_data:
            squad_slots_data = cloud_data.get("squad_slots_data", {})
            active_slot_id = cloud_data.get("active_slot_id", "slot_1")
            fpl_profile = cloud_data.get("fpl_profile")

    if not squad_slots_data and os.path.exists(SAVED_SQUAD_FILE):
        try:
            with open(SAVED_SQUAD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "fpl_profile" in data:
                fpl_profile = data.get("fpl_profile")
            if "squad_slots_data" in data and isinstance(data["squad_slots_data"], dict):
                squad_slots_data = data["squad_slots_data"]
                active_slot_id = data.get("active_slot_id", "slot_1")
                if active_slot_id not in SLOT_KEYS:
                    active_slot_id = "slot_1"
            elif "slots" in data and isinstance(data["slots"], dict) and len(data["slots"]) > 0:
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

    return active_slot_id, squad_slots_data, fpl_profile

def save_all_persisted_squads(active_slot_id, squad_slots_data, fpl_profile=None):
    if fpl_profile is None:
        fpl_profile = st.session_state.get("fpl_profile")

    if "user_token" in st.session_state and "user_uid" in st.session_state:
        save_squad_to_firestore(st.session_state["user_uid"], st.session_state["user_token"], active_slot_id, squad_slots_data, fpl_profile=fpl_profile)
        
    try:
        os.makedirs(os.path.dirname(SAVED_SQUAD_FILE), exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_slots = squad_slots_data.get(active_slot_id, {}).get("slots", {})
        data = {
            "active_slot_id": active_slot_id,
            "updated_at": now_str,
            "fpl_profile": fpl_profile,
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
            
        save_all_persisted_squads(active_slot_id, squad_slots_data, fpl_profile=st.session_state.get("fpl_profile"))
        st.session_state["squad_last_saved"] = now_str
        return True
    except Exception:
        return False

def render_tab_squad_planner(players_df, fpl_data, fdr_summary, current_gw, df_option_c=None):
    """
    Renders Tab: 15-Player Squad Planner with 3 Save Slots, Multi-Option xPoints Comparison & 10-Match FDR.
    """
    st.subheader("👥 Perencana Skuad 15 Pemain, Komparasi Multi-Option xPoin & FDR 10 Match")
    
    # ---- PROTECTED ROUTE / AUTHENTICATION GATE ----
    if "user_token" not in st.session_state:
        st.warning("🔒 Akses Ditolak: Anda harus login untuk menggunakan Squad Planner.")
        st.info("Fitur Squad Planner memanfaatkan Cloud Firestore untuk menyimpan 3 slot formasi independen Anda secara permanen. Silakan Login atau Daftar untuk melanjutkan.")
        
        st.markdown("### 🔑 Akses Akun Cloud Planner")
        col1, col2 = st.columns([1, 1])
        with col1:
            auth_mode = st.radio("Pilih Aksi:", ["Login", "Register", "Lupa Password"], horizontal=True, key="squad_planner_auth_mode")
            
            if auth_mode == "Login":
                auth_email = st.text_input("Email", key="squad_planner_auth_email")
                auth_pass = st.text_input("Password", type="password", key="squad_planner_auth_pass")
                if st.button("Masuk", key="squad_planner_btn_login", type="primary", use_container_width=True):
                    with st.spinner("Autentikasi..."):
                        success, msg = login_user(auth_email, auth_pass)
                        if success:
                            st.success("Login berhasil!")
                            st.rerun()
                        else:
                            st.error(msg)
                
                with st.expander("❓ Lupa Password?"):
                    st.caption("Jika Anda lupa password, pilih opsi radio **Lupa Password** di atas untuk mengirim link pemulihan ke email Anda.")
                                
            elif auth_mode == "Register":
                auth_email = st.text_input("Email", key="squad_planner_auth_email")
                auth_pass = st.text_input("Password (minimal 6 karakter)", type="password", key="squad_planner_auth_pass")
                if st.button("Daftar Akun Baru", key="squad_planner_btn_register", type="primary", use_container_width=True):
                    with st.spinner("Mendaftarkan akun..."):
                        success, msg = register_user(auth_email, auth_pass)
                        if success:
                            st.success("Registrasi berhasil! Anda telah otomatis masuk.")
                            st.rerun()
                        else:
                            st.error(msg)
                            
            else:  # Lupa Password
                st.markdown("#### 🔄 Pemulihan Kata Sandi (Lupa Password)")
                st.write("Masukkan email yang terdaftar pada akun Anda. Sistem Firebase akan mengirimkan tautan pemulihan untuk mengatur ulang password baru.")
                reset_email = st.text_input("Email Akun Anda", key="squad_planner_reset_email")
                if st.button("Kirim Link Reset Password", key="squad_planner_btn_reset", type="primary", use_container_width=True):
                    with st.spinner("Mengirim tautan reset..."):
                        success, msg = reset_password(reset_email)
                        if success:
                            st.success(msg)
                            st.info("💡 **Langkah berikutnya:** Buka email Anda, klik tautan dari Firebase, buat password baru, lalu kembali ke tab **Login** untuk masuk.")
                        else:
                            st.error(msg)
        return  # End execution here, protecting the route.
    # ---- END PROTECTED ROUTE ----

    # User is logged in, show user info and options
    col_auth1, col_auth2 = st.columns([3, 1])
    with col_auth1:
        st.success(f"Masuk sebagai: **{st.session_state.get('user_email')}**")
    with col_auth2:
        if st.button("Logout", key="squad_planner_btn_logout", use_container_width=True):
            logout_user()
            st.rerun()
            
    with st.expander("⚙️ Keamanan & Ganti Password Akun"):
        st.write(f"Email akun Anda: **{st.session_state.get('user_email')}**")
        st.caption("Ingin mengubah atau memperbarui password akun Anda? Klik tombol di bawah untuk menerima link ubah password di inbox email Anda.")
        if st.button("Kirim Link Ubah Password ke Email Saya", key="btn_auth_change_pw"):
            with st.spinner("Mengirim tautan ubah password..."):
                success, msg = reset_password(st.session_state.get("user_email"))
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            
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
        loaded_active_id, loaded_slots_data, loaded_fpl_profile = load_all_persisted_squads(players_df)
        st.session_state["active_slot_id"] = loaded_active_id
        st.session_state["squad_slots_data"] = loaded_slots_data
        if loaded_fpl_profile:
            st.session_state["fpl_profile"] = loaded_fpl_profile
        st.session_state["my_15_squad_slots"] = dict(loaded_slots_data[loaded_active_id]["slots"])
        st.session_state["squad_last_saved"] = loaded_slots_data[loaded_active_id].get("updated_at") or "Tersimpan Permanen"
        st.session_state["squad_revision"] = 0

    if "squad_revision" not in st.session_state:
        st.session_state["squad_revision"] = 0

    active_slot_id = st.session_state.get("active_slot_id", "slot_1")
    squad_slots_data = st.session_state.get("squad_slots_data", {})
    squad_slots = st.session_state["my_15_squad_slots"]
    squad_revision = st.session_state.get("squad_revision", 0)

    # =========================================================================
    # FPL REALTIME SQUAD SYNCHRONIZATION & MANAGER PROFILE
    # =========================================================================
    fpl_profile = st.session_state.get("fpl_profile")
    show_fpl_edit = st.session_state.get("show_fpl_edit_form", False)
    show_quick_text_sync = st.session_state.get("show_quick_text_sync", False)
    show_cookie_auth_sync = st.session_state.get("show_cookie_auth_sync", False)

    with st.container():
        if fpl_profile and not show_fpl_edit:
            # Connected Status Card
            fpl_team_name = fpl_profile.get("team_name", "Tim FPL")
            fpl_mgr_name = fpl_profile.get("manager_name", "Manajer FPL")
            fpl_team_id = fpl_profile.get("team_id", "-")
            fpl_overall_pts = fpl_profile.get("overall_points", 0)
            fpl_overall_rank = fpl_profile.get("overall_rank", 0)
            fpl_event = fpl_profile.get("current_event", current_gw)
            is_live_mt = fpl_profile.get("is_live_my_team", False)
            sync_source = fpl_profile.get("sync_source", "api_public")
            
            status_badge_text = "🟢 Terhubung Realtime Akun FPL (Live Pre-Deadline)" if is_live_mt else ("📋 Terhubung via Impor Teks Realtime" if sync_source == "text_import" else "📌 Skuad Terkunci di Deadline GW " + str(fpl_event))
            
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #064e3b 0%, #0f766e 100%); color: white; padding: 18px 22px; border-radius: 12px; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="background: rgba(255, 255, 255, 0.2); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                                {status_badge_text}
                            </span>
                            <h3 style="margin: 6px 0 2px 0; font-size: 1.35rem; font-weight: 700; color: #ffffff;">{fpl_team_name}</h3>
                            <p style="margin: 0; font-size: 0.88rem; color: #ccfbf1;">
                                👤 Manajer: <b>{fpl_mgr_name}</b> | 🆔 ID Tim: <b>#{fpl_team_id}</b>
                            </p>
                        </div>
                        <div style="display: flex; gap: 14px; text-align: center; flex-wrap: wrap;">
                            <div style="background: rgba(0, 0, 0, 0.25); padding: 8px 14px; border-radius: 8px;">
                                <div style="font-size: 0.75rem; color: #99f6e4;">Total Poin FPL</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{fpl_overall_pts:,}</div>
                            </div>
                            <div style="background: rgba(0, 0, 0, 0.25); padding: 8px 14px; border-radius: 8px;">
                                <div style="font-size: 0.75rem; color: #99f6e4;">Peringkat Global</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">#{fpl_overall_rank:,}</div>
                            </div>
                            <div style="background: rgba(0, 0, 0, 0.25); padding: 8px 14px; border-radius: 8px;">
                                <div style="font-size: 0.75rem; color: #99f6e4;">Gameweek</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">GW {fpl_event}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Informative notification regarding FPL privacy policy for pre-deadline transfers
            if not is_live_mt and sync_source != "text_import":
                with st.expander("ℹ️ **Kenapa pemain hasil transfer baru belum muncul otomatis via ID publik FPL?**", expanded=False):
                    st.markdown(
                        f"""
                        **Aturan Kerahasiaan Transfer FPL Resmi**:
                        - Premier League **secara resmi merahasiakan** setiap transfer yang Anda lakukan untuk Gameweek mendatang (GW {fpl_event + 1}) dari API publik hingga batas deadline tiba. Hal ini bertujuan agar lawan di mini-league tidak bisa mengintip strategi transfer Anda sebelum pertandingan.
                        - Endpoint publik `picks` hanya mengembalikan susunan pemain yang telah **terkunci pada deadline GW {fpl_event}**.
                        - **Solusi untuk memperbarui skuad realtime Anda saat ini**:
                          1. **📋 Tempel Teks Skuad (Paling Cepat & Mudah)**: Cukup salin 15 nama pemain dari aplikasi/web FPL dan tempel di bawah.
                          2. **🔐 Gunakan Cookie FPL (`pl_profile`)**: Mengakses data privat `/api/my-team/` langsung dari akun resmi Anda.
                          3. **⚡ Salin ke Slot 2**: Salin susunan GW {fpl_event} ke Slot 2 dan ganti 1–2 pemain yang baru saja ditransfer.
                        """
                    )

            # Quick Action Controls
            sync_cols = st.columns([2.8, 3.2, 2.5, 2.5])
            with sync_cols[0]:
                if st.button("📋 Perbarui via Tempel Teks", key="btn_open_quick_text", use_container_width=True, help="Tempel 15 nama pemain terkini Anda dari FPL"):
                    st.session_state["show_quick_text_sync"] = not show_quick_text_sync
                    st.rerun()

            with sync_cols[1]:
                if st.button("⚡ Salin ke Slot 2 (Perencana Transfer)", key="btn_copy_to_slot_2", type="primary", use_container_width=True, help="Salin susunan pemain realtime Slot 1 ke Slot 2 untuk merencanakan transfer dan rotasi pemain"):
                    slot1_curr = dict(squad_slots_data["slot_1"]["slots"])
                    squad_slots_data["slot_2"]["slots"] = slot1_curr
                    squad_slots_data["slot_2"]["name"] = f"Slot 2 (Rencana Transfer {fpl_team_name})".strip()
                    squad_slots_data["slot_2"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["active_slot_id"] = "slot_2"
                    st.session_state["my_15_squad_slots"] = slot1_curr
                    st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
                    save_all_persisted_squads("slot_2", squad_slots_data, fpl_profile=fpl_profile)
                    st.success(f"✅ Skuad realtime disalin ke Slot 2! Anda sekarang dapat mengganti pemain di Slot 2.")
                    st.rerun()

            with sync_cols[2]:
                if st.button("🔄 Tarik Ulang GW Resmi", key="btn_resync_fpl", use_container_width=True, help="Tarik ulang susunan pemain resmi yang terkunci dari server FPL ke Slot 1"):
                    with st.spinner("Menghubungi server resmi FPL..."):
                        squad_res, err = fetch_fpl_realtime_squad(fpl_profile["team_id"], current_gw)
                        if err:
                            st.error(err)
                        else:
                            slots_dict, meta, map_err = map_fpl_picks_to_squad_slots(squad_res["picks"], players_df)
                            if map_err:
                                st.error(map_err)
                            else:
                                squad_slots_data["slot_1"]["slots"] = slots_dict
                                squad_slots_data["slot_1"]["name"] = f"Slot 1 (Utama - {fpl_team_name})"
                                squad_slots_data["slot_1"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                fpl_profile["current_event"] = squad_res["resolved_gw"]
                                fpl_profile["captain_id"] = meta.get("captain_id")
                                fpl_profile["vice_captain_id"] = meta.get("vice_captain_id")
                                fpl_profile["is_live_my_team"] = False
                                fpl_profile["sync_source"] = "api_public"
                                st.session_state["fpl_profile"] = fpl_profile
                                if active_slot_id == "slot_1":
                                    st.session_state["my_15_squad_slots"] = dict(slots_dict)
                                save_all_persisted_squads(active_slot_id, squad_slots_data, fpl_profile=fpl_profile)
                                st.success(f"✅ Skuad resmi '{fpl_team_name}' berhasil diperbarui di Slot 1!")
                                st.rerun()

            with sync_cols[3]:
                if st.button("⚙️ Ganti Akun / Cookie", key="btn_edit_fpl_creds", use_container_width=True):
                    st.session_state["show_fpl_edit_form"] = True
                    st.rerun()

            # Inline Quick Text Matcher Modal / Container
            if show_quick_text_sync:
                with st.container():
                    st.markdown(
                        """
                        <div style="background-color: #f0fdf4; border: 1.5px solid #86efac; border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
                            <h5 style="margin: 0 0 6px 0; color: #166534;">📋 Impor Cepat via Tempel Teks Skuad Realtime</h5>
                            <p style="margin: 0; font-size: 0.86rem; color: #14532d;">
                                Salin 15 nama pemain dari aplikasi atau website FPL Anda, lalu tempelkan di kotak bawah. 
                                Sistem pintar akan otomatis mendeteksi nama pemain, membagi posisi (2 GKP, 5 DEF, 5 MID, 3 FWD), dan langsung memperbarui Slot 1.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    txt_input_val = st.text_area(
                        "Daftar 15 Pemain Terkini Anda:",
                        value="",
                        height=120,
                        placeholder="Contoh:\nTrafford, Dubravka\nTarkowski, White, De Cuyper, Calafiori, N.Williams\nSaka, B.Fernandes, Szoboszlai, Gomez, Groß\nIsak, Thiago, João Pedro",
                        help="Dapat berupa nama dipisahkan koma, baris baru, atau teks hasil copy-paste dari tampilan skuad FPL."
                    )
                    txt_btn_c1, txt_btn_c2 = st.columns([3, 2])
                    with txt_btn_c1:
                        if st.button("⚡ Pasang Skuad Realtime ke Slot 1", key="btn_submit_quick_text", type="primary", use_container_width=True):
                            parsed_slots, p_meta, p_err = parse_and_map_squad_from_text(txt_input_val, players_df)
                            if p_err:
                                st.error(f"❌ {p_err}")
                            else:
                                squad_slots_data["slot_1"]["slots"] = parsed_slots
                                squad_slots_data["slot_1"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                fpl_profile["is_live_my_team"] = True
                                fpl_profile["sync_source"] = "text_import"
                                st.session_state["fpl_profile"] = fpl_profile
                                st.session_state["show_quick_text_sync"] = False
                                if active_slot_id == "slot_1":
                                    st.session_state["my_15_squad_slots"] = dict(parsed_slots)
                                st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
                                save_all_persisted_squads(active_slot_id, squad_slots_data, fpl_profile=fpl_profile)
                                st.success(f"🎉 Berhasil memetakan {p_meta.get('matched_count', 15)} pemain ke Slot 1!")
                                st.rerun()
                    with txt_btn_c2:
                        if st.button("Tutup Panel Teks", key="btn_close_quick_text", use_container_width=True):
                            st.session_state["show_quick_text_sync"] = False
                            st.rerun()

        else:
            # Connection Form Card with 2 Intuitive Tabs
            st.markdown(
                """
                <div style="background: #f8fafc; border: 2px dashed #94a3b8; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 6px 0; color: #1e293b; font-weight: 700;">
                        🔗 Hubungkan Skuad Realtime FPL Resmi (Live Manager Sync)
                    </h4>
                    <p style="margin: 0 0 4px 0; font-size: 0.9rem; color: #475569;">
                        Tarik susunan 15 pemain tim FPL Anda secara otomatis ke <b>Slot 1</b>. Anda dapat mengetahui proyeksi <b>xPoin skuad resmi Anda</b>, 
                        lalu menyalinnya ke <b>Slot 2</b> untuk merencanakan transfer dan rotasi pemain.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            sync_method_tabs = st.tabs([
                "🌐 Tarik Otomatis via FPL ID / URL",
                "📋 Impor Cepat via Tempel Teks (Instan & Tanpa Login)"
            ])

            # TAB 1: SYNC VIA TEAM ID & OPTIONAL COOKIE
            with sync_method_tabs[0]:
                inp_c1, inp_c2, inp_c3 = st.columns([4, 3, 3])
                with inp_c1:
                    fpl_id_val = st.text_input(
                        "FPL Team ID atau URL Profil Tim:",
                        value=str(fpl_profile.get("team_id", "")) if (fpl_profile and show_fpl_edit) else "",
                        placeholder="Contoh: 2921195 atau https://fantasy.premierleague.com/entry/2921195/event/4",
                        help="Masukkan angka FPL Team ID Anda atau paste URL laman tim FPL Anda."
                    )
                with inp_c2:
                    fpl_mgr_val = st.text_input(
                        "Nama Manajer (Opsional):",
                        value=fpl_profile.get("manager_name", "") if (fpl_profile and show_fpl_edit) else "",
                        placeholder="Otomatis dari API jika kosong",
                        help="Nama manajer Anda di FPL. Jika dikosongkan, nama resmi dari profil FPL akan digunakan."
                    )
                with inp_c3:
                    fpl_team_val = st.text_input(
                        "Nama Tim FPL (Opsional):",
                        value=fpl_profile.get("team_name", "") if (fpl_profile and show_fpl_edit) else "",
                        placeholder="Otomatis dari API jika kosong",
                        help="Nama tim Anda di FPL. Jika dikosongkan, nama resmi dari profil FPL akan digunakan."
                    )

                # Cookie support for pre-deadline live squad
                fpl_cookie_val = st.text_input(
                    "🔑 Cookie Sesi FPL pl_profile (Opsional - Untuk Menarik Transfer Pra-Deadline GW):",
                    value="",
                    placeholder="Contoh: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... atau nilai cookie pl_profile Anda",
                    type="password",
                    help="FPL merahasiakan transfer yang dibuat sebelum deadline dari publik. Jika Anda memasukkan cookie sesi pl_profile Anda, aplikasi dapat menarik transfer GW5 langsung dari endpoint privat /api/my-team/."
                )

                with st.expander("❓ Cara Cepat Mengetahui FPL Team ID & Mengapa Transfer Belum Muncul di API Publik"):
                    st.markdown(
                        """
                        **1. Mengetahui FPL Team ID**:
                        - Buka situs resmi [fantasy.premierleague.com](https://fantasy.premierleague.com) dan login ke akun Anda.
                        - Masuk ke menu **Points** atau **Pick Team**.
                        - Perhatikan URL pada browser Anda: `https://fantasy.premierleague.com/entry/XXXXXXX/event/...`
                        - Angka setelah `/entry/` (misalnya **`2921195`**) adalah **FPL Team ID** Anda!
                        
                        **2. Mengapa Hasil Tarikan Tanpa Cookie Menampilkan GW 4?**:
                        - Di FPL resmi, **Premier League sengaja merahasiakan seluruh transfer sebelum deadline** dari publik agar lawan di liga Anda tidak bisa melihat transfer rahasia Anda.
                        - Endpoint publik hanya menampilkan susunan resmi yang terkunci pada deadline terakhir (GW 4).
                        - **Jika Anda tidak ingin repot mencari cookie**, gunakan tab **'📋 Impor Cepat via Tempel Teks'** di samping untuk memasukkan 15 nama pemain saat ini secara instan!
                        """
                    )

                btn_fpl_cols = st.columns([3, 2])
                with btn_fpl_cols[0]:
                    if st.button("🚀 Tarik Skuad FPL ke Slot 1", key="btn_do_fpl_sync", type="primary", use_container_width=True):
                        parsed_id = parse_fpl_team_id(fpl_id_val)
                        if not parsed_id:
                            st.error("❌ FPL Team ID tidak valid. Mohon masukkan angka Team ID atau paste URL profil tim FPL yang benar.")
                        else:
                            with st.spinner(f"Menghubungi server resmi FPL untuk Tim #{parsed_id}..."):
                                squad_res, err = fetch_fpl_realtime_squad(parsed_id, current_gw, cookie_str=fpl_cookie_val)
                                if err:
                                    st.error(f"❌ {err}")
                                else:
                                    profile = squad_res["profile"]
                                    team_name = fpl_team_val.strip() if fpl_team_val.strip() else profile["team_name"]
                                    mgr_name = fpl_mgr_val.strip() if fpl_mgr_val.strip() else profile["manager_name"]
                                    profile["team_name"] = team_name
                                    profile["manager_name"] = mgr_name
                                    profile["current_event"] = squad_res["resolved_gw"]
                                    profile["is_live_my_team"] = squad_res.get("is_live_my_team", False)
                                    profile["sync_source"] = "api_authenticated" if squad_res.get("is_live_my_team") else "api_public"

                                    slots_dict, meta, map_err = map_fpl_picks_to_squad_slots(squad_res["picks"], players_df)
                                    if map_err:
                                        st.error(f"❌ {map_err}")
                                    else:
                                        profile["captain_id"] = meta.get("captain_id")
                                        profile["vice_captain_id"] = meta.get("vice_captain_id")
                                        st.session_state["fpl_profile"] = profile
                                        st.session_state["show_fpl_edit_form"] = False

                                        # Update slot 1
                                        squad_slots_data["slot_1"]["slots"] = slots_dict
                                        squad_slots_data["slot_1"]["name"] = f"Slot 1 (Utama - {team_name})"
                                        squad_slots_data["slot_1"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                        # Set active to slot 1
                                        st.session_state["active_slot_id"] = "slot_1"
                                        st.session_state["my_15_squad_slots"] = dict(slots_dict)
                                        st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1

                                        save_all_persisted_squads("slot_1", squad_slots_data, fpl_profile=profile)
                                        success_msg = f"🎉 Berhasil terhubung ke tim '{team_name}' ({mgr_name})!"
                                        if profile["is_live_my_team"]:
                                            success_msg += " (Data realtime akun FPL termasuk transfer pra-deadline telah ditarik ke Slot 1)."
                                        else:
                                            success_msg += f" (Susunan resmi deadline GW {profile['current_event']} telah ditarik ke Slot 1)."
                                        st.success(success_msg)
                                        st.rerun()

                with btn_fpl_cols[1]:
                    if show_fpl_edit and st.button("Batal / Kembali", key="btn_cancel_fpl_edit", use_container_width=True):
                        st.session_state["show_fpl_edit_form"] = False
                        st.rerun()

            # TAB 2: SMART TEXT IMPORT (INSTANT & ZERO-SETUP)
            with sync_method_tabs[1]:
                st.markdown(
                    """
                    <div style="background-color: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                        <span style="font-weight: 700; color: #166534; font-size: 0.95rem;">💡 Impor Skuad Terkini Tanpa Login / Tanpa Cookie</span>
                        <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #14532d;">
                            Jika Anda baru saja melakukan transfer pemain untuk Gameweek depan dan ingin segera menganalisisnya: 
                            cukup salin atau ketik nama 15 pemain Anda di kotak teks di bawah. Sistem akan mencocokkan setiap pemain ke database dan menempatkannya ke formasi Slot 1.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    paste_team_name = st.text_input("Nama Tim Anda:", value=fpl_profile.get("team_name", "Tim FPL") if fpl_profile else "rdjm")
                with t_col2:
                    paste_mgr_name = st.text_input("Nama Manajer:", value=fpl_profile.get("manager_name", "Manajer FPL") if fpl_profile else "Richard Dimas")

                paste_squad_text = st.text_area(
                    "Tempelkan 15 Nama Pemain Skuad Anda:",
                    value="",
                    height=130,
                    placeholder="Contoh:\nTrafford, Dubravka\nTarkowski, White, De Cuyper, Calafiori, N.Williams\nSaka, B.Fernandes, Szoboszlai, Gomez, Groß\nIsak, Thiago, João Pedro",
                    help="Bisa dipisahkan koma atau baris baru. Nama klub atau tanda kurung akan otomatis dibersihkan oleh sistem."
                )

                if st.button("🚀 Pasang Skuad Realtime ke Slot 1", key="btn_do_text_sync", type="primary", use_container_width=True):
                    parsed_slots, p_meta, p_err = parse_and_map_squad_from_text(paste_squad_text, players_df)
                    if p_err:
                        st.error(f"❌ {p_err}")
                    else:
                        profile = fpl_profile or {}
                        profile["team_name"] = paste_team_name.strip() if paste_team_name.strip() else "Tim FPL"
                        profile["manager_name"] = paste_mgr_name.strip() if paste_mgr_name.strip() else "Manajer FPL"
                        profile["is_live_my_team"] = True
                        profile["sync_source"] = "text_import"
                        if "team_id" not in profile:
                            profile["team_id"] = "Custom"
                        if "overall_points" not in profile:
                            profile["overall_points"] = 0
                        if "overall_rank" not in profile:
                            profile["overall_rank"] = 0
                        profile["current_event"] = current_gw

                        st.session_state["fpl_profile"] = profile
                        st.session_state["show_fpl_edit_form"] = False

                        squad_slots_data["slot_1"]["slots"] = parsed_slots
                        squad_slots_data["slot_1"]["name"] = f"Slot 1 (Utama - {profile['team_name']})"
                        squad_slots_data["slot_1"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        st.session_state["active_slot_id"] = "slot_1"
                        st.session_state["my_15_squad_slots"] = dict(parsed_slots)
                        st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1

                        save_all_persisted_squads("slot_1", squad_slots_data, fpl_profile=profile)
                        st.success(f"🎉 Berhasil memasang {p_meta.get('matched_count', 15)} pemain realtime ke Slot 1!")
                        st.rerun()

    st.markdown("---")

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
                    st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
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
            st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
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
            
            # Show any pending import notification
            if "squad_import_msg" in st.session_state:
                st.success(st.session_state.pop("squad_import_msg"))

            uploaded_file = st.file_uploader(
                "Pilih file cadangan skuad (.json):", 
                type=["json"], 
                key=f"upload_squad_file_{active_slot_id}"
            )
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    imported_json = json.load(uploaded_file)
                    imp_slots = imported_json.get("slots", {})
                    valid_ids = set(players_df['id'].dropna().astype(int).tolist())
                    all_valid = len(imp_slots) == 15 and all(int(pid) in valid_ids for pid in imp_slots.values())
                    if all_valid:
                        formatted_slots = {k: int(v) for k, v in imp_slots.items()}
                        preview_cost = sum(player_lookup_cost.get(int(pid), 5.0) for pid in formatted_slots.values())
                        preview_xp = sum(player_lookup_xp.get(int(pid), 3.0) for pid in formatted_slots.values())
                        st.info(f"📄 **File Valid**: 15 pemain lengkap | Estimasi Biaya: **£{preview_cost:.1f}m** | Est. xPoin: **{preview_xp:.1f} pts**")
                        
                        if st.button(f"📥 Terapkan Skuad dari File ke {current_slot_name}", key="btn_apply_uploaded_squad", type="primary", use_container_width=True):
                            st.session_state["my_15_squad_slots"] = formatted_slots
                            st.session_state["squad_slots_data"][active_slot_id]["slots"] = formatted_slots
                            st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
                            save_persisted_squad(formatted_slots)
                            st.session_state["squad_import_msg"] = f"✅ Berhasil memulihkan skuad dari file cadangan ke {current_slot_name}!"
                            st.rerun()
                    else:
                        st.error("⚠️ Format file cadangan tidak sesuai atau ada ID pemain yang tidak valid di database.")
                except Exception as ex:
                    st.error(f"⚠️ Gagal membaca file cadangan: {ex}")

            st.markdown("---")
            st.markdown(f"##### 🔄 Reset **{current_slot_name}**")
            st.caption(f"Tindakan ini hanya mereset {current_slot_name} ke rekomendasi default algoritma tanpa mengganggu slot lainnya.")
            confirm_reset = st.checkbox(f"Saya yakin ingin mereset {current_slot_name}", key="confirm_reset_squad_check")
            if st.button("🚨 Jalankan Reset Slot Ini", disabled=not confirm_reset, use_container_width=True):
                default_slots = get_default_squad_ids(players_df)
                st.session_state["my_15_squad_slots"] = default_slots
                st.session_state["squad_slots_data"][active_slot_id]["slots"] = default_slots
                st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
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

    # =========================================================================
    # TRANSFER PLANNER: REALTIME SQUAD (SLOT 1) VS PLANNED SQUAD (SLOT 2)
    # =========================================================================
    slot1_info = squad_slots_data.get("slot_1", {})
    slot2_info = squad_slots_data.get("slot_2", {})
    s1_slots = slot1_info.get("slots", {})
    s2_slots = slot2_info.get("slots", {})

    if active_slot_id == "slot_2" or (active_slot_id != "slot_1" and len(s1_slots) == 15 and len(s2_slots) == 15):
        if len(s1_slots) == 15 and len(s2_slots) == 15:
            transfer_delta = calculate_squad_transfer_delta(s1_slots, s2_slots, df_merged)
            tf_count = transfer_delta["transfers_count"]

            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1.5px solid #94a3b8; border-radius: 12px; padding: 16px 20px; margin-top: 14px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.15rem; color: #0f172a;">
                                🔄 Perencana Transfer: Komparasi Slot 1 vs Slot 2
                            </span>
                            <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">
                                Slot 1: <b>{slot1_info.get('name', 'Realtime FPL')}</b> ➔ Slot 2: <b>{slot2_info.get('name', 'Rencana Transfer')}</b>
                            </div>
                        </div>
                        <span style="background-color: {'#dbeafe' if tf_count > 0 else '#f1f5f9'}; color: {'#1d4ed8' if tf_count > 0 else '#64748b'}; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;">
                            {tf_count} Transfer Direncanakan
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if tf_count == 0:
                st.info(
                    "ℹ️ **Susunan pemain di Slot 2 saat ini identik dengan Slot 1.** "
                    "Gunakan panel di bawah untuk mengganti pemain yang ingin Anda rotasi atau transfer keluar. "
                    "Aplikasi akan secara otomatis membandingkan delta xPoin, penghematan anggaran, dan tingkat kesulitan jadwal (FDR)."
                )
            else:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.markdown("###### 🔴 Pemain Keluar (Transfer OUT):")
                    for p in transfer_delta["players_out"]:
                        p_cost = float(p.get('Harga (£m)', 0.0))
                        p_xp = float(p.get('xPoin', 0.0))
                        p_club = p.get('Klub', '')
                        p_name = p.get('Nama Pemain', '')
                        st.markdown(
                            f"""
                            <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 700; color: #991b1b;">{p_name}</span>
                                    <span style="font-size: 0.8rem; color: #7f1d1d; margin-left: 6px;">({p_club})</span>
                                </div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: #b91c1c;">
                                    £{p_cost:.1f}m | xPoin: {p_xp:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with t_col2:
                    st.markdown("###### 🟢 Pemain Masuk (Transfer IN):")
                    for p in transfer_delta["players_in"]:
                        p_cost = float(p.get('Harga (£m)', 0.0))
                        p_xp = float(p.get('xPoin', 0.0))
                        p_club = p.get('Klub', '')
                        p_name = p.get('Nama Pemain', '')
                        st.markdown(
                            f"""
                            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 700; color: #166534;">{p_name}</span>
                                    <span style="font-size: 0.8rem; color: #14532d; margin-left: 6px;">({p_club})</span>
                                </div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: #15803d;">
                                    £{p_cost:.1f}m | xPoin: {p_xp:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Performance Delta Metrics
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                with d_c1:
                    dxp = transfer_delta["delta_xp_consensus"]
                    dxp_color = "normal" if dxp >= 0 else "inverse"
                    dxp_sign = "+" if dxp >= 0 else ""
                    st.metric("Δ Konsensus xPoin", f"{dxp_sign}{dxp:.2f} pts", delta=f"{dxp_sign}{dxp:.2f} pts", delta_color=dxp_color)
                with d_c2:
                    dcost = transfer_delta["delta_cost"]
                    dcost_sign = "+" if dcost >= 0 else ""
                    dcost_color = "inverse" if dcost > 0 else "normal"
                    st.metric("Δ Biaya Transfer", f"{dcost_sign}£{dcost:.1f}m", delta=f"{dcost_sign}£{dcost:.1f}m", delta_color=dcost_color)
                with d_c3:
                    dfdr = transfer_delta["delta_fdr"]
                    dfdr_sign = "+" if dfdr >= 0 else ""
                    dfdr_label = "Jadwal Lebih Mudah" if dfdr < 0 else ("Jadwal Lebih Sulit" if dfdr > 0 else "Netral")
                    st.metric("Δ Rata-rata FDR10", f"{dfdr_sign}{dfdr:.2f}", delta=dfdr_label, delta_color="inverse" if dfdr > 0 else "normal")
                with d_c4:
                    if tf_count <= 1:
                        st.metric("Estimasi Transfer Hit", "0 Poin", delta="Dalam 1 Free Transfer", delta_color="normal")
                    else:
                        hit_cost = (tf_count - 1) * 4
                        st.metric("Estimasi Transfer Hit", f"-{hit_cost} Poin", delta=f"Jika punya 1 FT (-4 pts/extra)", delta_color="inverse")

    elif active_slot_id == "slot_1" and fpl_profile:
        st.info(
            f"💡 **Tips Perencana Transfer**: Saat ini Anda sedang membuka **Slot 1 (Skuad Realtime FPL Resmi)**. "
            f"Untuk mencoba skenario transfer masuk/keluar tanpa mengubah susunan skuad asli, "
            f"klik tombol **'⚡ Salin ke Slot 2 (Perencana Transfer)'** di bagian atas."
        )

    # 3. INTERACTIVE SECTION: MEMILIH & MENGGANTI 15 PEMAIN
    with st.expander("🛠️ **Panel Penggantian Pemain (Ganti Pemain di Setiap Slot)**", expanded=True):
        st.write("Ubah pemain pada salah satu dari 15 slot di bawah. Daftar pilihan otomatis disaring sesuai posisi slot.")
        
        pos_tabs = st.tabs(["🧤 Kiper (2 GKP)", "🛡️ Bek (5 DEF)", "🎯 Gelandang (5 MID)", "⚡ Penyerang (3 FWD)", "🔁 Tukar Cepat (Swap Tool)"])
        
        # Helper to format player selectbox option
        player_dict_by_id = {int(r['id']): r for _, r in df_merged.iterrows()}
        
        def make_player_label(p_row):
            return f"{p_row['Klub']} | {p_row['Nama Pemain']} (£{p_row['Harga (£m)']:.1f}m) - xPoin: {p_row['xPoin']:.2f} | FDR1: {p_row['FDR1']:.1f}"

        def on_slot_select_change(slot_key, w_key):
            new_pid = st.session_state.get(w_key)
            if new_pid is not None:
                new_pid = int(new_pid)
                if "my_15_squad_slots" in st.session_state:
                    st.session_state["my_15_squad_slots"][slot_key] = new_pid
                act_slot = st.session_state.get("active_slot_id", "slot_1")
                sl_data = st.session_state.get("squad_slots_data", {})
                if act_slot in sl_data:
                    sl_data[act_slot]["slots"][slot_key] = new_pid
                    sl_data[act_slot]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_persisted_squad(st.session_state.get("my_15_squad_slots"))

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
                    w_key = f"sel_{active_slot_id}_{slot_key}_{squad_revision}"
                    st.selectbox(
                        f"Pilih Pemain {slot_key}",
                        options=gk_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=w_key,
                        on_change=on_slot_select_change,
                        args=(slot_key, w_key)
                    )

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
                    w_key = f"sel_{active_slot_id}_{slot_key}_{squad_revision}"
                    st.selectbox(
                        f"Pilih {slot_key}",
                        options=def_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=w_key,
                        on_change=on_slot_select_change,
                        args=(slot_key, w_key)
                    )

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
                    w_key = f"sel_{active_slot_id}_{slot_key}_{squad_revision}"
                    st.selectbox(
                        f"Pilih {slot_key}",
                        options=mid_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=w_key,
                        on_change=on_slot_select_change,
                        args=(slot_key, w_key)
                    )

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
                    w_key = f"sel_{active_slot_id}_{slot_key}_{squad_revision}"
                    st.selectbox(
                        f"Pilih {slot_key}",
                        options=fwd_options,
                        index=curr_idx,
                        format_func=lambda x: make_player_label(player_dict_by_id.get(x, {})),
                        key=w_key,
                        on_change=on_slot_select_change,
                        args=(slot_key, w_key)
                    )

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
                    act_slot = st.session_state.get("active_slot_id", "slot_1")
                    sl_data = st.session_state.get("squad_slots_data", {})
                    if act_slot in sl_data:
                        sl_data[act_slot]["slots"][swap_slot_choice] = replacement_choice
                        sl_data[act_slot]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["squad_revision"] = st.session_state.get("squad_revision", 0) + 1
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

    # Compare official FPL captain with predictor recommendation if available
    official_cap_id = fpl_profile.get("captain_id") if fpl_profile else None
    if official_cap_id and not squad_df[squad_df['id'] == official_cap_id].empty:
        off_cap_row = squad_df[squad_df['id'] == official_cap_id].iloc[0]
        if off_cap_row['id'] != top_captain['id']:
            diff_pts = top_captain['Konsensus xPoin'] - off_cap_row['Konsensus xPoin']
            st.warning(
                f"⚠️ **Evaluasi Kapten FPL**: Kapten resmi tim Anda di FPL saat ini adalah **{off_cap_row['Nama Pemain']}** "
                f"({off_cap_row['Klub']} - Konsensus: **{off_cap_row['Konsensus xPoin']:.2f} pts**). "
                f"Model memprediksi **{top_captain['Nama Pemain']}** berpotensi menghasilkan **+{diff_pts:.2f} pts** lebih tinggi."
            )
        else:
            st.success(
                f"🎯 **Pilihan Kapten Selaras**: Kapten resmi Anda di FPL (**{off_cap_row['Nama Pemain']}**) telah sesuai dengan pilihan algoritma proyeksi xPoin tertinggi!"
            )

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
