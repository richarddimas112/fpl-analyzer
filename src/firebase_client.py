"""
Firebase Firestore Client & Differential Synchronization Service for FPL App.
Handles reading and writing Gameweek match differentials with optimal caching to stay 100% within the free tier.
"""

import json
import os
import datetime
import requests
import streamlit as st

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "firebase-applet-config.json")

def load_firebase_config():
    """Load Firebase project credentials from st.secrets (Streamlit Cloud) or firebase-applet-config.json (local)."""
    # 1. Check Streamlit secrets first (ideal for Streamlit Community Cloud without committing secrets to GitHub)
    try:
        if hasattr(st, "secrets"):
            if "firebase" in st.secrets:
                return dict(st.secrets["firebase"])
            elif "projectId" in st.secrets:
                return dict(st.secrets)
    except Exception:
        pass

    # 2. Check local configuration file
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Gagal membaca firebase-applet-config.json: {e}")
    return None

def get_firestore_url(collection_name, doc_id=None):
    """Construct Firestore REST API URL for given collection or document."""
    cfg = load_firebase_config()
    if not cfg:
        return None, None
    project_id = cfg.get("projectId")
    db_id = cfg.get("firestoreDatabaseId", "(default)")
    api_key = cfg.get("apiKey")
    
    if doc_id:
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/{db_id}/documents/{collection_name}/{doc_id}?key={api_key}"
    else:
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/{db_id}/documents/{collection_name}?key={api_key}"
    return url, api_key

@st.cache_data(ttl=3600)
def fetch_gw_differentials_from_firestore(gw_id: int):
    """
    Fetch Gameweek differentials from Firestore with 1-hour memory caching
    to minimize read calls and keep Firebase strictly free.
    """
    url, _ = get_firestore_url("gameweek_differentials", f"gw_{gw_id}")
    if not url:
        return None
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            fields = data.get("fields", {})
            
            # Parse team differentials map
            team_diffs_raw = fields.get("team_differentials", {}).get("mapValue", {}).get("fields", {})
            parsed_team_diffs = {}
            for t_name, t_val in team_diffs_raw.items():
                props = t_val.get("mapValue", {}).get("fields", {})
                diff_att = float(props.get("diff_attack", {}).get("doubleValue", props.get("diff_attack", {}).get("integerValue", 0.0)))
                diff_def = float(props.get("diff_defense", {}).get("doubleValue", props.get("diff_defense", {}).get("integerValue", 0.0)))
                opp = props.get("opponent", {}).get("stringValue", "-")
                is_home = props.get("is_home", {}).get("booleanValue", True)
                parsed_team_diffs[t_name] = {
                    "diff_attack": diff_att,
                    "diff_defense": diff_def,
                    "opponent": opp,
                    "is_home": is_home
                }
            
            is_final = fields.get("is_final", {}).get("booleanValue", False)
            updated_at = fields.get("updated_at", {}).get("stringValue", "")
            
            return {
                "gameweek": gw_id,
                "is_final": is_final,
                "updated_at": updated_at,
                "team_differentials": parsed_team_diffs
            }
        elif r.status_code == 404:
            return None
    except Exception as e:
        print(f"Firestore read error for gw_{gw_id}: {e}")
    return None

def save_gw_differentials_to_firestore(gw_id: int, is_final: bool, team_diffs: dict, matches: list = None):
    """
    Write or update Gameweek differentials into Firestore via REST API.
    Uses structured map format conforming to firestore.rules.
    """
    url, _ = get_firestore_url("gameweek_differentials", f"gw_{gw_id}")
    if not url:
        return False, "Konfigurasi Firebase tidak ditemukan."
    
    # Build Firestore JSON payload
    team_diff_fields = {}
    for t_name, d in team_diffs.items():
        team_diff_fields[str(t_name)] = {
            "mapValue": {
                "fields": {
                    "diff_attack": {"doubleValue": float(d.get("diff_attack", 0.0))},
                    "diff_defense": {"doubleValue": float(d.get("diff_defense", 0.0))},
                    "opponent": {"stringValue": str(d.get("opponent", "-"))},
                    "is_home": {"booleanValue": bool(d.get("is_home", True))}
                }
            }
        }
    
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "fields": {
            "gameweek": {"integerValue": str(gw_id)},
            "is_final": {"booleanValue": is_final},
            "updated_at": {"stringValue": now_iso},
            "team_differentials": {
                "mapValue": {
                    "fields": team_diff_fields
                }
            }
        }
    }
    
    try:
        r = requests.patch(url, json=payload, timeout=6)
        if r.status_code == 200:
            # Clear Streamlit cache for this function so subsequent reads get fresh data
            fetch_gw_differentials_from_firestore.clear()
            return True, f"Sukses menyimpan data GW{gw_id} ke Firestore."
        else:
            return False, f"Firestore write error (HTTP {r.status_code}): {r.text[:200]}"
    except Exception as e:
        return False, f"Exception saat write ke Firestore: {e}"

def is_gw4_finished(fixtures: list) -> tuple[bool, str]:
    """
    Check if the last match of Gameweek 4 (Leeds vs Newcastle) has finished.
    Returns (is_finished, status_message).
    """
    gw4_matches = [f for f in fixtures if f.get("event") == 4]
    if not gw4_matches:
        return False, "Data match GW4 tidak ditemukan."
    
    # Check specifically for match involving Leeds and Newcastle
    # Or check if all 10 matches of GW4 are finished
    all_finished = True
    leeds_newcastle_status = "Belum Selesai"
    for m in gw4_matches:
        is_fin = bool(m.get("finished") or m.get("finished_provisional"))
        # Match 40: Leeds (13) vs Newcastle (17)
        if (m.get("team_h") == 13 and m.get("team_a") == 17) or (m.get("team_h") == 17 and m.get("team_a") == 13):
            if is_fin:
                leeds_newcastle_status = "Selesai"
            else:
                leeds_newcastle_status = "Belum Selesai (Kickoff: 14 Sep 2026, 19:00 WIB)"
        if not is_fin:
            all_finished = False
            
    if all_finished:
        return True, "Seluruh pertandingan GW4 (termasuk Leeds vs Newcastle) telah selesai."
    return False, f"Laga penutup GW4 (Leeds vs Newcastle): {leeds_newcastle_status}."

def compute_differentials_for_gw(gw_id: int, fixtures: list, df_teams, teams_dict: dict):
    """
    Compute differential attack & defense for all matches in a given Gameweek.
    Uses the latest team strength offensive & defensive scores.
    """
    gw_matches = [f for f in fixtures if f.get("event") == gw_id]
    team_stats = {}
    for _, row in df_teams.iterrows():
        t_name = row.get("Klub")
        team_stats[t_name] = {
            "att": float(row.get("Skor Serangan", 50.0)),
            "def": float(row.get("Skor Pertahanan", 50.0))
        }
        # Also map by ID if available
        t_id = row.get("team_id")
        if t_id:
            team_stats[t_id] = team_stats[t_name]

    team_diffs = {}
    match_records = []
    
    for m in gw_matches:
        h_id = m.get("team_h")
        a_id = m.get("team_a")
        h_name = teams_dict.get(h_id, f"Team {h_id}")
        a_name = teams_dict.get(a_id, f"Team {a_id}")
        
        h_info = team_stats.get(h_name, team_stats.get(h_id, {"att": 50.0, "def": 50.0}))
        a_info = team_stats.get(a_name, team_stats.get(a_id, {"att": 50.0, "def": 50.0}))
        
        h_att = h_info["att"]
        h_def = h_info["def"]
        a_att = a_info["att"]
        a_def = a_info["def"]
        
        # 1. Home Diff Attack = Home Attack - Away Defense
        h_att_diff = round(h_att - a_def, 1)
        # 2. Home Diff Def = Home Defense - Away Attack
        h_def_diff = round(h_def - a_att, 1)
        # 3. Away Diff Attack = Away Attack - Home Defense
        a_att_diff = round(a_att - h_def, 1)
        # 4. Away Diff Def = Away Defense - Home Attack
        a_def_diff = round(a_def - h_att, 1)
        
        # Store for Home Club
        team_diffs[h_name] = {
            "diff_attack": h_att_diff,
            "diff_defense": h_def_diff,
            "opponent": a_name,
            "is_home": True
        }
        # Store for Away Club
        team_diffs[a_name] = {
            "diff_attack": a_att_diff,
            "diff_defense": a_def_diff,
            "opponent": h_name,
            "is_home": False
        }
        
        match_records.append({
            "fixture_id": m.get("id"),
            "home_team": h_name,
            "away_team": a_name,
            "home_diff_attack": h_att_diff,
            "home_diff_defense": h_def_diff,
            "away_diff_attack": a_att_diff,
            "away_diff_defense": a_def_diff,
            "is_finished": bool(m.get("finished") or m.get("finished_provisional"))
        })
        
    return team_diffs, match_records

def sync_historical_gw1_to_gw4(fixtures: list, df_teams, teams_dict: dict, force: bool = False):
    """
    Initialize and save historical data for GW1 - GW4 into Firebase Firestore.
    Runs when Leeds vs Newcastle is finished OR when forced by user.
    """
    is_finished, msg = is_gw4_finished(fixtures)
    if not is_finished and not force:
        return False, f"Sinkronisasi ditunda: {msg}"
    
    results = []
    for gw in [1, 2, 3, 4]:
        team_diffs, matches = compute_differentials_for_gw(gw, fixtures, df_teams, teams_dict)
        if team_diffs:
            # GW1-3 are finished, GW4 is finished if triggered
            is_final = True if (gw < 4 or is_finished) else force
            ok, res_msg = save_gw_differentials_to_firestore(gw, is_final, team_diffs, matches)
            results.append(f"GW{gw}: {'Berhasil' if ok else 'Gagal'}")
        else:
            results.append(f"GW{gw}: Tidak ada data fixture")
            
    return True, f"Sinkronisasi GW1 - GW4 selesai: {', '.join(results)}"

def auto_sync_next_gw_differentials(current_gw_num: int, fixtures: list, df_teams, teams_dict: dict):
    """
    Checks if current_gw_num is fully finished.
    If yes, calculates and saves team differentials for current_gw_num + 1 to Firestore.
    """
    gw_matches = [f for f in fixtures if f.get("event") == current_gw_num]
    if not gw_matches:
        return False, "Tidak ada data match untuk GW saat ini."
    
    all_finished = True
    for m in gw_matches:
        is_fin = bool(m.get("finished") or m.get("finished_provisional"))
        if not is_fin:
            all_finished = False
            break
            
    if not all_finished:
        return False, f"Gameweek {current_gw_num} belum selesai sepenuhnya."
        
    next_gw = current_gw_num + 1
    
    # Check if next_gw is already saved
    existing_data = fetch_gw_differentials_from_firestore(next_gw)
    # If it exists and is marked as final, we don't need to re-save
    if existing_data and existing_data.get('is_final') == True:
        return True, f"Data GW {next_gw} sudah disinkronisasi sebelumnya."
        
    # Compute and save for next_gw
    team_diffs, matches = compute_differentials_for_gw(next_gw, fixtures, df_teams, teams_dict)
    if not team_diffs:
        return False, f"Tidak ada data fixture untuk GW {next_gw}."
        
    ok, msg = save_gw_differentials_to_firestore(next_gw, True, team_diffs, matches)
    return ok, f"Auto-sync GW {next_gw}: {msg}"

def load_user_squad(uid: str, token: str):
    """Load user's squad from Firestore."""
    url, _ = get_firestore_url("squads", uid)
    if not url:
        return []
    
    # Append auth token
    url = f"{url}&auth={token}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            fields = data.get("fields", {})
            squad_list = fields.get("squad", {}).get("arrayValue", {}).get("values", [])
            squad_ids = []
            for item in squad_list:
                val = item.get("integerValue") or item.get("doubleValue")
                if val is not None:
                    squad_ids.append(int(float(val)))
            return squad_ids
        return []
    except Exception as e:
        st.error(f"Error loading squad: {e}")
        return []

def save_user_squad(uid: str, token: str, squad_ids: list):
    """Save user's squad to Firestore."""
    url, _ = get_firestore_url("squads", uid)
    if not url:
        return False
    
    url = f"{url}&auth={token}"
    
    squad_values = [{"integerValue": str(p_id)} for p_id in squad_ids]
    
    payload = {
        "fields": {
            "uid": {"stringValue": uid},
            "squad": {"arrayValue": {"values": squad_values}},
            "updated_at": {"stringValue": datetime.datetime.utcnow().isoformat() + "Z"}
        }
    }
    
    try:
        r = requests.patch(url, json=payload, timeout=5)
        if r.status_code == 200:
            return True
        else:
            st.error(f"Error saving squad: {r.json()}")
            return False
    except Exception as e:
        st.error(f"Error saving squad: {e}")
        return False
