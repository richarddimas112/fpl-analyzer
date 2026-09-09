"""
API Fetching module for FPL Scout Analytics with caching and session reuse.
"""

import os
from datetime import datetime, timedelta
import requests
import streamlit as st
from src.constants import BOOTSTRAP_URL, FIXTURES_URL, ELEMENT_SUMMARY_URL, EVENT_LIVE_URL, HEADERS

_session = None

def get_http_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session

@st.cache_data(ttl=3600)
def fetch_fpl_data():
    """Mengambil dataset komprehensif FPL (pemain, tim, event)."""
    try:
        session = get_http_session()
        response = session.get(BOOTSTRAP_URL, timeout=12)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Gagal mengambil data FPL bootstrap: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_fixtures_data():
    """Mengambil jadwal pertandingan resmi dan FDR (Fixture Difficulty Rating)."""
    try:
        session = get_http_session()
        response = session.get(FIXTURES_URL, timeout=12)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Gagal mengambil data jadwal pertandingan (fixtures): {e}")
        return []

def fetch_player_history_raw(player_id):
    """Mengambil histori match-by-match individual pemain tanpa cache, menyaring fixture yang belum berlangsung."""
    try:
        session = get_http_session()
        url = ELEMENT_SUMMARY_URL.format(player_id)
        response = session.get(url, timeout=8)
        if response.status_code == 200:
            raw_hist = response.json().get('history', [])
            # Hanya sertakan pertandingan yang sudah dimulai/selesai (skor bukan None atau ada menit bermain yang tercatat)
            valid_hist = [
                m for m in raw_hist 
                if (m.get('team_h_score') is not None and m.get('team_a_score') is not None) or int(m.get('minutes', 0)) > 0
            ]
            return valid_hist
        return []
    except Exception:
        return []

@st.cache_data(ttl=86400)
def fetch_player_history(player_id):
    """Mengambil histori match-by-match individual pemain dengan cache."""
    return fetch_player_history_raw(player_id)

@st.cache_data(ttl=3600)
def fetch_player_element_summary(player_id):
    """Mengambil rangkuman elemen individual pemain lengkap (history, history_past, fixtures)."""
    try:
        session = get_http_session()
        url = ELEMENT_SUMMARY_URL.format(player_id)
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            raw_hist = data.get('history', [])
            data['history'] = [
                m for m in raw_hist 
                if (m.get('team_h_score') is not None and m.get('team_a_score') is not None) or int(m.get('minutes', 0)) > 0
            ]
            return data
        return {}
    except Exception as e:
        return {}

@st.cache_data(ttl=3600)
def fetch_gameweek_live_points(event_id):
    """Mengambil data performa match-by-match live pemain per Gameweek tertentu."""
    try:
        session = get_http_session()
        url = EVENT_LIVE_URL.format(event_id)
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            elements_map = {}
            for elem in data.get('elements', []):
                elements_map[int(elem['id'])] = elem.get('stats', {})
            return elements_map
        return {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_all_gameweeks_live_points(event_ids_tuple):
    """Mengambil dan menggabungkan data match-by-match seluruh Gameweek yang sudah/sedang berjalan."""
    all_gw_map = {}
    for ev_id in event_ids_tuple:
        all_gw_map[int(ev_id)] = fetch_gameweek_live_points(int(ev_id))
    return all_gw_map

# ==============================================================================
# FOOTBALL-DATA.ORG INTEGRATION (MULTI-COMPETITION FIXTURES)
# ==============================================================================

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# Pemetaan Nama Klub Premier League ke Team ID Football-Data.org
FOOTBALL_DATA_TEAM_MAP = {
    # Arsenal
    "Arsenal": 57,
    "Arsenal FC": 57,
    # Aston Villa
    "Aston Villa": 58,
    "Aston Villa FC": 58,
    # Chelsea
    "Chelsea": 61,
    "Chelsea FC": 61,
    # Everton
    "Everton": 62,
    "Everton FC": 62,
    # Fulham
    "Fulham": 63,
    "Fulham FC": 63,
    # Liverpool
    "Liverpool": 64,
    "Liverpool FC": 64,
    # Manchester City
    "Man City": 65,
    "Manchester City": 65,
    "Manchester City FC": 65,
    # Manchester United
    "Man Utd": 66,
    "Manchester United": 66,
    "Manchester United FC": 66,
    # Newcastle United
    "Newcastle": 67,
    "Newcastle United": 67,
    "Newcastle United FC": 67,
    # Tottenham Hotspur
    "Spurs": 73,
    "Tottenham": 73,
    "Tottenham Hotspur": 73,
    "Tottenham Hotspur FC": 73,
    # Wolverhampton Wanderers
    "Wolves": 76,
    "Wolverhampton": 76,
    "Wolverhampton Wanderers": 76,
    "Wolverhampton Wanderers FC": 76,
    # Leicester City
    "Leicester": 338,
    "Leicester City": 338,
    "Leicester City FC": 338,
    # Southampton
    "Southampton": 340,
    "Southampton FC": 340,
    # Ipswich Town
    "Ipswich": 349,
    "Ipswich Town": 349,
    "Ipswich Town FC": 349,
    # Nottingham Forest
    "Nott'm Forest": 351,
    "Nottingham Forest": 351,
    "Nottingham Forest FC": 351,
    # Crystal Palace
    "Crystal Palace": 354,
    "Crystal Palace FC": 354,
    # Brighton & Hove Albion
    "Brighton": 397,
    "Brighton & Hove Albion": 397,
    "Brighton & Hove Albion FC": 397,
    # Brentford
    "Brentford": 402,
    "Brentford FC": 402,
    # West Ham United
    "West Ham": 563,
    "West Ham United": 563,
    "West Ham United FC": 563,
    # AFC Bournemouth
    "Bournemouth": 1044,
    "AFC Bournemouth": 1044,
}

COMPETITION_BADGES = {
    "PL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",
    "CL": "⭐ UEFA Champions League",
    "FAC": "🛡️ FA Cup",
    "EFL": "🔴 Carabao Cup (EFL)",
    "EL": "🟠 UEFA Europa League",
    "ECL": "🟢 UEFA Conference League",
    "WC": "🌍 FIFA World Cup",
}

def get_football_data_api_key():
    """Mengambil API key Football-Data.org dari environment, st.secrets, atau file .env."""
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if key and key.strip() and key.strip() != "MY_FOOTBALL_DATA_API_KEY":
        return key.strip()
    try:
        if hasattr(st, "secrets") and "FOOTBALL_DATA_API_KEY" in st.secrets:
            s_key = st.secrets["FOOTBALL_DATA_API_KEY"]
            if s_key and str(s_key).strip() and str(s_key).strip() != "MY_FOOTBALL_DATA_API_KEY":
                return str(s_key).strip()
    except Exception:
        pass
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FOOTBALL_DATA_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and val != "MY_FOOTBALL_DATA_API_KEY":
                            return val
        except Exception:
            pass
    return None

def get_football_data_team_id(club_name: str):
    """Mencocokkan nama klub dari FPL ke Team ID Football-Data.org."""
    if not club_name:
        return None
    c_clean = str(club_name).strip()
    if c_clean in FOOTBALL_DATA_TEAM_MAP:
        return FOOTBALL_DATA_TEAM_MAP[c_clean]
    c_lower = c_clean.lower()
    for name, tid in FOOTBALL_DATA_TEAM_MAP.items():
        if name.lower() == c_lower or name.lower() in c_lower or c_lower in name.lower():
            return tid
    return None

def format_utc_to_wib(utc_str: str):
    """Mengonversi timestamp UTC dari football-data.org ke format Waktu Indonesia Barat (WIB)."""
    if not utc_str:
        return "-"
    try:
        clean_str = str(utc_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        wib_dt = dt + timedelta(hours=7)
        return wib_dt.strftime("%d %b %Y, %H:%M WIB")
    except Exception:
        return str(utc_str)[:16].replace("T", " ")

def get_competition_label(comp_dict: dict):
    """Memberikan label kompetisi ber-ikon menarik."""
    if not isinstance(comp_dict, dict):
        return "🏆 Pertandingan Resmi"
    code = comp_dict.get("code", "")
    name = comp_dict.get("name", "")
    if code in COMPETITION_BADGES:
        return COMPETITION_BADGES[code]
    if "Champions League" in name:
        return "⭐ UEFA Champions League"
    if "Europa League" in name:
        return "🟠 UEFA Europa League"
    if "Conference League" in name:
        return "🟢 UEFA Conference League"
    if "Premier League" in name:
        return "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League"
    if "FA Cup" in name:
        return "🛡️ FA Cup"
    if "EFL" in name or "League Cup" in name or "Carabao" in name:
        return "🔴 Carabao Cup"
    return f"🏆 {name}"

@st.cache_data(ttl=1800)
def fetch_football_data_upcoming_matches(team_id: int, api_token: str = None):
    """
    Mengambil jadwal pertandingan mendatang sebuah klub di semua kompetisi dari football-data.org.
    Header 'X-Auth-Token' digunakan untuk autentikasi personal token.
    """
    if not api_token:
        api_token = get_football_data_api_key()
    if not api_token:
        return None, "NO_TOKEN"
    
    url = f"{FOOTBALL_DATA_BASE_URL}/teams/{team_id}/matches"
    headers = {
        "X-Auth-Token": str(api_token).strip(),
        "User-Agent": "FPL-Scout-Analytics/1.0"
    }
    params = {
        "status": "SCHEDULED,TIMED"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("matches", [])
            matches = sorted(matches, key=lambda x: x.get("utcDate", ""))
            return matches, None
        elif resp.status_code == 403:
            return None, "403_FORBIDDEN"
        elif resp.status_code == 429:
            return None, "429_RATE_LIMIT"
        elif resp.status_code == 400:
            # Jika query parameter status ditolak, coba tanpa parameter status
            resp2 = requests.get(url, headers=headers, timeout=12)
            if resp2.status_code == 200:
                raw_m = resp2.json().get("matches", [])
                upcoming = [
                    m for m in raw_m 
                    if m.get("status") in ("SCHEDULED", "TIMED") or (m.get("score", {}).get("winner") is None and m.get("status") not in ("FINISHED", "CANCELLED"))
                ]
                upcoming = sorted(upcoming, key=lambda x: x.get("utcDate", ""))
                return upcoming, None
            return None, f"HTTP_{resp.status_code}"
        else:
            return None, f"HTTP_{resp.status_code}"
    except Exception as e:
        return None, f"EXCEPTION_{str(e)}"


