"""
API Fetching module for FPL Scout Analytics with caching and session reuse.
"""

import requests
import streamlit as st
from src.constants import BOOTSTRAP_URL, FIXTURES_URL, ELEMENT_SUMMARY_URL, HEADERS

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
    """Mengambil histori match-by-match individual pemain tanpa cache."""
    try:
        session = get_http_session()
        url = ELEMENT_SUMMARY_URL.format(player_id)
        response = session.get(url, timeout=8)
        if response.status_code == 200:
            return response.json().get('history', [])
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
            return response.json()
        return {}
    except Exception as e:
        return {}
