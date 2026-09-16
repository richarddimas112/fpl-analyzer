import requests
import streamlit as st
import datetime
from src.firebase_client import load_firebase_config

def get_auth_url(action="signInWithPassword"):
    cfg = load_firebase_config()
    if not cfg:
        return None
    api_key = cfg.get("apiKey")
    return f"https://identitytoolkit.googleapis.com/v1/accounts:{action}?key={api_key}"

def format_auth_error(raw_msg):
    if not raw_msg:
        return "Terjadi kesalahan autentikasi."
    if "EMAIL_NOT_FOUND" in raw_msg or "INVALID_LOGIN_CREDENTIALS" in raw_msg or "INVALID_PASSWORD" in raw_msg:
        return "Email atau password salah. Jika lupa password, gunakan opsi Lupa Password."
    if "EMAIL_EXISTS" in raw_msg:
        return "Email sudah terdaftar. Silakan pilih opsi Login atau gunakan Lupa Password."
    if "INVALID_EMAIL" in raw_msg:
        return "Format alamat email tidak valid."
    if "WEAK_PASSWORD" in raw_msg:
        return "Password terlalu lemah (minimal 6 karakter)."
    if "TOO_MANY_ATTEMPTS_TRY_LATER" in raw_msg:
        return "Terlalu banyak percobaan gagal. Silakan coba lagi nanti atau reset password Anda."
    if "OPERATION_NOT_ALLOWED" in raw_msg:
        return "Metode login Email/Password belum diaktifkan di Firebase Console."
    return f"Gagal: {raw_msg}"

def login_user(email, password):
    if not email or not password:
        return False, "Mohon isi email dan password."
    url = get_auth_url("signInWithPassword")
    if not url:
        return False, "Firebase config missing"
    
    payload = {
        "email": email.strip(),
        "password": password,
        "returnSecureToken": True
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        if r.status_code == 200:
            st.session_state["user_token"] = data["idToken"]
            st.session_state["user_uid"] = data["localId"]
            st.session_state["user_email"] = data["email"]
            return True, "Login berhasil"
        else:
            raw_err = data.get("error", {}).get("message", "")
            return False, format_auth_error(raw_err)
    except Exception as e:
        return False, str(e)

def register_user(email, password):
    if not email or not password:
        return False, "Mohon isi email dan password."
    url = get_auth_url("signUp")
    if not url:
        return False, "Firebase config missing"
        
    payload = {
        "email": email.strip(),
        "password": password,
        "returnSecureToken": True
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        if r.status_code == 200:
            st.session_state["user_token"] = data["idToken"]
            st.session_state["user_uid"] = data["localId"]
            st.session_state["user_email"] = data["email"]
            return True, "Registrasi sukses"
        else:
            raw_err = data.get("error", {}).get("message", "")
            return False, format_auth_error(raw_err)
    except Exception as e:
        return False, str(e)

def reset_password(email):
    if not email or "@" not in email:
        return False, "Mohon masukkan alamat email yang valid."
    url = get_auth_url("sendOobCode")
    if not url:
        return False, "Konfigurasi Firebase tidak ditemukan."
    
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": email.strip()
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        if r.status_code == 200:
            return True, f"Link reset password telah dikirim ke {email.strip()}! Silakan periksa kotak masuk (inbox) atau folder spam email Anda."
        else:
            raw_err = data.get("error", {}).get("message", "")
            return False, format_auth_error(raw_err)
    except Exception as e:
        return False, str(e)

def logout_user():
    if "user_token" in st.session_state:
        del st.session_state["user_token"]
    if "user_uid" in st.session_state:
        del st.session_state["user_uid"]
    if "user_email" in st.session_state:
        del st.session_state["user_email"]

