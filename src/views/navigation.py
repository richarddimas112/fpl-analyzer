"""
Navigation Module: Modern, responsive, categorized navigation hub for FPL Scout Analytics.
Eliminates horizontal scrolling while providing an intuitive, delightful UI/UX experience.
"""

import streamlit as st

NAV_MODULES = [
    {
        "id": "player_stats",
        "category": "⚽ Pemain & Skuad",
        "title": "📊 Player Stats & xPoin",
        "badge": "Default Model",
        "desc": "Eksplorasi performa statistik mendalam, analisis per 90 menit, dan model estimasi xPoin klasik FPL."
    },
    {
        "id": "squad_planner",
        "category": "⚽ Pemain & Skuad",
        "title": "👥 Squad Planner",
        "badge": "15 Skuad & FDR10",
        "desc": "Kelola skuad 15 pemain FPL pilihan Anda, bandingkan estimasi xPoin dari seluruh model (Default, Option B, Option C), dan pantau FDR 10 match."
    },
    {
        "id": "visualizations",
        "category": "⚽ Pemain & Skuad",
        "title": "📈 Visualisasi Data & Distribusi",
        "badge": "Box Plot, Scatter & Radar H2H",
        "desc": "Box plot komparasi distribusi poin antar posisi (filter per GW / All GW), scatter plot interaktif dengan korelasi Pearson, agregat klub, serta komparasi 2 pemain (radar chart)."
    },
    {
        "id": "hidden_gem",
        "category": "⚽ Pemain & Skuad",
        "title": "💎 Hidden Gem & Haul Predictor",
        "badge": "Diferensial & Haul",
        "desc": "Identifikasi pemain pembeda ber-ownership rendah dengan potensi poin meledak (haul) dan jadwal menguntungkan."
    },
    {
        "id": "option_b",
        "category": "🤖 Model xPoin AI",
        "title": "🧮 Option B: Component Model xPoin",
        "badge": "FPL Points Formula",
        "desc": "Model prediksi berbasis komponen FPL riil: xMins, xG Pts, xA Pts, Clean Sheet, Saves, DC Pts, dan Bonus Poin."
    },
    {
        "id": "option_c",
        "category": "🤖 Model xPoin AI",
        "title": "🔮 Option C: Current Season Model",
        "badge": "ML Ensemble",
        "desc": "Model Machine Learning musim berjalan (Gradient Boosting, Ridge, Linear Regression) berbobot tren terkini."
    },
    {
        "id": "team_strength",
        "category": "🏟️ Klub & Jadwal",
        "title": "🛡️ Team Strength Analysis",
        "badge": "xGC & Indeks Kekuatan",
        "desc": "Evaluasi kekuatan pertahanan, penyerangan, total xGC riil, dan indeks ketangguhan 20 klub Premier League."
    },
    {
        "id": "fixtures",
        "category": "🏟️ Klub & Jadwal",
        "title": "📅 Fixtures & FDR",
        "badge": "10 Match Matrix",
        "desc": "Tingkat kesulitan jadwal pertandingan (FDR1, FDR3, FDR5, FDR10) dan matriks ticker 10 laga ke depan seluruh klub."
    }
]

MODULE_MAP = {m["id"]: m for m in NAV_MODULES}
CATEGORIES = ["⚽ Pemain & Skuad", "🤖 Model xPoin AI", "🏟️ Klub & Jadwal"]

def get_modules_by_category(category):
    return [m for m in NAV_MODULES if m["category"] == category]

def render_navigation_bar():
    """
    Renders the categorized dashboard navigation.
    Returns the active module ID.
    """
    if "active_nav_id" not in st.session_state:
        st.session_state["active_nav_id"] = "player_stats"

    curr_id = st.session_state["active_nav_id"]
    if curr_id not in MODULE_MAP:
        curr_id = "player_stats"
        st.session_state["active_nav_id"] = curr_id

    st.markdown("""
    <style>
    div[data-baseweb="segmented-control"] {
        flex-wrap: wrap !important;
        gap: 6px !important;
    }
    div[data-baseweb="segmented-control"] button {
        border-radius: 999px !important;
        min-height: 38px !important;
    }
    .module-intro {
        background: linear-gradient(135deg, #111a2e 0%, #0e1728 100%);
        border: 1px solid rgba(255,255,255,.09);
        border-left: 3px solid #06b6d4;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0 20px;
    }
    .module-intro-category { color: #67e8f9; font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
    .module-intro-title { color: #f8fafc; font-size: 1.15rem; font-weight: 800; margin: 4px 0; }
    .module-intro-desc { color: #a8b5c9; font-size: .88rem; line-height: 1.5; margin: 0; }
    .module-intro-badge { color: #bae6fd; background: rgba(6,182,212,.1); border: 1px solid rgba(6,182,212,.22); border-radius: 999px; padding: 5px 11px; font-size: .74rem; font-weight: 700; white-space: nowrap; }
.module-group-label { color: #94a3b8; font-size: .75rem; font-weight: 750; letter-spacing: .07em; text-transform: uppercase; margin: 12px 0 3px; }
    @media (max-width: 700px) {
        div[data-baseweb="segmented-control"] button { white-space: normal !important; }
        .module-intro { padding: 14px 16px; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Jelajahi dashboard** · Pilih modul untuk langsung membukanya.")
    active_module = MODULE_MAP.get(st.session_state["active_nav_id"], NAV_MODULES[0])
    for category in CATEGORIES:
        modules = get_modules_by_category(category)
        widget_key = f"module_pills_{category}"
        title_to_id = {module["title"]: module["id"] for module in modules}
        st.session_state[widget_key] = (
            active_module["title"] if active_module["category"] == category else None
        )

        def on_module_selected(key=widget_key, lookup=title_to_id):
            chosen_id = lookup.get(st.session_state.get(key))
            if chosen_id:
                st.session_state["active_nav_id"] = chosen_id

        st.markdown(f'<div class="module-group-label">{category}</div>', unsafe_allow_html=True)
        st.pills(
            f"{category} modules",
            options=[module["title"] for module in modules],
            key=widget_key,
            on_change=on_module_selected,
            label_visibility="collapsed"
        )

    active_m = MODULE_MAP.get(st.session_state["active_nav_id"], NAV_MODULES[0])
    st.markdown(f"""
    <div class="module-intro">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <div class="module-intro-category">{active_m['category']}</div>
                <div class="module-intro-title">{active_m['title']}</div>
                <p class="module-intro-desc">{active_m['desc']}</p>
            </div>
            <div>
                <span class="module-intro-badge">{active_m['badge']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.session_state["active_nav_id"]
