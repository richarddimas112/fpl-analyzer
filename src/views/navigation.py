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
        "title": "👥 Perencana 15 Pemain & Multi-xPoin",
        "badge": "15 Skuad & FDR10",
        "desc": "Kelola skuad 15 pemain FPL pilihan Anda, bandingkan estimasi xPoin dari seluruh model (Default, Option B, Option C), dan pantau FDR 10 match."
    },
    {
        "id": "visualizations",
        "category": "⚽ Pemain & Skuad",
        "title": "📈 Visualisasi Data & Radar Pemain",
        "badge": "Scatter & Radar H2H",
        "desc": "Scatter plot interaktif, korelasi matriks Pearson, serta visualisasi Radar Chart perbandingan head-to-head 2 pemain."
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
TITLE_TO_ID = {m["title"]: m["id"] for m in NAV_MODULES}
CATEGORIES = ["⚽ Pemain & Skuad", "🤖 Model xPoin AI", "🏟️ Klub & Jadwal"]

def get_modules_by_category(category):
    return [m for m in NAV_MODULES if m["category"] == category]

def render_navigation_bar():
    """
    Renders the modern navigation hub.
    Returns the active module ID.
    """
    # 1. Ensure session state keys exist
    if "active_nav_id" not in st.session_state:
        st.session_state["active_nav_id"] = "player_stats"
    if "nav_view_mode" not in st.session_state:
        st.session_state["nav_view_mode"] = "🌐 Semua Modul (Wrap)"

    curr_id = st.session_state["active_nav_id"]
    if curr_id not in MODULE_MAP:
        curr_id = "player_stats"
        st.session_state["active_nav_id"] = curr_id

    curr_module = MODULE_MAP[curr_id]
    curr_category = curr_module["category"]

    # 2. Navigation Top Bar Container
    st.markdown("""
    <style>
    /* Styling for the Navigation Hub */
    .nav-hub-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .nav-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .nav-header-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .nav-active-badge {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    /* Auto wrap tabs & pills to completely eliminate horizontal scrolling */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        height: auto !important;
        gap: 6px !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        white-space: normal !important;
        height: auto !important;
        padding: 8px 14px !important;
        font-size: 0.88rem !important;
    }
    /* Streamlit Segmented Control / Pills Auto Wrap */
    div[data-baseweb="segmented-control"] {
        flex-wrap: wrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 3. Top Controls: View Mode Selector
    col_nav_title, col_view_mode = st.columns([3, 1])
    with col_nav_title:
        st.markdown(
            f"**🧭 Navigasi Modul Dashboard** &nbsp; "
            f"<span style='font-size: 0.85rem; color: #64748b; font-weight: normal;'>"
            f"Kategori: <b>{curr_category}</b> &bull; Modul: <b>{curr_module['title']}</b></span>",
            unsafe_allow_html=True
        )
    with col_view_mode:
        view_modes = ["🌐 Semua Modul (Wrap)", "🗂️ Kategori Terstruktur", "📋 Tab Klasik (Auto-Wrap)"]
        curr_mode = st.session_state["nav_view_mode"]
        mode = st.selectbox(
            "Tampilan Navigasi:",
            options=view_modes,
            index=view_modes.index(curr_mode) if curr_mode in view_modes else 0,
            key="sel_nav_view_mode_widget",
            label_visibility="collapsed"
        )
        if mode != st.session_state["nav_view_mode"]:
            st.session_state["nav_view_mode"] = mode
            st.rerun()

    # 4. RENDER BASED ON CHOSEN VIEW MODE
    active_mode = st.session_state["nav_view_mode"]

    if active_mode == "🗂️ Kategori Terstruktur":
        # Ensure category state is initialized and synchronized
        if "nav_selected_category" not in st.session_state or st.session_state["nav_selected_category"] not in CATEGORIES:
            st.session_state["nav_selected_category"] = curr_category
        elif curr_category != st.session_state["nav_selected_category"]:
            st.session_state["nav_selected_category"] = curr_category
            st.session_state["widget_cat_selector"] = curr_category
            for c in CATEGORIES:
                sub_k = f"widget_sub_mod_{c}"
                if sub_k in st.session_state:
                    del st.session_state[sub_k]

        def on_category_selected():
            new_cat = st.session_state.get("widget_cat_selector")
            if not new_cat:
                st.session_state["widget_cat_selector"] = st.session_state["nav_selected_category"]
                return
            if new_cat in CATEGORIES and new_cat != st.session_state["nav_selected_category"]:
                st.session_state["nav_selected_category"] = new_cat
                new_cat_modules = get_modules_by_category(new_cat)
                if new_cat_modules:
                    first_m = new_cat_modules[0]
                    st.session_state["active_nav_id"] = first_m["id"]
                    st.session_state["sidebar_nav_selectbox"] = first_m["title"]
                    st.session_state["pills_all_modules"] = first_m["title"]
                for c in CATEGORIES:
                    sub_k = f"widget_sub_mod_{c}"
                    if sub_k in st.session_state:
                        del st.session_state[sub_k]

        active_cat = st.session_state["nav_selected_category"]
        cat_modules = get_modules_by_category(active_cat)
        cat_module_titles = [m["title"] for m in cat_modules]
        cat_title_to_id = {m["title"]: m["id"] for m in cat_modules}

        curr_sub_title = curr_module["title"] if curr_module["title"] in cat_module_titles else cat_module_titles[0]

        def on_sub_module_selected():
            chosen = st.session_state.get(f"widget_sub_mod_{active_cat}")
            if not chosen:
                st.session_state[f"widget_sub_mod_{active_cat}"] = curr_sub_title
                return
            chosen_id = cat_title_to_id.get(chosen)
            if chosen_id and chosen_id != st.session_state.get("active_nav_id"):
                st.session_state["active_nav_id"] = chosen_id
                st.session_state["sidebar_nav_selectbox"] = chosen
                st.session_state["pills_all_modules"] = chosen

        if "widget_cat_selector" not in st.session_state:
            st.session_state["widget_cat_selector"] = active_cat

        # Keep sub-pill widget value in sync with active module if not yet set
        sub_key = f"widget_sub_mod_{active_cat}"
        if sub_key not in st.session_state or st.session_state[sub_key] not in cat_module_titles:
            st.session_state[sub_key] = curr_sub_title
        elif curr_module["title"] in cat_module_titles and st.session_state[sub_key] != curr_module["title"]:
            st.session_state[sub_key] = curr_module["title"]

        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em;'>1. Kategori Utama</p>", unsafe_allow_html=True)
        st.pills(
            "Kategori Utama",
            options=CATEGORIES,
            key="widget_cat_selector",
            on_change=on_category_selected,
            label_visibility="collapsed"
        )

        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748b; margin-top: 10px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em;'>2. Modul dalam Kategori</p>", unsafe_allow_html=True)
        st.pills(
            "Modul Spesifik",
            options=cat_module_titles,
            key=sub_key,
            on_change=on_sub_module_selected,
            label_visibility="collapsed"
        )

    elif active_mode == "🌐 Semua Modul (Wrap)":
        # All 8 modules as wrapping pills
        all_titles = [m["title"] for m in NAV_MODULES]

        if "pills_all_modules" not in st.session_state or st.session_state["pills_all_modules"] != curr_module["title"]:
            st.session_state["pills_all_modules"] = curr_module["title"]

        def on_all_modules_selected():
            chosen = st.session_state.get("pills_all_modules")
            if not chosen:
                st.session_state["pills_all_modules"] = curr_module["title"]
                return
            chosen_id = TITLE_TO_ID.get(chosen)
            if chosen_id and chosen_id != st.session_state.get("active_nav_id"):
                st.session_state["active_nav_id"] = chosen_id
                st.session_state["sidebar_nav_selectbox"] = chosen
                chosen_cat = MODULE_MAP[chosen_id]["category"]
                st.session_state["nav_selected_category"] = chosen_cat
                st.session_state["widget_cat_selector"] = chosen_cat
                for c in CATEGORIES:
                    sub_k = f"widget_sub_mod_{c}"
                    if sub_k in st.session_state:
                        del st.session_state[sub_k]

        st.pills(
            "Pilih Modul",
            options=all_titles,
            key="pills_all_modules",
            on_change=on_all_modules_selected,
            label_visibility="collapsed"
        )

    else:
        # Classical Tabs with CSS wrapping (no horizontal scrolling!)
        all_titles = [m["title"] for m in NAV_MODULES]
        curr_idx = [m["id"] for m in NAV_MODULES].index(curr_id) if curr_id in [m["id"] for m in NAV_MODULES] else 0
        # Return special signal so app.py can render the tab container or handle directly
        return "USE_ST_TABS"

    # Module Banner Card (Breadcrumb & Description)
    active_m = MODULE_MAP.get(st.session_state["active_nav_id"], NAV_MODULES[0])
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 18px; margin-top: 6px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.78rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.05em;">{active_m['category']}</span>
                <h3 style="margin: 2px 0 4px 0; font-size: 1.15rem; color: #0f172a; font-weight: 700;">{active_m['title']}</h3>
                <p style="margin: 0; font-size: 0.86rem; color: #64748b;">{active_m['desc']}</p>
            </div>
            <div>
                <span style="background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 12px;">{active_m['badge']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.session_state["active_nav_id"]

def render_sidebar_quick_nav():
    """
    Renders a quick module jumper in the sidebar.
    """
    if "active_nav_id" not in st.session_state:
        st.session_state["active_nav_id"] = "player_stats"

    curr_id = st.session_state["active_nav_id"]
    all_titles = [m["title"] for m in NAV_MODULES]
    curr_title = MODULE_MAP.get(curr_id, NAV_MODULES[0])["title"]

    st.sidebar.markdown("### 🧭 Menu Navigasi Modul")

    # Sync sidebar selectbox state with active module before rendering
    if "sidebar_nav_selectbox" not in st.session_state or st.session_state["sidebar_nav_selectbox"] != curr_title:
        st.session_state["sidebar_nav_selectbox"] = curr_title

    def on_sidebar_nav_change():
        sel_title = st.session_state.get("sidebar_nav_selectbox")
        if sel_title and sel_title in TITLE_TO_ID:
            chosen_id = TITLE_TO_ID[sel_title]
            if chosen_id != st.session_state.get("active_nav_id"):
                st.session_state["active_nav_id"] = chosen_id
                st.session_state["pills_all_modules"] = sel_title
                chosen_cat = MODULE_MAP[chosen_id]["category"]
                st.session_state["nav_selected_category"] = chosen_cat
                st.session_state["widget_cat_selector"] = chosen_cat
                for c in CATEGORIES:
                    sub_k = f"widget_sub_mod_{c}"
                    if sub_k in st.session_state:
                        del st.session_state[sub_k]

    st.sidebar.selectbox(
        "Pilih Modul Dashboard:",
        options=all_titles,
        key="sidebar_nav_selectbox",
        on_change=on_sidebar_nav_change
    )

    st.sidebar.markdown("---")

def render_module_footer_pager():
    """
    Renders Previous / Next buttons at the bottom of the active module.
    """
    curr_id = st.session_state.get("active_nav_id", "player_stats")
    id_list = [m["id"] for m in NAV_MODULES]
    if curr_id not in id_list:
        return

    curr_idx = id_list.index(curr_id)
    prev_idx = curr_idx - 1 if curr_idx > 0 else len(id_list) - 1
    next_idx = curr_idx + 1 if curr_idx < len(id_list) - 1 else 0

    prev_m = NAV_MODULES[prev_idx]
    next_m = NAV_MODULES[next_idx]

    st.markdown("---")
    c_prev, c_space, c_next = st.columns([3, 2, 3])
    with c_prev:
        if st.button(f"⬅️ {prev_m['title']}", use_container_width=True, key="pager_prev_btn"):
            st.session_state["active_nav_id"] = prev_m["id"]
            st.rerun()
    with c_next:
        if st.button(f"{next_m['title']} ➡️", use_container_width=True, key="pager_next_btn"):
            st.session_state["active_nav_id"] = next_m["id"]
            st.rerun()
