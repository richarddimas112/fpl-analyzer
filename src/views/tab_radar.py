"""
Radar and 2-Player Comparison View Tab & Instant Action Drawer Module.
High-tactile Head-to-Head comparison, Plotly radar chart, advantage breakdown,
instant action modal dialog, and upcoming fixtures comparison.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import percentileofscore

SHORT_CLUB_NAMES = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Bournemouth': 'BOU', 'Brentford': 'BRE',
    'Brighton': 'BHA', 'Brighton and Hove Albion': 'BHA', 'Chelsea': 'CHE', 'Crystal Palace': 'CRY',
    'Everton': 'EVE', 'Fulham': 'FUL', 'Ipswich': 'IPS', 'Ipswich Town': 'IPS', 'Leicester': 'LEI',
    'Leicester City': 'LEI', 'Liverpool': 'LIV', 'Man City': 'MCI', 'Manchester City': 'MCI',
    'Man Utd': 'MUN', 'Manchester Utd': 'MUN', 'Manchester United': 'MUN', 'Newcastle': 'NEW',
    'Newcastle United': 'NEW', "Nott'm Forest": 'NFO', 'Nottingham Forest': 'NFO',
    'Southampton': 'SOU', 'Spurs': 'TOT', 'Tottenham': 'TOT', 'Tottenham Hotspur': 'TOT',
    'West Ham': 'WHU', 'West Ham United': 'WHU', 'Wolves': 'WOL', 'Wolverhampton': 'WOL',
    'Wolverhampton Wanderers': 'WOL'
}

FDR_PALETTE = {
    1: {'bg': '#15803d', 'text': '#ffffff'},
    2: {'bg': '#10b981', 'text': '#ffffff'},
    3: {'bg': '#64748b', 'text': '#ffffff'},
    4: {'bg': '#f59e0b', 'text': '#ffffff'},
    5: {'bg': '#ef4444', 'text': '#ffffff'},
}

METRIC_META = {
    'xG': {'label': 'xG (Exp. Goals)', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.2f}'},
    'xA': {'label': 'xA (Exp. Assists)', 'category': 'Kreativitas', 'higher_better': True, 'fmt': '{:.2f}'},
    'xGI': {'label': 'xGI (Goal Involv.)', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.2f}'},
    'xG per 90': {'label': 'xG / 90 Mins', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.2f}'},
    'xA per 90': {'label': 'xA / 90 Mins', 'category': 'Kreativitas', 'higher_better': True, 'fmt': '{:.2f}'},
    'xGI per 90': {'label': 'xGI / 90 Mins', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.2f}'},
    'Gol': {'label': 'Total Gol', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.0f}'},
    'Asis': {'label': 'Total Asis', 'category': 'Kreativitas', 'higher_better': True, 'fmt': '{:.0f}'},
    'Total Poin': {'label': 'Total Poin', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.0f} pts'},
    'xPoin': {'label': 'Prediksi xPoin GW', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.2f} pts'},
    'xPoin (Option B)': {'label': 'xPoin (Opt B)', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.2f} pts'},
    'Form': {'label': 'Form Terkini', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.2f}'},
    'Harga (£m)': {'label': 'Harga (£m)', 'category': 'Value', 'higher_better': False, 'fmt': '£{:.1f}m'},
    'Avg Mins (L5M)': {'label': 'Avg Mins (L5M)', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.1f} m'},
    'Menit Bermain': {'label': 'Menit Bermain', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.0f} m'},
    'ICT Index': {'label': 'ICT Index', 'category': 'Kreativitas', 'higher_better': True, 'fmt': '{:.1f}'},
    'Influence': {'label': 'Influence Score', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.1f}'},
    'Creativity': {'label': 'Creativity Score', 'category': 'Kreativitas', 'higher_better': True, 'fmt': '{:.1f}'},
    'Threat': {'label': 'Threat Score', 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.1f}'},
    'Tackles': {'label': 'Tackles', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.0f}'},
    'Tackles per 90': {'label': 'Tackles / 90 Mins', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.2f}'},
    'Clearances': {'label': 'Clearances & Blocks', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.0f}'},
    'Recoveries': {'label': 'Ball Recoveries', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.0f}'},
    'Defensive Contribution': {'label': 'Defensive Contrib', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.1f}'},
    'BPS': {'label': 'BPS Score', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.0f}'},
    'Bonus Poin': {'label': 'Bonus Poin', 'category': 'Poin & Form', 'higher_better': True, 'fmt': '{:.0f}'},
    'Clean Sheet': {'label': 'Clean Sheet', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.0f}'},
    'Saves': {'label': 'Saves', 'category': 'Defensif', 'higher_better': True, 'fmt': '{:.0f}'},
    '% Ownership': {'label': '% Kepemilikan', 'category': 'Value', 'higher_better': True, 'fmt': '{:.1f}%'},
    'Poin per £m': {'label': 'Poin / £m', 'category': 'Value', 'higher_better': True, 'fmt': '{:.2f}'},
    'xPoin per £m': {'label': 'xPoin / £m', 'category': 'Value', 'higher_better': True, 'fmt': '{:.2f}'},
}

POPULAR_DUELS = [
    ("Haaland", "M.Salah", "👑 Battle of Premiums (Haaland vs Salah)"),
    ("Saka", "Palmer", "🪄 Creative Maestros (Saka vs Palmer)"),
    ("B.Fernandes", "Cherki", "🎯 Talisman Showdown (Fernandes vs Cherki)"),
    ("Alexander-Arnold", "Pedro Porro", "🛡️ Attacking Fullbacks Duel (Trent vs Porro)"),
    ("Raya", "Pickford", "🧤 Golden Glove Contenders (Raya vs Pickford)"),
    ("Joao Pedro", "Wood", "⚡ Budget Forward Kings (Pedro vs Wood)")
]

def player_label(p):
    if not p or not isinstance(p, dict):
        return ""
    name = p.get('Nama Pemain', '')
    klub = p.get('Klub', '')
    pos = p.get('Posisi', '')
    cost = p.get('Harga (£m)', 0.0)
    pts = p.get('Total Poin', 0)
    xpts = p.get('xPoin', 0.0)
    return f"{name} ({klub} - {pos}) · £{cost:.1f}m · {pts} pts (xP: {xpts:.1f})"

@st.dialog("⚔️ Head-to-Head Player Comparison Drawer (Aksi Instan)", width="large")
def show_h2h_comparison_dialog(df, fpl_data, teams_dict, fdr_summary=None):
    """
    Renders the instant modal dialog drawer for comparing 2 players.
    """
    render_player_comparison_radar_tab(df, fpl_data, teams_dict, fdr_summary=fdr_summary, is_drawer_mode=True)

def launch_h2h_for_player(target_player_id, df, fpl_data, teams_dict, fdr_summary=None):
    """
    Pre-configures target_player_id as Pemain 1, picks a contrasting Pemain 2,
    and opens the instant modal dialog drawer.
    """
    if df.empty:
        return
    player_records = df.to_dict('records')
    player_ids = [p['id'] for p in player_records]
    if target_player_id in player_ids:
        st.session_state['radar_p1_id'] = target_player_id
        st.session_state['radar_select_p1_drw'] = target_player_id
        st.session_state['radar_select_p1_tab'] = target_player_id
        
        # Ensure Pemain 2 is different from Pemain 1
        p2_curr = st.session_state.get('radar_p2_id')
        if p2_curr == target_player_id or not p2_curr or p2_curr not in player_ids:
            alt_candidates = [p['id'] for p in player_records if p['id'] != target_player_id]
            if alt_candidates:
                top_p2 = alt_candidates[0]
                for p in player_records:
                    if p['id'] != target_player_id and p.get('Nama Pemain') in ['Haaland', 'M.Salah', 'Saka', 'Palmer']:
                        top_p2 = p['id']
                        break
                st.session_state['radar_p2_id'] = top_p2
                st.session_state['radar_select_p2_drw'] = top_p2
                st.session_state['radar_select_p2_tab'] = top_p2

    show_h2h_comparison_dialog(df, fpl_data, teams_dict, fdr_summary=fdr_summary)

def render_instant_h2h_comparison_bar(df, fpl_data, teams_dict, fdr_summary=None, key_prefix="h2h_bar"):
    """
    Renders an instant action bar in Player Stats (or elsewhere).
    Offers 1-click popular duel chips, player pickers, a primary drawer dialog launcher, and an inline drawer expander.
    """
    if df.empty or len(df) < 2:
        return

    player_records = df.to_dict('records')
    player_by_id = {p['id']: p for p in player_records}
    player_ids = [p['id'] for p in player_records]

    # Initialize session state keys
    if 'radar_p1_id' not in st.session_state or st.session_state['radar_p1_id'] not in player_by_id:
        st.session_state['radar_p1_id'] = player_ids[0]
    if 'radar_p2_id' not in st.session_state or st.session_state['radar_p2_id'] not in player_by_id:
        st.session_state['radar_p2_id'] = player_ids[1] if len(player_ids) > 1 else player_ids[0]

    # Instant Action Bar Container
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border: 1.5px solid #cbd5e1; border-left: 5px solid #2563eb; border-radius: 12px; padding: 16px 20px; margin-top: 14px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.74rem; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em;">Aksi Instan Analisis Pemain</span>
                <h4 style="margin: 2px 0 4px 0; color: #0f172a; font-weight: 800; font-size: 1.15rem;">⚔️ Head-to-Head Player Comparison Drawer</h4>
                <p style="margin: 0; font-size: 0.84rem; color: #64748b;">
                    Bandingkan 2 pemain secara instan lewat grafik radar interaktif, keunggulan metrik per dimensi, dan perbandingan 5 jadwal laga mendatang.
                </p>
            </div>
            <div>
                <span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; font-size: 0.76rem; font-weight: 700; padding: 4px 10px; border-radius: 20px;">
                    ⚡ Akses Langsung
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Quick Duel Selector Pills
    matched_duels = []
    for t1, t2, tag in POPULAR_DUELS:
        p_m1 = next((p for p in player_records if t1.lower() in p.get('Nama Pemain', '').lower()), None)
        p_m2 = next((p for p in player_records if t2.lower() in p.get('Nama Pemain', '').lower()), None)
        if p_m1 and p_m2 and p_m1['id'] != p_m2['id']:
            matched_duels.append((p_m1, p_m2, tag))

    if matched_duels:
        st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #475569; margin-bottom: 6px;'>⚡ Duel Populer (Klik untuk Memuat Instan):</p>", unsafe_allow_html=True)
        duel_cols = st.columns(min(len(matched_duels), 3))
        for i, (pm1, pm2, tag) in enumerate(matched_duels):
            col = duel_cols[i % len(duel_cols)]
            with col:
                btn_title = f"{pm1.get('Nama Pemain')} vs {pm2.get('Nama Pemain')}"
                if st.button(btn_title, use_container_width=True, key=f"{key_prefix}_duel_{i}_{pm1['id']}_{pm2['id']}"):
                    st.session_state['radar_p1_id'] = pm1['id']
                    st.session_state['radar_p2_id'] = pm2['id']
                    st.session_state['radar_select_p1'] = pm1['id']
                    st.session_state['radar_select_p2'] = pm2['id']
                    show_h2h_comparison_dialog(df, fpl_data, teams_dict, fdr_summary=fdr_summary)

    # 2. Player Selector Controls & Launcher
    c_p1, c_swap, c_p2, c_btn = st.columns([3, 0.8, 3, 2.2])

    p1_idx = player_ids.index(st.session_state['radar_p1_id']) if st.session_state['radar_p1_id'] in player_ids else 0
    p2_idx = player_ids.index(st.session_state['radar_p2_id']) if st.session_state['radar_p2_id'] in player_ids else 1

    with c_p1:
        sel_p1 = st.selectbox(
            "🟢 Pemain 1:",
            options=player_ids,
            index=p1_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key=f"{key_prefix}_sel_p1"
        )
        st.session_state['radar_p1_id'] = sel_p1
        st.session_state['radar_select_p1'] = sel_p1

    with c_swap:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("↔️", help="Tukar Posisi Pemain 1 & 2", key=f"{key_prefix}_swap_btn", use_container_width=True):
            tmp = st.session_state['radar_p1_id']
            st.session_state['radar_p1_id'] = st.session_state['radar_p2_id']
            st.session_state['radar_p2_id'] = tmp
            st.session_state['radar_select_p1'] = st.session_state['radar_p1_id']
            st.session_state['radar_select_p2'] = st.session_state['radar_p2_id']
            st.rerun()

    with c_p2:
        sel_p2 = st.selectbox(
            "🔵 Pemain 2:",
            options=player_ids,
            index=p2_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key=f"{key_prefix}_sel_p2"
        )
        st.session_state['radar_p2_id'] = sel_p2
        st.session_state['radar_select_p2'] = sel_p2

    with c_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⚔️ Buka Drawer Komparasi", type="primary", use_container_width=True, key=f"{key_prefix}_open_dlg_btn"):
            show_h2h_comparison_dialog(df, fpl_data, teams_dict, fdr_summary=fdr_summary)

    # 3. Inline Drawer Expander (for users who prefer inline view without modal popup)
    with st.expander("🔍 Atau Periksa Drawer Komparasi Langsung di Sini (Inline View)", expanded=False):
        render_player_comparison_radar_tab(df, fpl_data, teams_dict, fdr_summary=fdr_summary, is_drawer_mode=False)


def render_player_comparison_radar_tab(df, fpl_data, teams_dict, fdr_summary=None, is_drawer_mode=False):
    """
    Renders the dedicated Player Comparison & Radar Chart tab or Drawer content.
    Compares 2 players across various metrics with percentiles, Min-Max, or Raw values,
    including advantage breakdown, win verdicts, and 5-match fixture comparisons.
    """
    if df.empty or len(df) < 2:
        st.warning("Data pemain tidak mencukupi untuk melakukan komparasi.")
        return

    # Build player lookup list
    player_records = df.to_dict('records')
    player_by_id = {p['id']: p for p in player_records}
    player_ids = [p['id'] for p in player_records]

    # Pre-select interesting defaults (e.g. Haaland vs Salah)
    p1_default_id = player_records[0]['id']
    p2_default_id = player_records[1]['id'] if len(player_records) > 1 else player_records[0]['id']
    
    for p in player_records:
        name = p.get('Nama Pemain', '')
        if name in ['Haaland', 'Erling Haaland']:
            p1_default_id = p['id']
        elif name in ['M.Salah', 'Salah', 'Mohamed Salah']:
            p2_default_id = p['id']

    if p1_default_id == p2_default_id and len(player_records) > 1:
        alt_p = [p['id'] for p in player_records if p['id'] != p1_default_id]
        if alt_p:
            p2_default_id = alt_p[0]

    # Ensure valid IDs in session state
    if 'radar_p1_id' not in st.session_state or st.session_state['radar_p1_id'] not in player_by_id:
        st.session_state['radar_p1_id'] = p1_default_id
    if 'radar_p2_id' not in st.session_state or st.session_state['radar_p2_id'] not in player_by_id:
        st.session_state['radar_p2_id'] = p2_default_id

    # 1. TOP HEADER / TITLE (only in tab view)
    if not is_drawer_mode:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 20px 24px; color: #ffffff; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);">
            <span style="font-size: 0.76rem; font-weight: 700; color: #38ef7d; text-transform: uppercase; letter-spacing: 0.08em;">Head-to-Head Player Comparison & Radar Analysis</span>
            <h2 style="margin: 4px 0 6px 0; font-size: 1.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">⚔️ Komparasi Head-to-Head & Radar Chart 2 Pemain</h2>
            <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                Analisis perbandingan komprehensif antara 2 aset FPL: metrik ofensif, playmaking, efisiensi harga, 
                distribusi persentil liga, skor keunggulan, serta jadwal pertandingan mendatang.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick duel recommendation pills
        matched_pairs = []
        for t1, t2, tag in POPULAR_DUELS:
            p_match1 = next((p for p in player_records if t1.lower() in p.get('Nama Pemain', '').lower()), None)
            p_match2 = next((p for p in player_records if t2.lower() in p.get('Nama Pemain', '').lower()), None)
            if p_match1 is not None and p_match2 is not None and p_match1['id'] != p_match2['id']:
                matched_pairs.append((p_match1, p_match2, tag))

        if matched_pairs:
            st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #475569; margin-bottom: 6px;'>⚡ Rekomendasi Duel Populer (Klik untuk Memuat Langsung):</p>", unsafe_allow_html=True)
            cols = st.columns(min(len(matched_pairs), 3))
            for i, (pm1, pm2, tag) in enumerate(matched_pairs):
                col = cols[i % len(cols)]
                with col:
                    btn_lbl = f"{pm1.get('Nama Pemain')} vs {pm2.get('Nama Pemain')}"
                    if st.button(btn_lbl, use_container_width=True, key=f"rt_btn_{i}_{pm1['id']}_{pm2['id']}_{'drw' if is_drawer_mode else 'tab'}"):
                        st.session_state['radar_p1_id'] = pm1['id']
                        st.session_state['radar_p2_id'] = pm2['id']
                        st.session_state['radar_select_p1'] = pm1['id']
                        st.session_state['radar_select_p2'] = pm2['id']
                        st.rerun()

    # 2. PLAYER SELECTION & QUICK SWAP CONTROLS
    c_s1, c_sw, c_s2 = st.columns([4.5, 1, 4.5])

    p1_idx = player_ids.index(st.session_state['radar_p1_id']) if st.session_state['radar_p1_id'] in player_ids else 0
    p2_idx = player_ids.index(st.session_state['radar_p2_id']) if st.session_state['radar_p2_id'] in player_ids else 1

    with c_s1:
        st.markdown("""
        <div style="padding: 8px 12px; background-color: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; margin-bottom: 6px;">
            <strong style="color: #166534; font-size: 0.92rem;">🟢 Pemain 1 (Warna Hijau Emerald)</strong>
        </div>
        """, unsafe_allow_html=True)
        selected_p1_id = st.selectbox(
            "Pilih Pemain Pertama:",
            options=player_ids,
            index=p1_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key=f"radar_select_p1_{'drw' if is_drawer_mode else 'tab'}",
            label_visibility="collapsed"
        )
        st.session_state['radar_p1_id'] = selected_p1_id

    with c_sw:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        if st.button("↔️", help="Balik Posisi Pemain 1 & 2", key=f"swap_p_btn_{'drw' if is_drawer_mode else 'tab'}", use_container_width=True):
            tmp = st.session_state['radar_p1_id']
            st.session_state['radar_p1_id'] = st.session_state['radar_p2_id']
            st.session_state['radar_p2_id'] = tmp
            st.rerun()

    with c_s2:
        st.markdown("""
        <div style="padding: 8px 12px; background-color: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; margin-bottom: 6px;">
            <strong style="color: #1e40af; font-size: 0.92rem;">🔵 Pemain 2 (Warna Biru Royal)</strong>
        </div>
        """, unsafe_allow_html=True)
        selected_p2_id = st.selectbox(
            "Pilih Pemain Kedua:",
            options=player_ids,
            index=p2_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key=f"radar_select_p2_{'drw' if is_drawer_mode else 'tab'}",
            label_visibility="collapsed"
        )
        st.session_state['radar_p2_id'] = selected_p2_id

    p1 = player_by_id.get(selected_p1_id, player_records[0])
    p2 = player_by_id.get(selected_p2_id, player_records[1] if len(player_records) > 1 else player_records[0])

    # 3. SIDE-BY-SIDE PROFILE CARDS
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    prof1, prof2 = st.columns(2)

    with prof1:
        p1_name = p1.get('Nama Pemain')
        p1_club = p1.get('Klub')
        p1_pos = p1.get('Posisi')
        p1_cost = float(p1.get('Harga (£m)', 0.0) or 0.0)
        p1_pts = int(p1.get('Total Poin', 0) or 0)
        p1_xpts = float(p1.get('xPoin', 0.0) or 0.0)
        p1_form = float(p1.get('Form', 0.0) or 0.0)
        p1_opp = p1.get('Lawan GW Berikutnya', '-')

        st.markdown(f"""
        <div style="padding: 16px; background-color: #ffffff; border: 1.5px solid #86efac; border-left: 6px solid #10b981; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem; font-weight: 800;">{p1_name}</h3>
                    <p style="margin: 2px 0 6px 0; font-weight: 700; color: #10b981; font-size: 0.88rem;">{p1_club} · {p1_pos}</p>
                </div>
                <div style="background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 20px; font-weight: 800; font-size: 0.78rem;">
                    🟢 Pemain A
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; text-align: center; border-top: 1px solid #f1f5f9; padding-top: 10px;">
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">HARGA</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">£{p1_cost:.1f}m</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">TOTAL POIN</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">{p1_pts} pts</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">xPOIN</div><div style="font-size: 1.05rem; font-weight: 800; color: #10b981;">{p1_xpts:.2f}</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">FORM</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">{p1_form:.2f}</div></div>
            </div>
            <div style="margin-top: 8px; font-size: 0.78rem; color: #475569; background: #f8fafc; padding: 4px 8px; border-radius: 6px;">
                📅 Lawan Berikutnya: <strong>{p1_opp}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prof2:
        p2_name = p2.get('Nama Pemain')
        p2_club = p2.get('Klub')
        p2_pos = p2.get('Posisi')
        p2_cost = float(p2.get('Harga (£m)', 0.0) or 0.0)
        p2_pts = int(p2.get('Total Poin', 0) or 0)
        p2_xpts = float(p2.get('xPoin', 0.0) or 0.0)
        p2_form = float(p2.get('Form', 0.0) or 0.0)
        p2_opp = p2.get('Lawan GW Berikutnya', '-')

        st.markdown(f"""
        <div style="padding: 16px; background-color: #ffffff; border: 1.5px solid #93c5fd; border-left: 6px solid #2563eb; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem; font-weight: 800;">{p2_name}</h3>
                    <p style="margin: 2px 0 6px 0; font-weight: 700; color: #2563eb; font-size: 0.88rem;">{p2_club} · {p2_pos}</p>
                </div>
                <div style="background: #eff6ff; color: #1d4ed8; padding: 3px 10px; border-radius: 20px; font-weight: 800; font-size: 0.78rem;">
                    🔵 Pemain B
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; text-align: center; border-top: 1px solid #f1f5f9; padding-top: 10px;">
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">HARGA</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">£{p2_cost:.1f}m</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">TOTAL POIN</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">{p2_pts} pts</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">xPOIN</div><div style="font-size: 1.05rem; font-weight: 800; color: #2563eb;">{p2_xpts:.2f}</div></div>
                <div><div style="font-size: 0.7rem; color: #64748b; font-weight: 700;">FORM</div><div style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">{p2_form:.2f}</div></div>
            </div>
            <div style="margin-top: 8px; font-size: 0.78rem; color: #475569; background: #f8fafc; padding: 4px 8px; border-radius: 6px;">
                📅 Lawan Berikutnya: <strong>{p2_opp}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. RADAR CONFIGURATION CONTROLS
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🕸️ Grafik Radar Komparasi Statistik")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        radar_preset = st.selectbox(
            "Pilih Preset Metrik Radar:",
            options=[
                "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, Threat, Creativity, xPoin)",
                "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Influence, Creativity, Threat, BPS, Poin/£m)",
                "🛡️ Defensif & Kiper (Clean Sheet, Defensive Contribution, Tackles, Clearances, Saves, BPS)",
                "🪄 Kreativitas & Ancaman Serangan (Creativity, Threat, Influence, xG, xA, xGI/90, ICT Index)",
                "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, Influence, BPS, % Ownership)",
                "🛠️ Kustom (Pilih Metrik Bebas)"
            ],
            key=f"radar_preset_choice_{'drw' if is_drawer_mode else 'tab'}"
        )
    with ctrl2:
        norm_mode = st.selectbox(
            "Skala Nilai Radar:",
            options=[
                "Persentil Liga (0 - 100%)",
                "Min-Max Normalisasi (0 - 100)",
                "Nilai Aktual (Raw Values)"
            ],
            help="Skala Persentil direkomendasikan agar metrik dengan skala berbeda dapat dibandingkan secara proporsional.",
            key=f"radar_norm_mode_{'drw' if is_drawer_mode else 'tab'}"
        )
    with ctrl3:
        cohort_mode_code = st.selectbox(
            "Basis Kohort Persentil:",
            options=["all", "pos1", "pos2"],
            format_func=lambda c: (
                "Seluruh Liga (All Players)" if c == "all" else
                f"Sesama {p1.get('Posisi')} ({p1.get('Nama Pemain')})" if c == "pos1" else
                f"Sesama {p2.get('Posisi')} ({p2.get('Nama Pemain')})"
            ),
            key=f"radar_cohort_mode_{'drw' if is_drawer_mode else 'tab'}"
        )

    preset_metric_map = {
        "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, Threat, Creativity, xPoin)": [
            'xG', 'xA', 'xGI per 90', 'Gol', 'Asis', 'Threat', 'Creativity', 'xPoin'
        ],
        "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Influence, Creativity, Threat, BPS, Poin/£m)": [
            'Total Poin', 'xPoin', 'Form', 'Influence', 'Creativity', 'Threat', 'BPS', 'Poin per £m'
        ],
        "🛡️ Defensif & Kiper (Clean Sheet, Defensive Contribution, Tackles, Clearances, Saves, BPS)": [
            'Clean Sheet', 'Defensive Contribution', 'Tackles', 'Clearances', 'Saves', 'BPS'
        ],
        "🪄 Kreativitas & Ancaman Serangan (Creativity, Threat, Influence, xG, xA, xGI/90, ICT Index)": [
            'Creativity', 'Threat', 'Influence', 'xG', 'xA', 'xGI per 90', 'ICT Index'
        ],
        "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, Influence, BPS, % Ownership)": [
            'Poin per £m', 'xPoin per £m', 'Form', 'Avg Mins (L5M)', 'Influence', 'BPS', '% Ownership'
        ]
    }

    all_available_metrics = list(METRIC_META.keys())

    if "Kustom" in radar_preset:
        active_metric_keys = st.multiselect(
            "Pilih minimal 3 metrik statistik:",
            options=all_available_metrics,
            default=['Total Poin', 'xPoin', 'Influence', 'Creativity', 'Threat', 'Defensive Contribution', 'BPS', 'xGI per 90'],
            key=f"radar_custom_metrics_{'drw' if is_drawer_mode else 'tab'}"
        )
    else:
        active_metric_keys = preset_metric_map.get(radar_preset, preset_metric_map["🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, Threat, Creativity, xPoin)"])

    if len(active_metric_keys) < 3:
        st.warning("Pilih minimal 3 metrik agar poligon radar chart dapat divisualisasikan dengan proporsional.")
        return

    # Cohort filtering
    if cohort_mode_code == "pos1":
        cohort_df = df[df['Posisi'] == p1.get('Posisi')]
    elif cohort_mode_code == "pos2":
        cohort_df = df[df['Posisi'] == p2.get('Posisi')]
    else:
        cohort_df = df

    active_cohort = cohort_df[cohort_df['Menit Bermain'] > 0]
    if not active_cohort.empty:
        cohort_df = active_cohort
    elif cohort_df.empty:
        cohort_df = df

    theta_labels = []
    r_p1 = []
    r_p2 = []
    customdata_p1 = []
    customdata_p2 = []
    h2h_rows = []

    p1_wins = 0
    p2_wins = 0
    ties = 0

    category_tallies = {
        'Ofensif': {'p1': 0, 'p2': 0},
        'Kreativitas': {'p1': 0, 'p2': 0},
        'Poin & Form': {'p1': 0, 'p2': 0},
        'Defensif': {'p1': 0, 'p2': 0},
        'Value': {'p1': 0, 'p2': 0},
    }

    for m_key in active_metric_keys:
        meta = METRIC_META.get(m_key, {'label': m_key, 'category': 'Ofensif', 'higher_better': True, 'fmt': '{:.2f}'})
        lbl = meta['label']
        cat = meta.get('category', 'Ofensif')
        higher_better = meta['higher_better']
        fmt = meta['fmt']

        val1 = float(p1.get(m_key, 0.0) or 0.0)
        val2 = float(p2.get(m_key, 0.0) or 0.0)

        try:
            val1_str = fmt.format(val1)
        except (ValueError, TypeError):
            val1_str = str(val1)

        try:
            val2_str = fmt.format(val2)
        except (ValueError, TypeError):
            val2_str = str(val2)

        # Normalization
        if "Persentil" in norm_mode:
            c_vals = cohort_df[m_key].dropna().astype(float).values if m_key in cohort_df.columns else np.array([val1, val2])
            if len(c_vals) > 0:
                pct1 = float(percentileofscore(c_vals, val1, kind='rank'))
                pct2 = float(percentileofscore(c_vals, val2, kind='rank'))
                if not higher_better:
                    pct1 = 100.0 - pct1
                    pct2 = 100.0 - pct2
            else:
                pct1, pct2 = 50.0, 50.0

            score1 = round(pct1, 1)
            score2 = round(pct2, 1)

        elif "Min-Max" in norm_mode:
            c_vals = cohort_df[m_key].dropna().astype(float).values if m_key in cohort_df.columns else np.array([val1, val2])
            if len(c_vals) > 0:
                c_min, c_max = float(np.min(c_vals)), float(np.max(c_vals))
                if c_max > c_min:
                    s1 = ((val1 - c_min) / (c_max - c_min)) * 100.0
                    s2 = ((val2 - c_min) / (c_max - c_min)) * 100.0
                    if not higher_better:
                        s1 = 100.0 - s1
                        s2 = 100.0 - s2
                else:
                    s1, s2 = 50.0, 50.0
            else:
                s1, s2 = 50.0, 50.0

            score1 = round(s1, 1)
            score2 = round(s2, 1)

        else: # Raw Values
            score1 = val1
            score2 = val2

        theta_labels.append(lbl)
        r_p1.append(score1)
        r_p2.append(score2)
        customdata_p1.append(val1_str)
        customdata_p2.append(val2_str)

        # Judgement
        if higher_better:
            if val1 > val2:
                winner_badge = f"🟢 {p1.get('Nama Pemain')}"
                p1_wins += 1
                if cat in category_tallies:
                    category_tallies[cat]['p1'] += 1
            elif val2 > val1:
                winner_badge = f"🔵 {p2.get('Nama Pemain')}"
                p2_wins += 1
                if cat in category_tallies:
                    category_tallies[cat]['p2'] += 1
            else:
                winner_badge = "🤝 Seimbang"
                ties += 1
        else: # Price (lower is better)
            if val1 < val2:
                winner_badge = f"🟢 {p1.get('Nama Pemain')}"
                p1_wins += 1
                if cat in category_tallies:
                    category_tallies[cat]['p1'] += 1
            elif val2 < val1:
                winner_badge = f"🔵 {p2.get('Nama Pemain')}"
                p2_wins += 1
                if cat in category_tallies:
                    category_tallies[cat]['p2'] += 1
            else:
                winner_badge = "🤝 Seimbang"
                ties += 1

        h2h_rows.append({
            'Kategori': cat,
            'Metrik Statistik': lbl,
            f"{p1.get('Nama Pemain')} (Aktual)": val1_str,
            f"{p2.get('Nama Pemain')} (Aktual)": val2_str,
            f"Skor {p1.get('Nama Pemain')}": score1,
            f"Skor {p2.get('Nama Pemain')}": score2,
            'Keunggulan': winner_badge
        })

    # Close the radar loop
    theta_closed = theta_labels + [theta_labels[0]]
    r_p1_closed = r_p1 + [r_p1[0]]
    r_p2_closed = r_p2 + [r_p2[0]]
    custom_p1_closed = customdata_p1 + [customdata_p1[0]]
    custom_p2_closed = customdata_p2 + [customdata_p2[0]]

    # 5. RENDER PLOTLY RADAR FIGURE
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=r_p1_closed,
        theta=theta_closed,
        fill='toself',
        name=f"🟢 {p1.get('Nama Pemain')} ({p1.get('Klub')})",
        line=dict(color='#10b981', width=3),
        fillcolor='rgba(16, 185, 129, 0.25)',
        customdata=custom_p1_closed,
        hovertemplate="<b>%{theta}</b><br>Skor: %{r}<br>Nilai Riil: %{customdata}<extra></extra>"
    ))

    fig.add_trace(go.Scatterpolar(
        r=r_p2_closed,
        theta=theta_closed,
        fill='toself',
        name=f"🔵 {p2.get('Nama Pemain')} ({p2.get('Klub')})",
        line=dict(color='#2563eb', width=3),
        fillcolor='rgba(37, 99, 235, 0.25)',
        customdata=custom_p2_closed,
        hovertemplate="<b>%{theta}</b><br>Skor: %{r}<br>Nilai Riil: %{customdata}<extra></extra>"
    ))

    radial_range = [0, 100] if ("Persentil" in norm_mode or "Min-Max" in norm_mode) else None

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=radial_range,
                tickfont=dict(size=10, color='#64748b'),
                gridcolor='#e2e8f0',
                linecolor='#cbd5e1'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#1e293b', family='Plus Jakarta Sans', weight='bold'),
                gridcolor='#e2e8f0',
                linecolor='#cbd5e1',
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='#ffffff'
        ),
        paper_bgcolor='#ffffff',
        font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Plus Jakarta Sans", weight='bold')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 6. INSTANT EDGE DETECTOR & VERDICT SUMMARY
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Analisis Keunggulan & Ringkasan Keputusan (Verdict)")

    w1, w2, w3 = st.columns(3)
    total_metrics = len(active_metric_keys)
    p1_pct = (p1_wins / total_metrics * 100) if total_metrics > 0 else 0
    p2_pct = (p2_wins / total_metrics * 100) if total_metrics > 0 else 0

    with w1:
        st.metric(
            f"Keunggulan {p1.get('Nama Pemain')}",
            f"{p1_wins} dari {total_metrics} Metrik",
            f"{p1_pct:.0f}% Keunggulan"
        )
    with w2:
        st.metric(
            f"Keunggulan {p2.get('Nama Pemain')}",
            f"{p2_wins} dari {total_metrics} Metrik",
            f"{p2_pct:.0f}% Keunggulan"
        )
    with w3:
        if p1_wins > p2_wins:
            verdict_text = f"🟢 {p1.get('Nama Pemain')} Lebih Diunggulkan"
            verdict_sub = f"Selisih +{p1_wins - p2_wins} metrik"
        elif p2_wins > p1_wins:
            verdict_text = f"🔵 {p2.get('Nama Pemain')} Lebih Diunggulkan"
            verdict_sub = f"Selisih +{p2_wins - p1_wins} metrik"
        else:
            verdict_text = "⚖️ Performa Sangat Seimbang"
            verdict_sub = f"{ties} metrik imbang"
        st.metric("Keputusan Rekomendasi", verdict_text, verdict_sub)

    # Dimension Breakdown Cards
    st.markdown("##### 🔍 Keunggulan Berdasarkan Dimensi Analisis:")
    cat_cols = st.columns(len(category_tallies))
    cat_names_ind = {
        'Ofensif': '⚽ Ofensif',
        'Kreativitas': '🪄 Kreativitas',
        'Poin & Form': '📈 Poin & Form',
        'Defensif': '🛡️ Defensif',
        'Value': '💰 Value/Harga'
    }

    for idx, (cat_key, t_vals) in enumerate(category_tallies.items()):
        with cat_cols[idx]:
            if t_vals['p1'] > t_vals['p2']:
                edge_label = f"🟢 {p1.get('Nama Pemain')}"
                edge_bg = "#dcfce7"
                edge_clr = "#15803d"
            elif t_vals['p2'] > t_vals['p1']:
                edge_label = f"🔵 {p2.get('Nama Pemain')}"
                edge_bg = "#eff6ff"
                edge_clr = "#1d4ed8"
            else:
                edge_label = "⚖️ Imbang"
                edge_bg = "#f1f5f9"
                edge_clr = "#475569"

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                <div style="font-size: 0.76rem; font-weight: 700; color: #64748b; margin-bottom: 4px;">{cat_names_ind.get(cat_key, cat_key)}</div>
                <div style="background: {edge_bg}; color: {edge_clr}; font-size: 0.8rem; font-weight: 800; padding: 3px 6px; border-radius: 6px;">
                    {edge_label}
                </div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px;">
                    {t_vals['p1']} vs {t_vals['p2']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. COMPARATIVE ADVANTAGE BREAKDOWN TABLE
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Rincian Nilai Statistik Lengkap & Selisih")
    df_h2h = pd.DataFrame(h2h_rows)
    st.dataframe(
        df_h2h,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # 8. UPCOMING 5 FIXTURES COMPARISON FOR BOTH PLAYERS
    # -------------------------------------------------------------------------
    if fdr_summary:
        p1_tid = p1.get('team')
        p2_tid = p2.get('team')

        if p1_tid in fdr_summary and p2_tid in fdr_summary:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🗓️ Komparasi 5 Jadwal Pertandingan Mendatang (Fixture Run)")
            
            p1_fxs = fdr_summary[p1_tid].get('upcoming_10', [])[:5]
            p2_fxs = fdr_summary[p2_tid].get('upcoming_10', [])[:5]

            f_cmp_rows = []
            for i in range(5):
                m1 = p1_fxs[i] if i < len(p1_fxs) else {}
                m2 = p2_fxs[i] if i < len(p2_fxs) else {}

                o1_short = SHORT_CLUB_NAMES.get(m1.get('opp_name', ''), m1.get('opp_name', '-')[:3].upper())
                o2_short = SHORT_CLUB_NAMES.get(m2.get('opp_name', ''), m2.get('opp_name', '-')[:3].upper())
                ha1 = 'H' if m1.get('is_home') == 1 else 'A'
                ha2 = 'H' if m2.get('is_home') == 1 else 'A'
                d1 = int(m1.get('fdr', 3)) if m1.get('fdr') is not None else 3
                d2 = int(m2.get('fdr', 3)) if m2.get('fdr') is not None else 3
                gw_txt = f"GW {m1.get('gw', i+1)}"

                if d1 < d2:
                    easier_badge = f"🟢 Lebih Mudah untuk {p1.get('Nama Pemain')}"
                elif d2 < d1:
                    easier_badge = f"🔵 Lebih Mudah untuk {p2.get('Nama Pemain')}"
                else:
                    easier_badge = "⚖️ Kesulitan Sama"

                f_cmp_rows.append({
                    'Gameweek': gw_txt,
                    f"{p1.get('Nama Pemain')} ({p1.get('Klub')})": f"{o1_short} ({ha1}) [FDR {d1}]",
                    f"{p2.get('Nama Pemain')} ({p2.get('Klub')})": f"{o2_short} ({ha2}) [FDR {d2}]",
                    'Jadwal Lebih Menguntungkan': easier_badge
                })

            st.dataframe(pd.DataFrame(f_cmp_rows), use_container_width=True, hide_index=True)
            
            p1_fdr3 = fdr_summary[p1_tid].get('FDR3', 3.0)
            p2_fdr3 = fdr_summary[p2_tid].get('FDR3', 3.0)
            if p1_fdr3 < p2_fdr3:
                adv_str = f"🟢 **{p1.get('Nama Pemain')} ({p1.get('Klub')})** memiliki rata-rata jadwal 3 laga mendatang lebih menguntungkan (FDR3: {p1_fdr3:.2f} vs {p2_fdr3:.2f})."
            elif p2_fdr3 < p1_fdr3:
                adv_str = f"🔵 **{p2.get('Nama Pemain')} ({p2.get('Klub')})** memiliki rata-rata jadwal 3 laga mendatang lebih menguntungkan (FDR3: {p2_fdr3:.2f} vs {p1_fdr3:.2f})."
            else:
                adv_str = f"⚖️ Kedua klub memiliki rata-rata jadwal 3 laga yang seimbang (FDR3: {p1_fdr3:.2f})."
            st.caption(f"💡 {adv_str}")
