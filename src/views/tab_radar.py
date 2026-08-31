"""
Radar and 2-Player Comparison view tab module.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import percentileofscore

def render_player_comparison_radar_tab(df, fpl_data, teams_dict):
    """
    Renders the dedicated Player Comparison & Radar Chart tab.
    Compares 2 players across various metrics with percentiles, Min-Max, or Raw values.
    """
    st.markdown("#### ⚔️ Komparasi Head-to-Head & Radar Chart 2 Pemain")
    st.write("Bandingkan profil performa, statistik lanjutan, metrik ofensif/defensif, dan proyeksi poin antara 2 pemain FPL pilihan Anda.")

    if df.empty or len(df) < 2:
        st.warning("Data pemain tidak mencukupi untuk melakukan komparasi.")
        return

    # Build player lookup list
    player_records = df.to_dict('records')
    player_by_id = {p['id']: p for p in player_records}
    player_ids = [p['id'] for p in player_records]

    # Pre-select interesting defaults (e.g. Haaland vs Salah, or Saka vs Palmer)
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

    def player_label(p):
        if not p or not isinstance(p, dict):
            return ""
        name = p.get('Nama Pemain', '')
        klub = p.get('Klub', '')
        pos = p.get('Posisi', '')
        cost = p.get('Harga (£m)', 0.0)
        pts = p.get('Total Poin', 0)
        return f"{name} ({klub} - {pos}) · £{cost:.1f}m · {pts} pts"

    # Pastikan session state terisi dengan ID yang valid
    if 'radar_p1_id' not in st.session_state or st.session_state['radar_p1_id'] not in player_by_id:
        st.session_state['radar_p1_id'] = p1_default_id
    if 'radar_p2_id' not in st.session_state or st.session_state['radar_p2_id'] not in player_by_id:
        st.session_state['radar_p2_id'] = p2_default_id

    # Sinkronisasi key widget selectbox jika belum ada
    if 'radar_select_p1' not in st.session_state or st.session_state['radar_select_p1'] not in player_by_id:
        st.session_state['radar_select_p1'] = st.session_state['radar_p1_id']
    if 'radar_select_p2' not in st.session_state or st.session_state['radar_select_p2'] not in player_by_id:
        st.session_state['radar_select_p2'] = st.session_state['radar_p2_id']

    # --- TOP 6 POPULAR MATCHUPS / QUICK DUEL SELECTOR ---
    st.markdown("##### ⚡ Rekomendasi Duel Populer (Klik untuk Membandingkan Langsung):")
    
    # Cari pasangan pemain yang benar-benar ada di dataset
    popular_pairs = [
        ("Haaland", "M.Salah", "👑 Battle of Premiums (Haaland vs Salah)"),
        ("Saka", "Palmer", "🪄 Creative Maestros (Saka vs Palmer)"),
        ("B.Fernandes", "Cherki", "🎯 Talisman Showdown (Fernandes vs Cherki)"),
        ("Alexander-Arnold", "Pedro Porro", "🛡️ Attacking Fullbacks Duel"),
        ("Raya", "Pickford", "🧤 Golden Glove Contenders (Raya vs Pickford)"),
        ("Joao Pedro", "Wood", "⚡ Budget Forward Kings (Pedro vs Wood)")
    ]

    matched_pairs = []
    for t1, t2, tag in popular_pairs:
        p_match1 = next((p for p in player_records if t1.lower() in p.get('Nama Pemain', '').lower()), None)
        p_match2 = next((p for p in player_records if t2.lower() in p.get('Nama Pemain', '').lower()), None)
        if p_match1 is not None and p_match2 is not None and p_match1['id'] != p_match2['id']:
            matched_pairs.append((p_match1, p_match2, tag, t1, t2))

    # Jika ada pasangan yang cocok, tampilkan grid tombol cepat
    if matched_pairs:
        cols = st.columns(min(len(matched_pairs), 3))
        for i, (pm1, pm2, tag, target1, target2) in enumerate(matched_pairs):
            col = cols[i % len(cols)]
            with col:
                btn_lbl = f"{pm1.get('Nama Pemain')} vs {pm2.get('Nama Pemain')}"
                if st.button(btn_lbl, use_container_width=True, key=f"rt_btn_{i}_{pm1['id']}_{pm2['id']}"):
                    st.session_state['radar_p1_id'] = pm1['id']
                    st.session_state['radar_p2_id'] = pm2['id']
                    st.session_state['radar_select_p1'] = pm1['id']
                    st.session_state['radar_select_p2'] = pm2['id']
                    st.rerun()

    def _sync_p1():
        if 'radar_select_p1' in st.session_state and st.session_state['radar_select_p1'] in player_by_id:
            st.session_state['radar_p1_id'] = st.session_state['radar_select_p1']

    def _sync_p2():
        if 'radar_select_p2' in st.session_state and st.session_state['radar_select_p2'] in player_by_id:
            st.session_state['radar_p2_id'] = st.session_state['radar_select_p2']

    p1_idx = player_ids.index(st.session_state['radar_p1_id'])
    p2_idx = player_ids.index(st.session_state['radar_p2_id'])

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        st.markdown("""
        <div style="padding: 10px; background-color: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; margin-bottom: 8px;">
            <strong style="color: #166534; font-size: 1rem;">🟢 Pemain 1 (Warna Hijau Emerald)</strong>
        </div>
        """, unsafe_allow_html=True)
        selected_p1_id = st.selectbox(
            "Pilih Pemain Pertama:",
            options=player_ids,
            index=p1_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key="radar_select_p1",
            on_change=_sync_p1
        )
        st.session_state['radar_p1_id'] = selected_p1_id

    with col_sel2:
        st.markdown("""
        <div style="padding: 10px; background-color: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 8px; margin-bottom: 8px;">
            <strong style="color: #1e40af; font-size: 1rem;">🔵 Pemain 2 (Warna Biru Royal)</strong>
        </div>
        """, unsafe_allow_html=True)
        selected_p2_id = st.selectbox(
            "Pilih Pemain Kedua:",
            options=player_ids,
            index=p2_idx,
            format_func=lambda pid: player_label(player_by_id.get(pid)),
            key="radar_select_p2",
            on_change=_sync_p2
        )
        st.session_state['radar_p2_id'] = selected_p2_id

    p1 = player_by_id.get(selected_p1_id, player_records[0])
    p2 = player_by_id.get(selected_p2_id, player_records[1] if len(player_records) > 1 else player_records[0])

    # Display Side-by-Side Quick Profiles
    st.markdown("---")
    prof1, prof2 = st.columns(2)

    with prof1:
        st.markdown(f"""
        <div style="padding: 14px; background-color: #f8fafc; border-left: 4px solid #10b981; border-radius: 8px;">
            <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem;">{p1.get('Nama Pemain')}</h3>
            <p style="margin: 2px 0 6px 0; font-weight: 600; color: #10b981;">{p1.get('Klub')} · {p1.get('Posisi')}</p>
            <p style="margin: 0 0 10px 0; font-size: 0.85rem; color: #64748b;">{p1.get('Nama Lengkap', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Harga", f"£{p1.get('Harga (£m)', 0):.1f}m")
        with m2:
            st.metric("Total Poin", f"{int(p1.get('Total Poin', 0))} pts")
        with m3:
            st.metric("Prediksi xPoin", f"{float(p1.get('xPoin', 0)):.2f} pts")
        with m4:
            st.metric("Form", f"{float(p1.get('Form', 0)):.2f}")

    with prof2:
        st.markdown(f"""
        <div style="padding: 14px; background-color: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 8px;">
            <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem;">{p2.get('Nama Pemain')}</h3>
            <p style="margin: 2px 0 6px 0; font-weight: 600; color: #3b82f6;">{p2.get('Klub')} · {p2.get('Posisi')}</p>
            <p style="margin: 0 0 10px 0; font-size: 0.85rem; color: #64748b;">{p2.get('Nama Lengkap', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Harga", f"£{p2.get('Harga (£m)', 0):.1f}m")
        with m2:
            st.metric("Total Poin", f"{int(p2.get('Total Poin', 0))} pts")
        with m3:
            st.metric("Prediksi xPoin", f"{float(p2.get('xPoin', 0)):.2f} pts")
        with m4:
            st.metric("Form", f"{float(p2.get('Form', 0)):.2f}")

    # Radar Configuration Controls
    st.markdown("### 🕸️ Grafik Radar Komparasi Statistik")
    
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        radar_preset = st.selectbox(
            "Pilih Preset Metrik Radar:",
            options=[
                "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)",
                "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Avg Mins L5M, BPS, xGI/90, ICT, Poin/£m)",
                "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, xGI/90, BPS, % Ownership)",
                "🛡️ Defensif & Kiper (Clean Sheet, Def Contrib/90, Saves, Avg Mins L5M, BPS, xPoin, Kemudahan Jadwal)",
                "🛠️ Kustom (Pilih Metrik Bebas)"
            ],
            key="radar_preset_choice"
        )
    with ctrl2:
        norm_mode = st.selectbox(
            "Skala Nilai Sumbu Radar:",
            options=[
                "Persentil Liga (0 - 100%)",
                "Min-Max Normalisasi (0 - 100)",
                "Nilai Aktual (Raw Values)"
            ],
            help="Skala Persentil direkomendasikan agar metrik dengan skala berbeda (misal menit, xG, poin, harga) dapat dibandingkan secara proporsional.",
            key="radar_norm_mode"
        )
    with ctrl3:
        cohort_mode_code = st.selectbox(
            "Basis Komparasi Persentil/Min-Max:",
            options=["all", "pos1", "pos2"],
            format_func=lambda c: (
                "Seluruh Pemain Liga (All Players)" if c == "all" else
                f"Sesama Posisi {p1.get('Nama Pemain')} ({p1.get('Posisi')})" if c == "pos1" else
                f"Sesama Posisi {p2.get('Nama Pemain')} ({p2.get('Posisi')})"
            ),
            key="radar_cohort_mode"
        )

    preset_metric_map = {
        "🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)": [
            'xG', 'xA', 'xGI per 90', 'Gol', 'Asis', 'ICT Index', 'xPoin', 'Form'
        ],
        "⭐ Profil Komprehensif FPL (Total Poin, xPoin, Form, Avg Mins L5M, BPS, xGI/90, ICT, Poin/£m)": [
            'Total Poin', 'xPoin', 'Form', 'Avg Mins (L5M)', 'BPS', 'xGI per 90', 'ICT Index', 'Poin per £m'
        ],
        "💰 Efisiensi Biaya / Value for Money (Poin/£m, xPoin/£m, Form, Avg Mins L5M, xGI/90, BPS, % Ownership)": [
            'Poin per £m', 'xPoin per £m', 'Form', 'Avg Mins (L5M)', 'xGI per 90', 'BPS', '% Ownership'
        ],
        "🛡️ Defensif & Kiper (Clean Sheet, Def Contrib/90, Saves, Avg Mins L5M, BPS, xPoin, Kemudahan Jadwal)": [
            'Clean Sheet', 'Defensive Contribution per 90', 'Saves', 'Avg Mins (L5M)', 'BPS', 'xPoin', 'Kemudahan Jadwal'
        ]
    }

    all_available_metrics = [
        'Total Poin', 'xPoin', 'xPoin (Option B)', 'Form', 'Harga (£m)', 'Avg Mins (L5M)', 
        'Menit Bermain', 'Gol', 'Asis', 'xG', 'xA', 'xGI', 'xG per 90', 'xA per 90', 'xGI per 90', 
        'ICT Index', 'BPS', 'Bonus Poin', 'Clean Sheet', 'Saves', 'Defensive Contribution per 90', 
        '% Ownership', 'Net Transfers GW', 'Poin per £m', 'xPoin per £m', 'Kemudahan Jadwal'
    ]

    if radar_preset == "🛠️ Kustom (Pilih Metrik Bebas)":
        active_metric_keys = st.multiselect(
            "Pilih minimal 3 metrik statistik untuk grafik radar:",
            options=all_available_metrics,
            default=['Total Poin', 'xPoin', 'xG', 'xA', 'Form', 'ICT Index', 'BPS', 'Poin per £m'],
            key="radar_custom_metrics"
        )
    else:
        active_metric_keys = preset_metric_map.get(radar_preset, preset_metric_map["🎯 Ofensif & Daya Serang (xG, xA, xGI/90, Gol, Asis, ICT, xPoin, Form)"])

    if len(active_metric_keys) < 3:
        st.warning("Pilih minimal 3 metrik agar poligon radar chart dapat divisualisasikan dengan proporsional.")
        return

    # Tentukan cohort data (persentil dihitung terhadap pemain aktif yang sudah bermain di EPL)
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

    METRIC_META = {
        'xG': {'label': 'xG (Exp. Goals)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xA': {'label': 'xA (Exp. Assists)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xGI': {'label': 'xGI (Goal Involv.)', 'higher_better': True, 'fmt': '{:.2f}'},
        'xG per 90': {'label': 'xG / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'xA per 90': {'label': 'xA / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'xGI per 90': {'label': 'xGI / 90 Mins', 'higher_better': True, 'fmt': '{:.2f}'},
        'Gol': {'label': 'Total Gol', 'higher_better': True, 'fmt': '{:.0f}'},
        'Asis': {'label': 'Total Asis', 'higher_better': True, 'fmt': '{:.0f}'},
        'Total Poin': {'label': 'Total Poin', 'higher_better': True, 'fmt': '{:.0f} pts'},
        'xPoin': {'label': 'Prediksi xPoin GW', 'higher_better': True, 'fmt': '{:.2f} pts'},
        'xPoin (Option B)': {'label': 'xPoin (Opt B)', 'higher_better': True, 'fmt': '{:.2f} pts'},
        'Form': {'label': 'Form Terkini', 'higher_better': True, 'fmt': '{:.2f}'},
        'Harga (£m)': {'label': 'Harga (£m)', 'higher_better': False, 'fmt': '£{:.1f}m'},
        'Avg Mins (L5M)': {'label': 'Avg Mins (L5M)', 'higher_better': True, 'fmt': '{:.1f} m'},
        'Menit Bermain': {'label': 'Menit Bermain', 'higher_better': True, 'fmt': '{:.0f} m'},
        'ICT Index': {'label': 'ICT Index', 'higher_better': True, 'fmt': '{:.1f}'},
        'BPS': {'label': 'BPS Score', 'higher_better': True, 'fmt': '{:.0f}'},
        'Bonus Poin': {'label': 'Bonus Poin', 'higher_better': True, 'fmt': '{:.0f}'},
        'Clean Sheet': {'label': 'Clean Sheet', 'higher_better': True, 'fmt': '{:.0f}'},
        'Saves': {'label': 'Saves', 'higher_better': True, 'fmt': '{:.0f}'},
        'Defensive Contribution per 90': {'label': 'Def Contrib / 90', 'higher_better': True, 'fmt': '{:.2f}'},
        '% Ownership': {'label': '% Ownership', 'higher_better': True, 'fmt': '{:.1f}%'},
        'Net Transfers GW': {'label': 'Net Transfers GW', 'higher_better': True, 'fmt': '{:+.0f}'},
        'Poin per £m': {'label': 'Poin / £m', 'higher_better': True, 'fmt': '{:.2f}'},
        'xPoin per £m': {'label': 'xPoin / £m', 'higher_better': True, 'fmt': '{:.2f}'},
        'Kemudahan Jadwal': {'label': 'Kemudahan Jadwal', 'higher_better': True, 'fmt': '{:.1f}'},
    }

    theta_labels = []
    r_p1 = []
    r_p2 = []
    customdata_p1 = []
    customdata_p2 = []
    h2h_rows = []

    p1_wins = 0
    p2_wins = 0
    ties = 0

    for m_key in active_metric_keys:
        meta = METRIC_META.get(m_key, {'label': m_key, 'higher_better': True, 'fmt': '{:.2f}'})
        lbl = meta['label']
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

        # Head-to-Head Judgement
        if higher_better:
            if val1 > val2:
                winner = f"🟢 {p1.get('Nama Pemain')} (+{abs(val1 - val2):.2f})"
                p1_wins += 1
            elif val2 > val1:
                winner = f"🔵 {p2.get('Nama Pemain')} (+{abs(val2 - val1):.2f})"
                p2_wins += 1
            else:
                winner = "🤝 Seimbang"
                ties += 1
        else: # Lower is better (e.g. Price)
            if val1 < val2:
                winner = f"🟢 {p1.get('Nama Pemain')} (£{val1:.1f}m vs £{val2:.1f}m)"
                p1_wins += 1
            elif val2 < val1:
                winner = f"🔵 {p2.get('Nama Pemain')} (£{val2:.1f}m vs £{val1:.1f}m)"
                p2_wins += 1
            else:
                winner = "🤝 Seimbang"
                ties += 1

        h2h_rows.append({
            'Metrik Statistik': lbl,
            f"{p1.get('Nama Pemain')} (Aktual)": val1_str,
            f"{p2.get('Nama Pemain')} (Aktual)": val2_str,
            f"Skor {p1.get('Nama Pemain')}": score1,
            f"Skor {p2.get('Nama Pemain')}": score2,
            'Keunggulan': winner
        })

    # Close the radar loop
    theta_closed = theta_labels + [theta_labels[0]]
    r_p1_closed = r_p1 + [r_p1[0]]
    r_p2_closed = r_p2 + [r_p2[0]]
    custom_p1_closed = customdata_p1 + [customdata_p1[0]]
    custom_p2_closed = customdata_p2 + [customdata_p2[0]]

    # Build Plotly Radar Figure
    fig = go.Figure()

    # Trace 1: Player 1
    fig.add_trace(go.Scatterpolar(
        r=r_p1_closed,
        theta=theta_closed,
        fill='toself',
        name=f"{p1.get('Nama Pemain')} ({p1.get('Klub')})",
        line=dict(color='#10b981', width=3),
        fillcolor='rgba(16, 185, 129, 0.28)',
        customdata=custom_p1_closed,
        hovertemplate="<b>%{theta}</b><br>Skor: %{r}<br>Nilai Riil: %{customdata}<extra></extra>"
    ))

    # Trace 2: Player 2
    fig.add_trace(go.Scatterpolar(
        r=r_p2_closed,
        theta=theta_closed,
        fill='toself',
        name=f"{p2.get('Nama Pemain')} ({p2.get('Klub')})",
        line=dict(color='#3b82f6', width=3),
        fillcolor='rgba(59, 130, 246, 0.28)',
        customdata=custom_p2_closed,
        hovertemplate="<b>%{theta}</b><br>Skor: %{r}<br>Nilai Riil: %{customdata}<extra></extra>"
    ))

    # Layout styling
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
                tickfont=dict(size=12, color='#1e293b', family='Plus Jakarta Sans'),
                gridcolor='#e2e8f0',
                linecolor='#cbd5e1',
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='#ffffff'
        ),
        paper_bgcolor='#ffffff',
        font=dict(family="Plus Jakarta Sans", size=12, color="#1e293b"),
        margin=dict(l=50, r=50, t=40, b=40),
        height=580,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family="Plus Jakarta Sans")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # HEAD-TO-HEAD WINNER SUMMARY & SCORECARD
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Ringkasan Hasil Komparasi Head-to-Head")
    
    w1, w2, w3 = st.columns(3)
    with w1:
        st.metric(
            f"Kemenangan {p1.get('Nama Pemain')}", 
            f"{p1_wins} Metrik", 
            f"{(p1_wins/len(active_metric_keys)*100):.0f}% Dominasi"
        )
    with w2:
        st.metric(
            f"Kemenangan {p2.get('Nama Pemain')}", 
            f"{p2_wins} Metrik", 
            f"{(p2_wins/len(active_metric_keys)*100):.0f}% Dominasi"
        )
    with w3:
        if p1_wins > p2_wins:
            h2h_verdict = f"🟢 {p1.get('Nama Pemain')} Lebih Unggul Secara Keseluruhan"
        elif p2_wins > p1_wins:
            h2h_verdict = f"🔵 {p2.get('Nama Pemain')} Lebih Unggul Secara Keseluruhan"
        else:
            h2h_verdict = "🤝 Kedua Pemain Sangat Berimbang"
        st.metric("Keputusan Rekomendasi", h2h_verdict, f"{ties} Metrik Seri")

    # Table breakdown
    st.markdown("#### 📋 Tabel Komparasi Nilai Metrik Lengkap")
    df_h2h = pd.DataFrame(h2h_rows)
    st.dataframe(
        df_h2h,
        use_container_width=True,
        hide_index=True
    )
    st.caption("💡 *Skor persentil 0 - 100 menunjukkan posisi pemain dibanding seluruh cohort liga. Semakin tinggi skor, semakin mendekati top 1% terbaik di EPL.*")
