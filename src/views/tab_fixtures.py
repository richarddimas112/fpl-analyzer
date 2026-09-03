"""
Fixtures and FDR Table View Tab.
"""

import pandas as pd
import streamlit as st

def render_tab_fixtures(fixtures_data, teams_dict, fdr_summary):
    """
    Renders Tab 4: Fixtures, Schedule and Fixture Difficulty Rating (FDR).
    """
    st.subheader("📅 Jadwal Pertandingan & Analisis Fixture Difficulty Rating (FDR)")
    st.write("Jadwal lengkap pertandingan mendatang beserta tingkat kesulitan (FDR) resmi Premier League untuk memudahkan perencanaan transfer jangka pendek (FDR3/FDR5) dan jangka panjang (FDR10).")

    # 1. Main FDR Comparison Table (FDR1, FDR3, FDR5, FDR10)
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown("#### 🏆 Peringkat Kemudahan Jadwal Klub (FDR1 - FDR10)")
    with col_hdr2:
        sort_fdr = st.selectbox(
            "Urutkan Berdasarkan:",
            options=["FDR3 (Rata-rata 3 Laga)", "FDR5 (Rata-rata 5 Laga)", "FDR10 (Rata-rata 10 Laga)", "FDR1 (Laga Terdekat)"],
            index=0,
            key="sort_fdr_sel"
        )

    fdr_table_data = []
    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        fdr_table_data.append({
            'Klub': t_name,
            'Lawan Laga Berikutnya': f_data.get('Next_Opponent_Fmt', '-'),
            'FDR1 (Laga Terdekat)': f_data.get('FDR1', 3.0),
            'FDR3 (Rata-rata 3 Laga)': f_data.get('FDR3', 3.0),
            'FDR5 (Rata-rata 5 Laga)': f_data.get('FDR5', 3.0),
            'FDR10 (Rata-rata 10 Laga)': f_data.get('FDR10', 3.0)
        })

    fdr_df = pd.DataFrame(fdr_table_data).sort_values(by=sort_fdr, ascending=True)

    st.dataframe(
        fdr_df,
        use_container_width=True,
        column_config={
            "FDR1 (Laga Terdekat)": st.column_config.NumberColumn(format="%.1f"),
            "FDR3 (Rata-rata 3 Laga)": st.column_config.NumberColumn(format="%.2f"),
            "FDR5 (Rata-rata 5 Laga)": st.column_config.NumberColumn(format="%.2f"),
            "FDR10 (Rata-rata 10 Laga)": st.column_config.NumberColumn(format="%.2f")
        }
    )
    st.caption("💡 *Catatan FDR: Skala 1 (Sangat Mudah) hingga 5 (Sangat Sulit). Nilai FDR yang lebih rendah menandakan jadwal pertandingan mendatang yang lebih menguntungkan.*")

    st.divider()

    # 2. Comprehensive 10-Match Fixture Matrix
    st.markdown("#### 🗓️ Matriks & Ticker 10 Pertandingan Mendatang (20 Klub Premier League)")
    st.write("Rincian lawan dan tingkat kesulitan (FDR) untuk 10 pertandingan mendatang setiap klub. Membantu strategi rotasi pemain dan identifikasi *fixture swing*.")

    # Legend for FDR
    st.markdown("""
    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; font-size: 0.85rem;">
        <span style="background: #22c55e; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">FDR 2 (Mudah)</span>
        <span style="background: #94a3b8; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">FDR 3 (Netral)</span>
        <span style="background: #f97316; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">FDR 4 (Sulit)</span>
        <span style="background: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600;">FDR 5 (Sangat Sulit)</span>
    </div>
    """, unsafe_allow_html=True)

    matrix_rows = []
    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        up10 = f_data.get('upcoming_10', [])
        row = {
            'Klub': t_name,
            'FDR10 Avg': f_data.get('FDR10', 3.0)
        }
        for idx in range(10):
            col_name = f"Match +{idx + 1}"
            if idx < len(up10):
                m = up10[idx]
                opp = m.get('opp_name', 'TBD')
                ha = "H" if m.get('is_home') == 1 else "A"
                diff = m.get('fdr', 3)
                row[col_name] = f"{opp} ({ha}) [{diff}]"
            else:
                row[col_name] = "-"
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows).sort_values(by="FDR10 Avg", ascending=True)

    st.dataframe(
        matrix_df,
        use_container_width=True,
        column_config={
            "FDR10 Avg": st.column_config.NumberColumn(format="%.2f", help="Rata-rata FDR untuk 10 pertandingan mendatang")
        }
    )

