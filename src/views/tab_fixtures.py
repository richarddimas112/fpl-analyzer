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
    st.write("Jadwal lengkap pertandingan mendatang beserta tingkat kesulitan (FDR) resmi Premier League untuk memudahkan perencanaan transfer dan kapten.")

    # Convert FDR summary into a readable table
    fdr_table_data = []
    for t_id, f_data in fdr_summary.items():
        t_name = teams_dict.get(t_id, f"Team {t_id}")
        fdr_table_data.append({
            'Klub': t_name,
            'Lawan Laga Berikutnya': f_data.get('Next_Opponent_Fmt', '-'),
            'FDR1 (Laga Terdekat)': f_data.get('FDR1', 3.0),
            'FDR3 (Rata-rata 3 Laga)': f_data.get('FDR3', 3.0),
            'FDR5 (Rata-rata 5 Laga)': f_data.get('FDR5', 3.0)
        })

    fdr_df = pd.DataFrame(fdr_table_data).sort_values(by="FDR3 (Rata-rata 3 Laga)", ascending=True)

    st.dataframe(
        fdr_df,
        use_container_width=True,
        column_config={
            "FDR1 (Laga Terdekat)": st.column_config.NumberColumn(format="%.1f"),
            "FDR3 (Rata-rata 3 Laga)": st.column_config.NumberColumn(format="%.2f"),
            "FDR5 (Rata-rata 5 Laga)": st.column_config.NumberColumn(format="%.2f")
        }
    )
    st.caption("💡 *Catatan FDR: Skala 1 (Sangat Mudah) hingga 5 (Sangat Sulit). Nilai FDR3 dan FDR5 yang lebih rendah menandakan jadwal pertandingan mendatang yang lebih menguntungkan.*")
