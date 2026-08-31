"""
Constants and configurations for FPL Scout Analytics.
"""

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

STATUS_MAP = {
    'a': 'Tersedia',
    'd': 'Diragukan (75%)',
    'i': 'Cedera',
    's': 'Skorsing',
    'u': 'Tidak Tersedia'
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

POS_MODEL_CONFIGS = {
    'FWD': {
        'element_type': 4,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'ICT Index'
        ]
    },
    'MID': {
        'element_type': 3,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'Defensive_Contribution_per_90', 'xGC_per_90', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'Kontribusi Defensif per 90',
            'xGC per 90 (Expected Goals Conceded)',
            'ICT Index'
        ]
    },
    'DEF': {
        'element_type': 2,
        'feature_cols': ['xG_per_90', 'xA_per_90', 'bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'Defensive_Contribution_per_90', 'xGC_per_90', 'ict_index'],
        'feature_labels': [
            'xG per 90 (Expected Goals)',
            'xA per 90 (Expected Assists)',
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'Kontribusi Defensif per 90',
            'xGC per 90 (Expected Goals Conceded)',
            'ICT Index'
        ]
    },
    'GK': {
        'element_type': 1,
        'feature_cols': ['bps_per_90', 'form', 'was_home', 'FDR', 'last_minutes_5_match', 'xGC_per_90', 'Saves_per_90'],
        'feature_labels': [
            'BPS per 90 (Bonus Points System)',
            'Form (Murni)',
            'Laga Kandang (Home)',
            'FDR Lawan',
            'Avg Mins (L5M)',
            'xGC per 90 (Expected Goals Conceded)',
            'Saves per 90'
        ]
    }
}
