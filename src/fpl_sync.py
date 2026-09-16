"""
FPL Realtime API Synchronization & Transfer Comparison Service.
Enables pulling live official FPL manager squads, mapping them to Squad Planner slots,
and analyzing transfer changes & xPoints differences between slots.
"""

import re
import requests
import pandas as pd
from typing import Tuple, Dict, Any, Optional

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def parse_fpl_team_id(val: Any) -> Optional[int]:
    """
    Extracts the numerical FPL Team/Entry ID from a string, integer, or URL.
    Examples supported:
      - 123456
      - "123456"
      - "https://fantasy.premierleague.com/entry/123456/event/4"
      - "fantasy.premierleague.com/entry/123456/"
    """
    if not val:
        return None
    s = str(val).strip()
    # Check for URL match first: /entry/(\d+)
    url_match = re.search(r'/entry/(\d+)', s)
    if url_match:
        try:
            return int(url_match.group(1))
        except ValueError:
            pass
            
    # Check for standalone integer or number in string
    digits = re.findall(r'\b\d+\b', s)
    if digits:
        try:
            return int(digits[0])
        except ValueError:
            pass
            
    return None

def fetch_fpl_manager_profile(team_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches official FPL manager profile information including team name,
    manager first and last name, overall rank, and current gameweek.
    """
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            first = data.get("player_first_name", "").strip()
            last = data.get("player_last_name", "").strip()
            mgr_name = f"{first} {last}".strip() if (first or last) else "Manajer FPL"
            team_name = data.get("name", f"Tim #{team_id}").strip()
            return {
                "team_id": team_id,
                "team_name": team_name,
                "manager_name": mgr_name,
                "region": data.get("player_region_name", "Global"),
                "overall_points": data.get("summary_overall_points", 0),
                "overall_rank": data.get("summary_overall_rank", 0),
                "current_event": data.get("current_event", 1),
                "raw_entry": data
            }, None
        elif r.status_code == 404:
            return None, f"Tim FPL dengan ID #{team_id} tidak ditemukan. Periksa kembali FPL Team ID Anda."
        else:
            return None, f"Server FPL merespons dengan status {r.status_code}. Silakan coba beberapa saat lagi."
    except requests.exceptions.Timeout:
        return None, "Koneksi ke server resmi FPL timeout (batas waktu habis). Coba kembali sebentar lagi."
    except Exception as e:
        return None, f"Gagal menghubungi server FPL: {str(e)}"

def fetch_fpl_realtime_squad(team_id: int, current_gw: int = 1, cookie_str: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Pulls 15-player squad picks from the FPL API.
    - If cookie_str is provided, attempts authenticated fetch from /api/my-team/{team_id}/
      to retrieve pre-deadline transfers for the upcoming gameweek in true realtime.
    - If cookie_str is not provided or auth fails, falls back to public /api/entry/{team_id}/event/{gw}/picks/
      (which contains official locked squad from the latest active gameweek).
    """
    # 1. Fetch manager profile to determine active gameweek and manager info
    profile, err = fetch_fpl_manager_profile(team_id)
    if err:
        return None, err
        
    active_event = profile.get("current_event") or current_gw
    picks_data = None
    resolved_gw = active_event
    is_live_my_team = False

    # A. If cookie is provided, attempt live /api/my-team/ first
    if cookie_str and cookie_str.strip():
        c_str = cookie_str.strip()
        if not c_str.startswith("pl_profile=") and "=" not in c_str:
            cookie_header = f"pl_profile={c_str}"
        else:
            cookie_header = c_str

        auth_headers = dict(HEADERS)
        auth_headers["Cookie"] = cookie_header
        my_team_url = f"https://fantasy.premierleague.com/api/my-team/{team_id}/"

        try:
            r_auth = requests.get(my_team_url, headers=auth_headers, timeout=8)
            if r_auth.status_code == 200:
                mt_json = r_auth.json()
                if "picks" in mt_json and len(mt_json["picks"]) > 0:
                    picks_data = mt_json
                    resolved_gw = active_event + 1  # Upcoming gameweek
                    is_live_my_team = True
            elif r_auth.status_code == 403:
                return None, "Autentikasi Cookie FPL ditolak (403). Cookie pl_profile tidak valid atau telah kedaluwarsa. Periksa kembali cookie Anda atau gunakan metode Sinkronisasi ID Publik / Tempel Teks."
        except Exception as ex:
            pass  # Fallback to public endpoints below

    # B. If picks_data not obtained yet, query public gameweek picks descending
    if not picks_data:
        for gw in range(active_event, 0, -1):
            url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
            try:
                r = requests.get(url, headers=HEADERS, timeout=8)
                if r.status_code == 200:
                    p_json = r.json()
                    if "picks" in p_json and len(p_json["picks"]) > 0:
                        picks_data = p_json
                        resolved_gw = gw
                        break
            except Exception:
                continue

    if not picks_data or "picks" not in picks_data:
        return None, f"Tidak ditemukan susunan pemain (picks) untuk tim '{profile['team_name']}' (ID #{team_id})."

    return {
        "profile": profile,
        "resolved_gw": resolved_gw,
        "picks": picks_data["picks"],
        "active_chip": picks_data.get("active_chip"),
        "entry_history": picks_data.get("entry_history", {}),
        "is_live_my_team": is_live_my_team,
        "transfers_info": picks_data.get("transfers", {})
    }, None

def parse_and_map_squad_from_text(text: str, players_df: pd.DataFrame) -> Tuple[Optional[Dict[str, int]], Dict[str, Any], Optional[str]]:
    """
    Intelligently parses user-pasted squad text (copied from FPL website, app, or typed)
    and maps the recognized players into standard 15 Squad Planner slots:
    2 GK, 5 DEF, 5 MID, 3 FWD.
    """
    import unicodedata
    def strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

    if not text or not text.strip():
        return None, {}, "Teks susunan pemain kosong. Silakan tempelkan daftar 15 pemain Anda."

    # Build normalized lookup catalog
    lookup = []
    for _, row in players_df.iterrows():
        p_id = int(row['id'])
        p_name = str(row.get('Nama Pemain', ''))
        p_web = str(row.get('Web Name', ''))
        p_pos = str(row.get('Posisi', ''))
        p_club = str(row.get('Klub', ''))
        clean_web = strip_accents(p_web).lower()
        clean_name = strip_accents(p_name).lower()
        lookup.append({
            'id': p_id,
            'name': p_name,
            'web_name': p_web,
            'pos': p_pos,
            'club': p_club,
            'clean_web': clean_web,
            'clean_name': clean_name,
        })

    # Split pasted text into lines/tokens
    lines = [l.strip() for l in re.split(r'[\r\n,;]+', text) if l.strip()]
    matched_players = []
    seen_ids = set()

    for raw in lines:
        # Strip club abbreviation in brackets like (ARS), (MCI), prices like £5.5m, position tags
        cleaned = re.sub(r'\([A-Za-z0-9\s-]+\)', '', raw).strip()
        cleaned = re.sub(r'^(gkp|gk|def|mid|fwd)\s*[:\-]?\s*', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'£[0-9.]+\s*m?', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\b(captain|vice|sub|bench|c|vc)\b', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned_norm = strip_accents(cleaned).lower().strip()

        if len(cleaned_norm) < 2:
            continue

        best = None
        # 1. Exact match on clean web_name
        for p in lookup:
            if p['clean_web'] == cleaned_norm:
                best = p
                break
        # 2. Exact match on clean full name
        if not best:
            for p in lookup:
                if p['clean_name'] == cleaned_norm:
                    best = p
                    break
        # 3. Substring match
        if not best and len(cleaned_norm) >= 3:
            for p in lookup:
                if cleaned_norm in p['clean_web'] or cleaned_norm in p['clean_name']:
                    best = p
                    break
        # 4. Word by word match
        if not best:
            words = [w for w in cleaned_norm.split() if len(w) > 2]
            for w in reversed(words):
                for p in lookup:
                    if p['clean_web'] == w or p['clean_web'].endswith(w):
                        best = p
                        break
                if best:
                    break

        if best and best['id'] not in seen_ids:
            matched_players.append(best)
            seen_ids.add(best['id'])

    if not matched_players:
        return None, {}, "Tidak ada nama pemain yang berhasil dikenali dari teks yang Anda tempelkan."

    # Group by positions
    gks = [p for p in matched_players if p['pos'] == 'GK']
    defs = [p for p in matched_players if p['pos'] == 'DEF']
    mids = [p for p in matched_players if p['pos'] == 'MID']
    fwds = [p for p in matched_players if p['pos'] == 'FWD']

    squad_slots = {}
    
    # Fill GKs (up to 2)
    for i in range(2):
        if i < len(gks):
            squad_slots[f"GKP {i+1}"] = gks[i]['id']
            
    # Fill DEFs (up to 5)
    for i in range(5):
        if i < len(defs):
            squad_slots[f"DEF {i+1}"] = defs[i]['id']
            
    # Fill MIDs (up to 5)
    for i in range(5):
        if i < len(mids):
            squad_slots[f"MID {i+1}"] = mids[i]['id']
            
    # Fill FWDs (up to 3)
    for i in range(3):
        if i < len(fwds):
            squad_slots[f"FWD {i+1}"] = fwds[i]['id']

    # Backfill missing slots with best alternatives from players_df if needed
    missing_count = 15 - len(squad_slots)
    for slot_name, expected_pos in [
        ('GKP 1', 'GK'), ('GKP 2', 'GK'),
        ('DEF 1', 'DEF'), ('DEF 2', 'DEF'), ('DEF 3', 'DEF'), ('DEF 4', 'DEF'), ('DEF 5', 'DEF'),
        ('MID 1', 'MID'), ('MID 2', 'MID'), ('MID 3', 'MID'), ('MID 4', 'MID'), ('MID 5', 'MID'),
        ('FWD 1', 'FWD'), ('FWD 2', 'FWD'), ('FWD 3', 'FWD')
    ]:
        if slot_name not in squad_slots:
            pos_cands = players_df[(players_df['Posisi'] == expected_pos) & (~players_df['id'].isin(seen_ids))]
            if not pos_cands.empty:
                chosen_id = int(pos_cands.iloc[0]['id'])
                squad_slots[slot_name] = chosen_id
                seen_ids.add(chosen_id)

    meta = {
        "matched_count": len(matched_players),
        "matched_players": matched_players,
        "missing_count": missing_count,
        "gks_count": len(gks),
        "defs_count": len(defs),
        "mids_count": len(mids),
        "fwds_count": len(fwds)
    }

    return squad_slots, meta, None

def map_fpl_picks_to_squad_slots(picks: list, players_df: pd.DataFrame) -> Tuple[Optional[Dict[str, int]], Dict[str, Any], Optional[str]]:
    """
    Converts official FPL 15 picks into the standard 15 Squad Planner slot dictionary:
    - 2 GK -> 'GKP 1', 'GKP 2'
    - 5 DEF -> 'DEF 1' .. 'DEF 5'
    - 5 MID -> 'MID 1' .. 'MID 5'
    - 3 FWD -> 'FWD 1' .. 'FWD 3'
    Also extracts captain, vice captain, and active starting 11 info.
    """
    if not picks or len(picks) != 15:
        return None, {}, "Jumlah pemain dari API FPL tidak sesuai standar 15 pemain."

    players_lookup = {int(r['id']): r for _, r in players_df.iterrows()}
    valid_ids = set(players_lookup.keys())

    gks = []
    defs = []
    mids = []
    fwds = []
    captain_id = None
    vc_id = None
    starters_ids = []
    bench_ids = []

    for idx, p in enumerate(picks):
        pid = int(p.get("element", 0))
        pos_order = p.get("position", idx + 1)
        is_cap = p.get("is_captain", False)
        is_vc = p.get("is_vice_captain", False)

        if is_cap:
            captain_id = pid
        if is_vc:
            vc_id = pid
        if pos_order <= 11:
            starters_ids.append(pid)
        else:
            bench_ids.append(pid)

        p_data = players_lookup.get(pid)
        pos = p_data['Posisi'] if p_data is not None else None
        
        # If not in players_lookup, infer from element_type in pick or fallback
        if not pos:
            el_type = p.get("element_type")
            pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            pos = pos_map.get(el_type, 'MID')

        if pos == 'GK':
            gks.append(pid)
        elif pos == 'DEF':
            defs.append(pid)
        elif pos == 'MID':
            mids.append(pid)
        elif pos == 'FWD':
            fwds.append(pid)

    # Validate slot quantities
    missing_notice = []
    if len(gks) != 2:
        missing_notice.append(f"Kiper: {len(gks)}/2")
    if len(defs) != 5:
        missing_notice.append(f"Bek: {len(defs)}/5")
    if len(mids) != 5:
        missing_notice.append(f"Gelandang: {len(mids)}/5")
    if len(fwds) != 3:
        missing_notice.append(f"Penyerang: {len(fwds)}/3")

    squad_slots = {}
    for i, pid in enumerate(gks[:2]):
        squad_slots[f"GKP {i+1}"] = pid
    for i, pid in enumerate(defs[:5]):
        squad_slots[f"DEF {i+1}"] = pid
    for i, pid in enumerate(mids[:5]):
        squad_slots[f"MID {i+1}"] = pid
    for i, pid in enumerate(fwds[:3]):
        squad_slots[f"FWD {i+1}"] = pid

    # In case of any position mismatch (e.g. unknown new player), fill with position fallback
    for slot_name, expected_pos in [
        ("GKP 1", "GK"), ("GKP 2", "GK"),
        ("DEF 1", "DEF"), ("DEF 2", "DEF"), ("DEF 3", "DEF"), ("DEF 4", "DEF"), ("DEF 5", "DEF"),
        ("MID 1", "MID"), ("MID 2", "MID"), ("MID 3", "MID"), ("MID 4", "MID"), ("MID 5", "MID"),
        ("FWD 1", "FWD"), ("FWD 2", "FWD"), ("FWD 3", "FWD")
    ]:
        if slot_name not in squad_slots or squad_slots[slot_name] not in valid_ids:
            pos_candidates = players_df[players_df['Posisi'] == expected_pos]
            if not pos_candidates.empty:
                squad_slots[slot_name] = int(pos_candidates.iloc[0]['id'])

    meta = {
        "captain_id": captain_id,
        "vice_captain_id": vc_id,
        "starters_ids": starters_ids,
        "bench_ids": bench_ids,
        "missing_notice": missing_notice
    }
    return squad_slots, meta, None

def calculate_squad_transfer_delta(squad1_slots: Dict[str, int], squad2_slots: Dict[str, int], players_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares two 15-player squads (e.g., Slot 1 Realtime vs Slot 2 Planned)
    to calculate players transferred out, players transferred in, and the net delta
    in xPoints (Default, Option B, Option C, Consensus), cost, and FDR.
    """
    lookup = {int(r['id']): r for _, r in players_df.iterrows()}
    
    set1 = set(squad1_slots.values())
    set2 = set(squad2_slots.values())
    
    out_ids = list(set1 - set2)
    in_ids = list(set2 - set1)
    
    players_out = [lookup.get(pid) for pid in out_ids if lookup.get(pid) is not None]
    players_in = [lookup.get(pid) for pid in in_ids if lookup.get(pid) is not None]
    
    # Calculate squad metrics
    def get_metrics(slots):
        cost = sum(float(lookup[pid]['Harga (£m)']) for pid in slots.values() if pid in lookup)
        xp_def = sum(float(lookup[pid]['xPoin']) for pid in slots.values() if pid in lookup)
        xp_b = sum(float(lookup[pid].get('xPoin (Option B)', lookup[pid]['xPoin'])) for pid in slots.values() if pid in lookup)
        xp_c = sum(float(lookup[pid].get('xPoin (Option C Ensemble)', lookup[pid]['xPoin'])) for pid in slots.values() if pid in lookup)
        consensus = (xp_def + xp_b + xp_c) / 3.0
        fdr = [float(lookup[pid].get('FDR10', 3.0)) for pid in slots.values() if pid in lookup]
        avg_fdr = sum(fdr) / len(fdr) if fdr else 3.0
        return {
            "cost": cost,
            "xp_default": xp_def,
            "xp_opt_b": xp_b,
            "xp_opt_c": xp_c,
            "xp_consensus": consensus,
            "avg_fdr": avg_fdr
        }
        
    m1 = get_metrics(squad1_slots)
    m2 = get_metrics(squad2_slots)
    
    delta = {
        "transfers_count": len(out_ids),
        "players_out": players_out,
        "players_in": players_in,
        "m1": m1,
        "m2": m2,
        "delta_cost": m2["cost"] - m1["cost"],
        "delta_xp_default": m2["xp_default"] - m1["xp_default"],
        "delta_xp_opt_b": m2["xp_opt_b"] - m1["xp_opt_b"],
        "delta_xp_opt_c": m2["xp_opt_c"] - m1["xp_opt_c"],
        "delta_xp_consensus": m2["xp_consensus"] - m1["xp_consensus"],
        "delta_fdr": m2["avg_fdr"] - m1["avg_fdr"]
    }
    return delta
