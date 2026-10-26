"""Build reproducible Option B forecasts from a frozen FPL input snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import train_option_b_models
from src.processors import (
    calculate_team_fdrs,
    compute_team_differentials_asof,
    get_current_gw,
    load_historical_training_data,
    process_players,
    apply_team_differentials_and_recalc_option_b,
)


PREDICTION_COLUMNS = [
    "id", "team", "Nama Pemain", "Klub", "Lawan GW Berikutnya", "Posisi", "Harga (£m)",
    "xPoin (Option B)", "xG Pred (90)", "xG Pred (Match)", "xA Pred (90)", "xA Pred (Match)", "xMins Pts", "xG Pts",
    "xA Pts", "xSaves Pts", "xDC Pts", "xCS Pts", "xBP", "Peluang CS (%)",
    "Diff Attack Team", "Diff Defense Team", "Avg Mins (L5M)", "FDR1",
    "option_b_form_prior", "option_b_started_prev", "option_b_threat_per_90", "option_b_creativity_per_90",
    "xG Mult", "xA Mult",
]


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def model_signature() -> str:
    digest = hashlib.sha256()
    for relative_path in (
        "src/models.py",
        "src/processors.py",
        "src/option_b_snapshot.py",
        "data/historical_train_data.csv",
    ):
        path = Path(__file__).resolve().parents[1] / relative_path
        digest.update(relative_path.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def make_option_b_predictions(
    fpl_data: dict[str, Any],
    fixtures: list[dict[str, Any]],
    histories_by_player_id: dict[int, list[dict[str, Any]]],
    target_gameweek: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the dashboard's Option B pipeline against captured, immutable inputs."""
    teams = fpl_data.get("teams", [])
    teams_dict = {int(team["id"]): team["name"] for team in teams}
    teams_short_dict = {int(team["id"]): team.get("short_name", team["name"][:3].upper()) for team in teams}
    fdr_summary = calculate_team_fdrs(fixtures, teams_dict, teams_short_dict=teams_short_dict)
    current_gw = int(target_gameweek or get_current_gw(fpl_data))
    elements = fpl_data.get("elements", [])

    opt_b_xg, opt_b_xa, _, _ = train_option_b_models(
        elements,
        fdr_summary,
        current_gw,
        load_historical_training_data(),
        fixtures_data=fixtures,
        histories_by_player_id=histories_by_player_id,
    )
    players_df, _ = process_players(
        fpl_data,
        fdr_summary,
        {},
        _opt_b_models=(opt_b_xg, opt_b_xa),
        histories_by_player_id=histories_by_player_id,
    )

    target_gw_by_team = {
        int(team_id): int(info["upcoming_all"][0]["gw"])
        for team_id, info in fdr_summary.items()
        if info.get("upcoming_all") and info["upcoming_all"][0].get("gw") is not None
    }
    target_gws = sorted({int(fixture["event"]) for fixture in fixtures if fixture.get("event") is not None})
    historical_diffs = compute_team_differentials_asof(
        elements,
        fixtures,
        target_gws=target_gws,
        histories_by_player_id=histories_by_player_id,
    )
    team_diffs_predict = {}
    for team_id, target_gw in target_gw_by_team.items():
        info = historical_diffs.get((target_gw, team_id))
        if info:
            team_name = teams_dict.get(team_id, f"Team {team_id}")
            opponent_name = teams_dict.get(info["opponent_id"], f"Team {info['opponent_id']}")
            team_diff = {**info, "opponent": opponent_name}
            team_diffs_predict[team_name] = team_diff
            team_diffs_predict[team_id] = team_diff

    players_df = apply_team_differentials_and_recalc_option_b(
        players_df, team_diffs_predict, _opt_b_models=(opt_b_xg, opt_b_xa)
    )
    target_gw = int(current_gw or 0)
    eligible_team_ids = {
        team_id for team_id, gameweek in target_gw_by_team.items()
        if gameweek == target_gw
    }
    if "team" in players_df.columns:
        players_df = players_df[players_df["team"].isin(eligible_team_ids)].copy()
    columns = [column for column in PREDICTION_COLUMNS if column in players_df.columns]
    predictions = []
    for row in players_df[columns].to_dict(orient="records"):
        prediction_row = {key: _json_scalar(value) for key, value in row.items()}
        prediction_row["prediction_gameweek"] = target_gw
        predictions.append(prediction_row)
    if not predictions:
        raise RuntimeError(f"Tidak ada pemain yang dapat diprediksi untuk GW {target_gw}.")

    metadata = {
        "status": "complete",
        "origin": "captured_with_snapshot",
        "model": "Option B",
        "model_signature": model_signature(),
        "prediction_count": len(predictions),
        "target_gameweeks": [target_gw] if target_gw else [],
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    return predictions, metadata
