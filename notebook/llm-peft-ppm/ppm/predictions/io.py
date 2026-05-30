"""Shared utilities for persisting per-prefix predictions across all baselines."""
from pathlib import Path
import pandas as pd


PREDICTION_SCHEMA = [
    "case_id",
    "pos",                       # prefix length within case (0-based)
    "case_len",                  # total events in this case
    "activity_id_current",       # the activity of the CURRENT event
    "y_true_next_activity",
    "y_pred_next_activity",
    "y_true_remaining_time",
    "y_pred_remaining_time",
    "y_true_time_to_next_event",
    "y_pred_time_to_next_event",
    "log",
    "backbone",
    "seed",
]


def predictions_path(base_dir: str, log_name: str, backbone: str, seed: int) -> Path:
    out_dir = Path(base_dir) / "predictions" / log_name / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"seed_{int(seed):02d}.parquet"


def build_predictions_df(
    dump: dict,
    log_name: str,
    backbone: str,
    seed: int,
) -> pd.DataFrame:
    """
    Construct the canonical predictions dataframe from a model's dump dict.
    `dump` must contain numpy arrays of equal length under these keys:
      case_ids, positions, case_lengths, activity_ids_current,
      y_true_next_activity, y_pred_next_activity,
      y_true_remaining_time, y_pred_remaining_time,
      y_true_time_to_next_event, y_pred_time_to_next_event
    """
    df = pd.DataFrame({
        "case_id": dump["case_ids"],
        "pos": dump["positions"],
        "case_len": dump["case_lengths"],
        "activity_id_current": dump["activity_ids_current"],
        "y_true_next_activity": dump["y_true_next_activity"],
        "y_pred_next_activity": dump["y_pred_next_activity"],
        "y_true_remaining_time": dump["y_true_remaining_time"],
        "y_pred_remaining_time": dump["y_pred_remaining_time"],
        "y_true_time_to_next_event": dump["y_true_time_to_next_event"],
        "y_pred_time_to_next_event": dump["y_pred_time_to_next_event"],
    })
    df["log"] = log_name
    df["backbone"] = backbone
    df["seed"] = int(seed)
    return df[PREDICTION_SCHEMA]


def persist_predictions(
    dump: dict,
    base_dir: str,
    log_name: str,
    backbone: str,
    seed: int,
) -> Path:
    df = build_predictions_df(dump, log_name=log_name, backbone=backbone, seed=seed)
    out_path = predictions_path(base_dir, log_name, backbone, seed)
    df.to_parquet(out_path, index=False)
    return out_path