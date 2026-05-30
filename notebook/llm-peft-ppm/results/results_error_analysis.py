#!/usr/bin/env python3
"""
Structured error and behavior analysis for per-prefix PPM prediction parquet files.

Expected input layout:
    results/predictions/<dataset>/<model>/seed_41.parquet

The script is deliberately schema-tolerant. It first inspects each parquet file,
normalizes known prediction-dump column variants, validates alignment, then writes
CSV/Parquet analysis outputs and optional plots.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

from skpm.event_logs import (
    BPI12, BPI17,
    BPI20PrepaidTravelCosts, BPI20TravelPermitData, BPI20RequestForPayment,
)
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
# also import your own prepare_data — adjust the path:
from fertig_lennart_next_event_prediction import prepare_data

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


PROJECT_DIR = Path(
    "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm"
)

RESULTS_DIR = PROJECT_DIR / "results"


DEFAULT_PRED_DIR = RESULTS_DIR / "predictions"

DEFAULT_OUT_DIR = RESULTS_DIR / "error_analysis"


DEFAULT_DATASETS = [
    "BPI12",
    "BPI17",
    "BPI20PrepaidTravelCosts",
    "BPI20TravelPermitData",
    "BPI20RequestForPayment",
]

DEFAULT_MODELS = [
    "rnn",
    "transformer",
    "tabpfn3",
    "saprpt",
    "llama32-1b",
    "gemma-2-2b",
]

MODEL_PRETTY = {
    "rnn": "RNN",
    "transformer": "Transformer",
    "tabpfn3": "TabPFN-3",
    "saprpt": "SAP-RPT",  # rename here if you want this displayed as ConTextTab
    "llama32-1b": "Llama-3.2-1B",
    "gemma-2-2b": "Gemma-2-2B",
}

MODEL_PARADIGM = {
    "rnn": "sequence",
    "transformer": "sequence",
    "tabpfn3": "tabular_fm",
    "saprpt": "tabular_fm",
    "llama32-1b": "llm",
    "gemma-2-2b": "llm",
}

TASKS = {
    "NA": {
        "true": "y_true_na",
        "pred": "y_pred_na",
        "kind": "classification",
        "label": "Next Activity",
    },
    "RT": {
        "true": "y_true_rt",
        "pred": "y_pred_rt",
        "kind": "regression",
        "label": "Remaining Time",
    },
    "NT": {
        "true": "y_true_nt",
        "pred": "y_pred_nt",
        "kind": "regression",
        "label": "Next Time to Next Event",
    },
}

GROUP_COLS_NA = [
    "prefix_length_bucket",
    "relative_prefix_bucket",
    "case_length_bucket",
    "activity_current",
    "y_true_na",
    "activity_frequency_bucket",
    "true_next_activity_frequency_bucket",
    "branching_degree_bucket",
    "branching_entropy_bucket",
]

GROUP_COLS_TEMPORAL = [
    "prefix_length_bucket",
    "relative_prefix_bucket",
    "case_length_bucket",
    "activity_current",
    "branching_degree_bucket",
    "branching_entropy_bucket",
    "rt_magnitude_bucket",
    "nt_magnitude_bucket",
]


# -----------------------------------------------------------------------------
# Schema normalization
# -----------------------------------------------------------------------------

COLUMN_ALIASES = {
    "case_id": [
        "case_id", "case_ids", "case:concept:name", "case", "trace_id", "trace_ids",
    ],
    "pos": [
        "pos", "position", "positions", "prefix_pos", "prefix_position", "event_position",
    ],
    "prefix_length": [
        "prefix_length", "prefix_len", "prefix_lengths",
    ],
    "case_length": [
        "case_len", "case_length", "case_lengths", "trace_len", "trace_length",
    ],
    "activity_current": [
        "activity_current", "activity_id_current", "activity_ids_current",
        "current_activity", "current_activity_id", "activity", "concept:name",
    ],
    "y_true_na": [
        "y_true_next_activity", "true_next_activity", "next_activity_true",
        "y_true_activity", "true_activity", "activity_true", "next_activity",
    ],
    "y_pred_na": [
        "y_pred_next_activity", "pred_next_activity", "next_activity_pred",
        "y_pred_activity", "pred_activity", "activity_pred",
    ],
    "y_true_rt": [
        "y_true_remaining_time", "true_remaining_time", "remaining_time_true",
        "y_true_next_remaining_time", "next_remaining_time_true", "remaining_time",
    ],
    "y_pred_rt": [
        "y_pred_remaining_time", "pred_remaining_time", "remaining_time_pred",
        "y_pred_next_remaining_time", "next_remaining_time_pred",
    ],
    "y_true_nt": [
        "y_true_time_to_next_event", "true_time_to_next_event", "time_to_next_event_true",
        "y_true_next_time_to_next_event", "next_time_to_next_event_true", "time_to_next_event",
    ],
    "y_pred_nt": [
        "y_pred_time_to_next_event", "pred_time_to_next_event", "time_to_next_event_pred",
        "y_pred_next_time_to_next_event", "next_time_to_next_event_pred",
    ],
}


@dataclass
class LoadedPrediction:
    dataset: str
    model: str
    path: Path
    exists: bool
    raw_columns: list[str]
    normalized_columns: list[str]
    row_count_raw: int
    row_count_normalized: int
    warnings: list[str]
    df: pd.DataFrame | None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def inspect_parquet_schema(path: Path) -> tuple[list[str], dict[str, str]]:
    """Return parquet columns and logical dtypes without loading all rows if possible."""
    try:
        import pyarrow.parquet as pq  # type: ignore
        schema = pq.read_schema(path)
        cols = list(schema.names)
        dtypes = {name: str(schema.field(name).type) for name in cols}
        return cols, dtypes
    except Exception:
        df = pd.read_parquet(path)
        return list(df.columns), {c: str(t) for c, t in df.dtypes.items()}


def first_existing(columns: Iterable[str], aliases: list[str]) -> str | None:
    cols = set(columns)
    for a in aliases:
        if a in cols:
            return a
    return None


def normalize_prediction_frame(raw: pd.DataFrame, dataset: str, model: str, path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Normalize one prediction parquet to canonical columns."""
    warnings_out: list[str] = []
    df = raw.copy()
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in COLUMN_ALIASES.items():
        col = first_existing(df.columns, aliases)
        if col is not None:
            out[canonical] = df[col]

    # Derive position / prefix length where possible.
    if "pos" not in out.columns and "prefix_length" in out.columns:
        out["pos"] = pd.to_numeric(out["prefix_length"], errors="coerce") - 1
        warnings_out.append("Derived pos from prefix_length - 1.")

    if "prefix_length" not in out.columns and "pos" in out.columns:
        out["prefix_length"] = pd.to_numeric(out["pos"], errors="coerce") + 1

    if "case_id" not in out.columns:
        out["case_id"] = np.arange(len(out)).astype(str)
        warnings_out.append("No case_id column found. Falling back to row index; cross-model alignment is weaker.")

    if "pos" not in out.columns:
        out["pos"] = np.arange(len(out))
        warnings_out.append("No position column found. Falling back to global row index; cross-model alignment is weaker.")

    if "case_length" not in out.columns:
        # This is only exact if the file contains all prefixes for each case.
        out["case_length"] = out.groupby("case_id")["pos"].transform("max") + 1
        warnings_out.append("No case_length column found. Estimated from max observed position per case.")

    if "activity_current" not in out.columns:
        warnings_out.append("No current activity column found. Activity/branching analyses will be incomplete.")

    # Metadata.
    out.insert(0, "dataset", dataset)
    out.insert(1, "model", model)
    out["model_pretty"] = MODEL_PRETTY.get(model, model)
    out["paradigm"] = MODEL_PARADIGM.get(model, "unknown")
    out["source_path"] = str(path)
    out["row_idx_in_file"] = np.arange(len(out))

    # Types.
    out["case_id"] = out["case_id"].astype(str)
    out["pos"] = pd.to_numeric(out["pos"], errors="coerce").astype("Int64")
    # Derive position / prefix length where possible.
    if "pos" not in out.columns and "prefix_length" in out.columns:
        out["pos"] = pd.to_numeric(out["prefix_length"], errors="coerce") - 1
        warnings_out.append("Derived pos from prefix_length - 1.")

    if "case_id" not in out.columns:
        out["case_id"] = np.arange(len(out)).astype(str)
        warnings_out.append("No case_id column found. Falling back to row index; cross-model alignment is weaker.")

    if "pos" not in out.columns:
        out["pos"] = np.arange(len(out))
        warnings_out.append("No position column found. Falling back to global row index; cross-model alignment is weaker.")

    if "prefix_length" not in out.columns:
        out["prefix_length"] = pd.to_numeric(out["pos"], errors="coerce") + 1
        warnings_out.append("Derived prefix_length from pos + 1.")
        
    out["case_length"] = pd.to_numeric(out["case_length"], errors="coerce")

    for c in ["activity_current", "y_true_na", "y_pred_na"]:
        if c in out.columns:
            # Keep strings if labels are non-numeric, otherwise use numeric where possible.
            as_num = pd.to_numeric(out[c], errors="coerce")
            if as_num.notna().mean() > 0.98:
                out[c] = as_num.astype("Int64")
            else:
                out[c] = out[c].astype(str)

    for c in ["y_true_rt", "y_pred_rt", "y_true_nt", "y_pred_nt"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Row key for alignment.
    out["row_key"] = out["case_id"].astype(str) + "::" + out["pos"].astype(str)

    # Remove rows without usable row key.
    before = len(out)
    out = out[out["pos"].notna() & out["case_id"].notna()].copy()
    if len(out) < before:
        warnings_out.append(f"Dropped {before - len(out)} rows without valid case_id/pos.")

    # Check duplicates.
    dup = int(out.duplicated("row_key").sum())
    if dup:
        warnings_out.append(f"Found {dup} duplicate row_key values. Keeping all rows; alignment reports will flag this.")

    required_any = ["y_true_na", "y_true_rt", "y_true_nt"]
    if not any(c in out.columns for c in required_any):
        warnings_out.append("No recognized target columns found.")

    return out.reset_index(drop=True), warnings_out


def load_prediction_file(pred_dir: Path, dataset: str, model: str, seed: int) -> LoadedPrediction:
    path = pred_dir / dataset / model / f"seed_{seed}.parquet"
    if not path.exists():
        return LoadedPrediction(dataset, model, path, False, [], [], 0, 0, ["Missing file."], None)

    raw = pd.read_parquet(path)
    norm, warn = normalize_prediction_frame(raw, dataset, model, path)
    return LoadedPrediction(
        dataset=dataset,
        model=model,
        path=path,
        exists=True,
        raw_columns=list(raw.columns),
        normalized_columns=list(norm.columns),
        row_count_raw=len(raw),
        row_count_normalized=len(norm),
        warnings=warn,
        df=norm,
    )


def load_all_predictions(pred_dir: Path, datasets: list[str], models: list[str], seed: int) -> list[LoadedPrediction]:
    return [load_prediction_file(pred_dir, d, m, seed) for d in datasets for m in models]


# -----------------------------------------------------------------------------
# Feature engineering: prefix, process structure, branching, magnitude bins
# -----------------------------------------------------------------------------

def safe_cut_numeric(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    try:
        return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", np.nan)
    except Exception:
        return pd.Series(np.nan, index=s.index)


def safe_qcut(s: pd.Series, q: int, prefix: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="object")
    mask = x.notna()
    if mask.sum() == 0 or x[mask].nunique() < 2:
        out.loc[mask] = f"{prefix}_all"
        return out
    try:
        cats = pd.qcut(x[mask], q=min(q, x[mask].nunique()), duplicates="drop")
        out.loc[mask] = [f"{prefix}_{str(c)}" for c in cats.astype(str)]
    except Exception:
        out.loc[mask] = f"{prefix}_all"
    return out


def entropy_base2(values: pd.Series) -> float:
    p = values.value_counts(normalize=True, dropna=True).to_numpy(dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return float("nan")
    return float(-(p * np.log2(p)).sum())


def select_reference_frame(df_all: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Pick the largest model dump as the reference test set for structure statistics.
    This avoids using smaller sampled tabular dumps as the default structure source.
    """
    sub = df_all[df_all["dataset"] == dataset].copy()
    counts = sub.groupby("model")["row_key"].nunique().sort_values(ascending=False)
    if counts.empty:
        return sub.iloc[0:0].copy()
    ref_model = counts.index[0]
    ref = sub[sub["model"] == ref_model].drop_duplicates("row_key").copy()
    return ref


def build_structure_features(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(df_all["dataset"].unique()):
        ref = select_reference_frame(df_all, dataset)
        needed = {"activity_current", "y_true_na"}
        if ref.empty or not needed.issubset(ref.columns):
            continue
        ref = ref.dropna(subset=["activity_current", "y_true_na"]).copy()
        if ref.empty:
            continue

        n_total = len(ref)
        current_counts = ref["activity_current"].value_counts(dropna=False).rename("activity_count")

        branch = (
            ref.groupby("activity_current", dropna=False)
            .agg(
                branching_degree=("y_true_na", "nunique"),
                branching_entropy=("y_true_na", entropy_base2),
                transition_count=("y_true_na", "size"),
            )
            .reset_index()
        )
        branch["dataset"] = dataset
        branch["activity_frequency"] = branch["activity_current"].map(current_counts).astype(float)
        branch["activity_frequency_share"] = branch["activity_frequency"] / max(n_total, 1)
        rows.append(branch)

        # Store next-activity frequencies separately via merge later.
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_true_next_frequency(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(df_all["dataset"].unique()):
        ref = select_reference_frame(df_all, dataset)
        if ref.empty or "y_true_na" not in ref.columns:
            continue
        n_total = len(ref)
        tmp = (
            ref.groupby("y_true_na", dropna=False)
            .size()
            .rename("true_next_activity_frequency")
            .reset_index()
        )
        tmp["true_next_activity_frequency_share"] = tmp["true_next_activity_frequency"] / max(n_total, 1)
        tmp["dataset"] = dataset
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_analysis_features(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()

    # Prefix and case-position features.
    df["prefix_length"] = pd.to_numeric(df["prefix_length"], errors="coerce")
    df["case_length"] = pd.to_numeric(df["case_length"], errors="coerce")
    df["relative_prefix_position"] = np.where(
        df["case_length"].gt(0),
        df["prefix_length"] / df["case_length"],
        np.nan,
    )
    df["remaining_events"] = df["case_length"] - df["prefix_length"]

    df["prefix_length_bucket"] = safe_cut_numeric(
        df["prefix_length"],
        bins=[0, 1, 2, 3, 5, 10, 20, 50, np.inf],
        labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"],
    )
    df["relative_prefix_bucket"] = safe_cut_numeric(
        df["relative_prefix_position"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", ">100%"],
    )
    df["case_length_bucket"] = safe_cut_numeric(
        df["case_length"],
        bins=[0, 3, 5, 10, 20, 50, 100, np.inf],
        labels=["1-3", "4-5", "6-10", "11-20", "21-50", "51-100", ">100"],
    )

    # Classification correctness.
    if {"y_true_na", "y_pred_na"}.issubset(df.columns):
        df["na_correct"] = df["y_true_na"].astype(str).eq(df["y_pred_na"].astype(str))
        df["na_error"] = ~df["na_correct"]
    else:
        df["na_correct"] = np.nan
        df["na_error"] = np.nan

    # Regression errors.
    for short, true_col, pred_col in [
        ("rt", "y_true_rt", "y_pred_rt"),
        ("nt", "y_true_nt", "y_pred_nt"),
    ]:
        if {true_col, pred_col}.issubset(df.columns):
            df[f"{short}_residual"] = df[pred_col] - df[true_col]
            df[f"{short}_abs_error"] = df[f"{short}_residual"].abs()
            df[f"{short}_sq_error"] = df[f"{short}_residual"] ** 2
        else:
            df[f"{short}_residual"] = np.nan
            df[f"{short}_abs_error"] = np.nan
            df[f"{short}_sq_error"] = np.nan

    # Process-structure features from the largest available dump per dataset.
    structure = build_structure_features(df)
    if not structure.empty and "activity_current" in df.columns:
        df = df.merge(
            structure,
            on=["dataset", "activity_current"],
            how="left",
            validate="many_to_one",
        )

    next_freq = build_true_next_frequency(df)
    if not next_freq.empty and "y_true_na" in df.columns:
        df = df.merge(
            next_freq,
            on=["dataset", "y_true_na"],
            how="left",
            validate="many_to_one",
        )

    # Dataset-wise quantile bins for structural and target magnitude variables.
    for col, out_col, prefix in [
        ("activity_frequency", "activity_frequency_bucket", "act_freq"),
        ("true_next_activity_frequency", "true_next_activity_frequency_bucket", "next_freq"),
        ("branching_degree", "branching_degree_bucket", "branch_deg"),
        ("branching_entropy", "branching_entropy_bucket", "branch_ent"),
        ("y_true_rt", "rt_magnitude_bucket", "rt_mag"),
        ("y_true_nt", "nt_magnitude_bucket", "nt_mag"),
    ]:
        if col in df.columns:
            df[out_col] = (
                df.groupby("dataset", group_keys=False)[col]
                .apply(lambda s: safe_qcut(s, q=4, prefix=prefix))
            )
        else:
            df[out_col] = np.nan

    return df


# -----------------------------------------------------------------------------
# Sanity checks and alignment
# -----------------------------------------------------------------------------

def write_inventory(loaded: list[LoadedPrediction], out_dir: Path) -> pd.DataFrame:
    rows = []
    schema_rows = []
    for item in loaded:
        rows.append({
            "dataset": item.dataset,
            "model": item.model,
            "path": str(item.path),
            "exists": item.exists,
            "row_count_raw": item.row_count_raw,
            "row_count_normalized": item.row_count_normalized,
            "warnings": " | ".join(item.warnings),
        })
        for c in item.raw_columns:
            schema_rows.append({
                "dataset": item.dataset,
                "model": item.model,
                "path": str(item.path),
                "column": c,
                "normalized": c in item.normalized_columns,
            })
    inv = pd.DataFrame(rows)
    schema = pd.DataFrame(schema_rows)
    inv.to_csv(out_dir / "00_file_inventory.csv", index=False)
    schema.to_csv(out_dir / "00_schema_columns.csv", index=False)
    return inv


def build_alignment_summary(df_all: pd.DataFrame, expected_models: list[str]) -> pd.DataFrame:
    rows = []
    true_cols = [c for c in ["y_true_na", "y_true_rt", "y_true_nt", "case_length", "activity_current"] if c in df_all.columns]

    for dataset, df_d in df_all.groupby("dataset"):
        model_counts = df_d.groupby("model")["row_key"].nunique().sort_values(ascending=False)
        if model_counts.empty:
            continue
        ref_model = model_counts.index[0]
        ref = df_d[df_d["model"] == ref_model].drop_duplicates("row_key")
        ref_keys = set(ref["row_key"])

        for model in expected_models:
            sub = df_d[df_d["model"] == model].drop_duplicates("row_key")
            keys = set(sub["row_key"])
            common_keys = ref_keys & keys
            row = {
                "dataset": dataset,
                "reference_model": ref_model,
                "model": model,
                "exists_in_loaded_data": not sub.empty,
                "n_rows_model": len(sub),
                "n_unique_row_keys_model": len(keys),
                "n_duplicate_row_keys_model": int(df_d[df_d["model"] == model].duplicated("row_key").sum()),
                "n_reference_rows": len(ref_keys),
                "n_common_with_reference": len(common_keys),
                "n_missing_vs_reference": len(ref_keys - keys),
                "n_extra_vs_reference": len(keys - ref_keys),
                "common_share_of_reference": len(common_keys) / max(len(ref_keys), 1),
            }
            if sub.empty:
                rows.append(row)
                continue

            merged = ref[["row_key"] + true_cols].merge(
                sub[["row_key"] + true_cols],
                on="row_key",
                how="inner",
                suffixes=("_ref", "_model"),
            )
            for c in true_cols:
                left = f"{c}_ref"
                right = f"{c}_model"
                if left in merged.columns and right in merged.columns:
                    if c.startswith("y_true_") and pd.api.types.is_numeric_dtype(merged[left]):
                        mismatch = ~np.isclose(
                            pd.to_numeric(merged[left], errors="coerce"),
                            pd.to_numeric(merged[right], errors="coerce"),
                            equal_nan=True,
                        )
                    else:
                        mismatch = merged[left].astype(str).ne(merged[right].astype(str))
                    row[f"{c}_mismatch_count"] = int(mismatch.sum())
                    row[f"{c}_mismatch_rate"] = float(mismatch.mean()) if len(mismatch) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def filter_common_rows(df_all: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Keep only row_keys present for every available expected model within each dataset."""
    chunks = []
    for dataset, df_d in df_all.groupby("dataset"):
        present_models = [m for m in models if m in set(df_d["model"])]
        if not present_models:
            continue
        key_sets = [set(df_d.loc[df_d["model"] == m, "row_key"]) for m in present_models]
        common = set.intersection(*key_sets) if key_sets else set()
        tmp = df_d[df_d["model"].isin(present_models) & df_d["row_key"].isin(common)].copy()
        tmp["comparison_scope"] = "common_rows_all_present_models"
        chunks.append(tmp)
    if not chunks:
        return df_all.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def overall_metrics(df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    rows = []
    for (dataset, model), g in df.groupby(["dataset", "model"], dropna=False):
        row: dict[str, Any] = {
            "scope": scope_name,
            "dataset": dataset,
            "model": model,
            "model_pretty": MODEL_PRETTY.get(model, model),
            "paradigm": MODEL_PARADIGM.get(model, "unknown"),
            "n_rows": len(g),
            "n_cases": g["case_id"].nunique() if "case_id" in g.columns else np.nan,
        }
        if {"y_true_na", "y_pred_na"}.issubset(g.columns):
            mask = g["y_true_na"].notna() & g["y_pred_na"].notna()
            row["na_n"] = int(mask.sum())
            row["na_accuracy"] = float(g.loc[mask, "y_true_na"].astype(str).eq(g.loc[mask, "y_pred_na"].astype(str)).mean()) if mask.sum() else np.nan
            row["na_error_rate"] = 1.0 - row["na_accuracy"] if pd.notna(row["na_accuracy"]) else np.nan
        for short, true_col, pred_col in [
            ("rt", "y_true_rt", "y_pred_rt"),
            ("nt", "y_true_nt", "y_pred_nt"),
        ]:
            if {true_col, pred_col}.issubset(g.columns):
                mask = g[true_col].notna() & g[pred_col].notna()
                y = g.loc[mask, true_col].astype(float)
                p = g.loc[mask, pred_col].astype(float)
                err = p - y
                ae = err.abs()
                se = err ** 2
                row[f"{short}_n"] = int(mask.sum())
                row[f"{short}_mse"] = float(se.mean()) if mask.sum() else np.nan
                row[f"{short}_rmse"] = float(math.sqrt(se.mean())) if mask.sum() else np.nan
                row[f"{short}_mae"] = float(ae.mean()) if mask.sum() else np.nan
                row[f"{short}_median_ae"] = float(ae.median()) if mask.sum() else np.nan
                row[f"{short}_p90_ae"] = float(ae.quantile(0.90)) if mask.sum() else np.nan
                row[f"{short}_bias"] = float(err.mean()) if mask.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def na_metrics_by_group(df: pd.DataFrame, group_col: str, min_n: int) -> pd.DataFrame:
    if not {"y_true_na", "y_pred_na", group_col}.issubset(df.columns):
        return pd.DataFrame()
    x = df.dropna(subset=[group_col, "y_true_na", "y_pred_na"]).copy()
    if x.empty:
        return pd.DataFrame()
    x["_correct"] = x["y_true_na"].astype(str).eq(x["y_pred_na"].astype(str))
    out = (
        x.groupby(["dataset", "model", "paradigm", group_col], dropna=False)
        .agg(
            n=("_correct", "size"),
            accuracy=("_correct", "mean"),
            error_rate=("_correct", lambda s: 1.0 - float(s.mean())),
            n_cases=("case_id", "nunique"),
        )
        .reset_index()
    )
    out = out[out["n"] >= min_n].copy()
    out.insert(3, "group_col", group_col)
    out = out.rename(columns={group_col: "group_value"})
    return out


def temporal_metrics_by_group(df: pd.DataFrame, group_col: str, task: str, min_n: int) -> pd.DataFrame:
    cfg = TASKS[task]
    true_col = cfg["true"]
    pred_col = cfg["pred"]
    short = task.lower()
    if not {true_col, pred_col, group_col}.issubset(df.columns):
        return pd.DataFrame()
    x = df.dropna(subset=[group_col, true_col, pred_col]).copy()
    if x.empty:
        return pd.DataFrame()
    x["_residual"] = x[pred_col].astype(float) - x[true_col].astype(float)
    x["_abs_error"] = x["_residual"].abs()
    x["_sq_error"] = x["_residual"] ** 2
    out = (
        x.groupby(["dataset", "model", "paradigm", group_col], dropna=False)
        .agg(
            n=("_abs_error", "size"),
            mae=("_abs_error", "mean"),
            mse=("_sq_error", "mean"),
            median_ae=("_abs_error", "median"),
            p90_ae=("_abs_error", lambda s: float(s.quantile(0.90))),
            bias=("_residual", "mean"),
            n_cases=("case_id", "nunique"),
        )
        .reset_index()
    )
    out = out[out["n"] >= min_n].copy()
    out.insert(3, "task", task)
    out.insert(4, "group_col", group_col)
    out = out.rename(columns={group_col: "group_value"})
    return out


def build_group_tables(df: pd.DataFrame, out_dir: Path, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    na_tables = []
    for group_col in GROUP_COLS_NA:
        tab = na_metrics_by_group(df, group_col, min_n=min_n)
        if not tab.empty:
            tab.to_csv(out_dir / f"20_na_by_{group_col}.csv", index=False)
            na_tables.append(tab)
    na_all = pd.concat(na_tables, ignore_index=True) if na_tables else pd.DataFrame()
    if not na_all.empty:
        na_all.to_csv(out_dir / "20_na_all_group_tables.csv", index=False)

    temp_tables = []
    for task in ["RT", "NT"]:
        for group_col in GROUP_COLS_TEMPORAL:
            tab = temporal_metrics_by_group(df, group_col, task=task, min_n=min_n)
            if not tab.empty:
                tab.to_csv(out_dir / f"30_{task.lower()}_by_{group_col}.csv", index=False)
                temp_tables.append(tab)
    temp_all = pd.concat(temp_tables, ignore_index=True) if temp_tables else pd.DataFrame()
    if not temp_all.empty:
        temp_all.to_csv(out_dir / "30_temporal_all_group_tables.csv", index=False)
    return na_all, temp_all


# -----------------------------------------------------------------------------
# Paradigm comparisons and evidence tables
# -----------------------------------------------------------------------------

def add_paradigm_means_from_pivot(pivot: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    out = pivot.copy()
    sequence = [m for m in ["rnn", "transformer"] if m in metric_cols]
    tabular = [m for m in ["tabpfn3", "saprpt"] if m in metric_cols]
    llm = [m for m in ["llama32-1b", "gemma-2-2b"] if m in metric_cols]
    out["sequence_mean"] = out[sequence].mean(axis=1) if sequence else np.nan
    out["tabular_fm_mean"] = out[tabular].mean(axis=1) if tabular else np.nan
    out["llm_mean"] = out[llm].mean(axis=1) if llm else np.nan
    return out


def paradigm_comparison_na(na_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if na_all.empty:
        return pd.DataFrame()
    idx_cols = ["dataset", "group_col", "group_value"]
    pivot = (
        na_all.pivot_table(
            index=idx_cols,
            columns="model",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.columns.name = None
    model_cols = [c for c in pivot.columns if c in DEFAULT_MODELS]
    pivot = add_paradigm_means_from_pivot(pivot, model_cols)
    pivot["sequence_minus_tabular_acc"] = pivot["sequence_mean"] - pivot["tabular_fm_mean"]
    pivot["sequence_minus_llm_acc"] = pivot["sequence_mean"] - pivot["llm_mean"]
    pivot["transformer_minus_llm_mean_acc"] = pivot.get("transformer", np.nan) - pivot["llm_mean"]
    pivot["rnn_minus_llm_mean_acc"] = pivot.get("rnn", np.nan) - pivot["llm_mean"]
    pivot.to_csv(out_dir / "40_paradigm_comparison_na_by_slices.csv", index=False)

    top_tabular_gaps = pivot.sort_values("sequence_minus_tabular_acc", ascending=False).head(100)
    top_llm_gaps = pivot.sort_values("sequence_minus_llm_acc", ascending=False).head(100)
    top_tabular_gaps.to_csv(out_dir / "41_top_sequence_over_tabular_na_slices.csv", index=False)
    top_llm_gaps.to_csv(out_dir / "42_top_sequence_over_llm_na_slices.csv", index=False)
    return pivot


def paradigm_comparison_temporal(temp_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if temp_all.empty:
        return pd.DataFrame()
    idx_cols = ["dataset", "task", "group_col", "group_value"]
    pivot = (
        temp_all.pivot_table(
            index=idx_cols,
            columns="model",
            values="mae",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivot.columns.name = None
    model_cols = [c for c in pivot.columns if c in DEFAULT_MODELS]
    pivot = add_paradigm_means_from_pivot(pivot, model_cols)
    # Positive means tabular/LLM has larger MAE than sequence. Negative means better than sequence.
    pivot["tabular_minus_sequence_mae"] = pivot["tabular_fm_mean"] - pivot["sequence_mean"]
    pivot["llm_minus_sequence_mae"] = pivot["llm_mean"] - pivot["sequence_mean"]
    pivot["tabular_relative_to_sequence_mae"] = pivot["tabular_fm_mean"] / pivot["sequence_mean"]
    pivot["llm_relative_to_sequence_mae"] = pivot["llm_mean"] / pivot["sequence_mean"]
    pivot.to_csv(out_dir / "50_paradigm_comparison_temporal_by_slices.csv", index=False)

    competitive = pivot.sort_values("tabular_minus_sequence_mae", ascending=True).head(100)
    competitive.to_csv(out_dir / "51_top_tabular_competitive_temporal_slices.csv", index=False)
    return pivot


def top_confusions(df: pd.DataFrame, out_dir: Path, top_k: int = 50) -> pd.DataFrame:
    if not {"y_true_na", "y_pred_na"}.issubset(df.columns):
        return pd.DataFrame()
    x = df.dropna(subset=["y_true_na", "y_pred_na"]).copy()
    x = x[x["y_true_na"].astype(str).ne(x["y_pred_na"].astype(str))]
    if x.empty:
        return pd.DataFrame()
    out = (
        x.groupby(["dataset", "model", "y_true_na", "y_pred_na"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = x.groupby(["dataset", "model"]).size().rename("n_errors_total").reset_index()
    out = out.merge(totals, on=["dataset", "model"], how="left")
    out["share_of_model_errors"] = out["n"] / out["n_errors_total"]
    out = (
        out.sort_values(["dataset", "model", "n"], ascending=[True, True, False])
        .groupby(["dataset", "model"], group_keys=False)
        .head(top_k)
    )
    out.to_csv(out_dir / "21_top_na_confusions.csv", index=False)
    return out


def disagreement_examples(df_common: pd.DataFrame, out_dir: Path, top_k: int = 500) -> pd.DataFrame:
    """Rows where sequence models are correct and LLM/tabular models fail, or vice versa."""
    if not {"y_true_na", "y_pred_na", "row_key"}.issubset(df_common.columns):
        return pd.DataFrame()
    x = df_common.copy()
    x["correct"] = x["y_true_na"].astype(str).eq(x["y_pred_na"].astype(str))
    wide = x.pivot_table(
        index=["dataset", "row_key", "case_id", "pos", "prefix_length", "case_length", "activity_current", "y_true_na"],
        columns="model",
        values="correct",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    for m in DEFAULT_MODELS:
        if m not in wide.columns:
            wide[m] = np.nan

    wide["n_sequence_correct"] = wide[["rnn", "transformer"]].sum(axis=1, min_count=1)
    wide["n_tabular_correct"] = wide[["tabpfn3", "saprpt"]].sum(axis=1, min_count=1)
    wide["n_llm_correct"] = wide[["llama32-1b", "gemma-2-2b"]].sum(axis=1, min_count=1)
    wide["sequence_correct_tabular_wrong"] = (wide["n_sequence_correct"] >= 1) & (wide["n_tabular_correct"] == 0)
    wide["sequence_correct_llm_wrong"] = (wide["n_sequence_correct"] >= 1) & (wide["n_llm_correct"] == 0)
    wide["llm_correct_sequence_wrong"] = (wide["n_llm_correct"] >= 1) & (wide["n_sequence_correct"] == 0)
    wide["tabular_correct_sequence_wrong"] = (wide["n_tabular_correct"] >= 1) & (wide["n_sequence_correct"] == 0)

    # Attach predictions for interpretability.
    pred_wide = x.pivot_table(
        index=["dataset", "row_key"],
        columns="model",
        values="y_pred_na",
        aggfunc="first",
    ).reset_index()
    pred_wide = pred_wide.rename(columns={m: f"pred_{m}" for m in DEFAULT_MODELS if m in pred_wide.columns})
    out = wide.merge(pred_wide, on=["dataset", "row_key"], how="left")

    flags = [
        "sequence_correct_tabular_wrong",
        "sequence_correct_llm_wrong",
        "llm_correct_sequence_wrong",
        "tabular_correct_sequence_wrong",
    ]
    out = out[out[flags].any(axis=1)].head(top_k).copy()
    out.to_csv(out_dir / "43_na_disagreement_examples_common_rows.csv", index=False)
    return out


def write_research_evidence_markdown(
    out_dir: Path,
    overall_all: pd.DataFrame,
    overall_common: pd.DataFrame,
    na_paradigm: pd.DataFrame,
    temp_paradigm: pd.DataFrame,
) -> None:
    """
    Writes a factual template. It intentionally does not invent conclusions;
    it points to the generated quantitative tables that support each RQ.
    """
    lines = []
    lines.append("# Error and behavior analysis: evidence map\n")
    lines.append("This file is generated from the prediction parquet files. Interpretations should be written only after inspecting the CSV tables referenced below.\n")

    lines.append("## Core metric tables\n")
    lines.append("- `10_overall_metrics_all_rows.csv`: native metrics on each model's available dump.\n")
    lines.append("- `11_overall_metrics_common_rows.csv`: paired metrics restricted to row keys present for all available final models per dataset. Use this for fair model deltas.\n")
    lines.append("- `00_alignment_summary.csv`: row-key and true-label alignment diagnostics.\n")

    if not overall_common.empty:
        lines.append("\n## Paired common-row overall snapshot\n")
        keep_cols = [
            "dataset", "model", "paradigm", "n_rows", "na_accuracy",
            "rt_mae", "rt_mse", "nt_mae", "nt_mse",
        ]
        keep_cols = [c for c in keep_cols if c in overall_common.columns]
        snapshot_df = overall_common[keep_cols].sort_values(["dataset", "model"])
        try:
            snapshot = snapshot_df.to_markdown(index=False)
        except Exception:
            snapshot = "```text\n" + snapshot_df.to_string(index=False) + "\n```"
        lines.append(snapshot + "\n")

    lines.append("\n## RQ1: Why do tabular foundation models perform worse on NA than sequence models?\n")
    lines.append("Empirical evidence to inspect first:\n")
    lines.append("- `40_paradigm_comparison_na_by_slices.csv`\n")
    lines.append("- `41_top_sequence_over_tabular_na_slices.csv`\n")
    lines.append("- `20_na_by_branching_entropy_bucket.csv`\n")
    lines.append("- `20_na_by_activity_frequency_bucket.csv`\n")
    lines.append("- `20_na_by_prefix_length_bucket.csv`\n")
    lines.append("Interpretation rule: claim a tabular control-flow limitation only if the sequence-minus-tabular NA gap increases on rare/high-entropy/high-branching states or longer prefixes on common rows.\n")

    lines.append("\n## RQ2: Why are tabular models more competitive on RT and NT?\n")
    lines.append("Empirical evidence to inspect first:\n")
    lines.append("- `50_paradigm_comparison_temporal_by_slices.csv`\n")
    lines.append("- `51_top_tabular_competitive_temporal_slices.csv`\n")
    lines.append("- `30_RT_by_rt_magnitude_bucket.csv`\n")
    lines.append("- `30_NT_by_nt_magnitude_bucket.csv`\n")
    lines.append("Interpretation rule: argue competitiveness only where tabular-minus-sequence MAE/MSE is close to zero or negative, preferably consistently across datasets or magnitude bins.\n")

    lines.append("\n## RQ3: Why do LLMs underperform sequence models on NA despite being pretrained sequence models?\n")
    lines.append("Empirical evidence to inspect first:\n")
    lines.append("- `42_top_sequence_over_llm_na_slices.csv`\n")
    lines.append("- `43_na_disagreement_examples_common_rows.csv`\n")
    lines.append("- `20_na_by_true_next_activity_frequency_bucket.csv`\n")
    lines.append("Interpretation rule: distinguish architecture/pretraining mismatch from data/hyperparameter effects. Only claim process-control-flow weakness if degradation is systematic across datasets and structural bins.\n")

    lines.append("\n## RQ4: Are LLMs better suited for temporal tasks than for discrete control-flow prediction?\n")
    lines.append("Compare each LLM's NA rank/gap with RT and NT MAE/MSE gaps in `11_overall_metrics_common_rows.csv` and `50_paradigm_comparison_temporal_by_slices.csv`.\n")

    lines.append("\n## RQ5: Paradigm limitation vs. dataset-specific effects\n")
    lines.append("Use cross-dataset consistency. A pattern is paradigm-level only if it appears in several datasets under paired common-row evaluation. A pattern isolated to one BPI log should be framed as dataset-specific.\n")

    (out_dir / "90_research_question_evidence_map.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Optional plots
# -----------------------------------------------------------------------------

def save_plot(fig: Any, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_plots(out_dir: Path, overall_common: pd.DataFrame, na_all: pd.DataFrame, temp_all: pd.DataFrame) -> None:
    if not PLOTTING_AVAILABLE:
        warnings.warn("Plotting libraries unavailable. Skipping plots.")
        return
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

    if not overall_common.empty and "na_accuracy" in overall_common.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=overall_common, x="dataset", y="na_accuracy", hue="model", ax=ax)
        ax.set_title("NA accuracy by dataset and model, common rows")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("NA accuracy")
        ax.tick_params(axis="x", rotation=30)
        save_plot(fig, plot_dir / "overall_na_accuracy_common_rows.png")

    for metric, label in [("rt_mae", "RT MAE"), ("nt_mae", "NT MAE")]:
        if not overall_common.empty and metric in overall_common.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=overall_common, x="dataset", y=metric, hue="model", ax=ax)
            ax.set_title(f"{label} by dataset and model, common rows")
            ax.set_xlabel("Dataset")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=30)
            save_plot(fig, plot_dir / f"overall_{metric}_common_rows.png")

    if not na_all.empty:
        for group_col in ["prefix_length_bucket", "branching_entropy_bucket", "activity_frequency_bucket"]:
            sub = na_all[na_all["group_col"] == group_col].copy()
            if sub.empty:
                continue
            for dataset, g in sub.groupby("dataset"):
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=g, x="group_value", y="accuracy", hue="model", marker="o", ax=ax)
                ax.set_title(f"{dataset}: NA accuracy by {group_col}")
                ax.set_xlabel(group_col)
                ax.set_ylabel("NA accuracy")
                ax.tick_params(axis="x", rotation=30)
                save_plot(fig, plot_dir / f"na_{dataset}_{group_col}.png")

    if not temp_all.empty:
        for task in ["RT", "NT"]:
            group_col = "rt_magnitude_bucket" if task == "RT" else "nt_magnitude_bucket"
            sub = temp_all[(temp_all["task"] == task) & (temp_all["group_col"] == group_col)].copy()
            if sub.empty:
                continue
            for dataset, g in sub.groupby("dataset"):
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=g, x="group_value", y="mae", hue="model", marker="o", ax=ax)
                ax.set_title(f"{dataset}: {task} MAE by target magnitude")
                ax.set_xlabel(group_col)
                ax.set_ylabel(f"{task} MAE")
                ax.tick_params(axis="x", rotation=30)
                save_plot(fig, plot_dir / f"{task.lower()}_{dataset}_{group_col}.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structured PPM prediction error analysis")

    p.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help=f"Prediction parquet base directory. Default: {DEFAULT_PRED_DIR}",
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for all generated analysis files. Default: {DEFAULT_OUT_DIR}",
    )

    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--min-group-n", type=int, default=30)

    p.add_argument(
        "--save-long",
        action="store_true",
        help="Persist normalized long prediction tables as parquet.",
    )

    p.add_argument("--make-plots", action="store_true")

    return p.parse_args()


EVENT_LOGS = {
    "BPI12": BPI12, "BPI17": BPI17,
    "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    "BPI20TravelPermitData": BPI20TravelPermitData,
    "BPI20RequestForPayment": BPI20RequestForPayment,
}

_vocab_cache: dict[str, dict[int, str]] = {}

def get_activity_vocab(dataset: str) -> dict[int, str]:
    """Rebuild the exact itos['activity'] mapping used at training time."""
    if dataset in _vocab_cache:
        return _vocab_cache[dataset]
    log = EVENT_LOGS[dataset]()
    train, _ = prepare_data(log.dataframe, log.unbiased_split_params)
    train_log = EventLog(
        dataframe=train, case_id="case_id",
        features=EventFeatures(categorical=["activity"], numerical=[]),
        targets=EventTargets(categorical=["activity"], numerical=[]),
        train_split=True, name=dataset,
    )
    itos = dict(train_log.itos["activity"])  # {0:<PAD>, 1:<UNK>, 2:<EOS>, 3:'A_SUBMITTED', ...}
    _vocab_cache[dataset] = itos
    return itos

def attach_activity_labels_to_csvs(out_dir: Path, datasets: list[str]) -> None:
    """Walk all CSVs in out_dir and add *_label columns next to activity IDs."""
    vocabs = {ds: get_activity_vocab(ds) for ds in datasets}

    def map_ids(ds: str, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").map(vocabs[ds])

    ACTIVITY_GROUP_COLS = {"activity_current", "y_true_na"}
    ID_COLS = ["activity_current", "y_true_na", "y_pred_na"]

    for path in sorted(out_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            continue
        changed = False
        for col in ID_COLS + [c for c in df.columns if c.startswith("pred_")]:
            if col in df.columns:
                df[f"{col}_label"] = df.apply(lambda r: vocabs[r["dataset"]].get(int(r[col]))
                                              if pd.notna(r[col]) else None, axis=1)
                changed = True
        if {"group_col", "group_value"}.issubset(df.columns):
            mask = df["group_col"].isin(ACTIVITY_GROUP_COLS)
            if mask.any():
                df.loc[mask, "group_value_label"] = df[mask].apply(
                    lambda r: vocabs[r["dataset"]].get(int(r["group_value"]))
                    if pd.notna(r["group_value"]) else None, axis=1)
                changed = True
        if path.name in {"20_na_by_activity_current.csv", "20_na_by_y_true_na.csv",
                         "30_rt_by_activity_current.csv", "30_nt_by_activity_current.csv"}:
            df["group_value_label"] = df.apply(
                lambda r: vocabs[r["dataset"]].get(int(r["group_value"]))
                if pd.notna(r["group_value"]) else None, axis=1)
            changed = True
        if changed:
            df.to_csv(path, index=False)
            print(f"labelled {path.name}")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # 1. Load and inspect all files.
    loaded = load_all_predictions(args.pred_dir, args.datasets, args.models, args.seed)
    inventory = write_inventory(loaded, args.out_dir)

    frames = [x.df for x in loaded if x.exists and x.df is not None and not x.df.empty]
    if not frames:
        raise RuntimeError(f"No prediction parquet files could be loaded from {args.pred_dir}")

    df_all_raw = pd.concat(frames, ignore_index=True)
    df_all = add_analysis_features(df_all_raw)

    if args.save_long:
        df_all.to_parquet(args.out_dir / "01_predictions_normalized_long.parquet", index=False)

    # 2. Alignment diagnostics.
    alignment = build_alignment_summary(df_all, args.models)
    alignment.to_csv(args.out_dir / "00_alignment_summary.csv", index=False)

    # 3. Native and paired-common metrics.
    df_common = filter_common_rows(df_all, args.models)
    if args.save_long:
        df_common.to_parquet(args.out_dir / "02_predictions_common_rows_long.parquet", index=False)

    overall_all = overall_metrics(df_all, scope_name="all_available_rows")
    overall_common = overall_metrics(df_common, scope_name="common_rows_all_present_models")
    overall_all.to_csv(args.out_dir / "10_overall_metrics_all_rows.csv", index=False)
    overall_common.to_csv(args.out_dir / "11_overall_metrics_common_rows.csv", index=False)

    # 4. Grouped error analysis. Use common rows for fair cross-model comparisons.
    grouped_base = df_common if not df_common.empty else df_all
    na_all, temp_all = build_group_tables(grouped_base, args.out_dir, min_n=args.min_group_n)

    # 5. NA confusion and disagreement examples.
    top_confusions(grouped_base, args.out_dir)
    disagreement_examples(df_common, args.out_dir)

    # 6. Paradigm comparison evidence tables.
    na_paradigm = paradigm_comparison_na(na_all, args.out_dir)
    temp_paradigm = paradigm_comparison_temporal(temp_all, args.out_dir)

    # 7. Research evidence map.
    write_research_evidence_markdown(args.out_dir, overall_all, overall_common, na_paradigm, temp_paradigm)

    # 8. Optional plots.
    if args.make_plots:
        make_plots(args.out_dir, overall_common, na_all, temp_all)
        
    attach_activity_labels_to_csvs(args.out_dir, args.datasets)

    # 9. Small machine-readable run summary.
    summary = {
        "pred_dir": str(args.pred_dir),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "datasets": args.datasets,
        "models": args.models,
        "n_expected_files": len(args.datasets) * len(args.models),
        "n_existing_files": int(inventory["exists"].sum()) if "exists" in inventory else None,
        "n_rows_all_loaded": int(len(df_all)),
        "n_rows_common": int(len(df_common)),
        "outputs": sorted(
            str(p.relative_to(args.out_dir))
            for p in args.out_dir.rglob("*")
            if p.is_file()
            ),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
