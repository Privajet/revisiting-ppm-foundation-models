import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from chronos import Chronos2Pipeline


def _resolve_torch_dtype(torch_dtype: Any) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype is None:
        return torch.float32
    if isinstance(torch_dtype, str):
        name = torch_dtype.replace("torch.", "")
        if not hasattr(torch, name):
            raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}")
        out = getattr(torch, name)
        if not isinstance(out, torch.dtype):
            raise ValueError(f"{torch_dtype!r} does not resolve to a torch.dtype")
        return out
    raise TypeError("torch_dtype must be either a torch.dtype or a string")


def _ordered_with_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve within-case event order from the dataframe row order.
    This is important because timestamps are no longer present in train_log/test_log.
    """
    out = df.reset_index(drop=False).rename(columns={"index": "_row_id"}).copy()
    out = out.sort_values(["case_id", "_row_id"], kind="mergesort").reset_index(drop=True)
    out["pos"] = out.groupby("case_id").cumcount()
    out["case_len"] = out.groupby("case_id")["pos"].transform("max") + 1
    out["remaining_events"] = out["case_len"] - out["pos"] - 1
    return out


def _decode_activity_array(activity_ids: np.ndarray, itos_activity: dict[int, str]) -> np.ndarray:
    return np.asarray(
        [itos_activity.get(int(x), "<UNK>") for x in activity_ids],
        dtype=object,
    )


def _singleton_cov_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray([value], dtype=object)
    return np.asarray([value], dtype=np.float32)


def _as_numpy(pred: Any) -> np.ndarray:
    if isinstance(pred, torch.Tensor):
        return pred.detach().cpu().numpy()
    return np.asarray(pred)


def _first_scalar(pred: Any) -> float:
    arr = _as_numpy(pred).reshape(-1)
    return float(arr[0])


def _sum_horizon(pred: Any, horizon: int) -> float:
    arr = _as_numpy(pred).reshape(-1)
    return float(arr[:horizon].sum())


def _hier_lookup(
    pos: int,
    activity_id: int,
    *,
    pos_act: dict[tuple[int, int], float],
    activity: dict[int, float],
    position: dict[int, float],
    global_default: float,
) -> float:
    """
    Hierarchical backoff:
      1) (prefix_length, activity)
      2) activity
      3) prefix_length
      4) global default
    """
    if (pos, activity_id) in pos_act:
        return float(pos_act[(pos, activity_id)])
    if activity_id in activity:
        return float(activity[activity_id])
    if pos in position:
        return float(position[pos])
    return float(global_default)


def _build_empirical_stats(train_df_ordered: pd.DataFrame) -> dict[str, Any]:
    df = train_df_ordered.copy()
    df["activity_id"] = df["activity"].astype(int)

    stats = {
        # Global fallbacks
        "nt_global": float(df["time_to_next_event"].astype(float).mean()),
        "rt_global": float(df["remaining_time"].astype(float).mean()),
        "rem_global": float(df["remaining_events"].astype(float).median()),

        # NT fallbacks
        "nt_by_pos_act": df.groupby(["pos", "activity_id"])["time_to_next_event"].mean().to_dict(),
        "nt_by_act": df.groupby("activity_id")["time_to_next_event"].mean().to_dict(),
        "nt_by_pos": df.groupby("pos")["time_to_next_event"].mean().to_dict(),

        # RT fallbacks
        "rt_by_pos_act": df.groupby(["pos", "activity_id"])["remaining_time"].mean().to_dict(),
        "rt_by_act": df.groupby("activity_id")["remaining_time"].mean().to_dict(),
        "rt_by_pos": df.groupby("pos")["remaining_time"].mean().to_dict(),

        # Remaining-event horizon estimator for RT rollout
        "rem_by_pos_act": df.groupby(["pos", "activity_id"])["remaining_events"].median().to_dict(),
        "rem_by_act": df.groupby("activity_id")["remaining_events"].median().to_dict(),
        "rem_by_pos": df.groupby("pos")["remaining_events"].median().to_dict(),
    }
    return stats


def _build_case_cache(
    df_ordered: pd.DataFrame,
    feature_cols: list[str],
    itos_activity: dict[int, str],
) -> dict[Any, dict[str, Any]]:
    cache = {}
    for cid, g in df_ordered.groupby("case_id", sort=False):
        g = g.sort_values("pos")
        features = {}
        for c in feature_cols:
            values = g[c].to_numpy()
            features[c] = values.astype(np.float32, copy=False)

        cache[cid] = {
            "nt": g["time_to_next_event"].astype(np.float32).to_numpy(),
            "rt": g["remaining_time"].astype(np.float32).to_numpy(),
            "features": features,
        }
    return cache


def run_chronos2_baseline(
    train_log,
    test_log,
    random_state: int,
    model_id: str = "amazon/chronos-2",
    device_map: str = "cuda",
    torch_dtype: Any = "float32",
    max_n_test: int = 2000,
    chronos_batch_size: int = 128,
    cross_learning: bool = False,
    na_strategy: str = "nan",      # "nan" (recommended) or "majority" (placeholder)
    evaluate_rt: bool = True,
):
    """
    Chronos-2 baseline for PPM.

    Recommended academic interpretation:
      - NT: direct Chronos-2 one-step forecast
      - RT: derived by summing an h-step NT rollout, where h is an empirical
            remaining-event estimate from the training split
      - NA: not supported natively; return NaN (recommended) or a majority placeholder

    Notes:
      - This baseline operates in the SAME standardized target space as your current code.
      - It uses only train-split statistics for fallbacks / RT horizon estimation.
      - It never uses past remaining_time values as model history (to avoid leakage).
    """
    start_time = time.perf_counter()

    if na_strategy not in {"nan", "majority"}:
        raise ValueError("na_strategy must be either 'nan' or 'majority'")

    dtype = _resolve_torch_dtype(torch_dtype)

    df_train = train_log.dataframe.copy()
    df_test = test_log.dataframe.copy()

    feature_cols = list(train_log.features.numerical)

    train_ord = _ordered_with_pos(df_train)
    test_ord = _ordered_with_pos(df_test)

    # Same max_n_test convention as your tabular baselines
    if len(test_ord) > max_n_test:
        df_eval = test_ord.sample(n=max_n_test, random_state=random_state).reset_index(drop=True)
    else:
        df_eval = test_ord.reset_index(drop=True)

    stats = _build_empirical_stats(train_ord)

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
    )

    itos_activity = train_log.itos["activity"]
    case_cache = _build_case_cache(
        test_ord,
        feature_cols=feature_cols,
        itos_activity=itos_activity,
    )

    # Ground truth arrays
    y_true_nt = df_eval["time_to_next_event"].astype(float).to_numpy()
    y_true_rt = df_eval["remaining_time"].astype(float).to_numpy()
    y_true_na = df_eval["next_activity"].astype(int).to_numpy()

    # Default predictions = train-split empirical fallbacks
    y_pred_nt = np.full(len(df_eval), stats["nt_global"], dtype=np.float32)
    y_pred_rt = np.full(len(df_eval), stats["rt_global"], dtype=np.float32)

    if na_strategy == "majority":
        eos_id = int(train_log.special_tokens["<EOS>"])
        mask = df_train["next_activity"] != eos_id
        majority_class_id = int(df_train.loc[mask, "next_activity"].mode().iloc[0])
        y_pred_na = np.full(len(df_eval), majority_class_id, dtype=np.int64)
        na_acc = float(accuracy_score(y_true_na, y_pred_na))
    else:
        y_pred_na = np.full(len(df_eval), -1, dtype=np.int64)
        na_acc = float("nan")

    # Batch containers
    nt_inputs = []
    nt_out_idx = []

    # RT inputs need to be bucketed by horizon, because Chronos-2 uses one
    # prediction_length per call.
    rt_inputs_by_horizon = defaultdict(list)
    rt_out_idx_by_horizon = defaultdict(list)

    for i, row in enumerate(df_eval.itertuples(index=False)):
        pos = int(row.pos)
        activity_id = int(row.activity)

        # Boundary case: first event of a case has no observable NT history
        if pos <= 0:
            y_pred_nt[i] = _hier_lookup(
                pos,
                activity_id,
                pos_act=stats["nt_by_pos_act"],
                activity=stats["nt_by_act"],
                position=stats["nt_by_pos"],
                global_default=stats["nt_global"],
            )
            y_pred_rt[i] = _hier_lookup(
                pos,
                activity_id,
                pos_act=stats["rt_by_pos_act"],
                activity=stats["rt_by_act"],
                position=stats["rt_by_pos"],
                global_default=stats["rt_global"],
            )
            continue

        case = case_cache[row.case_id]

        # Observable NT history only: gaps that have already occurred
        target_hist = case["nt"][:pos]

        # Observable past covariates aligned with target history
        past_cov = {c: case["features"][c][:pos] for c in feature_cols}

        # For one-step NT, the CURRENT event covariates are known and can be
        # passed as future_covariates for horizon=1.
        future_cov_one_step = {
            c: _singleton_cov_value(case["features"][c][pos])
            for c in feature_cols
        }

        nt_inputs.append(
            {
                "target": target_hist,
                "past_covariates": past_cov,
                "future_covariates": future_cov_one_step,
            }
        )
        nt_out_idx.append(i)

        if evaluate_rt:
            # Estimate how many future event gaps remain from this prefix.
            rem_steps = int(round(_hier_lookup(
                pos,
                activity_id,
                pos_act=stats["rem_by_pos_act"],
                activity=stats["rem_by_act"],
                position=stats["rem_by_pos"],
                global_default=stats["rem_global"],
            )))
            rem_steps = max(0, rem_steps)

            if rem_steps == 0:
                y_pred_rt[i] = 0.0
            else:
                # RT is derived from an h-step NT rollout.
                # No future covariates are used here because they are unknown
                # beyond the current event.
                rt_inputs_by_horizon[rem_steps].append(
                    {
                        "target": target_hist,
                        "past_covariates": past_cov,
                    }
                )
                rt_out_idx_by_horizon[rem_steps].append(i)

    # ---------- NT: direct one-step forecast ----------
    if nt_inputs:
        _, mean_nt = pipeline.predict_quantiles(
            inputs=nt_inputs,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=min(chronos_batch_size, len(nt_inputs)),
            cross_learning=cross_learning,
        )

        mean_iter = mean_nt if not isinstance(mean_nt, torch.Tensor) else list(mean_nt)
        for idx, pred in zip(nt_out_idx, mean_iter):
            y_pred_nt[idx] = _first_scalar(pred)

    # ---------- RT: h-step rollout over NT ----------
    if evaluate_rt:
        for horizon, inputs_h in rt_inputs_by_horizon.items():
            _, mean_rt = pipeline.predict_quantiles(
                inputs=inputs_h,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                batch_size=min(chronos_batch_size, len(inputs_h)),
                cross_learning=cross_learning,
            )

            mean_iter = mean_rt if not isinstance(mean_rt, torch.Tensor) else list(mean_rt)
            for idx, pred in zip(rt_out_idx_by_horizon[horizon], mean_iter):
                y_pred_rt[idx] = _sum_horizon(pred, horizon)


    return {
        "test_next_activity_acc": na_acc,
        "test_next_remaining_time_loss": float(mean_squared_error(y_true_rt, y_pred_rt)),
        "test_next_time_to_next_event_loss": float(mean_squared_error(y_true_nt, y_pred_nt)),
        "y_true_pred_dump": {
            "y_true_next_activity": y_true_na,
            "y_pred_next_activity": y_pred_na,
            "y_true_remaining_time": y_true_rt,
            "y_pred_remaining_time": y_pred_rt,
            "y_true_time_to_next_event": y_true_nt,
            "y_pred_time_to_next_event": y_pred_nt,
        },
    }