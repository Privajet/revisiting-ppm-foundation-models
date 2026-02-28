import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from chronos import Chronos2Pipeline


def run_chronos2_baseline(
    train_log,
    test_log,
    random_state: int,
    model_id: str = "amazon/chronos-2",
    device_map: str = "cuda",
    torch_dtype: str = "float32",
    max_n_test: int = 2000,
    chronos_batch_size: int = 128,
    cross_learning: bool = False,
):
    """
    Chronos-2 baseline for PPM:
      - NA (next_activity): majority class
      - RT (remaining_time): constant mean baseline (to avoid label leakage)
      - NT (time_to_next_event): Chronos-2 one-step forecast using past NT + covariates

    Chronos-2 dict input format supports:
      target: 1d array (history_length,)
      past_covariates: dict[name -> 1d array of length history_length]
      future_covariates: dict[name -> 1d array of length prediction_length]
    See upstream schema. :contentReference[oaicite:2]{index=2}
    """

    df_train = train_log.dataframe
    df_test = test_log.dataframe

    feature_cols = list(train_log.features.categorical) + list(train_log.features.numerical)

    # Build a deterministic within-case order for time-series slicing
    df_full = df_test.reset_index(drop=False).rename(columns={"index": "_row_id"})
    df_full = df_full.sort_values(["case_id", "_row_id"], kind="mergesort")
    df_full["pos"] = df_full.groupby("case_id").cumcount()

    # Sample evaluation rows (align with other tabular baselines)
    if len(df_full) > max_n_test:
        df_eval = df_full.sample(n=max_n_test, random_state=random_state).reset_index(drop=True)
    else:
        df_eval = df_full.reset_index(drop=True)

    # Majority class for NA (exclude EOS positions from training statistics)
    eos_id = int(train_log.special_tokens["<EOS>"])
    mask = df_train["next_activity"] != eos_id
    majority_class_id = int(df_train.loc[mask, "next_activity"].mode().iloc[0])

    y_true_na = df_eval["next_activity"].to_numpy()
    y_pred_na = np.full(len(df_eval), majority_class_id, dtype=np.int64)

    # Constant mean RT baseline (avoid leaking past remaining_time history)
    mean_rt = float(df_train.loc[mask, "remaining_time"].astype(float).mean())
    y_true_rt = df_eval["remaining_time"].astype(float).to_numpy()
    y_pred_rt = np.full(len(df_eval), mean_rt, dtype=np.float32)

    # Chronos-2 for NT
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )  # usage matches model card :contentReference[oaicite:3]{index=3}

    # Default NT (for first event in a case)
    mean_nt = float(df_train.loc[mask, "time_to_next_event"].astype(float).mean())
    y_true_nt = df_eval["time_to_next_event"].astype(float).to_numpy()
    y_pred_nt = np.full(len(df_eval), mean_nt, dtype=np.float32)

    # Cache per-case arrays for slicing
    case_cache = {}
    for cid, g in df_full.groupby("case_id", sort=False):
        g = g.sort_values("pos")
        case_cache[cid] = {
            "tne": g["time_to_next_event"].astype(float).to_numpy(),
            "features": {c: g[c].to_numpy() for c in feature_cols},
        }

    inputs = []
    out_idx = []

    for i, row in enumerate(df_eval.itertuples(index=False)):
        pos = int(row.pos)
        if pos <= 0:
            continue  # keep default mean_nt

        cid = row.case_id
        case = case_cache[cid]

        # history up to previous event (length = pos)
        target_hist = case["tne"][:pos]
        past_cov = {c: case["features"][c][:pos] for c in feature_cols}

        # known covariates at the forecast step (prediction_length=1)
        future_cov = {c: np.array([case["features"][c][pos]]) for c in feature_cols}

        inputs.append(
            {
                "target": target_hist,
                "past_covariates": past_cov,
                "future_covariates": future_cov,
            }
        )
        out_idx.append(i)

    if inputs:
        # returns (quantiles, mean); mean is the point forecast used by predict_df :contentReference[oaicite:4]{index=4}
        quantiles, mean = pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=1,
            quantile_levels=[0.5],
            batch_size=chronos_batch_size,
            cross_learning=cross_learning,
            limit_prediction_length=False,
        )
        for i, m in zip(out_idx, mean):
            # m is shape (n_variates, horizon); here (1,1)
            y_pred_nt[i] = float(m.reshape(-1)[0].cpu().item())

    metrics = {
        "test_next_activity_acc": float(accuracy_score(y_true_na, y_pred_na)),
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
    return metrics