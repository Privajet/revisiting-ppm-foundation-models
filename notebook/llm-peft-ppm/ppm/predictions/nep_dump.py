"""Dump per-prefix predictions from a trained NEP (RNN/Transformer/LLM) model."""
import numpy as np
import pandas as pd
import torch


# Wie die EventTargets-Namen auf die Spaltennamen in den Predictions mappen
CAT_TARGET_TO_DUMP_KEY = {
    "next_activity": "next_activity",  # Präfix beibehalten
    "activity": "next_activity",       # Fallback, falls jemand unprefixed nutzt
}
NUM_TARGET_TO_DUMP_KEY = {
    "next_remaining_time": "remaining_time",            # Präfix strippen
    "next_time_to_next_event": "time_to_next_event",    # Präfix strippen
    "remaining_time": "remaining_time",                  # Fallback
    "time_to_next_event": "time_to_next_event",          # Fallback
}


@torch.no_grad()
def dump_predictions_nep(model, test_loader, test_log, device: str = "cuda") -> dict:
    """
    Run trained NEP model over test_loader and return a y_true_pred_dump-style dict
    with per-prefix predictions and case/position metadata.

    Assumes test_loader was built with shuffle=False and that ContinuousTraces
    iterates cases in the same order as test_log.dataframe sorted by (case_id, _row_id).
    """
    model.eval()
    targets_cat = list(test_log.targets.categorical)  # e.g. ["activity"]
    targets_num = list(test_log.targets.numerical)    # e.g. ["remaining_time", "time_to_next_event"]
    print(f"[nep_dump] targets_cat={targets_cat}, targets_num={targets_num}")
    if not targets_cat and not targets_num:
        raise RuntimeError(
            "dump_predictions_nep: no targets detected on test_log. "
            f"categorical={targets_cat}, numerical={targets_num}"
        )

    # 1) Reihenfolge der Cases ableiten, wie der DataLoader sie ausgibt
    df_test = test_log.dataframe.reset_index(drop=False).rename(columns={"index": "_row_id"})
    df_test = df_test.sort_values(["case_id", "_row_id"], kind="mergesort").reset_index(drop=True)
    df_test["pos"] = df_test.groupby("case_id").cumcount()
    df_test["case_len"] = df_test.groupby("case_id")["pos"].transform("max") + 1
    case_ids_in_order = df_test.drop_duplicates(subset="case_id", keep="first")["case_id"].tolist()

    # 2) Inferenz batchweise, Predictions einsammeln
    records = []
    case_pointer = 0

    for batch in test_loader:
        x_cat, x_num, y_cat, y_num = batch
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y_cat = y_cat.to(device)
        y_num = y_num.to(device)

        attention_mask = (x_cat[..., 0] != 0).long()
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

        B = x_cat.size(0)
        for b in range(B):
            cid = case_ids_in_order[case_pointer + b]
            valid_positions = attention_mask[b].nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
            for t in valid_positions:
                rec = {
                    "case_id": cid,
                    "pos": int(t),
                    "activity_id_current": int(x_cat[b, t, 0].item()),
                }
                # Categorical: argmax über letzte Achse
                for ix, tname in enumerate(targets_cat):
                    dump_key = CAT_TARGET_TO_DUMP_KEY.get(tname, tname)
                    rec[f"y_pred_{dump_key}"] = int(out[tname][b, t].argmax().item())
                    rec[f"y_true_{dump_key}"] = int(y_cat[b, t, ix].item())
                # Numerical: direkter Wert
                for ix, tname in enumerate(targets_num):
                    dump_key = NUM_TARGET_TO_DUMP_KEY.get(tname, tname)
                    rec[f"y_pred_{dump_key}"] = float(out[tname][b, t].item())
                    rec[f"y_true_{dump_key}"] = float(y_num[b, t, ix].item())
                records.append(rec)
        case_pointer += B

    if not records:
        raise RuntimeError("dump_predictions_nep produced no records — empty test loader?")

    df_preds = pd.DataFrame.from_records(records)
    df_preds = df_preds.merge(
        df_test[["case_id", "pos", "case_len"]],
        on=["case_id", "pos"],
        how="left",
        validate="one_to_one",
    )

    # 3) Im selben Dict-Format zurückgeben wie tabulare Baselines
    dump = {
        "case_ids": df_preds["case_id"].to_numpy(),
        "positions": df_preds["pos"].to_numpy(),
        "case_lengths": df_preds["case_len"].to_numpy(),
        "activity_ids_current": df_preds["activity_id_current"].to_numpy(),
        "y_true_next_activity": df_preds["y_true_next_activity"].to_numpy(),
        "y_pred_next_activity": df_preds["y_pred_next_activity"].to_numpy(),
        "y_true_remaining_time": df_preds["y_true_remaining_time"].to_numpy(),
        "y_pred_remaining_time": df_preds["y_pred_remaining_time"].to_numpy(),
        "y_true_time_to_next_event": df_preds["y_true_time_to_next_event"].to_numpy(),
        "y_pred_time_to_next_event": df_preds["y_pred_time_to_next_event"].to_numpy(),
    }
    return dump