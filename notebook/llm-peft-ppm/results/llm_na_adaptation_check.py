from pathlib import Path
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and model groups
# ---------------------------------------------------------------------------

BASE = Path(
    "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/"
    "notebook/llm-peft-ppm/results/error_analysis"
)

SEQ = ["rnn", "transformer"]
TAB = ["saprpt", "tabpfn3"]
LLM = ["llama32-1b", "gemma-2-2b"]

THRESH = 0.03


def read_csv(name: str) -> pd.DataFrame:
    path = BASE / name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")


# ===========================================================================
# PART A: Does branching affect LLMs and tabular models similarly?
# ===========================================================================

slices = read_csv("40_paradigm_comparison_na_by_slices.csv")

branch = slices[
    slices["group_col"].isin(
        ["branching_entropy_bucket", "branching_degree_bucket"]
    )
].copy()

branch["edge"] = branch["group_value"].map(
    lambda value: float(re.search(r"\(([-0-9.]+)", value).group(1))
)

rows = []

for (dataset, measure), group in branch.groupby(["dataset", "group_col"]):
    group = group.sort_values("edge")

    rows.append({
        "dataset": dataset,
        "measure": measure.replace("_bucket", ""),
        "tab_growth": (
            group.iloc[-1]["sequence_minus_tabular_acc"]
            - group.iloc[0]["sequence_minus_tabular_acc"]
        ),
        "llm_growth": (
            group.iloc[-1]["sequence_minus_llm_acc"]
            - group.iloc[0]["sequence_minus_llm_acc"]
        ),
    })

growth = pd.DataFrame(rows)

print("=" * 90)
print("PART A — Growth of the gap vs sequence from least- to most-branching bucket")
print("=" * 90)
print(growth.round(2).to_string(index=False))

print("\nSummary:")
print(
    f"Tabular gap widens in {(growth['tab_growth'] > THRESH).sum()}/{len(growth)} cells; "
    f"mean growth = {growth['tab_growth'].mean():+.2f}"
)
print(
    f"LLM gap widens in {(growth['llm_growth'] > THRESH).sum()}/{len(growth)} cells; "
    f"mean growth = {growth['llm_growth'].mean():+.2f}"
)


# ===========================================================================
# PART B: Is the LLM deficit uniform or dataset-dependent?
# ===========================================================================

overall = read_csv("11_overall_metrics_common_rows.csv")

pivot = overall.pivot_table(
    index="dataset",
    columns="model",
    values="na_accuracy",
)

sequence_mean = pivot[SEQ].mean(axis=1)
tabular_mean = pivot[TAB].mean(axis=1)
llm_mean = pivot[LLM].mean(axis=1)

summary = pd.DataFrame({
    "sequence_mean": sequence_mean,
    "tabular_mean": tabular_mean,
    "llm_mean": llm_mean,
    "sequence_minus_tabular": sequence_mean - tabular_mean,
    "sequence_minus_llm": sequence_mean - llm_mean,
})

print("\n" + "=" * 90)
print("PART B — NA gaps under paired common-row evaluation")
print("=" * 90)
print(summary.round(2).to_string())

print("\nSummary:")
print(
    f"Tabular models trail sequence models on "
    f"{(summary['sequence_minus_tabular'] > 0).sum()}/{len(summary)} logs."
)
print(
    f"LLMs remain within 0.05 accuracy points of sequence models on "
    f"{(summary['sequence_minus_llm'] < 0.05).sum()}/{len(summary)} logs."
)
print(
    f"LLMs trail sequence models by more than 0.10 accuracy points on "
    f"{(summary['sequence_minus_llm'] > 0.10).sum()}/{len(summary)} logs."
)


# ===========================================================================
# PART C: Are premature <EOS> predictions a relevant LLM error pattern?
# ===========================================================================

confusions = read_csv("21_top_na_confusions.csv")

premature_eos = confusions[
    confusions["y_pred_na_label"].fillna("").str.contains("<EOS>")
    & ~confusions["y_true_na_label"].fillna("").str.contains("<EOS>")
].copy()

eos_share = (
    premature_eos
    .groupby(["dataset", "model"])["share_of_model_errors"]
    .sum()
    .unstack(fill_value=0.0)
)

ordered_models = [
    model for model in SEQ + LLM + TAB
    if model in eos_share.columns
]

eos_share = eos_share[ordered_models]

print("\n" + "=" * 90)
print("PART C — Share of NA errors caused by premature <EOS> predictions")
print("=" * 90)
print(eos_share.round(2).to_string())

print("\nBPI12:")
print(eos_share.loc["BPI12", [model for model in SEQ + LLM if model in eos_share.columns]].round(2))