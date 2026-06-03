"""
Temporal counterpart to the NA branching check.  (MAE version.)

Question: on the SAME branching axis where tabular fell behind sequence on NA,
does tabular ALSO fall behind on the temporal targets (RT, NT) as branching rises?

Metric: MAE -- robust to the few large-magnitude time outliers that dominate MSE
inside high-branching buckets, so the per-bucket picture is smoother.
Source: 30_temporal_all_group_tables.csv (per-model MAE per bucket). Each paradigm
score is the mean MAE of its two models (sequence = rnn, transformer;
tabular_fm = saprpt, tabpfn3).

Sign convention (matches the NA script): positive = SEQUENCE advantage.
tabular_minus_sequence_mae = tabular_MAE - sequence_MAE: positive => sequence ahead.
"""

import re
import pandas as pd

# --- Step 0: point this at your file -----------------------------------------
PATH = "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/results/error_analysis/30_temporal_all_group_tables.csv"

df = pd.read_csv(PATH)

# --- Step 1: keep branching slices, and only the two compared paradigms ------
branch = df[df["group_col"].isin(["branching_entropy_bucket", "branching_degree_bucket"])
            & df["paradigm"].isin(["sequence", "tabular_fm"])].copy()

# --- Step 2: aggregate each paradigm to its mean MAE per slice ----------------
agg = (branch.groupby(["dataset", "task", "group_col", "group_value", "paradigm"])["mae"]
             .mean().unstack("paradigm").reset_index())
agg["sequence_mean"] = agg["sequence"]
agg["tabular_fm_mean"] = agg["tabular_fm"]
agg["tabular_minus_sequence_mae"] = agg["tabular_fm"] - agg["sequence"]   # + => sequence ahead

# --- Step 3: order buckets least- to most-branchy and rank 1..K --------------
agg["edge"] = agg["group_value"].map(lambda s: float(re.search(r"\(([-0-9.]+)", s).group(1)))
agg["branch_rank"] = (agg.groupby(["dataset", "task", "group_col"])["edge"]
                         .rank(method="dense").astype(int))

# --- Step 4: per-dataset picture (entropy buckets), one block per task --------
print("=" * 86)
print("PER-DATASET: temporal error by branching level (entropy buckets), MAE")
print("Level 1 = LEAST branchy ... level K = MOST branchy.")
print("seq/tab are MAE (lower=better); diff = tab - seq  (positive => sequence ahead).")
print("=" * 86)
ent = agg[agg["group_col"] == "branching_entropy_bucket"]
for ds, g_ds in ent.groupby("dataset"):
    print(f"\n{ds}")
    for task, g in g_ds.groupby("task"):
        g = g.sort_values("branch_rank")
        K = int(g["branch_rank"].max())
        print(f"  [{task}]")
        for _, r in g.iterrows():
            k = int(r.branch_rank)
            tag = "(LEAST branchy)" if k == 1 else "(MOST branchy)" if k == K else ""
            print(f"    level {k} of {K} {tag:16s} "
                  f"seq={r.sequence_mean:.2f}  tab={r.tabular_fm_mean:.2f}  "
                  f"diff={r.tabular_minus_sequence_mae:+.2f}")

# --- Step 5: trend -- difference at least-branchy vs most-branchy level -------
print("\n" + "=" * 86)
print("TREND: sequence advantage at least- vs most-branchy level (per dataset x task)")
print("=" * 86)
summary_rows = []
for (ds, task, gc), g in agg.groupby(["dataset", "task", "group_col"]):
    g = g.sort_values("branch_rank")
    diff_low = g.iloc[0]["tabular_minus_sequence_mae"]
    diff_high = g.iloc[-1]["tabular_minus_sequence_mae"]
    summary_rows.append({
        "dataset": ds, "task": task, "metric": gc.replace("_bucket", ""),
        "diff_least_branchy": round(diff_low, 2),
        "diff_most_branchy": round(diff_high, 2),
        "growth": round(diff_high - diff_low, 2),
    })
summary = pd.DataFrame(summary_rows).sort_values(["task", "dataset", "metric"])
print(summary.to_string(index=False))

# --- Step 6: verdict ---------------------------------------------------------
THRESH = 0.03
print("\n" + "=" * 86)
print(f"VERDICT: cells where the sequence advantage does NOT grow with branching")
print(f"(growth <= {THRESH} MAE). High count => temporal behaves UNLIKE NA.")
print("=" * 86)
for task in ["RT", "NT"]:
    sub = summary[summary["task"] == task]
    n_flat = (sub["growth"] <= THRESH).sum()
    print(f"  {task}: {n_flat} / {len(sub)} (dataset x branching-measure) cells stay flat or shrink")