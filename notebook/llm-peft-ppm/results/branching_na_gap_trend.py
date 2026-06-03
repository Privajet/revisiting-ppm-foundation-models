"""
Q1 mechanism check: does the sequence-vs-tabular next-activity (NA) accuracy gap
grow as prefixes become more "branchy" (more possible next activities)?

The raw entropy/degree numbers are used ONLY to order the buckets from least- to
most-branchy. Each bucket is then shown as a plain "level k of K"
(level 1 = least branchy / on rails, level K = most branchy / a fork).
"""

import re
import pandas as pd

# --- Step 0: point this at your file -----------------------------------------
PATH = "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/results/error_analysis/40_paradigm_comparison_na_by_slices.csv"

df = pd.read_csv(PATH)

# --- Step 1: keep only the two branching slice-types -------------------------
# Each row is one "slice" (a subset of test predictions grouped by a prefix
# property). We want only the two slice-types that measure branching.
branch = df[df["group_col"].isin(["branching_entropy_bucket",
                                  "branching_degree_bucket"])].copy()

# --- Step 2: turn each bucket label into a sortable number -------------------
# A label looks like 'branch_ent_(-0.001, 1.29]'. The first number in the
# parentheses is the bucket's lower edge; sorting by it orders buckets from
# least-branchy to most-branchy.
branch["edge"] = branch["group_value"].map(
    lambda s: float(re.search(r"\(([-0-9.]+)", s).group(1)))

# --- Step 3: within each (dataset, branching-type), rank buckets 1..K --------
# level 1 = least branchy (on rails), level K = most branchy (a fork).
branch["branch_rank"] = (
    branch.groupby(["dataset", "group_col"])["edge"].rank(method="dense").astype(int))

# Columns we read per slice:
#   sequence_mean   = avg NA accuracy of RNN + Transformer        (0-1, higher=better)
#   tabular_fm_mean = avg NA accuracy of ConTextTab + TabPFN-3    (0-1, higher=better)
#   sequence_minus_tabular_acc = sequence_mean - tabular_fm_mean
#       > 0 sequence better,  ~0 tie,  < 0 tabular better

# --- Step 4: print the per-dataset picture for branching ENTROPY -------------
print("=" * 78)
print("PER-DATASET: NA accuracy by branching level (entropy buckets)")
print("Level 1 = LEAST branchy (on rails) ... level K = MOST branchy (fork).")
print("seq/tab are accuracy (fraction correct, 0-1); gap = seq - tab.")
print("=" * 78)
ent = branch[branch["group_col"] == "branching_entropy_bucket"]
for ds, g in ent.groupby("dataset"):
    g = g.sort_values("branch_rank")
    K = int(g["branch_rank"].max())
    print(f"\n{ds}")
    for _, r in g.iterrows():
        k = int(r.branch_rank)
        tag = "(LEAST branchy)" if k == 1 else "(MOST branchy)" if k == K else ""
        print(f"  level {k} of {K} {tag:16s} "
              f"seq_acc={r.sequence_mean:.2f}  tab_acc={r.tabular_fm_mean:.2f}  "
              f"gap={r.sequence_minus_tabular_acc:+.2f}")

# --- Step 5: the broader trend, one comparison per dataset -------------------
# For each dataset and branching-type, compare the gap at the LEAST-branchy
# level with the gap at the MOST-branchy level.
print("\n" + "=" * 78)
print("TREND: gap at least-branchy level vs most-branchy level")
print("=" * 78)
summary_rows = []
for (ds, gc), g in branch.groupby(["dataset", "group_col"]):
    g = g.sort_values("branch_rank")
    gap_low = g.iloc[0]["sequence_minus_tabular_acc"]   # least branchy
    gap_high = g.iloc[-1]["sequence_minus_tabular_acc"]  # most branchy
    summary_rows.append({
        "dataset": ds, "metric": gc.replace("_bucket", ""),
        "gap_least_branchy": round(gap_low, 2),
        "gap_most_branchy": round(gap_high, 2),
        "widened_by": round(gap_high - gap_low, 2),
    })
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# --- Step 6: verdict, counting only NON-TRIVIAL widening ---------------------
# A widening counts only if it exceeds 0.03 accuracy, so near-zero ties
# (e.g. a 0.000 -> 0.004 difference) are not mistaken for a real trend.
THRESH = 0.03
print("\n" + "=" * 78)
print(f"VERDICT: datasets where the gap widens by more than {THRESH} from least to most branchy")
print("=" * 78)
for gc in ["branching_entropy", "branching_degree"]:
    sub = summary[summary["metric"] == gc]
    n = (sub["widened_by"] > THRESH).sum()
    print(f"  {gc:18s}: {n} / {len(sub)} datasets")