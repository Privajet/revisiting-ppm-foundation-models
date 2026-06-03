"""
Figure: the branching dissociation, two panels, per model, one log.

LEFT  (NA, accuracy):  RNN & Transformer (sequence, blue/solid) vs
                       TabPFN-3 & ConTextTab (tabular, orange/dashed).
                       As branching rises, the tabular models fall away from the
                       sequence models -> the paradigms SEPARATE.
RIGHT (NT, MSE):       the same four models. The lines stay bunched and move
                       together as branching rises -> the paradigms do NOT separate.

So branching splits the models on next-activity prediction but not on time.

Sources: NA from 40_paradigm_comparison_na_by_slices.csv (per-model accuracy),
         NT from 30_temporal_all_group_tables.csv (per-model MAE).
Note: the NT panel uses MAE, which is robust to the large-time outliers that make MSE
spike inside middle buckets, so the four model lines stay smoothly bunched.
Switch TASK to "RT" or DATASET to another log as needed.
"""
import re
import pandas as pd
import matplotlib.pyplot as plt

NA_PATH  = "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/results/error_analysis/40_paradigm_comparison_na_by_slices.csv"
TMP_PATH = "/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/results/error_analysis/30_temporal_all_group_tables.csv"
DATASET  = "BPI12"
GROUP    = "branching_entropy_bucket"   # or "branching_degree_bucket"
TASK     = "NT"                          # temporal task for the right panel ("NT" or "RT")

def edge(s):
    return float(re.search(r"\(([-0-9.]+)", s).group(1))

def xlabels(ax, x):
    ax.set_xticks(x)
    lab = [str(i) for i in x]; lab[0] += "\nleast"; lab[-1] += "\nmost"
    ax.set_xticklabels(lab); ax.set_xlabel("branching level")

# --- NA per model ------------------------------------------------------------
na = pd.read_csv(NA_PATH)
na = na[(na["dataset"] == DATASET) & (na["group_col"] == GROUP)].copy()
na["edge"] = na["group_value"].map(edge); na = na.sort_values("edge")
xn = list(range(1, len(na) + 1))

# --- temporal per model (MSE) ------------------------------------------------
t = pd.read_csv(TMP_PATH)
t = t[(t["dataset"] == DATASET) & (t["group_col"] == GROUP) & (t["task"] == TASK)].copy()
t["edge"] = t["group_value"].map(edge)
piv = t.pivot_table(index="edge", columns="model", values="mae").sort_index()
xt = list(range(1, len(piv) + 1))

# --- plot --------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.3))

axL.plot(xn, na["rnn"],         marker="o", color="#1f4e79", ls="-",  label="RNN (seq.)")
axL.plot(xn, na["transformer"], marker="o", color="#5b9bd5", ls="-",  label="Transformer (seq.)")
axL.plot(xn, na["tabpfn3"],     marker="s", color="#c55a11", ls="--", label="TabPFN-3 (tab.)")
axL.plot(xn, na["saprpt"],      marker="s", color="#ed9b40", ls="--", label="ConTextTab (tab.)")
xlabels(axL, xn); axL.set_ylabel("NA accuracy (higher = better)")
axL.set_title("Next activity"); axL.legend(frameon=False, fontsize=7)

axR.plot(xt, piv["rnn"],         marker="o", color="#1f4e79", ls="-")
axR.plot(xt, piv["transformer"], marker="o", color="#5b9bd5", ls="-")
axR.plot(xt, piv["tabpfn3"],     marker="s", color="#c55a11", ls="--")
axR.plot(xt, piv["saprpt"],      marker="s", color="#ed9b40", ls="--")
xlabels(axR, xt); axR.set_ylabel(f"{TASK} error MAE (lower = better)")
axR.set_title("Next event time" if TASK == "NT" else "Remaining time")

fig.suptitle(f"{DATASET}: branching splits the models on NA but not on time")
fig.tight_layout()
fig.savefig(f"fig_dissociation_{DATASET}.pdf")
fig.savefig(f"fig_dissociation_{DATASET}.png", dpi=150)