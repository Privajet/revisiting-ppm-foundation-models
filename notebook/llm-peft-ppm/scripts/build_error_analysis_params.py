"""Generate compact error-analysis params files for RNN and Transformer.

Reads the best HP configuration per (log, backbone) from
results/csv/baseline_best_settings_mean_std.csv and writes:
  scripts/rnn_params_error_analysis.txt
  scripts/transformer_params_error_analysis.txt
Each file contains one line per dataset (5 lines total).
"""
from pathlib import Path
import pandas as pd

BASELINE_BEST_CSV = "results/csv/baseline_best_settings_mean_std.csv"

COMMON_ARGS = (
    "--categorical_features activity "
    "--categorical_targets activity "
    "--continuous_features all "
    "--continuous_targets remaining_time time_to_next_event"
)

HP_FLAGS_STR   = ["rnn_type", "strategy"]
HP_FLAGS_INT   = ["embedding_size", "hidden_size", "n_layers", "batch_size", "epochs"]
HP_FLAGS_FLOAT = ["lr", "weight_decay", "grad_clip"]


def build_args_line(row, backbone: str) -> str:
    parts = [
        f"--dataset {row['log']}",
        "--model nep",
        f"--backbone {backbone}",
    ]
    for col in HP_FLAGS_STR:
        val = row.get(col)
        if pd.notna(val) and str(val) != "":
            parts.append(f"--{col} {val}")
    for col in HP_FLAGS_INT:
        val = row.get(col)
        if pd.notna(val):
            parts.append(f"--{col} {int(val)}")
    for col in HP_FLAGS_FLOAT:
        val = row.get(col)
        if pd.notna(val):
            parts.append(f"--{col} {val}")
    parts.append(COMMON_ARGS)
    return " ".join(parts)


def main():
    if not Path(BASELINE_BEST_CSV).exists():
        raise FileNotFoundError(
            f"{BASELINE_BEST_CSV} not found. "
            "Run results.py first to (re)generate baseline_best."
        )

    df = pd.read_csv(BASELINE_BEST_CSV)
    print(f"Loaded {BASELINE_BEST_CSV} with {len(df)} rows total.")
    print(f"Backbones present: {sorted(df['backbone'].unique())}")
    print()

    for backbone in ["rnn", "transformer"]:
        sub = df[df["backbone"] == backbone].sort_values("log").reset_index(drop=True)
        if sub.empty:
            print(f"WARNING: no rows for backbone={backbone} — skipping")
            continue

        out_path = Path("scripts") / f"{backbone}_params_error_analysis.txt"
        lines = [build_args_line(row, backbone) for _, row in sub.iterrows()]
        out_path.write_text("\n".join(lines) + "\n")

        print(f"=== {backbone}: wrote {len(lines)} lines to {out_path} ===")
        for line in lines:
            # zur Verifikation: nur Dataset + Backbone + die ersten HPs zeigen
            preview = line.split("--categorical_features")[0].strip()
            print(f"  {preview}")
        print()


if __name__ == "__main__":
    main()