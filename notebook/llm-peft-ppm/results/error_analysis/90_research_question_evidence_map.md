# Error and behavior analysis: evidence map

This file is generated from the prediction parquet files. Interpretations should be written only after inspecting the CSV tables referenced below.

## Core metric tables

- `10_overall_metrics_all_rows.csv`: native metrics on each model's available dump.

- `11_overall_metrics_common_rows.csv`: paired metrics restricted to row keys present for all available final models per dataset. Use this for fair model deltas.

- `00_alignment_summary.csv`: row-key and true-label alignment diagnostics.


## Paired common-row overall snapshot

```text
                dataset       model   paradigm  n_rows  na_accuracy   rt_mae   rt_mse   nt_mae   nt_mse
                  BPI12  gemma-2-2b        llm    2000       0.5610 1.135479 2.360828 0.433558 1.548400
                  BPI12  llama32-1b        llm    2000       0.5735 1.220707 2.771052 0.400097 1.614602
                  BPI12         rnn   sequence    2000       0.7565 0.949652 1.776871 0.417669 1.426505
                  BPI12      saprpt tabular_fm    2000       0.6320 1.057015 2.155807 0.357855 1.555437
                  BPI12     tabpfn3 tabular_fm    2000       0.6385 1.098831 2.311418 0.348274 1.549496
                  BPI12 transformer   sequence    2000       0.7515 1.398618 3.157632 0.366390 1.543495
                  BPI17  gemma-2-2b        llm    2000       0.8250 0.885964 1.445085 0.320260 1.017445
                  BPI17  llama32-1b        llm    2000       0.7980 0.616512 0.634859 0.317713 0.801681
                  BPI17         rnn   sequence    2000       0.8570 0.625839 0.731441 0.272861 0.839827
                  BPI17      saprpt tabular_fm    2000       0.6190 0.830625 1.391884 0.316003 1.197492
                  BPI17     tabpfn3 tabular_fm    2000       0.6495 1.009050 1.906832 0.280488 1.226173
                  BPI17 transformer   sequence    2000       0.8520 0.764144 1.140526 0.241767 0.990716
BPI20PrepaidTravelCosts  gemma-2-2b        llm    2000       0.6230 1.063135 1.953729 0.406650 0.760532
BPI20PrepaidTravelCosts  llama32-1b        llm    2000       0.6105 0.610363 0.975183 0.628442 1.037790
BPI20PrepaidTravelCosts         rnn   sequence    2000       0.7835 0.673303 1.021807 0.452114 0.874359
BPI20PrepaidTravelCosts      saprpt tabular_fm    2000       0.7430 0.548601 0.967157 0.336144 0.769522
BPI20PrepaidTravelCosts     tabpfn3 tabular_fm    2000       0.6400 0.541590 0.915957 0.314726 0.810440
BPI20PrepaidTravelCosts transformer   sequence    2000       0.7795 0.575281 1.055000 0.557052 0.912088
 BPI20RequestForPayment  gemma-2-2b        llm    2000       0.8735 0.622616 0.737676 0.505769 0.821902
 BPI20RequestForPayment  llama32-1b        llm    2000       0.8520 0.572929 0.681384 0.479852 0.844611
 BPI20RequestForPayment         rnn   sequence    2000       0.8925 0.552536 0.723954 0.500623 0.846010
 BPI20RequestForPayment      saprpt tabular_fm    2000       0.7435 0.516660 0.662807 0.450277 0.803106
 BPI20RequestForPayment     tabpfn3 tabular_fm    2000       0.7275 0.498645 0.682652 0.431519 0.786521
 BPI20RequestForPayment transformer   sequence    2000       0.8945 0.528498 0.694739 0.474792 0.836444
  BPI20TravelPermitData  gemma-2-2b        llm    2000       0.6600 0.764409 1.157618 0.557173 1.104436
  BPI20TravelPermitData  llama32-1b        llm    2000       0.7370 0.643128 0.968328 0.473040 1.013363
  BPI20TravelPermitData         rnn   sequence    2000       0.7410 0.622359 0.955171 0.448531 1.109328
  BPI20TravelPermitData      saprpt tabular_fm    2000       0.6760 0.627649 0.930727 0.438622 1.073502
  BPI20TravelPermitData     tabpfn3 tabular_fm    2000       0.6040 0.640320 1.006921 0.430714 1.036362
  BPI20TravelPermitData transformer   sequence    2000       0.7250 0.577414 0.853220 0.433871 0.942435
```


## RQ1: Why do tabular foundation models perform worse on NA than sequence models?

Empirical evidence to inspect first:

- `40_paradigm_comparison_na_by_slices.csv`

- `41_top_sequence_over_tabular_na_slices.csv`

- `20_na_by_branching_entropy_bucket.csv`

- `20_na_by_activity_frequency_bucket.csv`

- `20_na_by_prefix_length_bucket.csv`

Interpretation rule: claim a tabular control-flow limitation only if the sequence-minus-tabular NA gap increases on rare/high-entropy/high-branching states or longer prefixes on common rows.


## RQ2: Why are tabular models more competitive on RT and NT?

Empirical evidence to inspect first:

- `50_paradigm_comparison_temporal_by_slices.csv`

- `51_top_tabular_competitive_temporal_slices.csv`

- `30_RT_by_rt_magnitude_bucket.csv`

- `30_NT_by_nt_magnitude_bucket.csv`

Interpretation rule: argue competitiveness only where tabular-minus-sequence MAE/MSE is close to zero or negative, preferably consistently across datasets or magnitude bins.


## RQ3: Why do LLMs underperform sequence models on NA despite being pretrained sequence models?

Empirical evidence to inspect first:

- `42_top_sequence_over_llm_na_slices.csv`

- `43_na_disagreement_examples_common_rows.csv`

- `20_na_by_true_next_activity_frequency_bucket.csv`

Interpretation rule: distinguish architecture/pretraining mismatch from data/hyperparameter effects. Only claim process-control-flow weakness if degradation is systematic across datasets and structural bins.


## RQ4: Are LLMs better suited for temporal tasks than for discrete control-flow prediction?

Compare each LLM's NA rank/gap with RT and NT MAE/MSE gaps in `11_overall_metrics_common_rows.csv` and `50_paradigm_comparison_temporal_by_slices.csv`.


## RQ5: Paradigm limitation vs. dataset-specific effects

Use cross-dataset consistency. A pattern is paradigm-level only if it appears in several datasets under paired common-row evaluation. A pattern isolated to one BPI log should be framed as dataset-specific.
