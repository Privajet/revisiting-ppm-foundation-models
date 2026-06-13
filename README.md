# Revisiting Predictive Process Monitoring in the Age of Foundation Models

This repository contains the implementation and experiment scripts for the comparative study:

**Revisiting Predictive Process Monitoring in the Age of Foundation Models: A Comparative Study of Sequence, Tabular, and LLM Approaches**

## Overview

The project benchmarks three modeling paradigms for predictive process monitoring (PPM):

1. **Sequence models trained from scratch**

   * LSTM
   * Transformer

2. **Decoder-only large language models adapted via LoRA**

   * Llama-3.2-1B
   * Gemma-2-2B

3. **Tabular foundation models using in-context learning**

   * TabPFN-3
   * ConTextTab

The models are evaluated on three prediction tasks:

* **NA**: next activity prediction
* **RT**: remaining time prediction
* **NT**: next event time prediction

The benchmark covers five public event logs and reports predictive performance, runtime, and slice-level error analyses.

## Origin and Attribution

This project builds on the original implementation by Oyamada et al.:

* Original repository: https://github.com/raseidi/llm-peft-ppm
* Original paper: https://doi.org/10.48550/arXiv.2509.03161

The original repository provides the foundation for adapting decoder-only LLMs to structured event-log representations for PPM.

## Main Extensions

Relative to the original implementation, this repository:

1. Adds **next event time prediction** as an additional temporal task.
2. Extends the benchmark to three prediction tasks: NA, RT, and NT.
3. Adds further decoder-only LLM backbones and evaluates LoRA, zero-shot, and few-shot configurations.
4. Adds two tabular foundation model baselines: TabPFN-3 and ConTextTab.
5. Includes slice-level analyses of prediction behavior, including branching-related difficulty and model-specific error patterns.

## Prediction Setup

The sequence models and LLM-based approaches jointly predict NA, RT, and NT in a multi-task setting.

The tabular foundation models use three task-specific predictors:

* one classifier for NA
* one regressor for RT
* one regressor for NT

TabPFN-3 results are unavailable for BPI17 because evaluation could not be completed due to computational constraints at this dataset scale.

## Data

The experiments use five public event logs from the 4TU.ResearchData repository:

* **BPI20PTC**: Prepaid Travel Costs
  https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165

* **BPI20RfP**: Request for Payment
  https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030

* **BPI20TPD**: Permit Data
  https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550

* **BPI12**
  https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

* **BPI17**
  https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11

The event logs are downloaded through `skpm` into:

```text
data/<LOG>/
```

Documentation for the data API:

https://skpm.readthedocs.io/en/latest/examples/01_data_api.html

## Input Features

The shared preprocessing pipeline uses the activity label as a categorical input feature and derives numerical features from event timestamps.

The evaluated configurations pass the configured timestamp-derived features through:

```bash
--continuous_features all
```

## System Requirements

The project was tested with:

* Ubuntu 24.04
* Python 3.12
* CUDA-capable GPU for LLM fine-tuning
* `uv` or Conda for environment management

Documentation for `uv`:

https://docs.astral.sh/uv/guides/install-python/

## Installation

### Option A: Install with `uv`

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install the dependencies:

```bash
uv pip install -r requirements.txt
```

### Option B: Install with Conda

The repository provides Conda environment files for reproducibility:

```text
env-llm-peft-ppm.yml
env-llm-peft-ppm-saprpt.yml
```

Create and activate the main environment:

```bash
conda env create -f env-llm-peft-ppm.yml
conda activate env-llm-peft-ppm
```

For ConTextTab experiments, use:

```bash
conda env create -f env-llm-peft-ppm-saprpt.yml
conda activate env-llm-peft-ppm-saprpt
```

## Repository Structure

```text
.
├── data/                                        # Downloaded event logs
├── scripts/                                     # Experiment scripts and parameter grids
│   ├── *.sh
│   └── *.txt
├── notebook/                                   # Analysis notebooks
├── ppm/                                         # Source code
├── results/                                     # Metrics, exports, and analyses
├── fertig_lennart_next_event_prediction.py      # Main experiment entry point
├── requirements.txt                             # Python dependencies
├── env-llm-peft-ppm.yml                         # Main Conda environment
├── env-llm-peft-ppm-saprpt.yml                  # ConTextTab environment
└── README.md
```

## Usage

### LSTM Baseline

```bash
python fertig_lennart_next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone rnn \
  --embedding_size 32 \
  --hidden_size 128 \
  --lr 0.0005 \
  --batch_size 64 \
  --epochs 25 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time next_event_time
```

### LLM Fine-Tuning with LoRA

Hugging Face-hosted model weights require a Hugging Face token.

Documentation:

https://huggingface.co/docs/hub/en/security-tokens

Provide the token through an `.env` file:

```text
HF_TOKEN=<YOUR_TOKEN>
```

or export it as an environment variable:

```bash
export HF_TOKEN="<YOUR_TOKEN>"
```

Minimal LoRA example:

```bash
python fertig_lennart_next_event_prediction.py \
  --dataset BPI20PrepaidTravelCosts \
  --backbone gemma-2-2b \
  --embedding_size 896 \
  --hidden_size 896 \
  --lr 0.00005 \
  --batch_size 64 \
  --epochs 1 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time next_event_time \
  --fine_tuning lora \
  --r 2 \
  --lora_alpha 4
```

Weights & Biases logging can be enabled with:

```bash
--wandb
```

## Reproducing the Benchmark

Experiment scripts and parameter grids are located under:

```text
scripts/
```

The directory contains separate configurations for:

* sequence baselines
* LLM-based approaches
* TabPFN-3
* ConTextTab
* result aggregation
* error analysis

Use the scripts and parameter files as the reference for reproducing the reported benchmark runs.

## Results and Analyses

Generated metrics and prediction outputs are stored under:

```text
results/
```

The repository also contains scripts for:

* benchmark aggregation
* slice-level error analysis
* branching-related analysis
* temporal dissociation analysis
* LLM adaptation analysis
  
For questions or feedback, please open an issue in this repository.
