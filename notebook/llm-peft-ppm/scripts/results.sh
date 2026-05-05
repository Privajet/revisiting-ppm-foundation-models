#!/bin/bash
#SBATCH --job-name=llm-peft-ppm_results
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --mail-user=lennart.fertig@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm
#SBATCH --output=/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/logs/%x_%j.out
#SBATCH --error=/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm/logs/%x_%j.err

set -euo pipefail

# Create log/cache dirs
mkdir -p logs .cache/huggingface .wandb

# Conda env
eval "$(/ceph/lfertig/miniconda3/bin/conda shell.bash hook)"
conda activate llm-peft-ppm

# Use libstdc++ from conda env first (fix CXXABI_1.3.15 errors)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# PYTHONPATH so ppm/ gefunden wird
export PYTHONPATH="/ceph/lfertig/Paper/revisiting-ppm-foundation-models/notebook/llm-peft-ppm:${PYTHONPATH:-}"

# W&B offline (kein Projekt nötig)
export WANDB_MODE=offline

# Runtime
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$PWD/.wandb"
export TOKENIZERS_PARALLELISM=false

# GPU Info
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch,sys; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true

# Configuration
PY_MAIN="results/results.py"
export ENTITY="privajet-university-of-mannheim"

echo ">>> RUN: python $PY_MAIN"
python "$PY_MAIN"