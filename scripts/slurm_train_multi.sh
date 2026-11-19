#!/bin/bash
#SBATCH --job-name=jepa-ddp
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:L4:4
#SBATCH --output=%x-%j.log
#
# Data-parallel training on one node with 4x L4 (DDP via torchrun, one process
# per GPU). Extra args go to `python -m jepa train`:
#   sbatch scripts/slurm_train_multi.sh
#   sbatch scripts/slurm_train_multi.sh --epochs 1
#
# The job uses one torchrun task that spawns a
# process per GPU, so --ntasks stays 1; CPUs (8 per GPU for dataloader
# workers) and memory (each rank loads and compiles its own model copy) are
# sized for 4 ranks.
#
# `--batch-size` is PER GPU. The default here is 6, so the global batch is
# 4 x 6 = 24: the same as single-GPU training, and the LR in train.py (6e-4)
# applies unchanged. Passing --batch-size overrides it (argparse keeps the
# last value); for a larger global batch, scale LEARNING_RATE by about
# sqrt(global batch / 24).

set -e

# Move to the repository root so `python -m jepa` resolves the package.
cd "$(dirname "$0")/.."
mkdir -p runs/jepa

VENV=${VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Grow allocator segments instead of fragmenting fixed blocks; avoids
# near-OOM allocator thrashing with the dynamic-shape batches.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# One process per allocated GPU. SLURM exports SLURM_GPUS_ON_NODE for the
# --gres request; fall back to 4 to match it.
NGPUS=${SLURM_GPUS_ON_NODE:-4}
echo "Launching DDP on $NGPUS GPUs (global batch = $NGPUS x per-GPU batch)"

# torchrun exports RANK/WORLD_SIZE/LOCAL_RANK, which is all
# `python -m jepa train` needs to switch to DDP.
torchrun --standalone --nproc_per_node="$NGPUS" -m jepa train --batch-size 6 "$@"
