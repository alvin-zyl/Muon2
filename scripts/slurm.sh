#!/bin/bash
# Multi-node launcher (written for NERSC Perlmutter, 4 GPUs per node).
# Add your account / QoS / constraint, then submit with e.g.
#   CONFIG=configs/gpt_base/muon2_ns5.yaml sbatch -N 8 scripts/slurm.sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1

# Activate your environment here, e.g.
#   module load conda && conda activate muon2

cd "${SLURM_SUBMIT_DIR}"
CONFIG=${CONFIG:-configs/gpt_small/muon2_ns3.yaml}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export DATA_DIR=${DATA_DIR:-./data}

# Per-job compiler caches on node-local scratch to avoid concurrent writes
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/torchinductor_${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"

MASTER_PORT=$(( RANDOM + 1000 ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun -u bash -c "mkdir -p \"$TRITON_CACHE_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" && \
    torchrun --nproc-per-node=$GPUS_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    run.py --config $CONFIG ${PREFIX:+--prefix $PREFIX}"
