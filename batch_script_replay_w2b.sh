#!/bin/bash
# SLURM batch job for Berzelius (MIG 1g.10gb, 2h)

############################
# SLURM RESOURCES
############################
#SBATCH -A Berzelius-2025-243
#SBATCH --reservation=1g.10gb     # Use the MIG reservation
#SBATCH --gpus=1                   # One MIG GPU in that reservation
#SBATCH -t 05:00:00                # 5 hours walltime
#SBATCH -n 1
#SBATCH -c 2                       # MIG slice provides 2 cores
#SBATCH --mem=32G                  # MIG slice provides ~32 GB RAM

############################
# SHELL SAFETY / LOGGING
############################
set -eo pipefail
export PYTHONUNBUFFERED=1

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"

############################
# CONDA (no ~/.bashrc)
############################
# Try common locations or your site module
module load Anaconda3/2023.03 2>/dev/null || true

# Find conda.sh
for C in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/etc/profile.d/conda.sh" \
  "/usr/local/etc/profile.d/conda.sh"
do
  if [ -f "$C" ]; then CONDASH="$C"; break; fi
done

if [ -z "${CONDASH:-}" ]; then
  echo "ERROR: conda.sh not found. Update the path above."
  exit 1
fi

# Guard against PS1 under nounset
set +u || true
: "${PS1:=}"
source "$CONDASH"
conda activate personalized_fl || { conda env list; exit 1; }
set -u 2>/dev/null || true

############################
# DIAGNOSTICS (optional)
############################
which python
python --version
nvidia-smi || true

############################
# RUN YOUR WORK
############################
srun --ntasks=1 --cpus-per-task=2 --gpus=1 bash ./run_replay_w2b.sh

echo "Job finished at $(date)"
