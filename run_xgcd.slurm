#!/bin/bash
#SBATCH --job-name=Mozaf
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=128gb
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%j.log

set -e

# ============================================================
# xGCD — Stage 1 + Phase 3 Pipeline (SLURM)
# ============================================================

echo "=========================================="
echo "Verifying GPU availability..."
echo "=========================================="
nvidia-smi || (echo "ERROR: GPU not accessible!" && exit 1)

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Navigate to project directory
cd /path/to/xGCD  # change this path

# Initialize conda
source $CONDA_PREFIX/etc/profile.d/conda.sh  # or your conda path

# Setup environment
echo "=========================================="
echo "Running setup.sh..."
echo "=========================================="
bash setup.sh
conda activate xGCD

# Verify installation
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__} / CUDA {torch.version.cuda}')"
python -c "import wandb; print(f'wandb {wandb.__version__}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

mkdir -p exp data/datasets logs/wandb

# ============================================================
# Configuration
# ============================================================
DATASET="cifar10"
EXP_NAME="stage1_cbl_cifar10"
CONF_THR="0.15"

WANDB_PROJECT="xGCD"
WANDB_ENTITY="ifrat-ikhtear-university-of-south-dakota"
export WANDB_DIR="logs"

# Fast integration run (set SMOKE=0 for full run)
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    STAGE1_EPOCHS=2; WARMUP=2; STAGE2_EPOCHS=4; REFRESH=2
else
    STAGE1_EPOCHS=50; WARMUP=0; STAGE2_EPOCHS=30; REFRESH=2
fi

# ---- Estimation-only configuration ----
FREEZE_BACKBONE="True"
CBL_LR="1e-3"
ALPHA_FIDELITY="0.5"
NOVELTY_ALPHA="0.2"
TAU_MODE="empirical"
PCL_KNOWN_UNLAB_W="1.0"
NORM_D2_BY_C="True"
PCL_WEIGHT="1.0"
GRAD_CLIP="1.0"
ESTIMATION_ONLY="True"
DPMM_COV_SCALE="1.0"
DPMM_ALPHA="1.0"
DPMM_SEED="42"
DPMM_SWEEPS="30"
DPMM_PATIENCE="3"
DPMM_ABSORB="True"
DPMM_MERGE="False"
ABSORB_SCALE="0.1"
MERGE_SCALE="1.0"
S2_EXP="stage2_estonly_seed42"

# ---- Phase 3 configuration ----
PEAK_GATE="True"
CE_WEIGHT="1.0"
HEAD_LR="1e-2"
LABELLED_WARMUP="0"
CONF_TEMP="0.0"
P3_EPOCHS="15"
P3_EXP="phase3_cifar10"

# ============================================================
# Stage 1: CBL Pre-training
# ============================================================
echo "=========================================="
echo "Stage 1: CBL pre-training | dataset=$DATASET epochs=$STAGE1_EPOCHS smoke=$SMOKE"
echo "=========================================="

CBL_OUT="exp/${EXP_NAME}"
if [ -f "$CBL_OUT/cbl_stage1.pt" ]; then
    echo "--- Stage 1 already done ($CBL_OUT/cbl_stage1.pt exists) — skipping ---"
else
    python -m methods.contrastive_training.stage1_cbl \
        --dataset_name "$DATASET" \
        --prop_train_labels 0.5 \
        --concept_conf_threshold "$CONF_THR" \
        --cbl_epochs "$STAGE1_EPOCHS" \
        --cbl_lr 1e-4 \
        --cbl_weight_decay 1e-5 \
        --cbl_batch_size 256 \
        --batch_size 128 \
        --num_workers 4 \
        --exp_name "$EXP_NAME" \
        --use_wandb True \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY"
fi

# Guard: confirm Stage-1 artifacts exist
if [ ! -f "$CBL_OUT/cbl_stage1.pt" ] || [ ! -f "$CBL_OUT/vocab.json" ]; then
    echo "ERROR: Stage 1 outputs missing in $CBL_OUT (cbl_stage1.pt / vocab.json)."
    exit 1
fi
echo "Stage 1 COMPLETED — CBL: $CBL_OUT"
ls -la "$CBL_OUT" 2>/dev/null || true

# ============================================================
# Phase 3: Parametric CE Refinement
# ============================================================
S3_OUT="exp/${P3_EXP}"
echo "=========================================="
echo "Phase 3: CE refinement | peak_gate=$PEAK_GATE epochs=$P3_EPOCHS head_lr=$HEAD_LR warmup=$LABELLED_WARMUP"
echo "=========================================="
python -m methods.contrastive_training.phase3 \
    --dataset_name "$DATASET" \
    --cbl_dir "$CBL_OUT" \
    --prop_train_labels 0.5 \
    --concept_conf_threshold "$CONF_THR" \
    --epochs "$P3_EPOCHS" \
    --refresh_period "$REFRESH" \
    --cbl_lr "$CBL_LR" \
    --alpha_fidelity "$ALPHA_FIDELITY" \
    --grad_clip "$GRAD_CLIP" \
    --novelty_alpha "$NOVELTY_ALPHA" \
    --tau_mode "$TAU_MODE" \
    --dpmm_cov_scale "$DPMM_COV_SCALE" \
    --dpmm_alpha "$DPMM_ALPHA" \
    --dpmm_seed "$DPMM_SEED" \
    --dpmm_absorb "$DPMM_ABSORB" \
    --dpmm_merge "$DPMM_MERGE" \
    --absorb_scale "$ABSORB_SCALE" \
    --peak_gate "$PEAK_GATE" \
    --peak_thresh 0.80 \
    --ce_weight "$CE_WEIGHT" \
    --head_lr "$HEAD_LR" \
    --labelled_warmup "$LABELLED_WARMUP" \
    --conf_temp "$CONF_TEMP" \
    --ref_acc 0.8707 \
    --batch_size 128 \
    --num_workers 4 \
    --exp_name "$P3_EXP" \
    --use_wandb True \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

echo ""
echo "=========================================="
echo "xGCD PIPELINE COMPLETED — Stage1: $CBL_OUT | Phase3: $S3_OUT"
echo "=========================================="
ls -la "$S3_OUT" 2>/dev/null || true

# Sync WandB runs
if [ -d "logs/wandb" ]; then
    for run_dir in logs/wandb/offline-run-*; do
        [ -d "$run_dir" ] || continue
        echo "Syncing offline wandb run: $run_dir"
        wandb sync "$run_dir" --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" \
            || echo "Warning: failed to sync $run_dir"
    done
fi
echo "View results: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "Job finished!"