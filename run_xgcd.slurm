#!/bin/bash
#SBATCH --job-name=Mozaf
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=128gb
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%j_XGCD_FRB_Ph_1-2-PCL_lr_0.1-max_0.5-dpmm_warm_start-phase1_lambda_0.5-anchor_known_1-freeze_known_protos.log

echo "====================================================="
pwd; hostname; date
echo "====================================================="

module load apptainer

# --- project + container paths (EDIT to match the cluster) --------------------
PROJECT="/home/mozafarinia/Documents/xGCD"
CONTAINER="/home/mozafarinia/Documents/xgcd.sif"
cd "$PROJECT" || { echo "project dir not found: $PROJECT"; exit 1; }

nvidia-smi

mkdir -p exp data/datasets logs/wandb

# --- verify the environment INSIDE the container -----------------------------
apptainer exec --nv "$CONTAINER" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
apptainer exec --nv "$CONTAINER" python -c "import wandb; print('wandb', wandb.__version__)"

# --- config ------------------------------------------------------------------
export WANDB_DIR="logs"
# wandb auth: "offline" logs locally with no API key (sync later with `wandb sync`);
# "online" needs a key -> set WANDB_API_KEY below; "disabled" turns wandb off entirely.
export WANDB_MODE=offline
export WANDB_SILENT=true   # hide the "wandb: Tracking run / syncing / ..." console lines
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # uncomment for online
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONWARNINGS="ignore"   # hide the DataLoader worker UserWarning
DATASET="cifar10"
EXP_NAME="stage1_cbl_cifar10"
CONF_THR="0.15"
WANDB_PROJECT="xGCD"
WANDB_ENTITY="ifrat-ikhtear-university-of-south-dakota"

# 1 = fast smoke test, 0 = full run
SMOKE=2
if [ "$SMOKE" = "1" ]; then
    STAGE1_EPOCHS=50; WARMUP=4; STAGE2_EPOCHS=4; REFRESH=2
else
    STAGE1_EPOCHS=0; WARMUP=5; STAGE2_EPOCHS=70; REFRESH=5
fi

CBL_OUT="exp/${EXP_NAME}"

# --- Stage 1: train the CBL only if its checkpoint is missing ----------------
if [ ! -f "${CBL_OUT}/cbl_stage1.pt" ]; then
    echo "Stage-1 CBL not found at ${CBL_OUT}/cbl_stage1.pt -> running Stage 1"
    apptainer exec --nv "$CONTAINER" \
        python -m methods.contrastive_training.stage1_cbl \
        --dataset_name "$DATASET" \
        --prop_train_labels 0.5 \
        --concept_conf_threshold "$CONF_THR" \
        --cbl_epochs "$STAGE1_EPOCHS" \
        --pos_weight_clip 20 \
        --exp_name "$EXP_NAME" \
        --use_wandb True \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
    || { echo "Stage 1 failed"; exit 1; }
else
    echo "Found existing Stage-1 CBL at ${CBL_OUT}/cbl_stage1.pt -> skipping Stage 1"
fi

# --- Stage 2: joint novel-class discovery + GCD ------------------------------
apptainer exec --nv "$CONTAINER" \
    python -m methods.contrastive_training.xgcd \
    --dataset_name "$DATASET" \
    --cbl_dir "$CBL_OUT" \
    --prop_train_labels 0.5 \
    --concept_conf_threshold "$CONF_THR" \
    --warmup_epochs "$WARMUP" \
    --epochs "$STAGE2_EPOCHS" \
    --refresh_period "$REFRESH" \
    --lambda_warmup 20 \
    --novelty_alpha 0.6 \
    --lr 0.1 \
    --cbl_lr_divisor 100 \
    --batch_size 128 \
    --num_workers 2 \
    --exp_name "stage2-FRB_Ph_1-2-PCL_lr_0.1-max_0.5-dpmm_warm_start-phase1_lambda_0.5-anchor_known_1-freeze_known_protos-hungarian_frozen_known" \
    --use_wandb True \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --pos_weight_clip 20 \
    --freeze_backbone True \
    --alpha_fidelity 0.1 \
    --phase1_lambda 0.5 \
    --lambda_max 0.5 \
    --w_cl 1
    --dpmm_warm_start True \
    --freeze_known_protos True \
    --anchor_known 1 \
    --protect_known_grad False \
    --protect_known_energy 0.99
    --hungarian_frozen_known True

date
