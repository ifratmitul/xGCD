#!/usr/bin/env bash
# Stage 2 — joint discovery + GCD on CIFAR-10 (needs Stage-1 output in --cbl_dir).
python -m methods.contrastive_training.xgcd \
    --dataset_name cifar10 \
    --cbl_dir exp/stage1_cbl_cifar10 \
    --prop_train_labels 0.5 \
    --concept_conf_threshold 0.15 \
    --warmup_epochs 20 \
    --epochs 200 \
    --refresh_period 5 \
    --lambda_warmup 20 \
    --novelty_alpha 0.05 \
    --alpha_fidelity 0.1 \
    --lr 0.1 \
    --cbl_lr_divisor 100 \
    --batch_size 128 \
    --exp_name stage2_xgcd_cifar10

# CIFAR-100:  --dataset_name cifar100 --num_labeled_classes 80  (or 50)
#             --cbl_dir exp/stage1_cbl_cifar100
