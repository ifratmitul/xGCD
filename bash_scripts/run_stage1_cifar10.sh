#!/usr/bin/env bash
# Stage 1 — CBL pre-training on CIFAR-10 (5 known / 5 novel).
# First run downloads CIFAR-10 (~170MB) and DINO ViT-B/16 (~343MB, cached).
python -m methods.contrastive_training.stage1_cbl \
    --dataset_name cifar10 \
    --prop_train_labels 0.5 \
    --concept_conf_threshold 0.15 \
    --cbl_epochs 50 \
    --cbl_lr 1e-4 \
    --cbl_weight_decay 1e-5 \
    --cbl_batch_size 256 \
    --batch_size 128 \
    --exp_name stage1_cbl_cifar10

# CIFAR-100 variants:
#   --dataset_name cifar100 --num_labeled_classes 80   # 80/20
#   --dataset_name cifar100 --num_labeled_classes 50   # 50/50
