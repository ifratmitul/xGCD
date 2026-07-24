# xGCD — Interpretable Generalized Category Discovery

Discovers **what** novel categories exist and **how many**, in an interpretable concept space.
Two stages: **(1)** train a Concept Bottleneck Layer (CBL) on labelled known classes, then
**(2)** a joint E-step (novelty test + DPMM to estimate `K^n`) / M-step discovery loop.

## Setup
```bash
bash setup.sh && conda activate xGCD      # GPU (cluster)
# local mac:  conda env create -f env.local.yml && conda activate xgcd
```
DINO weights and CIFAR download automatically on first run. Concept annotations must already
be in `data/annotations/{cifar10_train,cifar10_val,...}/{uq_idx}.json`.

## Run
```bash
# Stage 1 -> exp/stage1_cbl_cifar10/{cbl_stage1.pt, vocab.json}
python -m methods.contrastive_training.stage1_cbl --dataset_name cifar10

# Stage 2 -> exp/stage2_xgcd_cifar10/{backbone.pt, cbl.pt, gcd_state.pt}
python -m methods.contrastive_training.xgcd --dataset_name cifar10 --cbl_dir exp/stage1_cbl_cifar10
```
Cluster: `kubectl apply -f job.yml` (runs both stages; set `SMOKE=1` for a fast test).
CIFAR-100: add `--num_labeled_classes 80` (or `50`) to both commands.

## Reading the output
Stage 2 prints an epoch-0 baseline and a summary block every 10 epochs:
```
 xGCD SUMMARY epoch 30/220  [phase 2]
   eval : All=0.86  Old=0.95  New=0.78  |  K^n_hat=5 (true 5, err 0)
```
- **All/Old/New** = GCD accuracy; **K^n_hat/err** = estimated vs true novel-class count.
- Add `--use_wandb True` to stream to Weights & Biases (project `xGCD`).

## Common knobs
`--num_labeled_classes` `--concept_conf_threshold 0.15` `--warmup_epochs 20` `--epochs 200`
`--refresh_period 5` `--novelty_alpha 0.05` `--lda_ridge_gamma 0.1` `--lr 0.1` `--alpha_fidelity 0.1`

## Layout
```
models/   backbone (DINO ViT), cbl, concept_model, model_factory
data/     cifar, splits, concept_annotations (vocab + concept targets)
methods/contrastive_training/  stage1_cbl.py, xgcd.py (main), extract.py
methods/gcd/  lda_gaussian, novelty_test, dpmm, estep, losses, eval_gcd
doc/      papers + IMPLEMENTATION_PLAN.md
```

## Troubleshooting
- **CIFAR download slow** — Toronto server; ~1h once, then cached. Restart resumes.
- **`libjpeg` torchvision warning** — harmless (CIFAR uses PIL).
- **`K^n` collapses / low Old acc** — raise `--lda_ridge_gamma` (1.0), lower `--lr` (0.01),
  or raise `--alpha_fidelity`; watch `cond(Sigma)` in the summary.
