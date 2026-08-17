"""Stage 1 — Concept Bottleneck Layer pre-training (xGCD Alg 1 lines 1-4; draft Eq 1-2).

Backbone is frozen; only the linear CBL is trained on the *labelled known-class* images
with a class-imbalance-rebalanced BCE against the Grounding-DINO concept targets o_i
(VLG-CBM Eq 6). Because the backbone is frozen, its CLS features are extracted **once**
and the CBL (a linear probe) is trained on the cached features — fast and exact.

Run:
    python -m methods.contrastive_training.stage1_cbl --dataset_name cifar10
    python -m methods.contrastive_training.stage1_cbl --dataset_name cifar100 --num_labeled_classes 80
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from config import exp_root
from data.augmentations import get_transform
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.concept_annotations import get_concept_vocab_and_lookup
from data.splits import configure_splits
from methods.contrastive_training.extract import extract_backbone_features
from models.model_factory import build_concept_model
from project_utils.general_utils import str2bool


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def get_device(prefer_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(args, stage="stage1"):
    """Optionally start a Weights & Biases run. Returns the run or None."""
    if not getattr(args, "use_wandb", False):
        return None
    import wandb
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"mahsa_{stage}-{args.dataset_name}-{args.exp_name}",
        group=args.exp_name,
        config=vars(args),
    )
    logger.info(f"wandb: logging to {args.wandb_entity}/{args.wandb_project} as {run.name}")
    return run


# --------------------------------------------------------------------------- #
# metrics (multi-label)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def concept_metrics(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    tp = (preds * targets).sum(0)
    fp = (preds * (1 - targets)).sum(0)
    fn = ((1 - preds) * targets).sum(0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    present = targets.sum(0) > 0  # only score concepts that occur
    return {
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "pos_recall": float(recall[present].mean()) if present.any() else 0.0,
        "pos_precision": float(precision[present].mean()) if present.any() else 0.0,
    }


@torch.no_grad()
def logit_health(logits: torch.Tensor):
    """Distributional stats on concept logits to catch saturation / blow-up."""
    finite = torch.isfinite(logits)
    return {
        "logit_mean": float(logits[finite].mean()) if finite.any() else float("nan"),
        "logit_std": float(logits[finite].std()) if finite.any() else float("nan"),
        "logit_absmax": float(logits[finite].abs().max()) if finite.any() else float("nan"),
        "nonfinite_frac": float((~finite).float().mean()),
    }


# --------------------------------------------------------------------------- #
# CBL training on cached features
# --------------------------------------------------------------------------- #
def train_cbl_on_features(cbl, feats, targets, pos_weight, args, device,
                          val_feats=None, val_targets=None, wandb_run=None):
    cbl = cbl.to(device)
    ds = TensorDataset(feats, targets)
    loader = DataLoader(ds, batch_size=args.cbl_batch_size, shuffle=True, drop_last=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optim = torch.optim.Adam(cbl.parameters(), lr=args.cbl_lr, weight_decay=args.cbl_weight_decay)
    grad_clip = getattr(args, "grad_clip", 0.0)
    best_val_f1 = -1.0

    for epoch in range(args.cbl_epochs):
        cbl.train()
        running, gnorm_sum, gnorm_max, nsteps, nonfinite = 0.0, 0.0, 0.0, 0, 0
        for z, o in loader:
            z, o = z.to(device), o.to(device)
            logits = cbl(z)
            loss = criterion(logits, o)

            # --- numeric-stability guard ---
            if not torch.isfinite(loss):
                nonfinite += 1
                logger.error(f"[Stage1] non-finite loss at epoch {epoch+1} "
                             f"(loss={loss.item()}); aborting to avoid corrupting weights.")
                raise RuntimeError("Stage 1 CBL training diverged (non-finite loss).")

            optim.zero_grad()
            loss.backward()
            # total grad norm (also clips if grad_clip > 0)
            total_norm = torch.nn.utils.clip_grad_norm_(
                cbl.parameters(), max_norm=(grad_clip if grad_clip > 0 else 1e12))
            optim.step()

            running += loss.item() * len(z)
            gn = float(total_norm)
            gnorm_sum += gn
            gnorm_max = max(gnorm_max, gn)
            nsteps += 1
        running /= len(ds)

        log = {
            "stage1/bce": running,
            "stage1/epoch": epoch + 1,
            "stage1/lr": optim.param_groups[0]["lr"],
            "stage1/grad_norm_mean": gnorm_sum / max(nsteps, 1),
            "stage1/grad_norm_max": gnorm_max,
        }
        if (epoch + 1) % args.cbl_log_freq == 0 or epoch == args.cbl_epochs - 1:
            cbl.eval()
            with torch.no_grad():
                tr_logits = cbl(feats.to(device)).cpu()
                tr_m = concept_metrics(tr_logits, targets)
                lh = logit_health(tr_logits)
                log.update({"stage1/train_f1": tr_m["macro_f1"],
                            "stage1/train_recall": tr_m["pos_recall"],
                            "stage1/train_precision": tr_m["pos_precision"],
                            "stage1/logit_mean": lh["logit_mean"],
                            "stage1/logit_std": lh["logit_std"],
                            "stage1/logit_absmax": lh["logit_absmax"],
                            "stage1/nonfinite_frac": lh["nonfinite_frac"]})
                msg = (f"[Stage1] epoch {epoch+1}/{args.cbl_epochs} "
                       f"bce={running:.4f} train_F1={tr_m['macro_f1']:.3f} "
                       f"train_recall={tr_m['pos_recall']:.3f} "
                       f"| grad(mean/max)={log['stage1/grad_norm_mean']:.2f}/{gnorm_max:.2f} "
                       f"logit(|max|={lh['logit_absmax']:.1f})")
                if val_feats is not None:
                    val_m = concept_metrics(cbl(val_feats.to(device)).cpu(), val_targets)
                    best_val_f1 = max(best_val_f1, val_m["macro_f1"])
                    log.update({"stage1/val_f1": val_m["macro_f1"],
                                "stage1/val_recall": val_m["pos_recall"],
                                "stage1/val_precision": val_m["pos_precision"],
                                "stage1/best_val_f1": best_val_f1})
                    msg += f" | val_F1={val_m['macro_f1']:.3f} val_recall={val_m['pos_recall']:.3f}"
                logger.info(msg)
        if wandb_run is not None:
            wandb_run.log(log, step=epoch + 1)
    return cbl


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def _build_transforms(args):
    args.interpolation = 3
    args.crop_pct = 0.875
    args.use_strong_aug = False
    _, test_transform = get_transform("imagenet", image_size=args.image_size, args=args)
    return test_transform


def _get_datasets(args, transform):
    if args.dataset_name == "cifar10":
        return get_cifar_10_datasets(train_transform=transform, test_transform=transform,
                                     train_classes=args.train_classes,
                                     prop_train_labels=args.prop_train_labels)
    elif args.dataset_name == "cifar100":
        return get_cifar_100_datasets(train_transform=transform, test_transform=transform,
                                      train_classes=args.train_classes,
                                      prop_train_labels=args.prop_train_labels)
    raise ValueError(f"Stage 1 supports cifar10/cifar100, got {args.dataset_name}")


def run_stage1(args):
    device = get_device()
    logger.info(f"Stage 1 device: {device}")

    configure_splits(args)
    transform = _build_transforms(args)
    datasets = _get_datasets(args, transform)

    # vocabulary + targets from labelled known-class images (sets args.num_concepts)
    vocab, lookup = get_concept_vocab_and_lookup(args, datasets)

    # optional wandb (after num_concepts is known so it logs in the config)
    wandb_run = init_wandb(args, stage="stage1")

    # model (frozen backbone + fresh CBL)
    model = build_concept_model(args).to(device)

    # extract backbone features for the labelled set (once)
    labelled = datasets["train_labelled"]
    loader = DataLoader(labelled, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    feats, labels, uq_idxs = extract_backbone_features(model.backbone, loader, device)
    logger.info(f"Extracted labelled features: {tuple(feats.shape)}")

    # targets o_i aligned to the extracted uq order
    uq_list = uq_idxs.numpy().tolist()
    targets = lookup.batch(uq_list).float()
    pos_weight = lookup.pos_weight(uq_list)

    # ---- data / imbalance diagnostics (help interpret Stage-1 behaviour) ----
    concept_pos = targets.sum(0)                       # positives per concept
    n_dead = int((concept_pos == 0).sum())             # concepts never positive in labelled set
    pos_frac = float(targets.mean())                   # overall positive rate
    logger.info(
        f"Targets: {tuple(targets.shape)} | mean pos/img={float(targets.sum(1).mean()):.1f} "
        f"| pos_frac={pos_frac:.4f} | dead_concepts={n_dead}/{targets.shape[1]}"
    )
    logger.info(
        f"pos_weight (pre-clip): min={float(pos_weight.min()):.1f} "
        f"mean={float(pos_weight.mean()):.1f} max={float(pos_weight.max()):.1f}"
    )
    if getattr(args, "pos_weight_clip", 0) and args.pos_weight_clip > 0:
        n_clipped = int((pos_weight > args.pos_weight_clip).sum())
        pos_weight = pos_weight.clamp(max=args.pos_weight_clip)
        logger.info(f"Clipped pos_weight to <= {args.pos_weight_clip} ({n_clipped} concepts affected)")

    if wandb_run is not None:
        wandb_run.log({
            "data/num_concepts": targets.shape[1],
            "data/dead_concepts": n_dead,
            "data/pos_frac": pos_frac,
            "data/mean_pos_per_img": float(targets.sum(1).mean()),
            "data/pos_weight_max": float(pos_weight.max()),
            "data/pos_weight_mean": float(pos_weight.mean()),
            "data/n_labelled": len(uq_list),
        }, step=0)

    # held-out split of labelled for monitoring
    val_feats = val_targets = None
    if args.cbl_val_frac > 0:
        rng = np.random.RandomState(0)
        n = len(feats)
        perm = rng.permutation(n)
        n_val = int(args.cbl_val_frac * n)
        vi, ti = perm[:n_val], perm[n_val:]
        val_feats, val_targets = feats[vi], targets[vi]
        feats, targets = feats[ti], targets[ti]
        logger.info(f"CBL train/val split: {len(feats)}/{len(val_feats)}")

    train_cbl_on_features(model.cbl, feats, targets, pos_weight, args, device,
                          val_feats, val_targets, wandb_run=wandb_run)

    # save CBL + vocabulary + backbone (backbone unchanged, saved for a self-contained ckpt)
    save_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    model.cbl.save(save_dir, name="cbl_stage1.pt")
    from data.concept_annotations import save_vocabulary
    save_vocabulary(vocab, os.path.join(save_dir, "vocab.json"),
                    meta={"dataset": args.dataset_name, "num_labeled_classes": args.num_labeled_classes})
    logger.info(f"Stage 1 done. Saved CBL + vocab -> {save_dir}")

    if wandb_run is not None:
        wandb_run.finish()
    return model, vocab, lookup


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_stage1_parser():
    p = argparse.ArgumentParser(description="xGCD Stage 1: CBL pre-training")
    p.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--num_labeled_classes", type=int, default=None,
                   help="known classes: cifar10=5; cifar100=80 (80/20) or 50 (50/50)")
    p.add_argument("--prop_train_labels", type=float, default=0.5)
    p.add_argument("--concept_conf_threshold", type=float, default=0.15)
    p.add_argument("--rebuild_vocab", type=str2bool, default=False)
    # model
    p.add_argument("--model_arch", type=str, default="vit_base")
    p.add_argument("--grad_from_block", type=int, default=11)
    p.add_argument("--cbl_hidden_layers", type=int, default=0)
    p.add_argument("--pretrain", type=str, default="dino")
    p.add_argument("--pretrain_path", type=str, default=None)
    # cbl training
    p.add_argument("--cbl_epochs", type=int, default=50)
    p.add_argument("--cbl_lr", type=float, default=1e-4)
    p.add_argument("--cbl_weight_decay", type=float, default=1e-5)
    p.add_argument("--cbl_batch_size", type=int, default=256)
    p.add_argument("--cbl_val_frac", type=float, default=0.1)
    p.add_argument("--cbl_log_freq", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="max grad norm (0 = measure only, no clipping)")
    p.add_argument("--pos_weight_clip", type=float, default=50.0,
                   help="cap per-concept BCE pos_weight to avoid rare-concept instability (0 = off)")
    # io
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--exp_root", type=str, default=exp_root)
    p.add_argument("--exp_name", type=str, default="stage1_cbl")
    # wandb
    p.add_argument("--use_wandb", type=str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="xGCD")
    p.add_argument("--wandb_entity", type=str, default="ifrat-ikhtear-university-of-south-dakota")
    return p


if __name__ == "__main__":
    args = get_stage1_parser().parse_args()
    run_stage1(args)
