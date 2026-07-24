"""Stage 2 — joint novel-class discovery and GCD (xGCD draft Alg 1, Eq 12-16).

Loads the Stage-1 CBL and alternates, in an EM-like loop:
  * E-step (atomic, every R epochs): freeze the network, extract ALL concept logits in
    one pass, fit mu_k + tied Sigma on the labelled slice, gate the unlabelled slice
    (Stage A), run the DPMM on the novel pool (Stage B), and cache assignments. Every
    E-step quantity comes from the same representation snapshot.
  * M-step: train the last ViT block + CBL against the fixed prototypes/metric with
    L = L_CL + lambda(t) L_PCL + alpha_f L_BCE, all distances in the whitened space.

Phase 1 (T0 epochs, labelled only) establishes clean known geometry; Phase 2 turns on
the gate + DPMM and re-estimates K^n each refresh.
"""
import argparse
import copy
import os

import torch
from loguru import logger
from torch.utils.data import DataLoader

from config import exp_root
from data.augmentations import get_transform
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.concept_annotations import get_concept_vocab_and_lookup, load_vocabulary, ConceptTargetLookup
from data.data_utils import MergedDataset
from data.splits import configure_splits
from methods.contrastive_training.extract import extract_concept_logits
from methods.contrastive_training.stage1_cbl import get_device, init_wandb
from methods.gcd.estep import run_estep
from methods.gcd.lda_gaussian import LDAGaussian
from methods.gcd.losses import mahalanobis_contrastive_loss, prototypical_loss, fidelity_bce
from methods.gcd.eval_gcd import evaluate_gcd, explain_prototype
from models.model_factory import build_concept_model
from project_utils.general_utils import str2bool
from project_utils.loss_utils import ContrastiveLearningViewGenerator


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(args):
    args.interpolation = 3
    args.crop_pct = 0.875
    args.use_strong_aug = False
    train_tf, test_tf = get_transform("imagenet", image_size=args.image_size, args=args)
    two_view = ContrastiveLearningViewGenerator(base_transform=train_tf, n_views=2)

    get_ds = get_cifar_10_datasets if args.dataset_name == "cifar10" else get_cifar_100_datasets
    ds = get_ds(train_transform=two_view, test_transform=test_tf,
                train_classes=args.train_classes, prop_train_labels=args.prop_train_labels)

    # deterministic (single-view) copies for the E-step / eval extraction
    lab_extract = copy.deepcopy(ds["train_labelled"]); lab_extract.transform = test_tf
    unlab_extract = copy.deepcopy(ds["train_unlabelled"]); unlab_extract.transform = test_tf

    merged = MergedDataset(copy.deepcopy(ds["train_labelled"]), copy.deepcopy(ds["train_unlabelled"]))
    return ds, merged, lab_extract, unlab_extract


# --------------------------------------------------------------------------- #
# E-step (atomic snapshot)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_estep(model, lab_loader, unlab_loader, phase, args, device):
    model.eval()
    lab_logits, lab_labels, lab_uq, _ = extract_concept_logits(model, lab_loader, device)

    if phase == 1:
        lda = LDAGaussian(args.lda_ridge_gamma).fit(lab_logits, lab_labels)
        prototypes = lda.means
        state = dict(lda=lda.to(device), prototypes=prototypes.to(device),
                     assign_by_uq=None, k_known=len(lda.classes), k_novel=0,
                     eval_logits=None, eval_labels=None, novel_pool=0, tau=0.0,
                     cond_sigma=float(torch.linalg.cond(lda.cov)),
                     logit_std=float(lab_logits.std()))
        return state

    unlab_logits, unlab_labels, unlab_uq, _ = extract_concept_logits(model, unlab_loader, device)
    res = run_estep(lab_logits, lab_labels, unlab_logits, args)

    max_uq = int(max(int(lab_uq.max()), int(unlab_uq.max()))) + 1
    assign_by_uq = torch.zeros(max_uq, dtype=torch.long, device=device)
    assign_by_uq[unlab_uq.long()] = res.assignments.to(device)

    return dict(lda=res.lda.to(device), prototypes=res.prototypes.to(device),
                assign_by_uq=assign_by_uq, k_known=res.k_known, k_novel=res.k_novel,
                eval_logits=unlab_logits, eval_labels=unlab_labels,
                novel_pool=int(len(res.novel_idx)), tau=res.tau,
                cond_sigma=float(torch.linalg.cond(res.lda.cov)),
                novel_frac=float(len(res.novel_idx)) / max(len(unlab_logits), 1),
                logit_std=float(unlab_logits.std()))


# --------------------------------------------------------------------------- #
# M-step loss on one (2-view) batch
# --------------------------------------------------------------------------- #
def mstep_loss(model, images, labels, uq, mask_lab, state, lookup, pos_weight, lambda_t, args, device):
    v = torch.cat([images[0], images[1]], dim=0).to(device)     # [2B, ...]
    logits = model(v)
    B = images[0].shape[0]
    l1, l2 = logits[:B], logits[B:]

    lda, protos = state["lda"], state["prototypes"]
    z1, z2, protos_w = lda.whiten(l1), lda.whiten(l2), lda.whiten(protos)

    # L_CL (Eq 13) — whitened Euclidean contrastive between the two views
    L_cl = mahalanobis_contrastive_loss(z1, z2, precision=None, temperature=args.temperature)

    # per-image prototype target: GT class if labelled, cached E-step assignment if not
    labels = labels.to(device).long()
    mask_lab = mask_lab.reshape(-1).bool().to(device)
    uq = uq.reshape(-1).to(device)
    if state["assign_by_uq"] is not None:
        targets = torch.where(mask_lab, labels, state["assign_by_uq"][uq])
    else:
        targets = labels
    # sanitise: anything outside [0, K) becomes -1 -> ignored by L_PCL (e.g. unlabelled
    # samples the E-step left unassigned when the novel pool was too small).
    K = protos.shape[0]
    targets = torch.where((targets >= 0) & (targets < K), targets, torch.full_like(targets, -1))

    # L_PCL (Eq 14) — averaged over both views against the same per-image prototype
    L_pcl = 0.5 * (prototypical_loss(z1, protos_w, targets, None, args.temperature) +
                   prototypical_loss(z2, protos_w, targets, None, args.temperature))

    # concept-fidelity BCE (Eq 16) — labelled images only, averaged over views
    if mask_lab.any():
        o = lookup.batch(uq[mask_lab].cpu().tolist()).to(device)
        L_bce = 0.5 * (fidelity_bce(l1[mask_lab], o, pos_weight) +
                       fidelity_bce(l2[mask_lab], o, pos_weight))
    else:
        L_bce = torch.zeros((), device=device)

    loss = L_cl + lambda_t * L_pcl + args.alpha_fidelity * L_bce
    return loss, dict(L_cl=float(L_cl), L_pcl=float(L_pcl), L_bce=float(L_bce))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def run_xgcd(args):
    device = get_device()
    configure_splits(args)
    args.total_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # vocabulary: load the Stage-1 vocab so C matches the trained CBL
    vocab_path = os.path.join(args.cbl_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Stage-1 vocab not found at {vocab_path}. Run Stage 1 first "
            f"(it saves cbl_stage1.pt + vocab.json into --cbl_dir).")
    vocab = load_vocabulary(vocab_path)
    args.num_concepts = len(vocab)
    logger.info(f"xGCD Stage 2: device={device}, C={args.num_concepts}, "
                f"K^l={args.num_labeled_classes}, K^n(true)={args.num_unlabeled_classes}")

    ds, merged, lab_extract, unlab_extract = build_data(args)

    # concept-target lookup (for the fidelity BCE) + pos_weight from labelled
    from config import concept_annotation_root, concept_annotation_dirs
    ann_dir = os.path.join(concept_annotation_root, concept_annotation_dirs[args.dataset_name][0])
    lookup = ConceptTargetLookup(ann_dir, vocab, threshold=args.concept_conf_threshold)
    labelled_uq = list(ds["train_labelled"].uq_idxs)
    lookup.precompute(labelled_uq)          # avoid re-reading JSONs every M-step batch
    pos_weight = lookup.pos_weight(labelled_uq)
    if args.pos_weight_clip > 0:
        pos_weight = pos_weight.clamp(max=args.pos_weight_clip)

    # model + Stage-1 CBL
    model = build_concept_model(args).to(device)
    cbl_ckpt = os.path.join(args.cbl_dir, "cbl_stage1.pt")
    model.cbl.load_state_dict(torch.load(cbl_ckpt, map_location=device))
    logger.info(f"Loaded Stage-1 CBL from {cbl_ckpt}")

    wandb_run = init_wandb(args, stage="stage2")

    # loaders
    lab_loader = DataLoader(lab_extract, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    unlab_loader = DataLoader(unlab_extract, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    labelled_train = DataLoader(copy.deepcopy(ds["train_labelled"]), batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, drop_last=True)
    merged_train = DataLoader(merged, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)

    total_epochs = args.warmup_epochs + args.epochs
    optimizer = torch.optim.SGD(model.param_groups(args.lr, args.lr / args.cbl_lr_divisor),
                                momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # baseline eval (epoch 0): the starting point straight from the Stage-1 CBL, before
    # any Stage-2 training. Distinguishes "training collapsed a good rep" (baseline high,
    # then drops) from "concept space never separated the classes" (baseline already low).
    baseline = compute_estep(model, lab_loader, unlab_loader, phase=2, args=args, device=device)
    last_eval = _eval_and_log(baseline, args, wandb_run, 0, vocab)
    _print_summary(0, total_epochs, 2, None, 0.0, baseline, last_eval, baseline=True)

    state = None
    global_epoch = 0

    for epoch in range(total_epochs):
        phase = 1 if epoch < args.warmup_epochs else 2
        # ---- atomic E-step refresh ----
        if phase == 1:
            if state is None or epoch % args.refresh_period == 0:
                state = compute_estep(model, lab_loader, None, phase=1, args=args, device=device)
            lambda_t = 1.0
            loader = labelled_train
        else:
            t = epoch - args.warmup_epochs
            if state is None or state.get("k_novel", -1) < 0 or t % args.refresh_period == 0 or state["assign_by_uq"] is None:
                state = compute_estep(model, lab_loader, unlab_loader, phase=2, args=args, device=device)
                last_eval = _eval_and_log(state, args, wandb_run, epoch, vocab) or last_eval
            lambda_t = min(1.0, t / max(args.lambda_warmup, 1))
            loader = merged_train

        # ---- M-step ----
        model.train()
        agg = dict(L_cl=0.0, L_pcl=0.0, L_bce=0.0, loss=0.0)
        nb = 0
        for batch in loader:
            if phase == 1:
                images, labels, uq = batch
                mask_lab = torch.ones(images[0].shape[0])
            else:
                images, labels, uq, mask_lab = batch
            loss, parts = mstep_loss(model, images, labels, uq, mask_lab, state,
                                     lookup, pos_weight, lambda_t, args, device)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for k in parts:
                agg[k] += parts[k]
            agg["loss"] += float(loss)
            nb += 1
        scheduler.step()

        for k in agg:
            agg[k] /= max(nb, 1)
        logger.info(f"[xGCD] epoch {epoch+1}/{total_epochs} (phase {phase}) "
                    f"loss={agg['loss']:.4f} L_cl={agg['L_cl']:.4f} L_pcl={agg['L_pcl']:.4f} "
                    f"L_bce={agg['L_bce']:.4f} lambda={lambda_t:.2f} K^n={state.get('k_novel',0)}")
        # detailed, shareable summary block every 10 epochs (+ first & last)
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == total_epochs:
            _print_summary(epoch + 1, total_epochs, phase, agg, lambda_t, state, last_eval)
        if wandb_run is not None:
            wandb_run.log({f"stage2/{k}": v for k, v in agg.items()} |
                          {"stage2/phase": phase, "stage2/lambda": lambda_t,
                           "stage2/k_novel": state.get("k_novel", 0),
                           "stage2/novel_pool": state.get("novel_pool", 0),
                           "stage2/novel_frac": state.get("novel_frac", 0.0),
                           "stage2/cond_sigma": state.get("cond_sigma", 0.0),
                           "stage2/logit_std": state.get("logit_std", 0.0),
                           "stage2/lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)
        global_epoch = epoch + 1

    # final eval
    final = compute_estep(model, lab_loader, unlab_loader, phase=2, args=args, device=device)
    metrics = _eval_and_log(final, args, wandb_run, global_epoch, vocab, final=True)
    _print_summary(global_epoch, total_epochs, 2, None, 1.0, final, metrics, final=True)
    _save(model, final, vocab, args)
    if wandb_run is not None:
        wandb_run.finish()
    return metrics


def _print_summary(epoch, total, phase, agg, lambda_t, state, last_eval, baseline=False, final=False):
    """One compact, greppable block with everything needed to diagnose a run."""
    bar = "=" * 66
    logger.info(bar)
    if baseline:
        logger.info(" xGCD epoch 0  [BASELINE — straight from Stage-1 CBL, no Stage-2 training]")
    else:
        tag = " (FINAL)" if final else ""
        logger.info(f" xGCD SUMMARY epoch {epoch}/{total}  [phase {phase}]{tag}")
        if agg is not None:
            logger.info(f"   losses : total={agg['loss']:.4f}  L_cl={agg['L_cl']:.4f}  "
                        f"L_pcl={agg['L_pcl']:.4f}  L_bce={agg['L_bce']:.4f}  (lambda={lambda_t:.2f})")
    logger.info(f"   E-step : K^n={state.get('k_novel', 0)}  "
                f"novel_pool={state.get('novel_pool', 0)} ({100 * state.get('novel_frac', 0.0):.1f}%)  "
                f"cond(Sigma)={state.get('cond_sigma', 0.0):.1f}  logit_std={state.get('logit_std', 0.0):.2f}")
    if last_eval:
        logger.info(f"   eval   : All={last_eval['all_acc']:.4f}  Old={last_eval['old_acc']:.4f}  "
                    f"New={last_eval['new_acc']:.4f}  |  K^n_hat={last_eval['k_hat_novel']} "
                    f"(true {last_eval['k_novel_true']}, err {last_eval['k_novel_err']})")
    logger.info(bar)


def _eval_and_log(state, args, wandb_run, epoch, vocab, final=False):
    if state["eval_logits"] is None:
        return None
    m = evaluate_gcd(state["eval_logits"], state["eval_labels"], state["prototypes"],
                     state["lda"], args.num_labeled_classes, args.total_classes)
    tag = "final" if final else "eval"
    logger.info(f"[{tag}] epoch {epoch}: All {m['all_acc']:.4f} | Old {m['old_acc']:.4f} | "
                f"New {m['new_acc']:.4f} | K^n_hat={m['k_hat_novel']} (true {m['k_novel_true']}, "
                f"err {m['k_novel_err']})")
    if wandb_run is not None:
        wandb_run.log({f"{tag}/all_acc": m["all_acc"], f"{tag}/old_acc": m["old_acc"],
                       f"{tag}/new_acc": m["new_acc"], f"{tag}/k_hat_novel": m["k_hat_novel"],
                       f"{tag}/k_novel_err": m["k_novel_err"]}, step=epoch)
    if final:
        for k in range(min(state["k_known"] + state["k_novel"], state["prototypes"].shape[0])):
            kind = "known" if k < state["k_known"] else "novel"
            top = explain_prototype(state["prototypes"][k], vocab, top=5)
            logger.info(f"  proto[{k}] ({kind}): " + ", ".join(f"{c}:{p:.2f}" for c, p in top))
    return m


def _save(model, state, vocab, args):
    save_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    torch.save({"prototypes": state["prototypes"].cpu(), "k_known": state["k_known"],
                "k_novel": state["k_novel"], "cov": state["lda"].cov.cpu()},
               os.path.join(save_dir, "gcd_state.pt"))
    logger.info(f"Saved xGCD model + prototypes -> {save_dir}")


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_xgcd_parser():
    p = argparse.ArgumentParser(description="xGCD Stage 2: joint discovery + GCD")
    p.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--num_labeled_classes", type=int, default=None)
    p.add_argument("--prop_train_labels", type=float, default=0.5)
    p.add_argument("--concept_conf_threshold", type=float, default=0.15)
    p.add_argument("--cbl_dir", type=str, required=True, help="Stage-1 output dir (cbl_stage1.pt, vocab.json)")
    # model
    p.add_argument("--model_arch", type=str, default="vit_base")
    p.add_argument("--grad_from_block", type=int, default=11)
    p.add_argument("--cbl_hidden_layers", type=int, default=0)
    p.add_argument("--pretrain", type=str, default="dino")
    p.add_argument("--pretrain_path", type=str, default=None)
    # schedule
    p.add_argument("--warmup_epochs", type=int, default=20, help="Phase 1 (T0) epochs")
    p.add_argument("--epochs", type=int, default=200, help="Phase 2 joint epochs")
    p.add_argument("--refresh_period", type=int, default=5, help="E-step refresh period R")
    p.add_argument("--lambda_warmup", type=int, default=20, help="lambda(t)=min(1,t/T)")
    # losses / metric
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--alpha_fidelity", type=float, default=0.1)
    p.add_argument("--pos_weight_clip", type=float, default=50.0)
    # E-step hyperparameters
    p.add_argument("--novelty_alpha", type=float, default=0.05)
    p.add_argument("--lda_ridge_gamma", type=float, default=0.1)
    p.add_argument("--dpmm_alpha", type=float, default=1.0)
    p.add_argument("--dpmm_beta", type=float, default=1.0)
    p.add_argument("--dpmm_sweeps", type=int, default=30)
    p.add_argument("--dpmm_max_points", type=int, default=8000)
    p.add_argument("--dpmm_min_cluster_size", type=int, default=10)
    p.add_argument("--dpmm_min_pool", type=int, default=10)
    # optim
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--cbl_lr_divisor", type=float, default=100.0, help="CBL LR = lr / this")
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--grad_clip", type=float, default=0.0)
    # io
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--exp_root", type=str, default=exp_root)
    p.add_argument("--exp_name", type=str, default="stage2_xgcd")
    # wandb
    p.add_argument("--use_wandb", type=str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="xGCD")
    p.add_argument("--wandb_entity", type=str, default="ifrat-ikhtear-university-of-south-dakota")
    return p


if __name__ == "__main__":
    args = get_xgcd_parser().parse_args()
    run_xgcd(args)
