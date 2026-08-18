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
from methods.contrastive_training.extract import extract_concept_logits, extract_backbone_features
from methods.contrastive_training.stage1_cbl import get_device, init_wandb
from methods.gcd.estep import run_estep
from methods.gcd.lda_gaussian import LDAGaussian
from methods.gcd.losses import (mahalanobis_contrastive_loss, prototypical_loss,
                                 fidelity_bce, mahalanobis_sq_pairs)
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
def compute_estep(model, lab_loader, unlab_loader, phase, args, device, prev_state=None,
                  frozen_lda=None):
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

    # warm-start the DPMM from the previous E-step's assignments (if enabled + available)
    prev_assign_unlab = None
    if (getattr(args, "dpmm_warm_start", True) and prev_state is not None
            and prev_state.get("assign_by_uq") is not None):
        prev_assign_unlab = prev_state["assign_by_uq"][unlab_uq.long().to(device)].cpu()
    # unlab_labels are GT — passed for the gate-distance DIAGNOSTIC only (logging), never
    # used to make a training decision inside run_estep, so no leakage.
    res = run_estep(lab_logits, lab_labels, unlab_logits, args,
                    unlab_labels=unlab_labels,
                    prev_assign_unlab=prev_assign_unlab, frozen_lda=frozen_lda)

    max_uq = int(max(int(lab_uq.max()), int(unlab_uq.max()))) + 1
    assign_by_uq = torch.zeros(max_uq, dtype=torch.long, device=device)
    assign_by_uq[unlab_uq.long()] = res.assignments.to(device)

    return dict(lda=res.lda.to(device), prototypes=res.prototypes.to(device),
                assign_by_uq=assign_by_uq, k_known=res.k_known, k_novel=res.k_novel,
                eval_logits=unlab_logits, eval_labels=unlab_labels,
                novel_pool=int(len(res.novel_idx)), tau=res.tau,
                cond_sigma=float(torch.linalg.cond(res.lda.cov)),
                novel_frac=float(len(res.novel_idx)) / max(len(unlab_logits), 1),
                logit_std=float(unlab_logits.std()),
                tau_emp=res.tau_emp, tau_chi2=res.tau_chi2,
                tau_ratio=(res.tau_emp / res.tau_chi2 if res.tau_chi2 else 0.0),
                lab_reject=res.lab_reject, novel_recall=res.novel_recall)


# --------------------------------------------------------------------------- #
# M-step loss on one (2-view) batch
# --------------------------------------------------------------------------- #
def _known_feature_basis(feats: torch.Tensor, energy: float = 0.99) -> torch.Tensor:
    """Orthonormal basis [in, r] spanning the top known-feature directions (>= `energy` of
    the feature variance). Weight updates orthogonal to this span leave W @ f_known fixed."""
    feats = feats.float()
    _, S, Vt = torch.linalg.svd(feats, full_matrices=False)          # Vt: [k, in]
    cum = torch.cumsum(S ** 2, 0) / torch.clamp(torch.sum(S ** 2), min=1e-12)
    r = int((cum < energy).sum().item()) + 1
    r = max(1, min(r, Vt.shape[0]))
    return Vt[:r].t().contiguous()                                   # [in, r]


def _apply_known_protection(cbl, known_proj):
    """After an optimiser step, snap the CBL's known-feature directions back to their
    phase-1 values so W @ f_known (and the bias) stay exactly fixed. Robust to weight
    decay / momentum because it constrains the weight itself, not the gradient."""
    UUt, W0_known, b0 = known_proj
    lin = cbl.model[0]
    with torch.no_grad():
        W = lin.weight
        W.sub_(W @ UUt).add_(W0_known)          # replace known component with frozen phase-1 one
        if b0 is not None:
            lin.bias.copy_(b0)


def mstep_loss(model, images, labels, uq, mask_lab, state, lookup, pos_weight, lambda_t, args, device,
               frozen_cbl=None):
    v = torch.cat([images[0], images[1]], dim=0).to(device)     # [2B, ...]
    feat = model.backbone(v)                                    # backbone features (reused for anchor)
    logits = model.cbl(feat)                                    # == model(v)
    B = images[0].shape[0]
    l1, l2 = logits[:B], logits[B:]

    lda, protos = state["lda"], state["prototypes"]
    z1, z2, protos_w = lda.whiten(l1), lda.whiten(l2), lda.whiten(protos)

    # L_CL (Eq 13) — whitened Euclidean contrastive between the two views
    nc = getattr(args, "norm_d2_by_c", True)
    L_cl = mahalanobis_contrastive_loss(z1, z2, precision=None, temperature=args.temperature,
                                        normalize_by_c=nc)

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

    # anti-contamination weight: down-weight (default 0 = drop) L_PCL for unlabelled
    # samples the gate assigned to a KNOWN cluster. The gate misroutes many novel samples
    # as known; pulling them onto known prototypes corrupts them (catastrophic forgetting).
    # Known prototypes stay defined by labelled data; unlabelled drive L_PCL only via novel.
    weights = None
    if state["assign_by_uq"] is not None:
        k_known = state.get("k_known", 0)
        w = torch.ones(B, device=device)
        unlab_known = (~mask_lab) & (targets >= 0) & (targets < k_known)
        w[unlab_known] = getattr(args, "pcl_known_unlab_weight", 0.0)
        weights = w

    # L_PCL (Eq 14) — averaged over both views against the same per-image prototype
    L_pcl = 0.5 * (prototypical_loss(z1, protos_w, targets, None, args.temperature, weights, nc) +
                   prototypical_loss(z2, protos_w, targets, None, args.temperature, weights, nc))

    # concept-fidelity BCE (Eq 16) — labelled images only, averaged over views
    if mask_lab.any():
        o = lookup.batch(uq[mask_lab].cpu().tolist()).to(device)
        L_bce = 0.5 * (fidelity_bce(l1[mask_lab], o, pos_weight) +
                       fidelity_bce(l2[mask_lab], o, pos_weight))
    else:
        L_bce = torch.zeros((), device=device)

    # known-concept anchor: distil labelled concepts toward the frozen phase-1 CBL so the
    # known concepts don't drift off their (frozen) prototypes -> keeps Old from collapsing.
    if frozen_cbl is not None and mask_lab.any():
        with torch.no_grad():
            f_tar = frozen_cbl(feat)                 # phase-1 CBL output = fixed target
        ft1, ft2 = f_tar[:B], f_tar[B:]
        L_anchor = 0.5 * (((l1[mask_lab] - ft1[mask_lab]) ** 2).mean() +
                          ((l2[mask_lab] - ft2[mask_lab]) ** 2).mean())
    else:
        L_anchor = torch.zeros((), device=device)

    loss = (args.w_cl * L_cl + lambda_t * L_pcl + args.alpha_fidelity * L_bce
            + args.anchor_known * L_anchor)
    # [cl-scale] whitened-distance scale: pos = matching views (diag), neg = off-diagonal.
    with torch.no_grad():
        d2 = mahalanobis_sq_pairs(z1, z2, None)
        pos_d2 = float(d2.diag().mean())
        neg_d2 = float(d2.mean())
    parts = dict(L_cl=float(L_cl), L_pcl=float(L_pcl), L_bce=float(L_bce),
                 L_anchor=float(L_anchor), pos_d2=pos_d2, neg_d2=neg_d2)
    return loss, parts


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
    if args.freeze_backbone:
        # anti-forgetting: keep the DINO features fixed, adapt only the concept layer.
        # The CBL is a sensitive linear layer (Stage 1 trained it at 1e-4) -> use the
        # small, explicit --cbl_lr, NOT the backbone lr (0.1 diverges immediately).
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        param_groups = [{"params": list(model.cbl.parameters()), "lr": args.cbl_lr}]
        logger.info(f"Backbone FROZEN — training CBL only at lr={args.cbl_lr}")
    else:
        param_groups = model.param_groups(args.lr, args.lr / args.cbl_lr_divisor)
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # baseline eval (epoch 0): the starting point straight from the Stage-1 CBL, before
    # any Stage-2 training. Distinguishes "training collapsed a good rep" (baseline high,
    # then drops) from "concept space never separated the classes" (baseline already low).
    baseline = compute_estep(model, lab_loader, unlab_loader, phase=2, args=args, device=device)
    last_eval = _eval_and_log(baseline, args, wandb_run, 0, vocab)
    _print_summary(0, total_epochs, 2, None, 0.0, baseline, last_eval, baseline=True)

    state = None
    global_epoch = 0
    prev_phase = None
    frozen_lda = None          # known-class Gaussian snapshot (set once at start of phase 2)
    frozen_cbl = None          # phase-1 CBL snapshot for the known-concept anchor
    known_proj = None          # (UUt, W0_known, b0) for known-feature gradient protection

    for epoch in range(total_epochs):
        phase = 1 if epoch < args.warmup_epochs else 2

        # blank separator only when the phase changes (baseline->P1, P1->P2)
        if phase != prev_phase:
            desc = "warm-up, known classes only" if phase == 1 else "joint: gate + DPMM"
            logger.info("")
            logger.info("_" * 70)
            logger.info(f"PHASE {phase}  ({desc})")
            logger.info("_" * 70)
            prev_phase = phase

        # ---- atomic E-step refresh ----
        if phase == 1:
            # respect --freeze_backbone in phase 1 too. Previously this always set
            # requires_grad=True, so the last ViT block trained at full lambda during
            # warm-up and L_PCL collapsed the concept space (within-class var -> 0).
            for p in model.backbone.parameters():
                p.requires_grad = not args.freeze_backbone
                
            if state is None or epoch % args.refresh_period == 0:
                state = compute_estep(model, lab_loader, None, phase=1, args=args, device=device)
            lambda_t = args.phase1_lambda
            loader = labelled_train
        else:
            # freezing the backbone
            for p in model.backbone.parameters():
                p.requires_grad = not args.freeze_backbone

            # snapshot the phase-1 CBL once, to anchor known concepts through phase 2
            if args.anchor_known > 0 and frozen_cbl is None:
                frozen_cbl = copy.deepcopy(model.cbl).to(device).eval()
                for p in frozen_cbl.parameters():
                    p.requires_grad = False
                logger.info("snapshot phase-1 CBL for known-concept anchor (L_anchor)")

            # snapshot the known-feature subspace once, to hard-protect known outputs
            if args.protect_known_grad and known_proj is None:
                feats_k, _, _ = extract_backbone_features(model.backbone, lab_loader, device)
                U = _known_feature_basis(feats_k.to(device), args.protect_known_energy)  # [in, r]
                if len(model.cbl.model) > 1:
                    logger.warning("protect_known_grad exactly protects known only for num_hidden=0")
                lin0 = model.cbl.model[0]
                UUt = U @ U.t()                                       # [in, in]
                b0 = lin0.bias.detach().clone() if lin0.bias is not None else None
                known_proj = (UUt, (lin0.weight.detach() @ UUt).clone(), b0)
                logger.info(f"protect-known: fixing {U.shape[1]}/{feats_k.shape[1]} known feature dirs")

            t = epoch - args.warmup_epochs
            if state is None or state.get("k_novel", -1) < 0 or t % args.refresh_period == 0 or state["assign_by_uq"] is None:
                state = compute_estep(model, lab_loader, unlab_loader, phase=2, args=args,
                                      device=device, prev_state=state, frozen_lda=frozen_lda)
                # snapshot the known Gaussian once, at the first phase-2 E-step, then reuse
                # it so known prototypes + metric stay FIXED (only novel means update).
                if args.freeze_known_protos and frozen_lda is None:
                    frozen_lda = copy.deepcopy(state["lda"]).to("cpu")
                    logger.info("froze known prototypes + Sigma (only novel means will update)")
                last_eval = _eval_and_log(state, args, wandb_run, epoch + 1, vocab) or last_eval

            lambda_t = min(args.lambda_max, t / max(args.lambda_warmup, 1))
            loader = merged_train

        # ---- M-step ----
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()   # frozen backbone -> deterministic features (no DropPath)
        agg = dict(L_cl=0.0, L_pcl=0.0, L_bce=0.0, L_anchor=0.0,
                   pos_d2=0.0, neg_d2=0.0, loss=0.0)
        nb = 0
        for batch in loader:
            if phase == 1:
                images, labels, uq = batch
                mask_lab = torch.ones(images[0].shape[0])
            else:
                images, labels, uq, mask_lab = batch

            loss, parts = mstep_loss(model, images, labels, uq, mask_lab, state,
                                     lookup, pos_weight, lambda_t, args, device,
                                     frozen_cbl=frozen_cbl)
            if not torch.isfinite(loss):
                logger.error(f"[xGCD] non-finite loss at epoch {epoch+1} — diverged. "
                             f"Lower --cbl_lr / --lr or set --grad_clip.")
                raise RuntimeError("Stage 2 diverged (non-finite loss).")
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if known_proj is not None:
                _apply_known_protection(model.cbl, known_proj)   # snap known dirs back to phase-1
            for k in parts:
                agg[k] += parts[k]
            agg["loss"] += float(loss)
            nb += 1
        scheduler.step()

        for k in agg:
            agg[k] /= max(nb, 1)
        logger.info(f"[xGCD] epoch {epoch+1}/{total_epochs} (phase {phase}) "
                    f"loss={agg['loss']:.4f} L_cl={agg['L_cl']:.4f} L_pcl={agg['L_pcl']:.4f} "
                    f"L_bce={agg['L_bce']:.4f} L_anchor={agg['L_anchor']:.4f} "
                    f"lambda={lambda_t:.2f} K^n={state.get('k_novel',0)}")
        # detailed, shareable summary block every 10 epochs (+ first & last)
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == total_epochs:
            _print_summary(epoch + 1, total_epochs, phase, agg, lambda_t, state, last_eval)
        if wandb_run is not None:
            gate = {f"gate/{k}": state.get(k, 0.0) for k in
                    ("tau_emp", "tau_chi2", "tau_ratio", "lab_reject", "novel_recall")
                    if state.get(k) is not None}
            wandb_run.log({f"stage2/{k}": v for k, v in agg.items()} | gate |
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
    if baseline:
        logger.info(" xGCD epoch 0  [BASELINE — straight from Stage-1 CBL, no Stage-2 training]")
    else:
        tag = " (FINAL)" if final else ""
        logger.info(f" xGCD SUMMARY epoch {epoch}/{total}  [phase {phase}]{tag}")
        if agg is not None:
            logger.info(f"   losses : total={agg['loss']:.4f}  L_cl={agg['L_cl']:.4f}  "
                        f"L_pcl={agg['L_pcl']:.4f}  L_bce={agg['L_bce']:.4f}  "
                        f"L_anchor={agg.get('L_anchor', 0.0):.4f}  (lambda={lambda_t:.2f})")
            logger.info(f"   cl-scale: pos_d2={agg.get('pos_d2', 0.0):.1f}  "
                        f"neg_d2={agg.get('neg_d2', 0.0):.1f}  (want pos << neg for a live L_CL)")
    logger.info(f"   E-step : K^n={state.get('k_novel', 0)}  "
                f"novel_pool={state.get('novel_pool', 0)} ({100 * state.get('novel_frac', 0.0):.1f}%)  "
                f"cond(Sigma)={state.get('cond_sigma', 0.0):.1f}  logit_std={state.get('logit_std', 0.0):.2f}")
    if state.get("novel_recall") is not None:
        logger.info(f"   gate   : tau_emp={state.get('tau_emp',0):.1f} "
                    f"tau_ratio={state.get('tau_ratio',0):.2f} "
                    f"lab_reject={state.get('lab_reject',0):.3f} "
                    f"novel_recall={state.get('novel_recall',0):.3f}")
    if last_eval:
        logger.info(f"   eval   : All={last_eval['all_acc']:.4f}  Old={last_eval['old_acc']:.4f}  "
                    f"New={last_eval['new_acc']:.4f}  |  K^n_hat={last_eval['k_hat_novel']} "
                    f"(true {last_eval['k_novel_true']}, err {last_eval['k_novel_err']})")


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
    p.add_argument("--freeze_known_protos", type=str2bool, default=False,
                   help="freeze known means + Sigma at the start of phase 2 (only novel means update)")
    # schedule
    p.add_argument("--warmup_epochs", type=int, default=20, help="Phase 1 (T0) epochs")
    p.add_argument("--epochs", type=int, default=200, help="Phase 2 joint epochs")
    p.add_argument("--refresh_period", type=int, default=5, help="E-step refresh period R")
    p.add_argument("--lambda_warmup", type=int, default=20, help="lambda(t)=lambda_max*min(1,t/T)")
    p.add_argument("--lambda_max", type=float, default=0.1, help="phase-2 L_PCL ceiling (keeps it gentle so it doesn't slowly collapse the concept space)")
    p.add_argument("--phase1_lambda", type=float, default=1.0, help="L_PCL weight during phase 1 (lower to avoid warm-up collapse)")
    # losses / metric
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature; d^2 is already normalised by C in the losses")
    p.add_argument("--w_cl", type=float, default=1.0, help="weight of L_CL (view-contrastive); lower to reduce logit_std inflation")
    p.add_argument("--alpha_fidelity", type=float, default=0.1)
    p.add_argument("--anchor_known", type=float, default=0.0,
                   help="weight of L_anchor: distil labelled concepts toward the phase-1 CBL (0=off; try ~1.0)")
    p.add_argument("--protect_known_grad", type=str2bool, default=False,
                   help="hard-freeze known-feature directions in the CBL weight (projection); alternative to --anchor_known")
    p.add_argument("--protect_known_energy", type=float, default=0.99,
                   help="fraction of known-feature variance to protect (higher = more dirs frozen, less room for novel)")
    p.add_argument("--pos_weight_clip", type=float, default=50.0)
    p.add_argument("--pcl_known_unlab_weight", type=float, default=0.0,
                   help="L_PCL weight for unlabelled samples the gate assigned to a KNOWN "
                        "cluster (0 = drop them; anti-contamination for the forgetting issue)")
    # E-step hyperparameters
    p.add_argument("--novelty_alpha", type=float, default=0.2)
    p.add_argument("--norm_d2_by_c", type=str2bool, default=True,
                   help="normalize d^2 by C in L_CL/L_PCL softmax (fix #3). False = pre-fix behaviour")
    p.add_argument("--tau_mode", type=str, default="empirical", choices=["empirical", "chi2"],
                   help="novelty threshold: empirical (labelled quantile) or chi^2")
    p.add_argument("--lda_ridge_gamma", type=float, default=0.1)
    p.add_argument("--dpmm_alpha", type=float, default=0.5)
    p.add_argument("--dpmm_beta", type=float, default=1.0)
    p.add_argument("--dpmm_sweeps", type=int, default=30)
    p.add_argument("--dpmm_max_points", type=int, default=8000)
    p.add_argument("--dpmm_min_cluster_size", type=int, default=10)
    p.add_argument("--dpmm_min_pool", type=int, default=10)
    p.add_argument("--dpmm_warm_start", type=str2bool, default=True,
                   help="seed each DPMM refresh from the previous clusters (stops K^n churn)")
    # optim
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--cbl_lr_divisor", type=float, default=100.0, help="CBL LR = lr / this (non-frozen)")
    p.add_argument("--cbl_lr", type=float, default=1e-3,
                   help="explicit CBL LR used when --freeze_backbone (0.1 diverges; 1e-3 is safe)")
    p.add_argument("--freeze_backbone", type=str2bool, default=False,
                   help="freeze DINO backbone, train only the CBL at --cbl_lr (anti-forgetting)")
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
