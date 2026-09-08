"""Phase 3 — parametric CE refinement on the frozen discovery estimate (xGCD draft).

Pipeline: run Stage-2 estimation ONCE to fix the class set K = K_l + K_n; optionally drop
incoherent (junk) novel classes with a label-free peak-activation gate; initialise a linear
head from the prototypes (LDA-classifier identity) and train it with cross-entropy over the
FIXED class set. No DPMM / gate in the loop, so K cannot oscillate.

  * targets: GT for labelled samples; the head's own argmax for unlabelled (refreshed each
    few epochs so early mistakes get corrected as ell sharpens).
  * confidence weighting on temperature-scaled logits (head logits are Mahalanobis-scale, so
    a raw softmax saturates to one-hot -> the weight would silently be a no-op).
  * loss = CE + alpha_f * BCE.  L_CL / L_PCL are dropped (CE replaces them).
  * canaries: logit_std (divergence), Old-ACC (known-boundary erosion), refresh confidences.
  * eval both ways every refresh: head-argmax (headline) and nearest-prototype (monitor),
    against the frozen reference number.
"""
import argparse
import copy
import os

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from config import concept_annotation_root, concept_annotation_dirs
from data.concept_annotations import load_vocabulary, ConceptTargetLookup
from data.splits import configure_splits
from methods.contrastive_training.extract import extract_concept_logits
from methods.contrastive_training.stage1_cbl import get_device, init_wandb
from methods.contrastive_training.xgcd import build_data, get_xgcd_parser
from methods.gcd.estep import run_estep
from methods.gcd.losses import fidelity_bce
from methods.gcd.eval_gcd import assign_to_prototypes, explain_prototype
from models.classifier_head import ClassifierHead
from models.model_factory import build_concept_model
from project_utils.cluster_and_log_utils import split_cluster_acc_v2
from project_utils.general_utils import str2bool
from project_utils.ram_tagging_utils import DEFAULT_RAM_PRETRAINED
from project_utils.concept_discovery_utils import (
    ConceptDriftTracker, combined_fidelity_bce, discover_novel_concepts,
    warmup_cbl_and_refit_prototypes)


@torch.no_grad()
def _peak_gate(prototypes, k_known, thresh, do_gate):
    """Label-free coherence gate: keep a novel prototype iff its peak concept prob
    max_c sigma(mu)[c] >= thresh (junk clusters average to an incoherent ~0.5 peak).
    Returns the (possibly reduced) prototype set. Logs each novel prototype's decision."""
    novel = prototypes[k_known:]
    if novel.shape[0] == 0:
        return prototypes
    peaks = torch.sigmoid(novel).max(dim=1).values
    keep = peaks >= thresh if do_gate else torch.ones_like(peaks, dtype=torch.bool)
    for j, (pk, kp) in enumerate(zip(peaks.tolist(), keep.tolist())):
        tag = "GRADUATE" if kp else "DROP(junk)"
        logger.info(f"[phase3 peak-gate] novel {j}: peak={pk:.2f} thresh={thresh} -> {tag}")
    logger.info(f"[phase3 peak-gate] gate={'ON' if do_gate else 'OFF'} | "
                f"{int(keep.sum())}/{novel.shape[0]} novel graduate -> "
                f"K_total={k_known + int(keep.sum())}")
    return torch.cat([prototypes[:k_known], novel[keep]], dim=0)


@torch.no_grad()
def _eval(model, head, loader, prototypes, lda, k_lab, k_total, k_true, device, tag, epoch):
    """Dual eval on the unlabelled set: head-argmax (headline) + nearest-prototype (monitor)."""
    model.eval(); head.eval()
    ell, y, _, _ = extract_concept_logits(model, loader, device)
    ell = ell.to(device)
    y = y.cpu().numpy().astype(int)
    mask_old = y < k_lab
    hp = head(ell).argmax(1).cpu().numpy()
    h_all, h_old, h_new = split_cluster_acc_v2(y, hp, mask_old)
    npd = assign_to_prototypes(ell, prototypes, lda).cpu().numpy()
    n_all, n_old, n_new = split_cluster_acc_v2(y, npd, mask_old)
    k_hat = k_total - k_lab
    logger.info(f"[phase3 {tag}] epoch {epoch}: head All={h_all:.4f} Old={h_old:.4f} New={h_new:.4f} "
                f"| proto-monitor All={n_all:.4f} | K_hat={k_hat} (true {k_true}, err {abs(k_hat-k_true)}) "
                f"| logit_std={float(ell.std()):.2f}")
    return dict(all=h_all, old=h_old, new=h_new, logit_std=float(ell.std()))


def run_phase3(args):
    logger.info("Args:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}={v}")

    device = get_device()
    # reproducible CE refinement: without this the head/CBL training is unseeded and the
    # final All/Old/New vary run-to-run (K is already fixed by the seeded estimation).
    seed = int(getattr(args, "seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"[phase3] seeded: seed={seed} (cudnn.deterministic=True)")
    configure_splits(args)
    args.total_classes = args.num_labeled_classes + args.num_unlabeled_classes

    vocab = load_vocabulary(os.path.join(args.cbl_dir, "vocab.json"))
    args.num_concepts = len(vocab)
    logger.info(f"[phase3] device={device}, C={args.num_concepts}, K_l={args.num_labeled_classes}, "
                f"K_n(true)={args.num_unlabeled_classes}")

    ds, merged, lab_extract, unlab_extract = build_data(args)

    ann_dir = os.path.join(concept_annotation_root, concept_annotation_dirs[args.dataset_name][0])
    lookup = ConceptTargetLookup(ann_dir, vocab, threshold=args.concept_conf_threshold)
    labelled_uq = list(ds["train_labelled"].uq_idxs)
    lookup.precompute(labelled_uq)
    pos_weight = lookup.pos_weight(labelled_uq)
    if args.pos_weight_clip > 0:
        pos_weight = pos_weight.clamp(max=args.pos_weight_clip)

    model = build_concept_model(args).to(device)
    model.cbl.load_state_dict(torch.load(os.path.join(args.cbl_dir, "cbl_stage1.pt"), map_location=device))
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    logger.info("[phase3] backbone frozen; training CBL + head")

    wandb_run = init_wandb(args, stage="phase3")

    lab_loader = DataLoader(lab_extract, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    unlab_loader = DataLoader(unlab_extract, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    merge_gen = torch.Generator().manual_seed(seed)   # shuffle depends only on the seed
    merged_train = DataLoader(merged, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, generator=merge_gen)

    # ---- 1. frozen discovery estimate (single E-step; deterministic seed) ----
    model.eval()
    lab_logits, lab_labels, lab_uq, _ = extract_concept_logits(model, lab_loader, device)
    unlab_logits, unlab_labels, unlab_uq, _ = extract_concept_logits(model, unlab_loader, device)
    res = run_estep(lab_logits, lab_labels, unlab_logits, args, unlab_labels=unlab_labels)
    lda = res.lda.to(device)
    k_known = res.k_known
    logger.info(f"[phase3] frozen estimate: K_l={k_known} K_n={res.k_novel} "
                f"total_prototypes={res.prototypes.shape[0]}")

    # ---- 2. peak-activation gate (toggleable) -> the FIXED class set ----
    prototypes = _peak_gate(res.prototypes.to(device), k_known, args.peak_thresh, args.peak_gate)
    k_total = prototypes.shape[0]

    # ---- 2.b. novel-concept discovery (RAM++, no class label / vocabulary needed) ----
    # Grows vocab/CBL/prototypes/lda/lookup/pos_weight in place if args.novel_concepts;
    # a no-op passthrough otherwise (or if no novel cluster survived the peak-gate).
    num_old_concepts = args.num_concepts
    novel_o_by_uq, cluster_uqs = {}, {}
    if args.novel_concepts:
        vocab, model, lookup, pos_weight, prototypes, lda, novel_o_by_uq, cluster_uqs = discover_novel_concepts(
            args, model, vocab, lookup, pos_weight, prototypes, lda, k_known,
            unlab_logits, unlab_extract, lab_extract, device)
        args.num_concepts = len(vocab)

    # ---- 2.c. CBL warmup (all images, combined known+novel targets) + prototype/LDA refit ----
    # Off by default (--novel_cbl_warmup_epochs 0). When on, replaces the RAM-frequency
    # ASSUMPTION baked into new-dim prototypes/LDA at discovery time with real values measured
    # from a CBL that's actually been trained on those dims -- see
    # warmup_cbl_and_refit_prototypes's docstring for why that fixes the epoch-0 Old-ACC
    # collapse a freshly-grown, untrained CBL otherwise causes.
    new_prototypes, new_lda = warmup_cbl_and_refit_prototypes(
        model, merged_train, lab_extract, unlab_extract, lookup, novel_o_by_uq,
        num_old_concepts, k_known, k_total, cluster_uqs, pos_weight, args, device)
    if new_prototypes is not None:
        prototypes, lda = new_prototypes, new_lda

    drift = ConceptDriftTracker(k_known, k_total, num_old_concepts, vocab, prototypes)

    # ---- 3. head, initialised from the (gated) prototypes ----
    head = ClassifierHead(args.num_concepts, k_total).to(device)
    head.init_from_prototypes(prototypes, lda.precision)
    conf_temp = args.conf_temp if args.conf_temp > 0 else float(args.num_concepts)
    logger.info(f"[phase3] head init from {k_total} prototypes | conf_temp={conf_temp:.1f}")

    # ---- 3.b known-class rows use OLD concepts only (structural, not just trained-toward) ----
    # logit = W . ell + b; zeroing W's new-dim columns for known rows means a known class's
    # score literally cannot depend on the new concepts' values, whatever CE does elsewhere.
    # Novel rows are untouched -- they're supposed to use the full concept set. Re-applied
    # after every optimizer step below so training can't drift these back away from zero.
    known_rows_old_dims_only = args.novel_concepts and args.num_concepts > num_old_concepts and args.novel_known_rows_old_dims_only
    if known_rows_old_dims_only:
        with torch.no_grad():
            head.fc.weight[:k_known, num_old_concepts:].zero_()
            head.fc.bias[:k_known] = -0.5 * (head.fc.weight[:k_known] * prototypes[:k_known]).sum(dim=1)
        logger.info(f"[phase3] known-class head rows restricted to the original {num_old_concepts} "
                   f"concepts (new-dim weights zeroed, bias recomputed to match) -- known "
                   f"classification structurally unaffected by the new concepts")

    # ---- 4. pseudo-labels by uq: GT for labelled, head-argmax for unlabelled ----
    max_uq = int(max(int(lab_uq.max()), int(unlab_uq.max()))) + 1
    pseudo = torch.full((max_uq,), -1, dtype=torch.long, device=device)
    conf = torch.ones(max_uq, device=device)

    def refresh_pseudo():
        model.eval(); head.eval()
        with torch.no_grad():
            ell, _, uq, _ = extract_concept_logits(model, unlab_loader, device)
            probs = F.softmax(head(ell.to(device)) / conf_temp, dim=1)
            c, a = probs.max(dim=1)
            uq = uq.long().to(device)
            pseudo[uq] = a
            conf[uq] = c
        logger.info(f"[phase3 refresh] pseudo updated | conf mean={float(c.mean()):.3f} "
                    f"min={float(c.min()):.3f} max={float(c.max()):.3f}")

    refresh_pseudo()

    # ---- epoch-0: gate + init, NO training (its own result; != the 11-proto reference) ----
    base = _eval(model, head, unlab_loader, prototypes, lda, k_known, k_total,
                 args.num_unlabeled_classes, device, "epoch0", 0)
    logger.info(f"[phase3] chain: {args.ref_acc:.4f} (frozen-proto ref) -> "
                f"epoch-0 head {base['all']:.4f} (gated init, no training) -> trained head below")
    drift.log(model, lab_loader, unlab_loader, prototypes, lda, device, "epoch0")

    # ---- 5. optimiser: two LR groups (head fresh -> higher LR; CBL sensitive -> low) ----
    optimizer = torch.optim.SGD(
        [{"params": head.parameters(), "lr": args.head_lr},
         {"params": model.cbl.parameters(), "lr": args.cbl_lr}],
        momentum=0.9, weight_decay=args.weight_decay)

    # ---- 6. training loop ----
    m = base
    for epoch in range(args.epochs):
        labelled_only = epoch < args.labelled_warmup     # stabilise head on GT before self-training
        model.train(); model.backbone.eval(); head.train()
        agg = dict(ce=0.0, bce=0.0, loss=0.0); nb = 0
        for images, labels, uq, mask_lab in merged_train:
            x = images[0].to(device)                     # single augmented view (no contrastive term)
            ell = model(x)
            labels = labels.to(device).long()
            uq = uq.to(device).long()
            mask_lab = mask_lab.reshape(-1).bool().to(device)
            targets = torch.where(mask_lab, labels, pseudo[uq])
            weights = torch.where(mask_lab, torch.ones_like(conf[uq]), conf[uq])
            if labelled_only:
                weights = weights * mask_lab.float()     # unlabelled muted during warmup
            valid = (targets >= 0) & (targets < k_total)
            w = weights * valid.float()
            logits = head(ell)
            ce_all = F.cross_entropy(logits, targets.clamp(0, k_total - 1), reduction="none")
            denom = w.sum().clamp_min(1.0)
            L_ce = (ce_all * w).sum() / denom
            if args.novel_bce_include_unlabelled and novel_o_by_uq:
                # combined target: labelled (full C dims) + novel (new dims only, RAM-derived)
                # -- gives the CBL direct supervision from the images that actually motivated
                # each new concept, not just the labelled set (which rarely shows them).
                L_bce = combined_fidelity_bce(ell, uq, mask_lab, lookup, novel_o_by_uq,
                                              num_old_concepts, pos_weight, device)
            elif mask_lab.any():
                o = lookup.batch(uq[mask_lab].cpu().tolist()).to(device)
                L_bce = fidelity_bce(ell[mask_lab], o, pos_weight)
            else:
                L_bce = torch.zeros((), device=device)
            loss = args.ce_weight * L_ce + args.alpha_fidelity * L_bce
            if not torch.isfinite(loss):
                logger.error(f"[phase3] non-finite loss at epoch {epoch+1} — diverged.")
                raise RuntimeError("Phase 3 diverged.")
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(head.parameters()) + list(model.cbl.parameters()), args.grad_clip)
            optimizer.step()
            if known_rows_old_dims_only:
                with torch.no_grad():
                    head.fc.weight[:k_known, num_old_concepts:].zero_()
            agg["ce"] += float(L_ce); agg["bce"] += float(L_bce); agg["loss"] += float(loss); nb += 1
        for k in agg:
            agg[k] /= max(nb, 1)

        # refresh pseudo-labels + eval every refresh_period (and on the last epoch)
        do_refresh = (epoch + 1) % args.refresh_period == 0 or (epoch + 1) == args.epochs
        if do_refresh and not labelled_only:
            refresh_pseudo()
        if do_refresh:
            drift.log(model, lab_loader, unlab_loader, prototypes, lda, device, f"epoch{epoch+1}")
        m = _eval(model, head, unlab_loader, prototypes, lda, k_known, k_total,
                  args.num_unlabeled_classes, device, "eval", epoch + 1)
        logger.info(f"[phase3] epoch {epoch+1}/{args.epochs} loss={agg['loss']:.4f} "
                    f"ce={agg['ce']:.4f} bce={agg['bce']:.4f} lab_only={labelled_only}")
        # canaries
        if m["logit_std"] > args.logit_std_stop:
            logger.error(f"[phase3 canary] logit_std={m['logit_std']:.1f} > {args.logit_std_stop} — diverging, stop.")
            break
        if m["old"] < base["old"] - args.old_drop_stop:
            logger.warning(f"[phase3 canary] Old {base['old']:.3f}->{m['old']:.3f} "
                           f"(drop>{args.old_drop_stop}) — known-boundary erosion (confirmation bias?)")
        if wandb_run is not None:
            wandb_run.log({"phase3/loss": agg["loss"], "phase3/ce": agg["ce"], "phase3/bce": agg["bce"],
                           "phase3/all": m["all"], "phase3/old": m["old"], "phase3/new": m["new"],
                           "phase3/logit_std": m["logit_std"], "phase3/k_hat": k_total - k_known},
                          step=epoch + 1)

    # ---- 7. final: explanations + save ----
    logger.info(f"[phase3] FINAL vs reference {args.ref_acc:.4f}: "
                f"epoch-0 {base['all']:.4f} -> trained {m['all']:.4f} "
                f"({'BEAT' if m['all'] > args.ref_acc else 'did NOT beat'} the frozen pipeline)")
    for k in range(k_total):
        kind = "known" if k < k_known else "novel"
        top = explain_prototype(prototypes[k], vocab, top=5)
        logger.info(f"  proto[{k}] ({kind}): " + ", ".join(f"{c}:{p:.2f}" for c, p in top))
    save_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    torch.save({"head": head.state_dict(), "prototypes": prototypes.cpu(),
                "k_known": k_known, "k_total": k_total, "cov": lda.cov.cpu()},
               os.path.join(save_dir, "phase3_state.pt"))
    logger.info(f"[phase3] saved -> {save_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    return m


def get_phase3_parser():
    p = get_xgcd_parser()
    p.add_argument("--peak_gate", type=str2bool, default=True,
                   help="drop novel classes with prototype peak sigma(mu) < peak_thresh (junk). "
                        "Toggle OFF for datasets where the coherence gate is not yet calibrated")
    p.add_argument("--ce_weight", type=float, default=1.0, help="weight on cross-entropy")
    p.add_argument("--head_lr", type=float, default=1e-2, help="LR for the fresh classifier head")
    p.add_argument("--labelled_warmup", type=int, default=2,
                   help="epochs of labelled-only CE before trusting pseudo-labels")
    p.add_argument("--conf_temp", type=float, default=0.0,
                   help="temperature for the confidence softmax (0 = use C, the concept count)")
    p.add_argument("--logit_std_stop", type=float, default=9.0, help="canary: stop if logit_std exceeds")
    p.add_argument("--old_drop_stop", type=float, default=0.05, help="canary: warn if Old-ACC drops by more")
    p.add_argument("--ref_acc", type=float, default=0.8707,
                   help="frozen-pipeline reference All-ACC to beat (for the comparison chain)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for reproducible CE refinement (shuffle + init + cudnn deterministic)")
    # novel-concept discovery (RAM++; see concept_discovery.py)
    p.add_argument("--novel_concepts", type=str2bool, default=False,
                   help="RAM++-discover concepts for novel DPMM clusters and grow the CBL for them")
    p.add_argument("--novel_min_images_per_cluster", type=int, default=2,
                   help="a RAM tag must appear in >= this many of a cluster's own images to be a candidate")
    p.add_argument("--novel_drop_known_universal_thresh", type=float, default=0.0,
                   help="drop a candidate concept if it appears in >= this fraction of some "
                        "EXISTING known class's own labelled images (0 = disabled). A concept "
                        "near-universal for a known class is a poor discriminator regardless of "
                        "how common it is within the discovering novel cluster")
    p.add_argument("--ram_pretrained", type=str, default=DEFAULT_RAM_PRETRAINED)
    p.add_argument("--ram_image_size", type=int, default=384)
    p.add_argument("--ram_batch_size", type=int, default=32)
    p.add_argument("--cluster_images_dir", type=str, default="cluster_images",
                   help="per novel cluster: nearest-to-center images + finalized concepts -> <this>/<cluster_id>/")
    p.add_argument("--cluster_images_n", type=int, default=20,
                   help="how many nearest-to-center member images to save per novel cluster")
    p.add_argument("--novel_cbl_warmup_epochs", type=int, default=0,
                   help="epochs to pre-train the (just-grown) CBL on ALL images against combined "
                        "known+novel targets before building the head from prototypes (0 = off, "
                        "keeps the RAM-frequency-assumption prototypes/LDA from discovery as-is)")
    p.add_argument("--novel_cbl_warmup_lr", type=float, default=1e-3,
                   help="Adam LR for the CBL-only warmup stage")
    p.add_argument("--novel_bce_include_unlabelled", type=str2bool, default=False,
                   help="in the main training loop, also supervise the new concept dims on "
                        "unlabelled/novel images (RAM-derived targets), not just labelled ones")
    p.add_argument("--novel_known_rows_old_dims_only", type=str2bool, default=False,
                   help="zero the head's new-concept weights for KNOWN class rows (kept at zero "
                        "throughout training) so known classification structurally can't depend "
                        "on the new concepts, even though the CBL still learns them from all images")
    # Grounding DINO (labelled-set targets for newly discovered concepts; not installed by
    # default here -- see grounding_dino_tagging.py's docstring)
    p.add_argument("--gdino_config", type=str,
                   default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino_checkpoint", type=str,
                   default="GroundingDINO/weights/groundingdino_swint_ogc.pth")
    p.add_argument("--gdino_box_threshold", type=float, default=0.35)
    p.add_argument("--gdino_text_threshold", type=float, default=0.25)
    return p


if __name__ == "__main__":
    run_phase3(get_phase3_parser().parse_args())
