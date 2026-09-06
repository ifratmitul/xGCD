"""Novel-concept discovery for phase 3 (the "N" in N-concept-Gen).

Stage 1's concept vocabulary only covers labelled known classes: build_concept_vocabulary
(data/concept_annotations.py) reads only the labelled Grounding-DINO annotations, because
Grounding DINO needs a candidate text vocabulary up front to ground against. Phase 3's DPMM
step discovers novel *clusters*, but the CBL was never trained to detect anything outside
that known vocabulary, so those clusters have no concepts describing them.

This module closes that gap once, right after the class set K is fixed (post peak-gate,
phase3.py step 2) and before the classifier head is initialised (step 3):

  1. RAM++-tag every unlabelled image that fell in a *novel* cluster — no class label and no
     candidate vocabulary needed (see N-concept-Gen.py's docstring for why RAM++ is the model
     for this, as opposed to Grounding DINO which needs the vocabulary supplied up front).
  2. Per novel cluster, keep tags seen in >= --novel_min_images_per_cluster of that cluster's
     own images -> that cluster's candidate concepts. Union across novel clusters, drop
     anything already in the vocab -> the actual new concepts being added.
  3. Grow the CBL: a new nn.Linear with the old weight rows copied over verbatim and
     `delta_c` freshly-initialised new rows appended.
  4. Extend the LDA metric (Sigma, Sigma^-1, and its Cholesky factor used by `lda.whiten`)
     block-diagonally with an identity block for the new dims: there is no covariance data
     for them yet, so they start uncorrelated with everything else, unit variance.
  5. Extend `prototypes` with one new column per new concept, valued at the logit of that
     row's own empirical concept frequency (novel cluster: frequency among its RAM-tagged
     members; known class: frequency among its labelled members, RAM-tagged the same way
     for now — see point 6) rather than a flat zero — informative rather than "neutral"
     where data exists for it.
  6. Extend the labelled-set BCE fidelity targets (`ConceptTargetLookup`) and `pos_weight`
     for the new dims — NOT forced to 0. This is a deliberate choice, not a shortcut: a
     known image CAN exhibit a concept discovered from a novel cluster (e.g. "wheels",
     discovered from a novel "truck" cluster, is obviously also true of a known "car").
     Forcing that target to 0 would teach the CBL to suppress a concept it can plainly see,
     fighting its own visual evidence on every known-class batch. Currently done by running
     RAM++ over the *labelled* set too (no candidate vocabulary needed, but also no
     localization/grounding signal). The better-fit alternative -- Grounding DINO, since
     `new_concepts` is by now a short, closed vocabulary, and it's the model already used
     everywhere else in this repo for labelled-set concept targets, e.g.
     data/annotations/*.json and Stage 1's BCE targets -- is written but commented out
     in the code until groundingdino is installed/vendored here; switch back then.

Also saves, per novel cluster, the `--cluster_images_n` member images nearest the cluster's
own (pre-discovery) prototype plus its finalized concept set, to
`--cluster_images_dir/<cluster_id>/` — a human-checkable "does this cluster + its concepts
actually make sense" snapshot.

Everything here runs ONCE, right after the class set is fixed — not every epoch. The returned
`ExtendedConceptLookup` is a drop-in replacement for the plain `ConceptTargetLookup`: the
training loop's `lookup.batch(uq)` call is unchanged either way.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from loguru import logger
from PIL import Image

# grounding_dino_tagging is imported lazily, inside discover_novel_concepts, so that importing
# THIS module (e.g. from phase3.py at startup) doesn't require the groundingdino package to be
# installed unless --novel_concepts is actually turned on.
from project_utils.ram_tagging_utils import load_ram_plus, tag_by_uq
from methods.gcd.eval_gcd import assign_to_prototypes
from models.cbl import ConceptBottleneckLayer


# --------------------------------------------------------------------------- #
# CBL growth
# --------------------------------------------------------------------------- #
def _grow_cbl(cbl: ConceptBottleneckLayer, delta_c: int, device) -> ConceptBottleneckLayer:
    """New CBL with `delta_c` extra output concepts: old rows copied, new rows fresh-init."""
    if len(cbl.model) != 1:
        raise NotImplementedError("concept discovery only supports num_hidden=0 (a single Linear)")
    old_linear = cbl.model[0]
    new_cbl = ConceptBottleneckLayer(cbl.in_features, cbl.num_concepts + delta_c,
                                     bias=old_linear.bias is not None)
    new_linear = new_cbl.model[0]
    with torch.no_grad():
        new_linear.weight[:cbl.num_concepts].copy_(old_linear.weight)
        if old_linear.bias is not None:
            new_linear.bias[:cbl.num_concepts].copy_(old_linear.bias)
    return new_cbl.to(device)


# --------------------------------------------------------------------------- #
# LDA metric growth
# --------------------------------------------------------------------------- #
def _extend_lda_block_diag(lda, delta_c: int, device):
    """Extend Sigma / Sigma^-1 / its Cholesky factor with an identity block for `delta_c`
    new, as-yet-uncorrelated, unit-variance concept dims. Mutates `lda` in place."""
    C = lda.C
    dtype = lda.cov.dtype
    eye = torch.eye(delta_c, device=device, dtype=dtype)
    zeros = torch.zeros(C, delta_c, device=device, dtype=dtype)

    def block(top_left):
        return torch.cat([torch.cat([top_left, zeros], dim=1),
                          torch.cat([zeros.T, eye], dim=1)], dim=0)

    lda.cov = block(lda.cov)
    lda.precision = block(lda.precision)
    lda.chol_L = block(lda.chol_L)
    lda.C = C + delta_c
    return lda


# --------------------------------------------------------------------------- #
# extended lookup (old Grounding-DINO dims + new RAM-tag dims)
# --------------------------------------------------------------------------- #
class ExtendedConceptLookup:
    """`base_lookup` (Grounding-DINO, C_old dims) + RAM-tag targets for the new ΔC dims,
    concatenated so `lookup.batch(uq)` returns C_old+ΔC targets — the training loop's BCE
    call doesn't need to know discovery happened."""

    def __init__(self, base_lookup, new_concepts: List[str], new_o_by_uq: Dict[int, torch.Tensor]):
        self.base = base_lookup
        self.new_concepts = new_concepts
        self.new_o_by_uq = new_o_by_uq
        self.delta_c = len(new_concepts)

    def batch(self, uq_list) -> torch.Tensor:
        base_o = self.base.batch(uq_list)
        zeros = torch.zeros(self.delta_c)
        new_o = torch.stack([self.new_o_by_uq.get(int(u), zeros) for u in uq_list])
        return torch.cat([base_o, new_o.to(base_o.dtype)], dim=1)


def _tags_to_o(tags: List[str], concepts: List[str]) -> torch.Tensor:
    present = set(tags)
    return torch.tensor([1.0 if c in present else 0.0 for c in concepts])


# --------------------------------------------------------------------------- #
# per-cluster snapshot: nearest-to-center images + finalized concept set
# --------------------------------------------------------------------------- #
def _save_cluster_snapshots(cluster_uqs: Dict[int, List[int]], candidate_by_cluster: Dict[int, set],
                            cluster_tag_counts: Dict[int, Counter], vocab_set: set,
                            novel_uq, novel_arrays, novel_logits: torch.Tensor,
                            prototypes: torch.Tensor, lda, out_dir: str, n_images: int):
    """For each novel cluster: save the `n_images` member images nearest the cluster's own
    (pre-discovery) prototype under the fitted Mahalanobis metric, plus a concepts.json of
    that cluster's finalized (post-threshold) concept set — to `out_dir/<cluster_id>/`.

    Uses the ORIGINAL geometry (unlab_logits/prototypes/lda from before the CBL/LDA growth
    below): the post-growth metric's new dims are an untrained identity-block placeholder,
    not yet informative about which images are actually "central"."""
    uq_to_pos = {int(u): i for i, u in enumerate(novel_uq.tolist())}
    for cid, member_uq in cluster_uqs.items():
        idx = [uq_to_pos[u] for u in member_uq]
        z = lda.whiten(novel_logits[idx])
        center = lda.whiten(prototypes[cid:cid + 1])
        dist = torch.cdist(z, center).squeeze(1)
        k = min(n_images, len(idx))
        nearest = [idx[i] for i in dist.topk(k, largest=False).indices.tolist()]

        cluster_dir = Path(out_dir) / str(cid)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for rank, pos in enumerate(nearest):
            uq = int(novel_uq[pos])
            img = Image.fromarray(novel_arrays[pos]).convert("RGB").resize((128, 128), Image.NEAREST)
            img.save(cluster_dir / f"{rank:02d}_uq{uq}.png")

        concepts = candidate_by_cluster.get(cid, set())
        counts = cluster_tag_counts.get(cid, Counter())
        payload = {
            "cluster_id": cid,
            "n_members": len(member_uq),
            "n_images_saved": len(nearest),
            "finalized_concepts": [
                {"concept": c, "count_in_cluster": counts[c], "new_to_vocab": c not in vocab_set}
                for c in sorted(concepts, key=lambda c: -counts[c])
            ],
        }
        with open(cluster_dir / "concepts.json", "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[concept-discovery] cluster {cid}: saved {len(nearest)} nearest-to-center "
                   f"images + {len(concepts)} finalized concepts -> {cluster_dir}")


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def discover_novel_concepts(args, model, vocab: List[str], lookup, pos_weight: torch.Tensor,
                            prototypes: torch.Tensor, lda, k_known: int,
                            unlab_logits: torch.Tensor, unlab_extract, lab_extract, device
                            ) -> Tuple[List[str], object, object, torch.Tensor, torch.Tensor, object]:
    """Returns (vocab, model, lookup, pos_weight, prototypes, lda), all possibly grown."""
    k_total = prototypes.shape[0]
    if k_total <= k_known:
        logger.info("[concept-discovery] no novel clusters survived the peak-gate; skipping")
        return vocab, model, lookup, pos_weight, prototypes, lda

    assignments = assign_to_prototypes(unlab_logits, prototypes, lda).cpu().numpy()
    novel_mask = assignments >= k_known
    n_novel_imgs = int(novel_mask.sum())
    logger.info(f"[concept-discovery] {n_novel_imgs}/{len(assignments)} unlabelled images "
               f"fell in a novel cluster (k_known={k_known}, k_total={k_total})")
    if n_novel_imgs == 0:
        logger.info("[concept-discovery] no unlabelled images landed in a novel cluster; skipping")
        return vocab, model, lookup, pos_weight, prototypes, lda

    ram_model, ram_transform = load_ram_plus(args.ram_pretrained, args.ram_image_size, device)

    # ---- 1. tag every novel-cluster image ----
    novel_arrays = unlab_extract.data[novel_mask]
    novel_uq = unlab_extract.uq_idxs[novel_mask]
    novel_cluster_ids = assignments[novel_mask]
    tags_by_uq = tag_by_uq(ram_model, ram_transform, novel_arrays, novel_uq, device,
                           batch_size=args.ram_batch_size)

    # ---- 2. per-cluster candidates -> union -> drop anything already known ----
    vocab_set = set(vocab)
    cluster_tag_counts: Dict[int, Counter] = {}
    cluster_uqs: Dict[int, List[int]] = {}
    for cid in sorted(set(novel_cluster_ids.tolist())):
        member_uq = novel_uq[novel_cluster_ids == cid].tolist()
        cluster_uqs[cid] = member_uq
        counts = Counter()
        for uq in member_uq:
            counts.update(tags_by_uq[int(uq)])
        cluster_tag_counts[cid] = counts

    min_imgs = args.novel_min_images_per_cluster
    candidate_by_cluster = {cid: {t for t, n in counts.items() if n >= min_imgs}
                            for cid, counts in cluster_tag_counts.items()}
    all_candidates = set().union(*candidate_by_cluster.values()) if candidate_by_cluster else set()
    new_concepts = sorted(all_candidates - vocab_set)
    delta_c = len(new_concepts)

    logger.info(f"[concept-discovery] {len(cluster_tag_counts)} novel clusters -> "
               f"{delta_c} new concepts (min_images_per_cluster={min_imgs}): {new_concepts}")
    for cid, counts in cluster_tag_counts.items():
        top = counts.most_common(8)
        logger.info(f"  cluster {cid} (n={len(cluster_uqs[cid])}): " +
                   ", ".join(f"{t}({n})" for t, n in top))

    # ---- snapshot: nearest-to-center images + finalized concepts, per novel cluster ----
    # Uses the PRE-growth geometry (unlab_logits/prototypes/lda as passed in) -- see
    # _save_cluster_snapshots's docstring for why.
    novel_logits = unlab_logits[torch.from_numpy(novel_mask)]
    _save_cluster_snapshots(cluster_uqs, candidate_by_cluster, cluster_tag_counts, vocab_set,
                            novel_uq, novel_arrays, novel_logits, prototypes, lda,
                            args.cluster_images_dir, args.cluster_images_n)

    if delta_c == 0:
        logger.info("[concept-discovery] nothing new to add; CBL/vocab unchanged")
        return vocab, model, lookup, pos_weight, prototypes, lda

    # ---- 3. grow the CBL ----
    model.cbl = _grow_cbl(model.cbl, delta_c, device)
    new_vocab = vocab + new_concepts
    logger.info(f"[concept-discovery] CBL grown: {len(vocab)} -> {len(new_vocab)} concepts")

    # ---- 4. extend the LDA metric ----
    _extend_lda_block_diag(lda, delta_c, device)

    # ---- 5/6. get the LABELLED set's targets for the new concept dims (recompute, not
    #           force-zero — see module docstring point 6) ----
    # Grounding DINO version (not used for now — groundingdino isn't installed/vendored here
    # yet; switch back once it's set up on the GPU machine. `new_concepts` is a short, closed
    # vocabulary at this point, which is exactly what Grounding DINO needs supplied up front,
    # and it's the model already used everywhere else in this repo for labelled-set concept
    # targets (data/annotations/*.json, Stage 1's BCE targets)):
    #   from methods.contrastive_training.grounding_dino_tagging import ground_by_uq, load_grounding_dino
    #   gdino_model = load_grounding_dino(args.gdino_config, args.gdino_checkpoint, device)
    #   lab_tags_by_uq = ground_by_uq(gdino_model, lab_extract.data, lab_extract.uq_idxs, new_concepts,
    #                                 device, box_threshold=args.gdino_box_threshold,
    #                                 text_threshold=args.gdino_text_threshold)
    #
    # RAM++ version (used for now): no candidate vocabulary needed either, but also no
    # localization/grounding signal -- good enough to get the pipeline running end to end.
    lab_tags_by_uq = tag_by_uq(ram_model, ram_transform, lab_extract.data, lab_extract.uq_idxs,
                              device, batch_size=args.ram_batch_size)
    lab_o_by_uq = {int(u): _tags_to_o(tags, new_concepts) for u, tags in lab_tags_by_uq.items()}

    new_cols = torch.zeros(k_total, delta_c, device=device)
    for cid, member_uq in cluster_uqs.items():
        freqs = torch.stack([_tags_to_o(tags_by_uq[int(u)], new_concepts) for u in member_uq]).mean(0)
        new_cols[cid] = torch.logit(freqs.clamp(0.02, 0.98)).to(device)
    for k in range(k_known):
        member_uq = [int(u) for u, lbl in zip(lab_extract.uq_idxs.tolist(), lab_extract.targets) if lbl == k]
        if member_uq:
            freqs = torch.stack([lab_o_by_uq[u] for u in member_uq]).mean(0)
            new_cols[k] = torch.logit(freqs.clamp(0.02, 0.98)).to(device)
    prototypes = torch.cat([prototypes, new_cols], dim=1)

    lab_o_all = torch.stack(list(lab_o_by_uq.values())) if lab_o_by_uq else torch.zeros(0, delta_c)
    pos = lab_o_all.sum(0)
    neg = lab_o_all.shape[0] - pos
    new_pos_weight = torch.where(pos > 0, neg / pos.clamp_min(1.0), torch.ones_like(pos))
    if getattr(args, "pos_weight_clip", 0) > 0:
        new_pos_weight = new_pos_weight.clamp(max=args.pos_weight_clip)
    pos_weight = torch.cat([pos_weight.cpu(), new_pos_weight])
    lookup = ExtendedConceptLookup(lookup, new_concepts, lab_o_by_uq)

    known_hit_frac = float((lab_o_all.sum(1) > 0).float().mean()) if len(lab_o_all) else 0.0
    logger.info(f"[concept-discovery] {known_hit_frac*100:.1f}% of labelled/known images show "
               f">=1 of the {delta_c} newly discovered concepts (RAM-recomputed, not forced to 0)")
    for c, n in zip(new_concepts, pos.tolist()):
        logger.info(f"  new concept '{c}': positive in {int(n)}/{len(lab_o_all)} labelled images "
                   f"(pos_weight={float(new_pos_weight[new_concepts.index(c)]):.1f})")

    return new_vocab, model, lookup, pos_weight, prototypes, lda


# --------------------------------------------------------------------------- #
# training-time diagnostics (the "is this doing something sensible?" checks)
# --------------------------------------------------------------------------- #
class ConceptDriftTracker:
    """Prints, at each call: (a) whether known/labelled samples activate on the newly
    discovered concepts, and (b) how each novel cluster's mean concept activation compares
    to its discovery-time prototype. Call once right after discovery (epoch 0 baseline) and
    again at each refresh — the same cadence the pseudo-label refresh already uses."""

    def __init__(self, k_known: int, k_total: int, num_old_concepts: int, vocab: List[str],
                initial_prototypes: torch.Tensor):
        self.k_known = k_known
        self.k_total = k_total
        self.c_old = num_old_concepts
        self.vocab = vocab
        self.initial_sigmoid = torch.sigmoid(initial_prototypes).cpu()   # [k_total, C] snapshot at discovery
        self.prev_cluster_mean = {}                                      # cid -> [C] running previous mean

    @torch.no_grad()
    def log(self, model, lab_loader, unlab_loader, prototypes, lda, device, tag: str):
        if self.k_total <= self.k_known or self.initial_sigmoid.shape[1] <= self.c_old:
            return  # no novel concepts were ever discovered this run
        from methods.contrastive_training.extract import extract_concept_logits

        model.eval()
        # (a) do KNOWN samples fire on any newly discovered concept?
        lab_ell, _, _, _ = extract_concept_logits(model, lab_loader, device)
        lab_new_act = torch.sigmoid(lab_ell[:, self.c_old:]).cpu()
        lab_hit_frac = float((lab_new_act > 0.5).any(dim=1).float().mean())
        logger.info(f"[concept-drift {tag}] known/labelled images firing on >=1 novel concept: "
                   f"{lab_hit_frac*100:.1f}% | mean activation={float(lab_new_act.mean()):.3f}")

        # (b) per novel cluster: current mean activation vs its discovery-time prototype
        unlab_ell, _, _, _ = extract_concept_logits(model, unlab_loader, device)
        unlab_ell = unlab_ell.to(device)
        cluster_ids = assign_to_prototypes(unlab_ell, prototypes, lda).cpu()
        cur_sigmoid = torch.sigmoid(unlab_ell).cpu()
        for cid in range(self.k_known, self.k_total):
            member = cluster_ids == cid
            n = int(member.sum())
            if n == 0:
                continue
            cur_mean = cur_sigmoid[member].mean(0)                       # [C]
            drift = float((cur_mean - self.initial_sigmoid[cid]).abs().mean())
            prev = self.prev_cluster_mean.get(cid)
            step = float((cur_mean - prev).abs().mean()) if prev is not None else float("nan")
            self.prev_cluster_mean[cid] = cur_mean
            top_new = cur_mean[self.c_old:].topk(min(3, cur_mean.shape[0] - self.c_old))
            top_new_str = ", ".join(f"{self.vocab[self.c_old + i]}:{v:.2f}"
                                    for v, i in zip(top_new.values.tolist(), top_new.indices.tolist()))
            logger.info(f"[concept-drift {tag}] cluster {cid} (n={n}): "
                       f"drift-from-init={drift:.3f} step-since-last={step:.3f} | top new concepts: {top_new_str}")
