# Novel-Concept Discovery for CBL-based GCD

This branch (`main-novel-concepts`) extends xGCD's Concept Bottleneck Layer (CBL) so it can
describe **novel, undiscovered classes** with new concepts, not just the known classes it was
trained on in Stage 1.

Stage 1 CBL concepts come from Grounding DINO annotations of the *labelled known classes* —
by construction, that vocabulary has no reason to contain words that describe a novel class
(e.g. "truck", "gecko"). This means a novel cluster's LDA prototype and its
`explain_prototype` output are stuck describing it in terms of known-class concepts, even
when none of them really apply. This feature closes that gap: after novel clusters are
discovered (Phase 2 of xGCD), we use a vision-language model to look at the novel images
themselves, propose new concept words with no class label required, clean and verify them,
grow the CBL to output them, and train the new dimensions carefully so they don't disturb the
already-correct known-class behavior.

## Pipeline

![Novel-concept discovery and training pipeline](novel_concept_pipeline.svg)

**Phase 1 — Concept Discovery.** Starting from the clusters produced by phase 3's DPMM +
peak-gate (already fixed at this point — no DPMM is re-run), we take the images closest to
each novel cluster's center and tag them with **RAM++** (Recognize Anything Model++), an
open-vocabulary image-tagging VLM. Unlike Grounding DINO, RAM++ needs no candidate vocabulary
and no class label — it proposes concept words directly from the image. Per-cluster tag
frequency is aggregated into a candidate list.

**Phase 2 — Verification & Expansion.** Candidate concepts that are near-universal in an
*existing known class* (e.g. "blue", "mammal") are dropped — they don't help distinguish
anything and only add noise to the CBL. The surviving concepts are then checked against the
*full* image set (known + novel) with a grounding/tagging model to build the expanded
concept-presence matrix `O = [O_known ; O_novel]`. The CBL's final linear layer is grown from
`M_known` to `M_known + M_novel` output dimensions to make room for them.

**Phase 3 — Expanded CBL Training.** This is the part that needs the most care, because a
naive joint retrain on the enlarged concept space measurably hurt known-class accuracy in
early experiments (Old-ACC collapsing at epoch 0). The final design uses three mechanisms
together:

1. **Warm-up.** Before the classifier head is (re-)initialized from LDA prototypes, the CBL
   linear layer is trained for a few epochs on the expanded concept space so its output
   for the *new* dimensions is no longer an untrained, essentially-random guess. Skipping
   this caused the prototype for new dimensions to mismatch the CBL's actual (untrained)
   output, which is what caused the Old-ACC collapse.
2. **Freeze the original CBL rows.** The pre-growth rows of the CBL weight/bias are
   snapshotted once and restored after every optimizer step (`FrozenCBLRows`, in
   [`freeze_old_concepts_cbl_utils.py`](project_utils/freeze_old_concepts_cbl_utils.py)).
   This is not gradient-zeroing — SGD's weight decay still nudges a zero-gradient row — so
   restoring the snapshot after each step is what actually keeps known-concept detection
   byte-for-byte identical to Stage 1 throughout Phase 3, regardless of what the optimizer
   does internally.
3. **Novel-only training for new concepts.** The new CBL dimensions are trained only on
   images from novel clusters, never on labelled known-class images. Loss is computed with
   `combined_fidelity_bce(..., novel_only=True)`. The positive/negative class balance
   (`pos_weight` in the BCE loss) for new concepts is computed **from the novel population
   only**, not mixed with labelled images — a concept's weight is the inverse of how often it
   is actually present among the novel images that will supply its training signal.

Together, these three mean the new concepts are learned entirely from what the novel images
actually look like, and the known-class CBL rows — and therefore known-class classification —
are provably unaffected by any of this.

## Key files

| File | Purpose |
|---|---|
| [`project_utils/concept_discovery_utils.py`](project_utils/concept_discovery_utils.py) | RAM++ tagging, candidate filtering, CBL growth, LDA extension, `pos_weight` computation, `combined_fidelity_bce` |
| [`project_utils/freeze_old_concepts_cbl_utils.py`](project_utils/freeze_old_concepts_cbl_utils.py) | `FrozenCBLRows` — snapshot/restore of the original CBL rows |
| [`project_utils/ram_tagging_utils.py`](project_utils/ram_tagging_utils.py) | RAM++ model loading and batched image tagging |
| [`project_utils/grounding_dino_tagging_utils.py`](project_utils/grounding_dino_tagging_utils.py) | Grounding DINO tagging (written, not currently wired in — candidate for the verification step) |
| [`methods/contrastive_training/phase3.py`](methods/contrastive_training/phase3.py) | Orchestrates discovery + warm-up + freezing + training inside the existing Phase 3 loop |

## Relevant CLI args

All new behavior is opt-in — everything defaults to off/0 and reproduces the original Phase 3
training exactly unless explicitly enabled.

| Arg | Default | Effect |
|---|---|---|
| `--novel_concepts` | `False` | Master switch for the whole feature |
| `--novel_min_images_per_cluster` | `2` | Minimum images tagged per novel cluster before it contributes candidate concepts |
| `--novel_drop_known_universal_thresh` | `0.0` | Drop a candidate concept if its frequency in any known class is >= this threshold |
| `--novel_cbl_warmup_epochs` | `0` | Epochs of CBL-only warm-up before prototype/LDA refit and head init |
| `--novel_cbl_warmup_lr` | `1e-3` | Learning rate for the warm-up |
| `--novel_bce_include_unlabelled` | `False` | Include novel-cluster images in the main BCE loss (not just labelled images) |
| `--novel_freeze_old_cbl` | `False` | Enable `FrozenCBLRows` + novel-only training for new concept dims |
| `--novel_pos_weight_from_novel` | `False` | Compute new concepts' `pos_weight` from the novel population instead of the labelled population |
| `--ram_pretrained`, `--ram_image_size`, `--ram_batch_size` | — | RAM++ model/loading config |
| `--cluster_images_dir`, `--cluster_images_n` | — / `20` | Where to save representative per-cluster images + their finalized concepts for inspection |

## Getting started

```bash
python methods/contrastive_training/phase3.py \
    --novel_concepts \
    --novel_min_images_per_cluster 200 \
    --novel_drop_known_universal_thresh 0.5 \
    --novel_cbl_warmup_epochs 10 \
    --novel_bce_include_unlabelled \
    --novel_freeze_old_cbl \
    --novel_pos_weight_from_novel \
    <...other existing phase3.py args...>
```

See [`novel_concept_experiments.txt`](novel_concept_experiments.txt) for ablation results on
these flags (CIFAR-10 GCD split: 5 known / 5 novel classes).
