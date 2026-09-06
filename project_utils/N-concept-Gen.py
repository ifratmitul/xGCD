"""N-concept-Gen — RAM++ concept-bag generation for images with no class label.

Stage 1's concept vocabulary (data/concept_annotations.py::build_concept_vocabulary) is built
only from the *labelled* known-class images, because Grounding DINO needs a candidate text
vocabulary up front to ground against. That leaves the novel-class ("N") pool uncovered: we
don't have labels for it, so there's no obvious vocabulary to hand Grounding DINO.

RAM++ closes that gap because it needs neither a class label nor a candidate vocabulary — it
tags any image directly from its own open ~4.5k-tag space. This script runs RAM++ over an
unlabelled/novel-class split and saves, per image, the concept bag it sees. The union of all
bags is the candidate vocabulary to hand to Grounding DINO next, the same role the labelled
set's vocabulary plays for data/annotations/ today.

(concept_discovery.py runs the same RAM++ tagging, scoped per DPMM cluster, live inside
phase 3 — this script is for offline exploration/sanity-checking on a whole split.)

Run:
    python methods/contrastive_training/N-concept-Gen.py --dataset_name cifar10
    python methods/contrastive_training/N-concept-Gen.py --dataset_name cifar10 --split test --limit 50
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # this file can't use `python -m` (hyphen in the name)

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets  # noqa: E402
from data.splits import configure_splits  # noqa: E402
from project_utils.ram_tagging_utils import (  # noqa: E402
    DEFAULT_RAM_PRETRAINED, load_ram_plus, tag_by_uq)


def _get_datasets(args):
    if args.dataset_name == "cifar10":
        return get_cifar_10_datasets(train_transform=None, test_transform=None,
                                     train_classes=args.train_classes,
                                     prop_train_labels=args.prop_train_labels)
    elif args.dataset_name == "cifar100":
        return get_cifar_100_datasets(train_transform=None, test_transform=None,
                                      train_classes=args.train_classes,
                                      prop_train_labels=args.prop_train_labels)
    raise ValueError(f"N-concept-Gen supports cifar10/cifar100, got {args.dataset_name}")


def generate_concept_bags(args):
    configure_splits(args)
    datasets = _get_datasets(args)
    cifar_dataset = datasets.get(args.split)
    if cifar_dataset is None:
        raise ValueError(f"Split '{args.split}' unavailable for {args.dataset_name} "
                          f"(have: {[k for k, v in datasets.items() if v is not None]})")
    logger.info(f"{args.dataset_name}/{args.split}: {len(cifar_dataset)} images "
               f"(no class labels used from here on)")

    data, uq_idxs = cifar_dataset.data, cifar_dataset.uq_idxs
    if args.limit:
        data, uq_idxs = data[:args.limit], uq_idxs[:args.limit]

    model, transform = load_ram_plus(args.pretrained, args.ram_image_size)
    tags_by_uq = tag_by_uq(model, transform, data, uq_idxs, next(model.parameters()).device,
                           batch_size=args.batch_size, num_workers=args.num_workers)
    tag_counts = Counter(t for tags in tags_by_uq.values() for t in tags)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags_path = out_dir / f"{args.dataset_name}_{args.split}_ram_tags.json"
    vocab_path = out_dir / f"{args.dataset_name}_{args.split}_ram_vocab.json"

    with open(tags_path, "w") as f:
        json.dump({str(uq): tags for uq, tags in tags_by_uq.items()}, f, indent=2)

    vocab = sorted(t for t, n in tag_counts.items() if n >= args.min_images)
    with open(vocab_path, "w") as f:
        json.dump({
            "vocab": vocab,
            "meta": {"dataset": args.dataset_name, "split": args.split,
                     "n_images": len(tags_by_uq), "min_images": args.min_images},
        }, f, indent=2)

    logger.info(f"Saved per-image concept bags -> {tags_path}")
    logger.info(f"Saved candidate vocab ({len(vocab)} concepts, min_images={args.min_images}) -> {vocab_path}")
    logger.info("Top 20 concepts: " + ", ".join(f"{t}({n})" for t, n in tag_counts.most_common(20)))
    return tags_by_uq, vocab


def get_parser():
    p = argparse.ArgumentParser(description="RAM++ concept-bag generation for label-free images")
    p.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--num_labeled_classes", type=int, default=None,
                   help="known classes: cifar10=5; cifar100=80 (80/20) or 50 (50/50)")
    p.add_argument("--prop_train_labels", type=float, default=0.5)
    p.add_argument("--split", type=str, default="train_unlabelled",
                   choices=["train_labelled", "train_unlabelled", "test"],
                   help="train_unlabelled = the known+novel pool with no usable class label")
    p.add_argument("--pretrained", type=str, default=DEFAULT_RAM_PRETRAINED)
    p.add_argument("--ram_image_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_images", type=int, default=1,
                   help="drop tags seen in fewer than this many images from the saved vocab")
    p.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all); for a quick check")
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "data" / "concept_bags"))
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate_concept_bags(args)
