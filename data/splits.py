"""GCD class splits for xGCD (CIFAR-10 / CIFAR-100).

Single place that fixes `args.train_classes` (known K^l) and `args.unlabeled_classes`
(novel K^n). The unlabelled training set is later built as (whole_train - labelled), so it
contains the *held-out* known-class images **and** all novel-class images — the known/novel
mixture that GCD assumes.

CIFAR-100 supports two splits via `args.num_labeled_classes`:
  * 80  -> 80/20  (standard GCD split)
  * 50  -> 50/50
"""
from loguru import logger

# dataset_name -> total number of classes
_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

# default number of labelled (known) classes
_DEFAULT_LABELED = {"cifar10": 5, "cifar100": 80}


def configure_splits(args):
    """Set args.image_size, args.train_classes, args.unlabeled_classes for the dataset."""
    name = args.dataset_name
    if name not in _NUM_CLASSES:
        raise ValueError(f"configure_splits: unsupported dataset '{name}'")

    total = _NUM_CLASSES[name]
    n_lab = getattr(args, "num_labeled_classes", None) or _DEFAULT_LABELED[name]
    if not (0 < n_lab < total):
        raise ValueError(f"num_labeled_classes={n_lab} invalid for {name} (total {total})")

    args.image_size = 224  # ViT-B/16 input; CIFAR is upscaled 32->224 by the transform
    args.num_labeled_classes = n_lab
    args.num_unlabeled_classes = total - n_lab
    args.train_classes = range(n_lab)
    args.unlabeled_classes = range(n_lab, total)

    logger.info(
        f"Split [{name}]: known K^l={n_lab} (classes 0-{n_lab-1}), "
        f"novel K^n={total - n_lab} (classes {n_lab}-{total-1})"
    )
    return args
