"""Single construction point for the xGCD model.

Every entry script (Stage 1, joint training, eval) calls `build_concept_model(args)`
so the backbone/CBL are wired identically everywhere.
"""
from loguru import logger

from models.backbone import DinoViTBackbone
from models.cbl import ConceptBottleneckLayer
from models.concept_model import ConceptModel


def build_backbone(args) -> DinoViTBackbone:
    # Default: auto-download official DINO weights (pretrain="dino").
    # Set args.pretrain_path to override with a local checkpoint.
    return DinoViTBackbone(
        arch=getattr(args, "model_arch", "vit_base"),
        patch_size=getattr(args, "patch_size", 16),
        pretrain=getattr(args, "pretrain", "dino"),
        pretrain_path=getattr(args, "pretrain_path", None),
        grad_from_block=getattr(args, "grad_from_block", 11),
        mae_checkpoint=getattr(args, "use_mae", False),
    )


def build_concept_model(args, backbone: DinoViTBackbone = None) -> ConceptModel:
    """Build (or reuse) a backbone and attach a CBL sized to `args.num_concepts`.

    `args.num_concepts` (= C = |vocabulary|) must be set by the data layer first.
    """
    if not getattr(args, "num_concepts", None):
        raise ValueError(
            "args.num_concepts is not set — build the concept vocabulary "
            "(Stage D) before constructing the model."
        )
    if backbone is None:
        backbone = build_backbone(args)

    cbl = ConceptBottleneckLayer(
        in_features=backbone.feat_dim,
        num_concepts=args.num_concepts,
        bias=True,
        num_hidden=getattr(args, "cbl_hidden_layers", 0),
    )
    model = ConceptModel(backbone, cbl)
    logger.info(
        f"Built ConceptModel: backbone feat_dim={backbone.feat_dim}, "
        f"num_concepts={args.num_concepts}"
    )
    return model
