"""Quick check: what concepts does RAM++ see in one image, with no class labels?

Run:
    conda run -n xGCD python ram_concept_demo.py --image path/to/image.jpg

Expects the RAM++ checkpoint at recognize-anything/pretrained/ram_plus_swin_large_14m.pth
(override with --pretrained).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).parent
RAM_DIR = REPO_ROOT / "recognize-anything"
sys.path.insert(0, str(RAM_DIR))

from ram import get_transform, inference_ram as inference  # noqa: E402
from ram.models import ram_plus  # noqa: E402

DEFAULT_IMAGE = RAM_DIR / "images" / "demo" / "demo1.jpg"
DEFAULT_CKPT = RAM_DIR / "pretrained" / "ram_plus_swin_large_14m.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument(
        "--pretrained", default=str(DEFAULT_CKPT),
        help="local .pth path, or a URL to auto-download + cache on first run",
    )
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(image_size=args.image_size)

    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit="swin_l")
    model.eval().to(device)

    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    tags, _ = inference(image, model)

    print(f"Image: {args.image}")
    print("Concepts:", tags)


if __name__ == "__main__":
    main()
