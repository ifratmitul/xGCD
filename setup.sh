#!/bin/bash
set -e

ENV_NAME="xGCD"

echo "=========================================="
echo "Setting up xGCD environment ($ENV_NAME)"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create (or update) conda environment from env.yml
echo ""
echo "Creating conda environment from env.yml..."
conda env create -f env.yml || conda env update -f env.yml --prune

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p exp data/datasets logs/wandb

# Verify installation
echo ""
echo "Verifying installation..."
conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch: {torch.__version__}')"
conda run -n $ENV_NAME python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
conda run -n $ENV_NAME python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
conda run -n $ENV_NAME python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
conda run -n $ENV_NAME python -c "import scipy; print(f'scipy: {scipy.__version__}')"
conda run -n $ENV_NAME python -c "import loguru; print('loguru: available')"
conda run -n $ENV_NAME python -c "import yacs; print('yacs: available')"
conda run -n $ENV_NAME python -c "from PIL import Image; print('Pillow: available')"
conda run -n $ENV_NAME python -c "import wandb; print(f'wandb: {wandb.__version__}')"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
conda run -n $ENV_NAME python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Pre-cache the DINO ViT-B/16 weights that models/backbone.py downloads
# (torch.hub.load_state_dict_from_url -> ~/.cache/torch/hub/checkpoints).
echo ""
echo "Pre-caching DINO ViT-B/16 weights..."
conda run -n $ENV_NAME python -c "import torch; torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', map_location='cpu'); print('DINO weights cached.')"

echo ""
echo "=========================================="
echo "Environment setup complete!  ->  conda activate $ENV_NAME"
echo "=========================================="
