#!/bin/bash
set -e

echo "=== Setting up DTensor transfer test environment ==="

# Create venv
if [ ! -d /root/.venv ]; then
    echo "Creating venv..."
    uv venv /root/.venv --python python3
fi

export PATH="/root/.venv/bin:$PATH"

# Install PyTorch nightlies
echo "Installing PyTorch nightlies..."
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Clone/update tensordict
if [ ! -d /root/tensordict ]; then
    echo "Cloning tensordict..."
    cd /root
    git clone https://github.com/pytorch/tensordict.git
    cd tensordict
else
    echo "Updating tensordict..."
    cd /root/tensordict
    git fetch origin
fi

echo "Checking out dtensor-transfer branch..."
git checkout dtensor-transfer
git pull origin dtensor-transfer

# Install tensordict from source
echo "Installing tensordict from source..."
uv pip install -e /root/tensordict

echo "=== Setup complete ==="
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "TensorDict: $(python -c 'import tensordict; print(tensordict.__version__)')"
