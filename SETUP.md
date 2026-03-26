# Setup Instructions

## Set Up Conda Environment

```bash
# (Recommended) Put large caches/checkpoints/datasets on scratch
cd /home/svu/yuchen.yan/yuchen_workspace/openvla-oft
source ./env.sh

# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install LIBERO (simulation benchmark dependency)
# We recommend keeping third-party repos in your workspace (not inside openvla-oft).
cd /home/svu/yuchen.yan/yuchen_workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

# Install extra LIBERO eval dependencies used by this repo
cd /home/svu/yuchen.yan/yuchen_workspace/openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```