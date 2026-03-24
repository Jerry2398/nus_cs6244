#!/bin/bash
# Download the full LIBERO benchmark datasets (~10 GB) from HuggingFace.
# Contains: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10 in RLDS format.
#
# IMPORTANT: This dataset uses HuggingFace Xet storage. If you only get ~368MB instead of ~10GB,
# your huggingface_hub is too old and downloaded LFS pointer files instead of real content.
# Fix: pip install -U 'huggingface_hub>=0.32.0'  (0.32+ includes hf_xet for Xet support)

set -e

TARGET_DIR="/scratch/yuchen.yan/open_vla/datasets"
REPO_ID="openvla/modified_libero_rlds"
LOCAL_DIR="modified_libero_rlds"
MIN_HF_HUB_VERSION="0.32.0"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Ensure huggingface_hub supports Xet storage (required for full ~10GB download)
echo "Checking huggingface_hub version (need >= ${MIN_HF_HUB_VERSION} for Xet storage)..."
python -c "
import sys
def ver_tuple(s):
    return tuple(int(x) for x in s.split('.')[:3])
try:
    import huggingface_hub as hfh
    v = hfh.__version__
    if ver_tuple(v) < ver_tuple('${MIN_HF_HUB_VERSION}'):
        print(f'ERROR: huggingface_hub {v} is too old. Xet storage requires >= ${MIN_HF_HUB_VERSION}.', file=sys.stderr)
        print('Run: pip install -U \"huggingface_hub>=${MIN_HF_HUB_VERSION}\"', file=sys.stderr)
        sys.exit(1)
    print(f'huggingface_hub {v} OK')
except ImportError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"

# Disable hf_transfer to avoid potential issues (uses standard requests)
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Downloading LIBERO benchmark datasets (~10 GB) to ${TARGET_DIR}/${LOCAL_DIR}..."
echo "This may take a while depending on your network connection."

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=\"${REPO_ID}\",
    repo_type=\"dataset\",
    local_dir=\"${LOCAL_DIR}\",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Done: ${TARGET_DIR}/${LOCAL_DIR}')
"

echo "Download complete. Dataset contains: libero_spatial_no_noops, libero_object_no_noops, libero_goal_no_noops, libero_10_no_noops"
echo "Use: --data_root_dir ${TARGET_DIR}/${LOCAL_DIR} --dataset_name libero_<suite>_no_noops"
