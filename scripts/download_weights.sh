#!/bin/bash
# Download model weights into weights/ (not tracked in git).
#
#   weights/classifier/      clutter-level classifier (best_model.pth, config.json)   [this repo's release]
#   weights/ppo_policies/    per-object PPO policies for the Contactile hand          [this repo's release]
#   weights/ggcnn/           GG-CNN trained on Cornell (Panda baseline grasp pose)    [dougsm/ggcnn release]
#   weights/sam/             Segment Anything ViT-B (Panda baseline target masks)     [Meta AI]
#
# Usage:  bash scripts/download_weights.sh [classifier] [ppo] [ggcnn] [sam]   (default: all)

set -euo pipefail

RELEASE_URL="https://github.com/JinW25/ObjectToScenes_EvaluateGraspInCluttered/releases/download/v1.0"
GGCNN_URL="https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/weights"
mkdir -p "${WEIGHTS_DIR}"
cd "${WEIGHTS_DIR}"

fetch() {  # fetch <url> <output file>
    echo "Downloading $(basename "$2") ..."
    if command -v wget >/dev/null; then wget -q --show-progress -O "$2" "$1"; else curl -L --fail -o "$2" "$1"; fi
}

want() { [ $# -eq 0 ] || [[ " ${ARGS[*]} " == *" $1 "* ]]; }
ARGS=("$@")

if want classifier "$@"; then
    fetch "${RELEASE_URL}/classifier.zip" classifier.zip && unzip -q -o classifier.zip && rm classifier.zip
fi
if want ppo "$@"; then
    fetch "${RELEASE_URL}/ppo_policies.zip" ppo_policies.zip && unzip -q -o ppo_policies.zip && rm ppo_policies.zip
fi
if want ggcnn "$@"; then
    mkdir -p ggcnn
    fetch "${GGCNN_URL}" ggcnn/ggcnn_weights_cornell.zip
    unzip -q -o -j ggcnn/ggcnn_weights_cornell.zip "*statedict.pt" -d ggcnn && rm ggcnn/ggcnn_weights_cornell.zip
fi
if want sam "$@"; then
    mkdir -p sam
    fetch "${SAM_URL}" sam/sam_vit_b_01ec64.pth
fi

echo ""
echo "Weights in ${WEIGHTS_DIR}:"
find . -maxdepth 2 -mindepth 1 | sort | head -40
