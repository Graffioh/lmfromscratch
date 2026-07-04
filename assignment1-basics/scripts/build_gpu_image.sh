#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/build_gpu_image.sh DOCKER_IMAGE_TAG"
  echo "Example: bash scripts/build_gpu_image.sh your-dockerhub-user/cs336-gpu:cuda13"
  exit 2
fi

IMAGE_TAG="$1"

cd "$(dirname "$0")/.."

docker build -f docker/Dockerfile.gpu -t "${IMAGE_TAG}" .

echo
echo "Built ${IMAGE_TAG}"
echo "Push it with:"
echo "  docker push ${IMAGE_TAG}"
