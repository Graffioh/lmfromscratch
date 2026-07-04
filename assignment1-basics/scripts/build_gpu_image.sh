#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/build_gpu_image.sh DOCKER_IMAGE_TAG"
  echo "Example: bash scripts/build_gpu_image.sh your-dockerhub-user/cs336-gpu:cuda13"
  exit 2
fi

IMAGE_TAG="$1"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BUILDER="${DOCKER_BUILDER:-cs336-gpu-builder}"

cd "$(dirname "$0")/.."

if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER}" --driver docker-container --use >/dev/null
else
  docker buildx use "${BUILDER}"
fi

docker buildx inspect --bootstrap >/dev/null
docker buildx build --platform "${PLATFORM}" -f docker/Dockerfile.gpu -t "${IMAGE_TAG}" --push .

echo
echo "Built and pushed ${IMAGE_TAG} for ${PLATFORM} with builder ${BUILDER}"
