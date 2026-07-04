#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/build_gpu_image.sh DOCKER_IMAGE_TAG"
  echo "Example: bash scripts/build_gpu_image.sh your-dockerhub-user/cs336-gpu:cuda13"
  exit 2
fi

IMAGE_TAG="$1"
# GPU cloud machines are x86, so cross-compile for linux/amd64 even when
# building on an ARM Mac. Without this the image silently builds for arm64
# and fails to start on the remote host.
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BUILDER="${DOCKER_BUILDER:-cs336-gpu-builder}"

# Run from the repo root so the build context (the trailing ".") is the repo.
cd "$(dirname "$0")/.."

# Cross-platform builds need a buildx builder (docker-container driver).
# Create it once, reuse it on later runs.
if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER}" --driver docker-container --use >/dev/null
else
  docker buildx use "${BUILDER}"
fi

# --bootstrap starts the builder container now instead of lazily mid-build.
docker buildx inspect --bootstrap >/dev/null
# --push uploads straight to the registry; cross-built images cannot be
# loaded into the local docker daemon anyway.
docker buildx build --platform "${PLATFORM}" -f docker/Dockerfile.gpu -t "${IMAGE_TAG}" --push .

echo
echo "Built and pushed ${IMAGE_TAG} for ${PLATFORM} with builder ${BUILDER}"
