#!/usr/bin/env bash
# Push WasteVision Docker image to Amazon ECR
#
# Usage: ./ecr-push.sh <aws-account-id> <region> [tag]
#
# Prerequisites: aws CLI configured, Docker running

set -euo pipefail

ACCOUNT_ID="${1:?Usage: $0 <account-id> <region> [tag]}"
REGION="${2:?Usage: $0 <account-id> <region> [tag]}"
TAG="${3:-latest}"

REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/wastevision"

# Authenticate Docker to ECR
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REPO}"

# Build and tag
docker build -t "wastevision:${TAG}" -f deploy/Dockerfile .
docker tag "wastevision:${TAG}" "${REPO}:${TAG}"

# Push
docker push "${REPO}:${TAG}"

echo "Pushed ${REPO}:${TAG}"
