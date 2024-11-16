#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export OPENAI_API_KEY=$(cat ${SECRETS_DIR}/API_KEY.txt)
export IMAGE_NAME="llm-rag-cli"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports -e OPENAI_API_KEY="$OPENAI_API_KEY" $IMAGE_NAME
