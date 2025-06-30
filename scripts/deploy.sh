#!/bin/bash

# Simple Docker deployment script for LLaMA2 Inference Service

set -e

echo "🚀 Building and starting LLaMA2 Inference Service"

# Configuration
IMAGE_NAME="llama2-inference"
CONTAINER_NAME="llama2-service"
PORT="8000"

# Build image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Stop existing container
echo "🛑 Stopping existing container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Start new container
echo "🚀 Starting container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  -p ${PORT}:${PORT} \
  ${IMAGE_NAME}

echo "⏳ Waiting for service..."
sleep 10

# Health check
if curl -s http://localhost:${PORT}/health > /dev/null; then
  echo "✅ Service is running at http://localhost:${PORT}"
else
  echo "❌ Service failed to start"
  docker logs ${CONTAINER_NAME}
  exit 1
fi
