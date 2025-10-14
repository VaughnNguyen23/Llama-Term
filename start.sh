#!/bin/bash

# Quick start script for Ollama TUI

echo "Starting Ollama Docker container..."
docker-compose up -d

echo "Waiting for Ollama to be ready..."
sleep 5

# Check if container is running
if ! docker ps | grep -q ollama; then
    echo "Error: Ollama container failed to start"
    exit 1
fi

echo "Ollama is running!"
echo ""
echo "Checking for models..."
if ! docker exec ollama ollama list | grep -q llama2; then
    echo "No models found. Would you like to pull llama2:latest? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Pulling llama2:latest (this may take a while)..."
        docker exec -it ollama ollama pull llama2:latest
    else
        echo "You can download models using F3 in the TUI or run:"
        echo "  docker exec -it ollama ollama pull <model-name>"
    fi
fi

echo ""
echo "Starting TUI application..."
cargo run --release
