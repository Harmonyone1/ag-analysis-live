#!/bin/bash
# AG Analyzer - Start Script

set -e

echo "==================================="
echo "  AG Analyzer - Starting Services  "
echo "==================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check for .env file
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your TradeLocker credentials before running again."
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models

# Pull latest images and build
echo ""
echo "Building Docker images..."
docker-compose build

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 5

# Check service status
echo ""
echo "Service Status:"
docker-compose ps

# Show logs
echo ""
echo "==================================="
echo "  Services Started Successfully!   "
echo "==================================="
echo ""
echo "Access points:"
echo "  - UI:      http://localhost:3000"
echo "  - API:     http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
