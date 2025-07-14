#!/bin/bash

# Microservices OCR system startup script

echo "Starting microservices OCR system..."
echo "=============================="

# Check Python environment
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found"
    exit 1
fi

# Create log directory
mkdir -p logs

export PATH=$PATH:.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:.venv/lib/python3.10/site-packages/nvidia/cudnn/lib

# Start detection service
echo "Starting text detection service (port 5006)..."
uv run service_entry_point.py -s detection -p 5006 > logs/detection.log 2>&1 &
DETECTION_PID=$!
echo "Detection service PID: $DETECTION_PID"

# Wait for service to start
sleep 2

# Start recognition service
echo "Starting text recognition service (port 5007)..."
uv run service_entry_point.py -s recognition -p 5007 > logs/recognition.log 2>&1 &
RECOGNITION_PID=$!
echo "Recognition service PID: $RECOGNITION_PID"

# Wait for service to start
sleep 2

# Start classification service
echo "Starting text classification service (port 5008)..."
uv run service_entry_point.py -s classification -p 5008 > logs/classification.log 2>&1 &
CLASSIFICATION_PID=$!
echo "Classification service PID: $CLASSIFICATION_PID"

# Wait for service to start
sleep 2

# Start orchestrator service
echo "Starting OCR orchestrator service (port 5009)..."
uv run microservices/orchestrator_service.py > logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "Orchestrator service PID: $ORCHESTRATOR_PID"

# Save PIDs to files
echo $DETECTION_PID > logs/detection.pid
echo $RECOGNITION_PID > logs/recognition.pid
echo $CLASSIFICATION_PID > logs/classification.pid
echo $ORCHESTRATOR_PID > logs/orchestrator.pid

echo ""
echo "All services started successfully!"
echo "=============================="
echo "Service ports:"
echo "  Detection service:   http://localhost:5006"
echo "  Recognition service: http://localhost:5007"
echo "  Classification service: http://localhost:5008"
echo "  Orchestrator service: http://localhost:5009"
echo ""
echo "Log files location: logs/"
echo "To stop services run: ./stop_services.sh"
echo ""

# Wait a moment for services to fully start
echo "Waiting for services to start..."
sleep 5

# Check if services started properly
echo "Checking service status..."
echo "=============================="

check_service() {
    local service_name=$1
    local port=$2
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/ --connect-timeout 3)
    
    if [ "$response" = "200" ]; then
        echo "✓ $service_name: Running normally"
    else
        echo "✗ $service_name: Failed to start or abnormal response"
    fi
}

check_service "Detection service" 5006
check_service "Recognition service" 5007
check_service "Classification service" 5008
check_service "Orchestrator service" 5009

echo ""
echo "Now you can run tests: uv run test_microservices.py"
