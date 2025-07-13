#!/bin/bash

# Microservices OCR system stop script

echo "Stopping microservices OCR system..."
echo "=============================="

# Function to stop services
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            sleep 1
            
            # If still running, force kill
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Force stopping $service_name..."
                kill -9 "$pid"
            fi
        else
            echo "$service_name already stopped"
        fi
        rm -f "$pid_file"
    else
        echo "$service_name PID file does not exist"
    fi
}

# Stop all services
stop_service "Orchestrator service" "logs/orchestrator.pid"
stop_service "Classification service" "logs/classification.pid"
stop_service "Recognition service" "logs/recognition.pid"
stop_service "Detection service" "logs/detection.pid"

# Additionally ensure killing remaining processes by port
echo ""
echo "Checking for remaining processes..."

# Find and kill processes occupying specified ports
kill_port() {
    local port=$1
    local service_name=$2
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo "Found remaining $service_name process (PID: $pid), stopping..."
        kill -9 "$pid" 2>/dev/null
    fi
}

kill_port 5006 "Detection service"
kill_port 5007 "Recognition service"
kill_port 5008 "Classification service"
kill_port 5009 "Orchestrator service"

echo ""
echo "All services stopped!"
echo "=============================="
