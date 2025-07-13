#!/bin/bash

for i in 2 5 10 20; do
  
    echo "Sending all request within $i s"

    docker restart ocr_onnx

    echo "Waiting for the container to be ready"
    sleep 60

    echo "Starting profiling for ocr_onnx_$i"
    curl -X POST http://localhost:8091/profiling/start \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"ocr_onnx_full_$i\"}"

    uv run test.py -t $i -w 50

    echo "Stopping profiling for ocr_onnx_$i"
    curl -X POST http://localhost:8091/profiling/stop \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"ocr_onnx_full_$i\"}"

    mv timestamp.csv "timestamp_$i.csv"
done