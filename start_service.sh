#!/bin/bash

# OCR服务启动脚本 (串行版本 - 无Ray依赖)

echo "Starting Serial OCR Service..."
echo "Service will be available at: http://localhost:5005"
echo ""
echo "✨ This version uses serial execution (no Ray dependency)"
echo ""
echo "Available endpoints:"
echo "  POST /detection      - Text detection only"
echo "  POST /classification - Text angle classification"
echo "  POST /recognition    - Text recognition only"
echo "  POST /inference      - Complete OCR pipeline"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# 启动服务
python app-service.py
