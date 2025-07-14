#!/usr/bin/env python3
"""
FastAPI服务启动脚本
使用uvicorn启动FastAPI应用
"""

import uvicorn

if __name__ == "__main__":
    # 启动FastAPI应用
    uvicorn.run(
        "app-service:app",  # 应用模块:应用实例
        host="0.0.0.0",
        port=5005,
        reload=True,  # 开发模式下自动重载
        log_level="info",
    )
