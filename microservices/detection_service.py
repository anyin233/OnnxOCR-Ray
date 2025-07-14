#!/usr/bin/env python3
"""
Text Detection Service - Independent detection service
"""

import cv2
import time
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from onnxocr.predict_det import TextDetector
from onnxocr.utils import infer_args as init_args
import argparse

# Initialize FastAPI application
app = FastAPI(title="TextDetectionService", description="Text Detection Service using ONNX")

# Initialize detection model
parser = init_args()
inference_args_dict = {}
for action in parser._actions:
    inference_args_dict[action.dest] = action.default
params = argparse.Namespace(**inference_args_dict)
params.use_gpu = True

detector = TextDetector(params)

# Define request model
class DetectionRequest(BaseModel):
    image: str  # base64 encoded image

# Define response model
class BoundingBox(BaseModel):
    coordinates: List[List[float]]  # 4 points coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

class DetectionResponse(BaseModel):
    processing_time: float
    bounding_boxes: List[BoundingBox]

@app.get("/")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "text_detection"}

@app.post("/inference", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest):
    """文本检测服务"""
    try:
        # 解码 base64 图像
        try:
            image_bytes = base64.b64decode(request.image)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to decode image from base64.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

        # 执行文本检测
        start_time = time.time()
        dt_boxes = detector(img)
        end_time = time.time()
        processing_time = end_time - start_time

        # 格式化结果
        bounding_boxes = []
        if dt_boxes is not None:
            for box in dt_boxes:
                # 确保坐标是4个点的格式
                if isinstance(box, (list, np.ndarray)):
                    coordinates = np.array(box).reshape(4, 2).tolist()
                    bounding_boxes.append(BoundingBox(coordinates=coordinates))

        return DetectionResponse(
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/inference_no_crop", response_model=DetectionResponse)
async def detect_from_crops(request: DetectionRequest):
    return await detect_text(request)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)
