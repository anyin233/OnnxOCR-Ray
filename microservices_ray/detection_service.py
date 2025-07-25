#!/usr/bin/env python3
"""
Text Detection Service - Independent detection service
"""

import cv2
import time
import base64
import numpy as np
from pydantic import BaseModel
from typing import List
from onnxocr.predict_det import TextDetector
from onnxocr.utils import infer_args as init_args
import argparse
from microservices_ray.common import decode_image, BoundingBox


# Define request model
class DetectionRequest(BaseModel):
    image: str  # base64 encoded image





class DetectionResponse(BaseModel):
    processing_time: float
    bounding_boxes: List[BoundingBox]

class DetectionService:
    def __init__(self):
        # Initialize detection model
        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)
        params.use_gpu = True

        self.detector = TextDetector(params)
        
    async def detect_text(self, request: DetectionRequest):
        """文本检测服务"""
        try:
            # 解码 base64 图像
            try:
                image_bytes = base64.b64decode(request.image)
                image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Failed to decode image from base64.")
            except Exception as e:
                raise RuntimeError(f"Image decoding failed: {str(e)}")

            # 执行文本检测
            start_time = time.time()
            dt_boxes = self.detector(img)
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
                processing_time=processing_time, bounding_boxes=bounding_boxes
            )
        except Exception as e:
            raise RuntimeError(f"Detection error: {str(e)}")

    async def detect_from_crops(self, request: DetectionRequest):
        return await detect_text(request)

