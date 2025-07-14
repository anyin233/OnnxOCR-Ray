#!/usr/bin/env python3
"""
Text Classification Service - Independent classification service (angle classification)
"""

import cv2
import time
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from onnxocr.predict_cls import TextClassifier
from onnxocr.utils import infer_args as init_args
import argparse
from microservices.common import decode_image, encode_image, crop_image_from_box

# Initialize FastAPI application
app = FastAPI(
    title="TextClassificationService",
    description="Text Classification Service using ONNX",
)

# Initialize classification model
parser = init_args()
inference_args_dict = {}
for action in parser._actions:
    inference_args_dict[action.dest] = action.default
params = argparse.Namespace(**inference_args_dict)
params.use_gpu = True

classifier = TextClassifier(params)


# Define request models
class BoundingBox(BaseModel):
    coordinates: List[
        List[float]
    ]  # 4 points coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


class ClassificationRequest(BaseModel):
    image: str  # base64 encoded original image
    bounding_boxes: List[BoundingBox]  # List of detected bounding boxes


class LegacyClassificationRequest(BaseModel):
    images: List[str]  # base64 encoded image list (backward compatibility)


class SingleImageRequest(BaseModel):
    image: str  # base64 encoded single image


# Define response models
class ClassificationResult(BaseModel):
    angle: int  # angle (0, 90, 180, 270)
    confidence: float
    rotated_image: Optional[str] = None  # rotated image (base64 encoded)
    bounding_box: Optional[List[List[float]]] = None  # corresponding bounding box


class ClassificationResponse(BaseModel):
    processing_time: float
    results: List[ClassificationResult]


@app.get("/")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "text_classification"}


@app.post("/inference", response_model=ClassificationResponse)
async def classify_text_angle(request: ClassificationRequest):
    """Text angle classification service - based on detection bounding boxes"""
    try:
        # Decode original image
        img = decode_image(request.image)

        # Crop images based on bounding boxes
        img_crop_list = []
        valid_boxes = []

        for bbox in request.bounding_boxes:
            img_crop = crop_image_from_box(img, bbox.coordinates)
            if img_crop is not None:
                img_crop_list.append(img_crop)
                valid_boxes.append(bbox.coordinates)

        if not img_crop_list:
            return ClassificationResponse(processing_time=0.0, results=[])

        # Execute text angle classification
        start_time = time.time()
        rotated_imgs, cls_res = classifier(img_crop_list)
        end_time = time.time()
        processing_time = end_time - start_time

        # Format results
        results = []
        for i, (rotated_img, cls_result, bbox) in enumerate(
            zip(rotated_imgs, cls_res, valid_boxes)
        ):
            if isinstance(cls_result, (list, tuple)) and len(cls_result) >= 2:
                angle = int(cls_result[0])  # angle
                confidence = float(cls_result[1])  # confidence
            else:
                angle = 0
                confidence = 1.0

            # Encode rotated image
            rotated_image_base64 = encode_image(rotated_img)

            results.append(
                ClassificationResult(
                    angle=angle,
                    confidence=confidence,
                    rotated_image=rotated_image_base64,
                    bounding_box=bbox,
                )
            )

        return ClassificationResponse(processing_time=processing_time, results=results)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/inference_single", response_model=ClassificationResult)
async def classify_single_text_angle(request: SingleImageRequest):
    """Text angle classification service - single image"""
    try:
        # Decode image
        img = decode_image(request.image)

        # Execute text angle classification
        start_time = time.time()
        rotated_imgs, cls_res = classifier([img])
        end_time = time.time()

        # Format results
        if rotated_imgs and cls_res and len(rotated_imgs) > 0 and len(cls_res) > 0:
            rotated_img = rotated_imgs[0]
            cls_result = cls_res[0]

            if isinstance(cls_result, (list, tuple)) and len(cls_result) >= 2:
                angle = int(cls_result[0])
                confidence = float(cls_result[1])
            else:
                angle = 0
                confidence = 1.0

            # Encode rotated image
            rotated_image_base64 = encode_image(rotated_img)
        else:
            angle = 0
            confidence = 0.0
            rotated_image_base64 = request.image

        return ClassificationResult(
            angle=angle,
            confidence=confidence,
            rotated_image=rotated_image_base64,
            bounding_box=None,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Start classification service on port 5008
    uvicorn.run(app, host="0.0.0.0", port=5008)
