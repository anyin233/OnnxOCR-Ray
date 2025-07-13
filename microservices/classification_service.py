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

# Initialize FastAPI application
app = FastAPI(title="TextClassificationService", description="Text Classification Service using ONNX")

# Initialize classification model
parser = init_args()
inference_args_dict = {}
for action in parser._actions:
    inference_args_dict[action.dest] = action.default
params = argparse.Namespace(**inference_args_dict)
params.use_gpu = False

classifier = TextClassifier(params)

# Define request models
class BoundingBox(BaseModel):
    coordinates: List[List[float]]  # 4 points coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

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

def decode_image(image_base64: str):
    """Decode base64 image"""
    try:
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64")
        return img
    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")

def encode_image(img):
    """Encode image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        raise ValueError(f"Image encoding failed: {str(e)}")

def crop_image_from_box(img, box):
    """Crop image based on bounding box"""
    try:
        # Import necessary functions
        import copy
        from onnxocr.utils import get_rotate_crop_image
        
        # Ensure box is in correct format: 4 points, each point has 2 coordinates
        if len(box) != 4:
            print(f"Invalid bounding box: expected 4 points, got {len(box)}")
            return None
        
        # Convert to numpy array, ensure it's float32 type with shape (4, 2)
        box_array = np.array(box, dtype=np.float32)
        if box_array.shape != (4, 2):
            print(f"Invalid bounding box shape: expected (4, 2), got {box_array.shape}")
            return None
        
        # Use quadrilateral cropping
        img_crop = get_rotate_crop_image(img, box_array)
        return img_crop
    except Exception as e:
        print(f"Crop error: {e}")
        return None

@app.post("/classify", response_model=ClassificationResponse)
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
        for i, (rotated_img, cls_result, bbox) in enumerate(zip(rotated_imgs, cls_res, valid_boxes)):
            if isinstance(cls_result, (list, tuple)) and len(cls_result) >= 2:
                angle = int(cls_result[0])  # angle
                confidence = float(cls_result[1])  # confidence
            else:
                angle = 0
                confidence = 1.0

            # Encode rotated image
            rotated_image_base64 = encode_image(rotated_img)

            results.append(ClassificationResult(
                angle=angle,
                confidence=confidence,
                rotated_image=rotated_image_base64,
                bounding_box=bbox
            ))

        return ClassificationResponse(
            processing_time=processing_time,
            results=results
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify_legacy", response_model=ClassificationResponse)
async def classify_text_angle_legacy(request: LegacyClassificationRequest):
    """Text angle classification service - batch processing (backward compatibility)"""
    try:
        # Decode all images
        img_list = []
        for image_base64 in request.images:
            img = decode_image(image_base64)
            img_list.append(img)

        if not img_list:
            return ClassificationResponse(processing_time=0.0, results=[])

        # Execute text angle classification
        start_time = time.time()
        rotated_imgs, cls_res = classifier(img_list)
        end_time = time.time()
        processing_time = end_time - start_time

        # Format results
        results = []
        for i, (rotated_img, cls_result) in enumerate(zip(rotated_imgs, cls_res)):
            if isinstance(cls_result, (list, tuple)) and len(cls_result) >= 2:
                angle = int(cls_result[0])  # angle
                confidence = float(cls_result[1])  # confidence
            else:
                angle = 0
                confidence = 1.0

            # Encode rotated image
            rotated_image_base64 = encode_image(rotated_img)

            results.append(ClassificationResult(
                angle=angle,
                confidence=confidence,
                rotated_image=rotated_image_base64,
                bounding_box=None  # No bounding box in compatibility mode
            ))

        return ClassificationResponse(
            processing_time=processing_time,
            results=results
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify_single", response_model=ClassificationResult)
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
            bounding_box=None
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    # Start classification service on port 5008
    uvicorn.run(app, host="0.0.0.0", port=5008)
