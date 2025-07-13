#!/usr/bin/env python3
"""
Text Recognition Service - Independent recognition service
"""

import cv2
import time
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from onnxocr.predict_rec import TextRecognizer
from onnxocr.utils import infer_args as init_args
import argparse

# Initialize FastAPI application
app = FastAPI(title="TextRecognitionService", description="Text Recognition Service using ONNX")

# Initialize recognition model
parser = init_args()
inference_args_dict = {}
for action in parser._actions:
    inference_args_dict[action.dest] = action.default
params = argparse.Namespace(**inference_args_dict)
params.use_gpu = False
params.rec_image_shape = "3, 48, 320"

recognizer = TextRecognizer(params)

# Define request models
class BoundingBox(BaseModel):
    coordinates: List[List[float]]  # 4 points coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

class ClassificationInfo(BaseModel):
    angle: int  # angle
    confidence: float
    rotated_image: str  # rotated image (base64 encoded)

class RecognitionRequest(BaseModel):
    image: str  # base64 encoded original image
    bounding_boxes: List[BoundingBox]  # List of detected bounding boxes
    classification_results: Optional[List[ClassificationInfo]] = None  # Classification results (optional)

class LegacyRecognitionRequest(BaseModel):
    images: List[str]  # base64 encoded image list (backward compatibility)

class SingleImageRequest(BaseModel):
    image: str  # base64 encoded single image

# Define response models
class RecognitionResult(BaseModel):
    text: str
    confidence: float
    bounding_box: Optional[List[List[float]]] = None  # corresponding bounding box
    angle: Optional[int] = None  # corresponding angle information

class RecognitionResponse(BaseModel):
    processing_time: float
    results: List[RecognitionResult]

@app.get("/")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "text_recognition"}

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

@app.post("/inference", response_model=RecognitionResponse)
async def recognize_text(request: RecognitionRequest):
    """Text recognition service - based on detection bounding boxes and classification results"""
    try:
        # Decode original image
        img = decode_image(request.image)
        
        # Process image list
        img_list = []
        valid_boxes = []
        angle_info = []
        
        # If classification results available, prioritize using rotated images
        if request.classification_results and len(request.classification_results) == len(request.bounding_boxes):
            for i, (bbox, cls_result) in enumerate(zip(request.bounding_boxes, request.classification_results)):
                # Use rotated image from classification service
                rotated_img = decode_image(cls_result.rotated_image)
                if rotated_img is not None:
                    img_list.append(rotated_img)
                    valid_boxes.append(bbox.coordinates)
                    angle_info.append({"angle": cls_result.angle, "confidence": cls_result.confidence})
        else:
            # No classification results, crop directly from original image
            for bbox in request.bounding_boxes:
                img_crop = crop_image_from_box(img, bbox.coordinates)
                if img_crop is not None:
                    img_list.append(img_crop)
                    valid_boxes.append(bbox.coordinates)
                    angle_info.append({"angle": 0, "confidence": 1.0})
        
        if not img_list:
            return RecognitionResponse(processing_time=0.0, results=[])

        # Execute text recognition
        start_time = time.time()
        rec_res = recognizer(img_list)
        end_time = time.time()
        processing_time = end_time - start_time

        # Format results
        results = []
        for i, (res, bbox, angle) in enumerate(zip(rec_res, valid_boxes, angle_info)):
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                text = str(res[0])
                confidence = float(res[1])
            else:
                text = str(res)
                confidence = 1.0
            
            results.append(RecognitionResult(
                text=text,
                confidence=confidence,
                bounding_box=bbox,
                angle=angle.get("angle")
            ))

        return RecognitionResponse(
            processing_time=processing_time,
            results=results
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")


@app.post("/inference_single", response_model=RecognitionResult)
async def recognize_single_text(request: SingleImageRequest):
    """Text recognition service - single image"""
    try:
        # Decode image
        img = decode_image(request.image)

        # Execute text recognition
        start_time = time.time()
        rec_res = recognizer([img])
        end_time = time.time()

        # Format results
        if rec_res and len(rec_res) > 0:
            res = rec_res[0]
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                text = str(res[0])
                confidence = float(res[1])
            else:
                text = str(res)
                confidence = 1.0
        else:
            text = ""
            confidence = 0.0

        return RecognitionResult(
            text=text,
            confidence=confidence,
            bounding_box=None,
            angle=None
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    # Start recognition service on port 5007
    uvicorn.run(app, host="0.0.0.0", port=5007)
