#!/usr/bin/env python3
"""
OCR Orchestrator Service - Coordinates detection, classification and recognition services
"""

import cv2
import time
import base64
import numpy as np
import requests
import copy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from onnxocr.utils import get_rotate_crop_image, get_minarea_rect_crop

# Initialize FastAPI application
app = FastAPI(title="OCROrchestrator", description="OCR Orchestrator Service")

# Service configuration
DETECTION_SERVICE_URL = "http://localhost:5006"
RECOGNITION_SERVICE_URL = "http://localhost:5007"
CLASSIFICATION_SERVICE_URL = "http://localhost:5008"

# Define request models
class OCRRequest(BaseModel):
    image: str  # base64 encoded image
    use_detection: bool = True
    use_recognition: bool = True
    use_classification: bool = True

# Define response models
class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_box: List[List[float]]
    angle: Optional[int] = None
    angle_confidence: Optional[float] = None

class OCRResponse(BaseModel):
    processing_time: float
    detection_time: float
    classification_time: float
    recognition_time: float
    results: List[OCRResult]

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

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[:, 1].min(), x[:, 0].min()))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][:, 1].min() - _boxes[j][:, 1].min()) < 10 and \
               (_boxes[j + 1][:, 0].min() < _boxes[j][:, 0].min()):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

@app.get("/")
async def health_check():
    """Health check"""
    # Check if all sub-services are available
    services_status = {}
    
    try:
        response = requests.get(f"{DETECTION_SERVICE_URL}/", timeout=2)
        services_status["detection"] = response.status_code == 200
    except:
        services_status["detection"] = False
    
    try:
        response = requests.get(f"{RECOGNITION_SERVICE_URL}/", timeout=2)
        services_status["recognition"] = response.status_code == 200
    except:
        services_status["recognition"] = False
    
    try:
        response = requests.get(f"{CLASSIFICATION_SERVICE_URL}/", timeout=2)
        services_status["classification"] = response.status_code == 200
    except:
        services_status["classification"] = False
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "ocr_orchestrator",
        "services": services_status
    }

@app.post("/ocr", response_model=OCRResponse)
async def orchestrate_ocr(request: OCRRequest):
    """OCR orchestration service - coordinates all sub-services"""
    try:
        total_start_time = time.time()
        detection_time = 0.0
        classification_time = 0.0
        recognition_time = 0.0
        
        # Decode image
        img = decode_image(request.image)
        
        ocr_results = []
        
        if request.use_detection:
            # 1. Text detection
            detection_start = time.time()
            try:
                detection_response = requests.post(
                    f"{DETECTION_SERVICE_URL}/detect",
                    json={"image": request.image},
                    timeout=30
                )
                detection_response.raise_for_status()
                detection_result = detection_response.json()
                detection_time = detection_result["processing_time"]
                dt_boxes = [np.array(bbox["coordinates"]) for bbox in detection_result["bounding_boxes"]]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Detection service error: {str(e)}")
            
            if not dt_boxes:
                return OCRResponse(
                    processing_time=time.time() - total_start_time,
                    detection_time=detection_time,
                    classification_time=0.0,
                    recognition_time=0.0,
                    results=[]
                )
            
            # Sort text boxes
            dt_boxes = sorted_boxes(np.array(dt_boxes))
            
            # Crop images
            img_crop_list = []
            valid_boxes = []
            
            for box in dt_boxes:
                img_crop = crop_image_from_box(img, box)
                if img_crop is not None:
                    img_crop_list.append(img_crop)
                    valid_boxes.append(box)
            
            if not img_crop_list:
                return OCRResponse(
                    processing_time=time.time() - total_start_time,
                    detection_time=detection_time,
                    classification_time=0.0,
                    recognition_time=0.0,
                    results=[]
                )
            
            # 2. Text angle classification (if enabled)
            angle_results = []
            
            if request.use_classification:
                classification_start = time.time()
                try:
                    # Build classification request - using new API
                    bounding_boxes_data = [{"coordinates": box.tolist()} for box in valid_boxes]
                    
                    classification_response = requests.post(
                        f"{CLASSIFICATION_SERVICE_URL}/classify",
                        json={
                            "image": request.image,
                            "bounding_boxes": bounding_boxes_data
                        },
                        timeout=30
                    )
                    classification_response.raise_for_status()
                    classification_result = classification_response.json()
                    classification_time = classification_result["processing_time"]
                    angle_results = classification_result["results"]
                    
                except Exception as e:
                    print(f"Classification service error: {e}")
                    # If classification fails, continue using original image
                    angle_results = [{"angle": 0, "confidence": 0.0, "rotated_image": None, "bounding_box": box.tolist()} for box in valid_boxes]

            # 3. Text recognition
            if request.use_recognition:
                recognition_start = time.time()
                try:
                    # Build recognition request - using new API
                    bounding_boxes_data = [{"coordinates": box.tolist()} for box in valid_boxes]
                    
                    # Prepare classification results (if available)
                    classification_info = None
                    if angle_results:
                        classification_info = [
                            {
                                "angle": result.get("angle", 0),
                                "confidence": result.get("confidence", 1.0),
                                "rotated_image": result.get("rotated_image", "")
                            }
                            for result in angle_results
                            if result.get("rotated_image")  # Only include results with rotated images
                        ]
                    
                    recognition_request_data = {
                        "image": request.image,
                        "bounding_boxes": bounding_boxes_data
                    }
                    
                    if classification_info:
                        recognition_request_data["classification_results"] = classification_info
                    
                    recognition_response = requests.post(
                        f"{RECOGNITION_SERVICE_URL}/recognize",
                        json=recognition_request_data,
                        timeout=30
                    )
                    recognition_response.raise_for_status()
                    recognition_result = recognition_response.json()
                    recognition_time = recognition_result["processing_time"]
                    rec_results = recognition_result["results"]
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Recognition service error: {str(e)}")
                
                # Assemble final results
                for i, rec_result in enumerate(rec_results):
                    angle_info = angle_results[i] if i < len(angle_results) else {"angle": 0, "confidence": 0.0}
                    
                    ocr_results.append(OCRResult(
                        text=rec_result["text"],
                        confidence=rec_result["confidence"],
                        bounding_box=rec_result.get("bounding_box", valid_boxes[i].tolist() if i < len(valid_boxes) else []),
                        angle=angle_info.get("angle"),
                        angle_confidence=angle_info.get("confidence")
                    ))
            else:
                # Only return detection results
                for box in valid_boxes:
                    ocr_results.append(OCRResult(
                        text="",
                        confidence=0.0,
                        bounding_box=box.tolist(),
                        angle=None,
                        angle_confidence=None
                    ))
        
        else:
            # If not using detection, directly recognize the entire image
            if request.use_recognition:
                recognition_start = time.time()
                try:
                    recognition_response = requests.post(
                        f"{RECOGNITION_SERVICE_URL}/recognize_single",
                        json={"image": request.image},
                        timeout=30
                    )
                    recognition_response.raise_for_status()
                    recognition_result = recognition_response.json()
                    recognition_time = time.time() - recognition_start
                    
                    # Create full image result
                    h, w = img.shape[:2]
                    full_box = [[0, 0], [w, 0], [w, h], [0, h]]
                    
                    ocr_results.append(OCRResult(
                        text=recognition_result["text"],
                        confidence=recognition_result["confidence"],
                        bounding_box=full_box,
                        angle=None,
                        angle_confidence=None
                    ))
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Recognition service error: {str(e)}")
        
        total_processing_time = time.time() - total_start_time
        
        return OCRResponse(
            processing_time=total_processing_time,
            detection_time=detection_time,
            classification_time=classification_time,
            recognition_time=recognition_time,
            results=ocr_results
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    # Start orchestrator service on port 5009
    uvicorn.run(app, host="0.0.0.0", port=5009)
