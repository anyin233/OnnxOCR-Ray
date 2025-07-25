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
from microservices_ray.common import decode_image, crop_image_from_box, BoundingBox

# Initialize FastAPI application
app = FastAPI(
    title="TextRecognitionService", description="Text Recognition Service using ONNX"
)


class ClassificationInfo(BaseModel):
    angle: int  # angle
    confidence: float
    rotated_image: str  # rotated image (base64 encoded)


class RecognitionRequest(BaseModel):
    image: str  # base64 encoded original image
    bounding_boxes: List[BoundingBox]  # List of detected bounding boxes
    classification_results: Optional[List[ClassificationInfo]] = (
        None  # Classification results (optional)
    )


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


class RecognitionService:
    def __init__(self):
        # Initialize recognition model
        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)
        params.use_gpu = True

        self.recognizer = TextRecognizer(params)

    async def recognize_text(self, request: RecognitionRequest):
        """Text recognition service - based on detection bounding boxes and classification results"""
        try:
            # Decode original image
            img = decode_image(request.image)

            # Process image list
            img_list = []
            valid_boxes = []
            angle_info = []

            # If classification results available, prioritize using rotated images
            if request.classification_results and len(
                request.classification_results
            ) == len(request.bounding_boxes):
                for i, (bbox, cls_result) in enumerate(
                    zip(request.bounding_boxes, request.classification_results)
                ):
                    # Use rotated image from classification service
                    rotated_img = decode_image(cls_result.rotated_image)
                    if rotated_img is not None:
                        img_list.append(rotated_img)
                        valid_boxes.append(bbox.coordinates)
                        angle_info.append(
                            {
                                "angle": cls_result.angle,
                                "confidence": cls_result.confidence,
                            }
                        )
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
            rec_res = self.recognizer(img_list)
            end_time = time.time()
            processing_time = end_time - start_time

            # Format results
            results = []
            for i, (res, bbox, angle) in enumerate(
                zip(rec_res, valid_boxes, angle_info)
            ):
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    text = str(res[0])
                    confidence = float(res[1])
                else:
                    text = str(res)
                    confidence = 1.0

                results.append(
                    RecognitionResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=bbox,
                        angle=angle.get("angle"),
                    )
                )

            return RecognitionResponse(processing_time=processing_time, results=results)

        except ValueError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            raise RuntimeError(f"Recognition error: {str(e)}")

    async def recognize_single_text(self, request: SingleImageRequest):
        """Text recognition service - single image"""
        try:
            # Decode image
            img = decode_image(request.image)

            # Execute text recognition
            start_time = time.time()
            rec_res = self.recognizer([img])
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
                text=text, confidence=confidence, bounding_box=None, angle=None
            )

        except ValueError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            raise RuntimeError(f"Recognition error: {str(e)}")
