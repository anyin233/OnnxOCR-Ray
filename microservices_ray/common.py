import base64
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List


# Define response model
class BoundingBox(BaseModel):
    coordinates: List[
        List[float]
    ]  # 4 points coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


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
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
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
