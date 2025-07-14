from .classification_service import app as classification_app
from .detection_service import app as detection_app
from .recognition_service import app as recognition_app

__all__ = [
    "classification_app",
    "detection_app",
    "recognition_app",
]
