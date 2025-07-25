import ray
import ray.serve
from pydantic import BaseModel
from fastapi import FastAPI

# Import and deploy the services as Ray Serve deployments
from microservices_ray.detection_service import DetectionService
from microservices_ray.classification_service import ClassificationService, ClassificationRequest
from microservices_ray.recognition_service import RecognitionService, RecognitionRequest, ClassificationInfo, RecognitionResponse


app = FastAPI()

@ray.serve.deployment(name="detection_service", ray_actor_options={"num_cpus": 0, "num_gpus": 0.1})
class DetectionServiceDeployment(DetectionService):
    pass

@ray.serve.deployment(name="classification_service", ray_actor_options={"num_cpus": 0, "num_gpus": 0.1})
class ClassificationServiceDeployment(ClassificationService):
    pass

@ray.serve.deployment(name="recognition_service", ray_actor_options={"num_cpus": 0, "num_gpus": 0.1})
class RecognitionServiceDeployment(RecognitionService):
    pass

class OCRRequest(BaseModel):
    image: str  # base64 encoded image
    

    


@ray.serve.deployment(
  name="main_service")
@ray.serve.ingress(app)
class OCRService:
  def __init__(self, detection_service, classification_service, recognition_service):
    # Get handles to the deployed services
    self.detection_service = detection_service
    self.classification_service = classification_service
    self.recognition_service = recognition_service

  
  @app.post("/ocr")
  async def ocr(self, request: OCRRequest) -> RecognitionResponse:
    """Combined OCR service"""
    detection_response = await self.detection_service.detect_text.remote(request)
    classification_request = ClassificationRequest(
        image=request.image,
        bounding_boxes=detection_response.bounding_boxes
    )
    classification_response = await self.classification_service.classify_text_angle.remote(classification_request)
    
    classification_infos = [
        ClassificationInfo(
            angle=result.angle,
            confidence=result.confidence,
            rotated_image=result.rotated_image,
        ) for result in classification_response.results 
    ]
    recognition_request = RecognitionRequest(
        image=request.image,
        bounding_boxes=detection_response.bounding_boxes,
        classification_results=classification_infos
    )
    recognition_response = await self.recognition_service.recognize_text.remote(recognition_request)

    return recognition_response
    
detection_service = DetectionServiceDeployment.bind()
classification_service = ClassificationServiceDeployment.bind()
recognition_service = RecognitionServiceDeployment.bind()
ocr_app = OCRService.bind(
    detection_service=detection_service,
    classification_service=classification_service,
    recognition_service=recognition_service
)
# ray.serve.run(ocr_app, name="ocr_service", route_prefix="/")
