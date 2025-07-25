from fastapi import FastAPI
from typing import List, Optional, Any
from pydantic import BaseModel
import uuid
from service_entry_point import find_neareast_available_port
from service_entry_point import start_service as start_ocr_component
import asyncio
import threading
import requests
import uvicorn
import os
import subprocess
import sys
import sys
from service_entry_point import start_service

class StartServiceRequest(BaseModel):
    device_id: str # "cuda:0" or "cpu"
    service_type: str  # e.g., "detection", "classification", "recognition"
    port: int = 5005
    
class StartServiceResponse(BaseModel):
    model_id: str # uuid
    service_type: str  # e.g., "detection", "classification", "recognition"
    port: int = 5005
    
class StopServiceRequest(BaseModel):
    model_id: str  # uuid
    
class StopServiceResponse(BaseModel):
    status_code: int
    model_id: str
    message: Optional[str] = None
    
class InferenceRequest(BaseModel):
    model_id: str  # uuid
    request_data: Any  # This can be a dict or any other type depending on the service
  
class InferenceResponse(BaseModel):
    status_code: int
    model_id: str  # uuid
    response_data: Any  # This can be a dict or any other type depending on the service
    message: Optional[str] = None

class ServiceInfo(BaseModel):
    model_id: str  # uuid
    service_type: str  # e.g., "detection", "classification", "recognition"
    port: int  # Port on which the service is running
    
class ListServicesResponse(BaseModel):
    services: List[ServiceInfo]  # List of running services
    
app = FastAPI()

modelid_to_port = {}  # Dictionary to map model_id to port
modelid_to_thread = {}  # Dictionary to keep track of threads for each model_id
modelid_to_service_type = {}  # Dictionary to map model_id to service_type

def parse_device_id(device_id: str) -> (str, int):
    """Parse device_id to get device type and GPU ID if applicable."""
    if device_id.startswith("cuda:"):
        gpu_id = int(device_id.split(":")[1])
        return "cuda", gpu_id
    elif device_id == "cpu":
        return "cpu", -1
    else:
        raise ValueError(f"Invalid device_id format: {device_id}")
      
@app.post("/start", response_model=StartServiceResponse)
async def start_service(request: StartServiceRequest):
    """Start a new service instance"""
    model_id = str(uuid.uuid4())
    modelid_to_port[model_id] = find_neareast_available_port(request.port)
    modelid_to_service_type[model_id] = request.service_type
    device_type, gpu_id = parse_device_id(request.device_id)
    # Start the service in a background thread
    def run_service_process():
      if device_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
      else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
      start_ocr_component(service=request.service_type, port=modelid_to_port[model_id], host="0.0.0.0")

    
    # 创建独立进程而不是线程
    process = subprocess.Popen([
      sys.executable, "-c",
      f"""import sys
import os
from service_entry_point import start_service
sys.path.insert(0, '{os.getcwd()}')

if "{device_type}" == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

start_service(service="{request.service_type}", port={modelid_to_port[model_id]}, host="0.0.0.0")
  """
    ])
    
    modelid_to_thread[model_id] = process  # 存储进程对象而不是线程对象
    

    return StartServiceResponse(
      model_id=model_id,
      service_type=request.service_type,
      port=modelid_to_port[model_id]  # 返回实际使用的端口
    )
    
@app.post("/stop", response_model=StopServiceResponse)
async def stop_service(request: StopServiceRequest):
    """Stop a service instance"""
    if request.model_id in modelid_to_port:
        port = modelid_to_port.pop(request.model_id)
        modelid_to_service_type.pop(request.model_id, None)  # Remove service type mapping
        # Here you would implement the logic to stop the service running on that port
        # For example, you might send a signal to the process running on that port
        target_service_url = f"http://localhost:{port}/stop"
        try:
          response = requests.get(target_service_url)
          if response.status_code == 200:
              message = f"Service on port {port} stopped successfully."
          else:
              message = f"Failed to stop service on port {port}. Response: {response.text}"
        except Exception as e:
          message = f"Error stopping service on port {port}: {str(e)}"
                
        process = modelid_to_thread.pop(request.model_id, None)
        if process:
            # Terminate the process
          try:
            process.terminate()
            # Wait for process to terminate gracefully
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
          # Force kill if it doesn't terminate gracefully
            process.kill()
            process.wait()
          except Exception as e:
            message = f"Error terminating process: {str(e)}"
            
        return StopServiceResponse(
            status_code=200,
            model_id=request.model_id,
            message=f"Service on port {port} stopped successfully."
        )
    else:
        return StopServiceResponse(
            status_code=404,
            model_id=request.model_id,
            message="Service not found."
        )

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Send inference request to the appropriate service instance"""
    if request.model_id in modelid_to_port:
        port = modelid_to_port[request.model_id]
        target_service_url = f"http://localhost:{port}/inference"
        
        try:
            response = requests.post(target_service_url, json=request.request_data)
            if response.status_code == 200:
                return InferenceResponse(
                    status_code=200,
                    model_id=request.model_id,
                    response_data=response.json()
                )
            else:
                return InferenceResponse(
                    status_code=response.status_code,
                    model_id=request.model_id,
                    message=f"Error from service: {response.text}"
                )
        except Exception as e:
            return InferenceResponse(
                status_code=500,
                model_id=request.model_id,
                message=str(e)
            )
    else:
        return InferenceResponse(
            status_code=404,
            model_id=request.model_id,
            message="Service not found."
        )
        
@app.get("/list_services", response_model=ListServicesResponse)
async def list_services():
    """List all running services"""
    services = []
    for model_id, port in modelid_to_port.items():
        service_type = modelid_to_service_type.get(model_id, "unknown")
        service_info = ServiceInfo(
            model_id=model_id,
            service_type=service_type,
            port=port
        )
        services.append(service_info)
    
    return ListServicesResponse(services=services)

uvicorn.run(app, host="0.0.0.0", port=8000)