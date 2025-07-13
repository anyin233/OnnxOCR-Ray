# OCR Microservices Architecture

This project splits the original monolithic OCR service into independent microservices, including text detection, text recognition, text classification, and orchestration services.

## Architecture Overview

```
┌─────────────────┐
│   Frontend/Client│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Orchestrator     │───▶│ Detection       │───▶│ Classification  │───▶│ Recognition     │
│ Service          │    │ Service         │    │ Service         │    │ Service         │
│  Port: 5009     │    │  Port: 5006     │    │  Port: 5008     │    │  Port: 5007     │
│                 │    │                 │    │                 │    │                 │
│ Coordinates      │    │ Text region     │    │ Text angle      │    │ Text content    │
│ all services     │    │ detection       │    │ classification  │    │ recognition     │
│ Handles complete │    │ Output bounding │    │ Angle correction│    │ Output text     │
│ OCR workflow     │    │ boxes           │    │                 │    │ content         │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Service Details

### 1. Detection Service (detection_service.py)
- **Port**: 5006
- **Function**: Text region detection
- **Input**: Base64 encoded image
- **Output**: List of text bounding box coordinates
- **Endpoints**:
  - `GET /` - Health check
  - `POST /detect` - Text detection

### 3. Classification Service (classification_service.py)
- **Port**: 5008
- **Function**: Text angle classification and correction
- **Input**: Base64 encoded original image and detection bounding boxes
- **Output**: Angle information and rotated image
- **Endpoints**:
  - `GET /` - Health check
  - `POST /classify` - Angle classification based on bounding boxes (new interface)
  - `POST /classify_legacy` - Batch angle classification (backward compatibility)
  - `POST /classify_single` - Single angle classification

### 2. Recognition Service (recognition_service.py)
- **Port**: 5007
- **Function**: Text content recognition
- **Input**: Base64 encoded original image, detection bounding boxes and classification results (optional)
- **Output**: Recognized text content and confidence
- **Endpoints**:
  - `GET /` - Health check
  - `POST /recognize` - Text recognition based on bounding boxes and classification results (new interface)
  - `POST /recognize_legacy` - Batch text recognition (backward compatibility)
  - `POST /recognize_single` - Single text recognition

### 4. Orchestrator Service (orchestrator_service.py)
- **Port**: 5009
- **Function**: Coordinates all services, provides complete OCR workflow
- **Input**: Base64 encoded image and configuration options
- **Output**: Complete OCR results
- **Endpoints**:
  - `GET /` - Health check (includes sub-service status)
  - `POST /ocr` - Complete OCR processing

## Quick Start

### 1. Requirements
- Python 3.7+
- Project dependencies installed (`pip install -r requirements.txt`)

### 2. Start All Services

#### Method 1: Using Start Script (Recommended)
```bash
# Start all services
./start_services.sh

# Stop all services
./stop_services.sh
```

#### Method 2: Manual Start
```bash
# Terminal 1: Start detection service
python detection_service.py

# Terminal 2: Start recognition service
python recognition_service.py

# Terminal 3: Start classification service
python classification_service.py

# Terminal 4: Start orchestrator service
python orchestrator_service.py
```

### 3. Test Services
```bash
# Test updated microservices system
python test_updated_microservices.py

# Test compatibility (automatically detects available services)
python test_service.py

# Performance comparison test
python performance_comparison.py

# Test original microservices (if needed)
python test_microservices.py
```

## API Usage Examples

### Orchestrator Service - Complete OCR
```python
import requests
import base64

# Read image and convert to base64
with open('demo_text_ocr.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Call OCR service
response = requests.post('http://localhost:5009/ocr', json={
    'image': image_base64,
    'use_detection': True,
    'use_recognition': True, 
    'use_classification': True
})

result = response.json()
print(f"Processing time: {result['processing_time']} seconds")
for item in result['results']:
    print(f"Text: {item['text']}, Confidence: {item['confidence']}")
```

### New Workflow Example
```python
# 1. Text detection
detection_response = requests.post('http://localhost:5006/detect', json={
    'image': image_base64
})
bounding_boxes = detection_response.json()['bounding_boxes']

# 2. Angle classification (based on bounding boxes)
classification_response = requests.post('http://localhost:5008/classify', json={
    'image': image_base64,
    'bounding_boxes': bounding_boxes
})
classification_results = classification_response.json()['results']

# 3. Text recognition (based on bounding boxes and classification results)
recognition_response = requests.post('http://localhost:5007/recognize', json={
    'image': image_base64,
    'bounding_boxes': bounding_boxes,
    'classification_results': [
        {
            'angle': r['angle'],
            'confidence': r['confidence'],
            'rotated_image': r['rotated_image']
        }
        for r in classification_results
    ]
})
```

### Backward Compatible Service Calls
```python
# Using compatible interfaces
response = requests.post('http://localhost:5007/recognize_legacy', json={
    'images': [image_base64]  # List of cropped images
})

response = requests.post('http://localhost:5008/classify_legacy', json={
    'images': [image_base64]  # List of images
})
```

## Service Configuration

### Port Configuration
To modify ports, edit the following files:
- `detection_service.py` - line 86
- `recognition_service.py` - line 135  
- `classification_service.py` - line 147
- `orchestrator_service.py` - lines 26-28 (service address configuration) and line 290

### Model Configuration
Model parameters for each service can be modified in the corresponding service files:
- Detection model parameters in `detection_service.py`
- Recognition model parameters in `recognition_service.py`
- Classification model parameters in `classification_service.py`

## Performance Monitoring

### Health Checks
```bash
# Check individual services
curl http://localhost:5006/  # Detection service
curl http://localhost:5007/  # Recognition service
curl http://localhost:5008/  # Classification service
curl http://localhost:5009/  # Orchestrator service (includes all sub-service status)
```

### Log Viewing
Service logs are saved in the `logs/` directory:
```bash
tail -f logs/detection.log      # Detection service logs
tail -f logs/recognition.log    # Recognition service logs
tail -f logs/classification.log # Classification service logs
tail -f logs/orchestrator.log   # Orchestrator service logs
```

## Advantages

### 1. Service Decoupling
- Each service can be deployed and scaled independently
- Independent upgrades and maintenance
- Fault isolation - single service failure doesn't affect others

### 2. Flexibility
- Can use only specific services as needed
- Supports different service combinations
- Easy to add new features and services

### 3. Scalability
- Can independently scale high-load services
- Supports load balancing
- Easy horizontal scaling

### 4. Development Efficiency
- Different teams can develop different services independently
- Easier unit testing
- Facilitates CI/CD workflows

## Troubleshooting

### Common Issues

1. **Service startup failure**
   - Check if ports are occupied
   - Review log files for error information
   - Ensure Python environment and dependencies are correctly installed

2. **Inter-service communication failure**
   - Check if services are running normally
   - Verify network connectivity
   - Check firewall settings

3. **Out of memory**
   - Consider reducing concurrent request count
   - Optimize model loading strategy
   - Increase system memory

### Debugging Suggestions
- Use `test_microservices.py` for comprehensive testing
- Check health check endpoints of each service
- Review detailed error information in log files

## Compatibility with Original Service

The new orchestrator service (`orchestrator_service.py`) maintains compatibility with the original API and can be used as a drop-in replacement for the original `app-service.py`. Main API compatibility:
- `POST /ocr` - Maintains same input/output format
- Response format is fully compatible
- Can seamlessly replace the original service

## Deployment Recommendations

### Development Environment
Use the provided startup scripts to quickly start all services for development and testing.

### Production Environment
Recommend containerized deployment:
1. Create Docker images for each service
2. Use Docker Compose or Kubernetes for orchestration
3. Configure load balancing and service discovery
4. Add monitoring and log collection
