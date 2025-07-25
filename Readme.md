# OCR Microservices

## Usage

### Run Single Service

```
uv run service_entry_point.py -s <service> -p <port> -h <host>
```

Support services:
- classification: Angle classification
- detection: Text detection
- recognition: Text recognition

### Run All Services

```
./start_services.sh
```

### Run All Services With Ray

```
uv run serve run ocr_app_scale.yaml
```

### Run Services with Service Manager

```
uv run ocr_service_manager.py
```

Docs -> OCR_SERVICE_MANAGER_API_DOCS.md

## API Reference

### Common APIs

#### GET `/stop`
Stop current microservice

**Response**
```
{"message": "Service is stopping"}
```

### Text Classification Service API Reference

A FastAPI-based text angle classification service using ONNX models to detect and correct text orientation.

### Endpoints

#### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "text_classification"
}
```

#### POST `/inference`
Classify text angles from detected bounding boxes.

**Request Body:**
```json
{
  "image": "string",           // base64 encoded image
  "bounding_boxes": [
    {
      "coordinates": [           // 4 corner points
        [x1, y1], [x2, y2], [x3, y3], [x4, y4]
      ]
    }
  ]
}
```

**Response:**
```json
{
  "processing_time": 0.0,
  "results": [
    {
      "angle": 0,                      // 0, 90, 180, or 270
      "confidence": 0.95,
      "rotated_image": "string",       // base64 encoded corrected image
      "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

#### POST `/inference_single`
Classify text angle for a single image.

**Request Body:**
```json
{
  "image": "string"             // base64 encoded image
}
```

**Response:**
```json
{
  "angle": 0,                   // 0, 90, 180, or 270
  "confidence": 0.95,
  "rotated_image": "string",    // base64 encoded corrected image
  "bounding_box": null
}
```

### Error Responses
- `400`: Invalid image format or decoding error
- `500`: Internal classification error

---

### Text Detection Service API Reference

A FastAPI-based text detection service using ONNX models to detect text regions in images.

#### Base URL
`http://localhost:5006`

#### Endpoints

##### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "text_detection"
}
```

##### POST `/inference`
Detect text regions in an image.

**Request Body:**
```json
{
  "image": "string"             // base64 encoded image
}
```

**Response:**
```json
{
  "processing_time": 0.0,
  "bounding_boxes": [
    {
      "coordinates": [           // 4 corner points for each detected text region
        [x1, y1], [x2, y2], [x3, y3], [x4, y4]
      ]
    }
  ]
}
```

##### POST `/inference_no_crop`
Alternative endpoint with identical functionality to `/inference`.

**Request Body:**
```json
{
  "image": "string"             // base64 encoded image
}
```

**Response:**
Same as `/inference` endpoint.

#### Error Responses
- `400`: Invalid image format or decoding error
- `500`: Internal detection

---

### Text Recognition Service API Reference

A FastAPI-based text recognition service using ONNX models to extract text content from detected text regions.

#### Base URL
`http://localhost:5007`

#### Endpoints

##### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "text_recognition"
}
```

##### POST `/inference`
Recognize text from detected bounding boxes with optional classification results.

**Request Body:**
```json
{
  "image": "string",           // base64 encoded image
  "bounding_boxes": [
    {
      "coordinates": [           // 4 corner points
        [x1, y1], [x2, y2], [x3, y3], [x4, y4]
      ]
    }
  ],
  "classification_results": [    // optional angle correction results
    {
      "angle": 0,                // 0, 90, 180, or 270
      "confidence": 0.95,
      "rotated_image": "string"  // base64 encoded corrected image
    }
  ]
}
```

**Response:**
```json
{
  "processing_time": 0.0,
  "results": [
    {
      "text": "recognized text",
      "confidence": 0.95,
      "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "angle": 0
    }
  ]
}
```

##### POST `/inference_single`
Recognize text from a single image.

**Request Body:**
```json
{
  "image": "string"             // base64 encoded image
}
```

**Response:**
```json
{
  "text": "recognized text",
  "confidence": 0.95,
  "bounding_box": null,
  "angle": null
}
```

#### Error Responses
- `400`: Invalid image format or dec