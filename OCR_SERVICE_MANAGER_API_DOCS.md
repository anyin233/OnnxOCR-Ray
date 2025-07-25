# OCR Service Manager API Documentation

## Overview

The OCR Service Manager is a FastAPI-based microservice orchestrator that manages multiple OCR service instances. It provides centralized management for starting, stopping, and routing inference requests to OCR services running on different devices (CPU/GPU).

**Base URL**: `http://localhost:8000`

## Features

- **Dynamic Service Management**: Start and stop OCR service instances on-demand
- **Multi-Device Support**: Deploy services on CPU or specific GPU devices
- **Service Discovery**: List and track all running service instances
- **Load Balancing**: Automatic port assignment to avoid conflicts
- **Process Isolation**: Each service runs in its own process for stability

## Service Types

The manager supports three types of OCR services:
- **detection**: Text detection service
- **classification**: Text classification service  
- **recognition**: Text recognition service

## API Endpoints

### 1. Start Service

**Endpoint**: `POST /start`

**Description**: Starts a new OCR service instance on the specified device.

**Request Body**:
```json
{
    "device_id": "string",     // Device identifier: "cpu" or "cuda:N" (where N is GPU ID)
    "service_type": "string",  // Service type: "detection", "classification", or "recognition"
    "port": 5005              // Preferred port number (optional, defaults to 5005)
}
```

**Response**:
```json
{
    "model_id": "string",      // Unique UUID identifier for the service instance
    "service_type": "string",  // Confirmed service type
    "port": 5005              // Actual port number assigned to the service
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "cuda:0",
    "service_type": "detection",
    "port": 5005
  }'
```

**Example Response**:
```json
{
    "model_id": "123e4567-e89b-12d3-a456-426614174000",
    "service_type": "detection",
    "port": 5005
}
```

**Status Codes**:
- `200 OK`: Service started successfully
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Failed to start service

---

### 2. Stop Service

**Endpoint**: `POST /stop`

**Description**: Stops a running OCR service instance identified by its model ID.

**Request Body**:
```json
{
    "model_id": "string"  // UUID of the service instance to stop
}
```

**Response**:
```json
{
    "status_code": 200,
    "model_id": "string",     // UUID of the stopped service
    "message": "string"       // Status message (optional)
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/stop" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

**Example Response**:
```json
{
    "status_code": 200,
    "model_id": "123e4567-e89b-12d3-a456-426614174000",
    "message": "Service on port 5005 stopped successfully."
}
```

**Status Codes**:
- `200 OK`: Service stopped successfully
- `404 Not Found`: Service with specified model_id not found
- `500 Internal Server Error`: Failed to stop service

---

### 3. Inference

**Endpoint**: `POST /inference`

**Description**: Routes an inference request to the appropriate OCR service instance.

**Request Body**:
```json
{
    "model_id": "string",     // UUID of the target service instance
    "request_data": {}        // Service-specific request data (varies by service type)
}
```

**Response**:
```json
{
    "status_code": 200,
    "model_id": "string",     // UUID of the service that processed the request
    "response_data": {},      // Service-specific response data
    "message": "string"       // Error message if applicable (optional)
}
```

**Example Request** (Detection Service):
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "123e4567-e89b-12d3-a456-426614174000",
    "request_data": {
      "image": "base64_encoded_image_data",
      "parameters": {
        "threshold": 0.5
      }
    }
  }'
```

**Example Response**:
```json
{
    "status_code": 200,
    "model_id": "123e4567-e89b-12d3-a456-426614174000",
    "response_data": {
      "boxes": [[100, 100, 200, 150], [300, 200, 400, 250]],
      "confidence": [0.95, 0.87]
    }
}
```

**Status Codes**:
- `200 OK`: Inference completed successfully
- `404 Not Found`: Service with specified model_id not found
- `500 Internal Server Error`: Inference failed or service error

---

### 4. List Services

**Endpoint**: `GET /list_services`

**Description**: Retrieves a list of all currently running OCR service instances.

**Request**: No request body required.

**Response**:
```json
{
    "services": [
        {
            "model_id": "string",      // Unique service identifier
            "service_type": "string",  // Type of OCR service
            "port": 5005              // Port number where service is running
        }
    ]
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/list_services"
```

**Example Response**:
```json
{
    "services": [
        {
            "model_id": "123e4567-e89b-12d3-a456-426614174000",
            "service_type": "detection",
            "port": 5005
        },
        {
            "model_id": "987fcdeb-51a2-43d7-8f9e-123456789abc",
            "service_type": "recognition",
            "port": 5006
        }
    ]
}
```

**Status Codes**:
- `200 OK`: List retrieved successfully

## Data Models

### StartServiceRequest
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| device_id | string | Yes | Device identifier: "cpu" or "cuda:N" |
| service_type | string | Yes | OCR service type |
| port | integer | No | Preferred port (default: 5005) |

### StartServiceResponse
| Field | Type | Description |
|-------|------|-------------|
| model_id | string | Unique service identifier (UUID) |
| service_type | string | Type of OCR service started |
| port | integer | Actual port assigned to the service |

### StopServiceRequest
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_id | string | Yes | UUID of service to stop |

### StopServiceResponse
| Field | Type | Description |
|-------|------|-------------|
| status_code | integer | HTTP status code |
| model_id | string | UUID of the stopped service |
| message | string | Optional status message |

### InferenceRequest
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_id | string | Yes | UUID of target service |
| request_data | any | Yes | Service-specific request payload |

### InferenceResponse
| Field | Type | Description |
|-------|------|-------------|
| status_code | integer | HTTP status code |
| model_id | string | UUID of the service that processed the request |
| response_data | any | Service-specific response payload |
| message | string | Optional error message |

### ServiceInfo
| Field | Type | Description |
|-------|------|-------------|
| model_id | string | Unique service identifier |
| service_type | string | Type of OCR service |
| port | integer | Port where service is running |

### ListServicesResponse
| Field | Type | Description |
|-------|------|-------------|
| services | array | List of ServiceInfo objects |

## Device Configuration

### CPU Configuration
```json
{
    "device_id": "cpu"
}
```

### GPU Configuration
```json
{
    "device_id": "cuda:0"  // Use GPU 0
}
```
```json
{
    "device_id": "cuda:1"  // Use GPU 1
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages in the response body.

### Common Error Responses

**400 Bad Request**:
```json
{
    "detail": "Invalid device_id format: invalid_device"
}
```

**404 Not Found**:
```json
{
    "status_code": 404,
    "model_id": "non-existent-uuid",
    "message": "Service not found."
}
```

**500 Internal Server Error**:
```json
{
    "status_code": 500,
    "model_id": "service-uuid",
    "message": "Failed to start service process"
}
```

## Usage Examples

### Complete Workflow Example

1. **Start a detection service on GPU 0**:
```bash
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "cuda:0",
    "service_type": "detection",
    "port": 5005
  }'
```

2. **List all running services**:
```bash
curl -X GET "http://localhost:8000/list_services"
```

3. **Send an inference request**:
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "returned-uuid-from-step-1",
    "request_data": {
      "image": "base64_image_data"
    }
  }'
```

4. **Stop the service**:
```bash
curl -X POST "http://localhost:8000/stop" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "returned-uuid-from-step-1"
  }'
```

## Technical Details

### Port Management
- The manager automatically finds available ports starting from the requested port
- If the requested port is unavailable, it will find the nearest available port
- Port assignments are tracked and managed internally

### Process Management
- Each OCR service runs in an isolated subprocess
- Services are gracefully terminated with a 5-second timeout
- Force termination is applied if graceful shutdown fails

### Environment Variables
- `CUDA_VISIBLE_DEVICES` is set automatically based on the device_id
- For CPU services: `CUDA_VISIBLE_DEVICES=""`
- For GPU services: `CUDA_VISIBLE_DEVICES="N"` (where N is the GPU ID)

### Service Communication
- Manager communicates with individual services via HTTP REST API
- Each service exposes `/inference` and `/stop` endpoints
- Request routing is handled transparently by the manager

## Deployment

The service manager runs on port 8000 by default:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

To change the port or host, modify the last line in `ocr_service_manager.py`.

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation
- Uvicorn: ASGI server
- Requests: HTTP client library
- UUID: Unique identifier generation
- Subprocess: Process management

## Security Considerations

- The API currently runs without authentication
- Services are bound to all interfaces (0.0.0.0)
- Consider implementing authentication and authorization for production use
- Network security should be configured at the infrastructure level
