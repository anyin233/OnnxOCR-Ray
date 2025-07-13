#!/usr/bin/env python3
"""
Updated microservice OCR system test script
Test new API interfaces: detection -> classification -> recognition
"""

import cv2
import base64
import requests
import json
import time
import os


def image_to_base64(image_path):
    """Convert image file to base64 encoding"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def test_new_workflow(image_path):
    """Test new workflow: detection -> classification -> recognition"""
    print("=" * 60)
    print("Testing new microservice workflow")
    print("=" * 60)
    
    image_base64 = image_to_base64(image_path)
    
    # Step 1: Text detection
    print("\n1. Text detection...")
    try:
        response = requests.post(
            "http://localhost:5006/detect",
            json={"image": image_base64},
            timeout=30
        )
        if response.status_code == 200:
            detection_result = response.json()
            print(f"✓ Detection successful")
            print(f"  Processing time: {detection_result['processing_time']:.3f} seconds")
            print(f"  Text boxes detected: {len(detection_result['bounding_boxes'])}")
            
            if not detection_result['bounding_boxes']:
                print("  No text regions detected")
                return
                
            bounding_boxes = detection_result['bounding_boxes']
        else:
            print(f"✗ Detection failed: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Detection service error: {e}")
        return
    
    # Step 2: Text classification (angle correction)
    print("\n2. Text classification (angle correction)...")
    try:
        response = requests.post(
            "http://localhost:5008/classify",
            json={
                "image": image_base64,
                "bounding_boxes": bounding_boxes
            },
            timeout=30
        )
        if response.status_code == 200:
            classification_result = response.json()
            print(f"✓ Classification successful")
            print(f"  Processing time: {classification_result['processing_time']:.3f} seconds")
            print(f"  Classification results:")
            for i, result in enumerate(classification_result['results']):
                print(f"    Region {i+1}: angle={result['angle']}°, confidence={result['confidence']:.4f}")
                
            classification_info = [
                {
                    "angle": result["angle"],
                    "confidence": result["confidence"],
                    "rotated_image": result["rotated_image"]
                }
                for result in classification_result['results']
            ]
        else:
            print(f"✗ Classification failed: {response.status_code}")
            classification_info = None
    except Exception as e:
        print(f"✗ Classification service error: {e}")
        classification_info = None
    
    # Step 3: Text recognition
    print("\n3. Text recognition...")
    try:
        recognition_request = {
            "image": image_base64,
            "bounding_boxes": bounding_boxes
        }
        
        # If classification results are available, add them to the request
        if classification_info:
            recognition_request["classification_results"] = classification_info
        
        response = requests.post(
            "http://localhost:5007/recognize",
            json=recognition_request,
            timeout=30
        )
        if response.status_code == 200:
            recognition_result = response.json()
            print(f"✓ Recognition successful")
            print(f"  Processing time: {recognition_result['processing_time']:.3f} seconds")
            print(f"  Recognition results:")
            for i, result in enumerate(recognition_result['results']):
                print(f"    Region {i+1}: '{result['text']}' (confidence: {result['confidence']:.4f})")
                if result.get('angle') is not None:
                    print(f"             angle: {result['angle']}°")
        else:
            print(f"✗ Recognition failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Recognition service error: {e}")


def test_legacy_workflow(image_path):
    """Test backward compatible workflow"""
    print("\n" + "=" * 60)
    print("Testing backward compatible workflow")
    print("=" * 60)
    
    image_base64 = image_to_base64(image_path)
    
    # First detect
    print("\n1. Detecting text regions...")
    try:
        response = requests.post(
            "http://localhost:5006/detect",
            json={"image": image_base64},
            timeout=30
        )
        if response.status_code == 200:
            detection_result = response.json()
            bounding_boxes = detection_result['bounding_boxes']
            print(f"✓ Detected {len(bounding_boxes)} text regions")
        else:
            print("✗ Detection failed")
            return
    except Exception as e:
        print(f"✗ Detection error: {e}")
        return
    
    # Manually crop images for compatibility testing
    print("\n2. Cropping images and testing compatibility interfaces...")
    try:
        # Read original image
        img = cv2.imread(image_path)
        cropped_images = []
        
        for i, bbox in enumerate(bounding_boxes):
            coords = bbox['coordinates']
            # Simple rectangular cropping (for demonstration)
            x_coords = [point[0] for point in coords]
            y_coords = [point[1] for point in coords]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            
            if x2 > x1 and y2 > y1:
                cropped = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.jpg', cropped)
                cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                cropped_images.append(cropped_base64)
        
        if cropped_images:
            # Test compatible classification interface
            response = requests.post(
                "http://localhost:5008/classify_legacy",
                json={"images": cropped_images},
                timeout=30
            )
            if response.status_code == 200:
                print("✓ Compatible classification interface test successful")
            
            # Test compatible recognition interface
            response = requests.post(
                "http://localhost:5007/recognize_legacy",
                json={"images": cropped_images},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print("✓ Compatible recognition interface test successful")
                for i, res in enumerate(result['results']):
                    print(f"  Text {i+1}: '{res['text']}' (confidence: {res['confidence']:.4f})")
                    
    except Exception as e:
        print(f"✗ Compatibility test error: {e}")


def test_orchestrator_service(image_path):
    """Test orchestrator service"""
    print("\n" + "=" * 60)
    print("Testing OCR orchestrator service (new version)")
    print("=" * 60)
    
    image_base64 = image_to_base64(image_path)
    
    print("\nTesting complete OCR workflow...")
    try:
        response = requests.post(
            "http://localhost:5009/ocr",
            json={
                "image": image_base64,
                "use_detection": True,
                "use_recognition": True,
                "use_classification": True
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Orchestrator service successful")
            print(f"  Total processing time: {result['processing_time']:.3f} seconds")
            print(f"  Detection time: {result['detection_time']:.3f} seconds")
            print(f"  Classification time: {result['classification_time']:.3f} seconds")
            print(f"  Recognition time: {result['recognition_time']:.3f} seconds")
            print(f"  Number of recognition results: {len(result['results'])}")
            
            # Print detailed results
            print("\nRecognition results:")
            print("-" * 50)
            for i, item in enumerate(result['results']):
                print(f"[{i+1}] Text: {item['text']}")
                print(f"    Confidence: {item['confidence']:.4f}")
                if item.get('angle') is not None:
                    print(f"    Angle: {item['angle']}° (confidence: {item['angle_confidence']:.4f})")
                print(f"    Bounding box: {item['bounding_box']}")
                print()
                
        else:
            print(f"✗ Orchestrator service failed: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except Exception as e:
        print(f"✗ Orchestrator service error: {e}")


def check_services_available():
    """Check if all services are available"""
    services = [
        ("Detection service", "http://localhost:5006"),
        ("Recognition service", "http://localhost:5007"),
        ("Classification service", "http://localhost:5008"),
        ("Orchestrator service", "http://localhost:5009")
    ]
    
    print("Checking service status...")
    print("-" * 30)
    
    all_available = True
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✓ {service_name}: Running normally")
            else:
                print(f"✗ {service_name}: Abnormal response ({response.status_code})")
                all_available = False
        except:
            print(f"✗ {service_name}: Cannot connect")
            all_available = False
    
    return all_available


def main():
    """Main function"""
    # Test image
    image_path = 'demo_text_ocr.jpg'
    
    if not os.path.exists(image_path):
        print(f"✗ Test image file does not exist: {image_path}")
        return
    
    print("Updated microservice OCR system test")
    print("=" * 60)
    print("New features:")
    print("- classification accepts image and bounding box")
    print("- recognition accepts image, bounding box and classification results")
    print("- Maintains backward compatibility")
    print("=" * 60)
    
    # Check if services are available
    if not check_services_available():
        print("\n✗ Some services are unavailable, please start all services first")
        print("\nStart command: ./start_services.sh")
        return
    
    # Test new workflow
    test_new_workflow(image_path)
    
    # Test backward compatibility
    test_legacy_workflow(image_path)
    
    # Test orchestrator service
    test_orchestrator_service(image_path)
    
    print("\n" + "=" * 60)
    print("Updated microservice OCR system test completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
