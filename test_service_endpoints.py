"""
测试OCR服务各个端点的示例代码
"""
import requests
import base64
import json
from pathlib import Path

# 服务基础URL
BASE_URL = "http://localhost:5005"

def encode_image_to_base64(image_path: str) -> str:
    """将图像文件编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_detection_service(image_path: str):
    """测试文本检测服务"""
    print("Testing Detection Service...")
    
    # 准备请求数据
    image_base64 = encode_image_to_base64(image_path)
    data = {
        "image": image_base64
    }
    
    # 发送请求
    response = requests.post(f"{BASE_URL}/detection", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detection successful!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"Found {len(result['bounding_boxes'])} text regions")
        return result['bounding_boxes']
    else:
        print(f"Detection failed: {response.text}")
        return []

def test_classification_service(image_path: str, bounding_boxes: list):
    """测试文本角度分类服务"""
    print("\nTesting Classification Service...")
    
    if not bounding_boxes:
        print("No bounding boxes available for classification")
        return []
    
    # 准备请求数据
    image_base64 = encode_image_to_base64(image_path)
    data = {
        "image": image_base64,
        "bounding_boxes": bounding_boxes
    }
    
    # 发送请求
    response = requests.post(f"{BASE_URL}/classification", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Classification successful!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        for i, cls_result in enumerate(result['results']):
            print(f"Region {i+1}: angle={cls_result['angle']:.1f}°, confidence={cls_result['confidence']:.3f}")
        return result['results']
    else:
        print(f"Classification failed: {response.text}")
        return []

def test_recognition_service(image_path: str, bounding_boxes: list):
    """测试文本识别服务"""
    print("\nTesting Recognition Service...")
    
    if not bounding_boxes:
        print("No bounding boxes available for recognition")
        return []
    
    # 准备请求数据 - 需要转换为新的数据格式
    image_base64 = encode_image_to_base64(image_path)
    
    # 将简单的边界框列表转换为包含角度信息的格式
    bounding_boxes_with_angle = []
    for bbox in bounding_boxes:
        bounding_boxes_with_angle.append({
            "angle": 0.0,  # 默认角度为0
            "bounding_box": bbox
        })
    
    data = {
        "image": image_base64,
        "bounding_boxes": bounding_boxes_with_angle
    }
    
    # 发送请求
    response = requests.post(f"{BASE_URL}/recognition", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Recognition successful!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        for i, rec_result in enumerate(result['results']):
            print(f"Region {i+1}: '{rec_result['text']}' (confidence: {rec_result['confidence']:.3f})")
        return result['results']
    else:
        print(f"Recognition failed: {response.text}")
        return []

def test_recognition_with_classification(image_path: str, bounding_boxes: list, classification_results: list):
    """测试带有角度分类结果的文本识别服务"""
    print("\nTesting Recognition Service with Classification Results...")
    
    if not bounding_boxes or not classification_results:
        print("No bounding boxes or classification results available")
        return []
    
    # 准备请求数据 - 使用分类结果中的角度信息
    image_base64 = encode_image_to_base64(image_path)
    
    # 将边界框和角度信息组合
    bounding_boxes_with_angle = []
    for i, (bbox, cls_result) in enumerate(zip(bounding_boxes, classification_results)):
        bounding_boxes_with_angle.append({
            "angle": cls_result.get('angle', 0.0),
            "bounding_box": bbox
        })
    
    data = {
        "image": image_base64,
        "bounding_boxes": bounding_boxes_with_angle
    }
    
    # 发送请求
    response = requests.post(f"{BASE_URL}/recognition", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Recognition with classification successful!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        for i, rec_result in enumerate(result['results']):
            angle = bounding_boxes_with_angle[i]['angle']
            print(f"Region {i+1} (angle: {angle}°): '{rec_result['text']}' (confidence: {rec_result['confidence']:.3f})")
        return result['results']
    else:
        print(f"Recognition with classification failed: {response.text}")
        return []

def test_full_ocr_service(image_path: str):
    """测试完整OCR服务"""
    print("\nTesting Full OCR Service...")
    
    # 准备请求数据
    image_base64 = encode_image_to_base64(image_path)
    data = {
        "image": image_base64
    }
    
    # 发送请求
    response = requests.post(f"{BASE_URL}/inference", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Full OCR successful!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"Recognized {len(result['results'])} text regions:")
        for i, ocr_result in enumerate(result['results']):
            print(f"  {i+1}. '{ocr_result['text']}' (confidence: {ocr_result['confidence']:.3f})")
        return result['results']
    else:
        print(f"Full OCR failed: {response.text}")
        return []

def main():
    """主测试函数"""
    # 使用测试图像
    image_path = "onnxocr/test_images/1.jpg"
    
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        print("Please make sure the image exists or update the path")
        return
    
    print(f"Testing with image: {image_path}")
    print("=" * 50)
    
    # 测试各个服务
    try:
        # 1. 测试检测服务
        bounding_boxes = test_detection_service(image_path)
        
        # 2. 测试分类服务
        classification_results = []
        if bounding_boxes:
            classification_results = test_classification_service(image_path, bounding_boxes)
            
            # 3a. 测试基础识别服务（不使用角度信息）
            test_recognition_service(image_path, bounding_boxes)
            
            # 3b. 测试带角度信息的识别服务
            if classification_results:
                test_recognition_with_classification(image_path, bounding_boxes, classification_results)
        
        # 4. 测试完整OCR服务
        test_full_ocr_service(image_path)
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the service. Make sure the service is running on http://localhost:5005")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
