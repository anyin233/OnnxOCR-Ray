#!/usr/bin/env python3
"""
测试demo_text_ocr.jpg文件的OCR识别
"""

import requests
import base64
import time
import sys

# OCR服务配置
OCR_SERVICE_URL = "http://localhost:8000/ocr"


def test_demo_image():
    """测试demo_text_ocr.jpg文件"""
    demo_image_path = "/home/yanweiye/Project/OnnxOCR-Ray/demo_text_ocr.jpg"

    try:
        # 读取demo图像文件
        with open(demo_image_path, "rb") as f:
            image_data = f.read()

        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # 准备请求数据
        payload = {"img": image_base64}

        print(f"正在测试demo图像: {demo_image_path}")
        print(f"图像大小: {len(image_data)} 字节")
        print(f"Base64编码大小: {len(image_base64)} 字符")
        print("发送OCR请求...")

        start_time = time.time()

        # 发送OCR请求
        response = requests.post(OCR_SERVICE_URL, json=payload)

        end_time = time.time()
        processing_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ OCR识别成功！")
            print(f"客户端总耗时: {processing_time:.3f}秒")
            print(f"服务器处理时间: {result.get('processing_time', 'N/A'):.3f}秒")
            print(f"识别到 {len(result.get('ocr_res', []))} 个文本区域:")
            print("-" * 60)

            for i, text_result in enumerate(result.get("ocr_res", [])[0], 1):
                text = text_result.get("res", "")[0]
                confidence = text_result.get("res", [])[1] if len(text_result.get("res", [])) > 1 else 0.0
                bbox = text_result.get("boxes", [])

                print(f"文本区域 {i}:")
                print(f"  📝 文本: '{text}'")
                print(f"  📊 置信度: {confidence:.4f}")
                print(f"  📍 边界框: {bbox}")
                print()

            # 保存结果到文件
            import json

            result_file = "demo_ocr_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到: {result_file}")

        else:
            print(f"❌ OCR请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {demo_image_path}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 错误: 无法连接到OCR服务 {OCR_SERVICE_URL}")
        print("请确保OCR服务已启动 (python app-service.py)")
        return False
    except Exception as e:
        print(f"❌ 测试demo图像时发生错误: {e}")
        return False

    return True


def check_service_status():
    """检查OCR服务是否正在运行"""
    try:
        # 尝试访问服务，看是否响应
        response = requests.get("http://localhost:5005/", timeout=2)
        return True
    except:
        return False


def main():
    print("🚀 OnnxOCR Demo测试工具")
    print("=" * 50)

    # 检查服务状态
    print("检查OCR服务状态...")
    if not check_service_status():
        print("❌ OCR服务未启动或无法访问")
        print("请先启动OCR服务:")
        print("  python app-service.py")
        print()
        print("服务启动后，再次运行此脚本")
        sys.exit(1)
    else:
        print("✅ OCR服务运行正常")

    print()

    # 执行测试
    success = test_demo_image()

    if success:
        print("\n🎉 Demo测试完成！")
    else:
        print("\n💥 Demo测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
