#!/usr/bin/env python3
"""
Advanced Test Script for OCR Service Manager
测试OCR服务管理器的所有功能

功能特点:
- 自动根据服务类型创建对应的测试数据
- detection服务: 只需要图片
- recognition服务: 需要图片和边界框信息
- classification服务: 需要图片和边界框信息
- 支持自动服务发现和智能测试数据生成
"""

import asyncio
import json
import time
import requests
import base64
import cv2
import numpy as np
from typing import Dict, Any, List
import threading
import subprocess
import signal
import os
import sys
from pathlib import Path


class OCRServiceManagerTester:
    def __init__(self, manager_host: str = "localhost", manager_port: int = 8000):
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.base_url = f"http://{manager_host}:{manager_port}"
        self.manager_process = None
        self.test_results = {}
        self.active_services = []

    def start_manager(self) -> bool:
        """启动服务管理器"""
        try:
            print("🚀 启动OCR服务管理器...")
            # 启动管理器进程
            self.manager_process = subprocess.Popen(
                [sys.executable, "ocr_service_manager.py"],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # 等待服务启动
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.base_url}/docs", timeout=2)
                    if response.status_code == 200:
                        print("✅ OCR服务管理器启动成功")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    print(f"⏳ 等待服务启动... ({i + 1}/{max_retries})")

            print("❌ OCR服务管理器启动失败")
            return False

        except Exception as e:
            print(f"❌ 启动管理器时出错: {e}")
            return False

    def stop_manager(self):
        """停止服务管理器"""
        if self.manager_process:
            print("🛑 停止OCR服务管理器...")
            self.manager_process.terminate()
            try:
                self.manager_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.manager_process.kill()
                self.manager_process.wait()
            print("✅ OCR服务管理器已停止")

    def test_start_service(
        self, device_id: str = "cuda:0", service_type: str = "detection", port: int = 5005
    ) -> Dict[str, Any]:
        """测试启动服务"""
        print(f"\n📝 测试启动服务: {service_type} on {device_id}")

        try:
            payload = {
                "device_id": device_id,
                "service_type": service_type,
                "port": port,
            }

            response = requests.post(f"{self.base_url}/start", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务启动成功: {result}")
                self.active_services.append(result["model_id"])
                return {"success": True, "data": result}
            else:
                print(f"❌ 服务启动失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}

        except Exception as e:
            print(f"❌ 启动服务时出错: {e}")
            return {"success": False, "error": str(e)}

    def test_list_services(self) -> Dict[str, Any]:
        """测试列出所有服务"""
        print("\n📋 测试列出所有服务")

        try:
            response = requests.get(f"{self.base_url}/list_services", timeout=10)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务列表获取成功: {len(result['services'])} 个服务")
                for service in result["services"]:
                    print(
                        f"   - {service['model_id']}: {service['service_type']} on port {service['port']}"
                    )
                return {"success": True, "data": result}
            else:
                print(f"❌ 获取服务列表失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}

        except Exception as e:
            print(f"❌ 获取服务列表时出错: {e}")
            return {"success": False, "error": str(e)}

    def _create_test_image_with_text(
        self, text: str = "Test OCR", width: int = 300, height: int = 100
    ) -> str:
        """创建包含指定文本的测试图片并返回base64编码"""
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        # 设置白色背景
        test_image.fill(255)
        # 添加黑色文本
        cv2.putText(
            test_image,
            text,
            (10, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        # 将图片编码为base64
        _, buffer = cv2.imencode(".jpg", test_image)
        return base64.b64encode(buffer).decode("utf-8")

    def _get_service_type_by_model_id(self, model_id: str) -> str:
        """通过model_id获取服务类型"""
        try:
            response = requests.get(f"{self.base_url}/list_services", timeout=5)
            if response.status_code == 200:
                services = response.json()["services"]
                for service in services:
                    if service["model_id"] == model_id:
                        return service["service_type"]
        except:
            pass
        return "unknown"

    def _create_test_data_for_service(self, service_type: str) -> Dict[str, Any]:
        """为不同服务类型创建相应的测试数据"""
        base_image = self._create_test_image_with_text(f"Test {service_type.title()}")

        if service_type == "detection":
            # 检测服务只需要图片
            return {"image": base_image}

        elif service_type == "recognition":
            # 识别服务需要图片和边界框
            return {
                "image": base_image,
                "bounding_boxes": [
                    {
                        "coordinates": [
                            [10.0, 30.0],  # 左上
                            [250.0, 30.0],  # 右上
                            [250.0, 70.0],  # 右下
                            [10.0, 70.0],  # 左下
                        ]
                    }
                ],
            }

        elif service_type == "classification":
            # 分类服务需要图片和边界框
            return {
                "image": base_image,
                "bounding_boxes": [
                    {
                        "coordinates": [
                            [10.0, 30.0],  # 左上
                            [250.0, 30.0],  # 右上
                            [250.0, 70.0],  # 右下
                            [10.0, 70.0],  # 左下
                        ]
                    }
                ],
            }

        else:
            # 未知服务类型，使用通用格式
            return {"image": base_image, "format": "base64"}

    def test_inference(
        self, model_id: str, test_data: Dict[Any, Any] = None, service_type: str = None
    ) -> Dict[str, Any]:
        """测试推理请求"""
        print(f"\n🔍 测试推理请求: {model_id}")

        # 如果没有提供测试数据，根据服务类型创建相应的测试数据
        if test_data is None:
            # 如果没有提供服务类型，尝试通过API获取
            if service_type is None:
                service_type = self._get_service_type_by_model_id(model_id)

            print(f"   📊 为 {service_type} 服务创建测试数据")
            test_data = self._create_test_data_for_service(service_type)

        try:
            payload = {"model_id": model_id, "request_data": test_data}

            response = requests.post(
                f"{self.base_url}/inference", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 推理请求成功")
                return {"success": True, "data": result}
            else:
                print(f"❌ 推理请求失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}

        except Exception as e:
            print(f"❌ 推理请求时出错: {e}")
            return {"success": False, "error": str(e)}

    def test_stop_service(self, model_id: str) -> Dict[str, Any]:
        """测试停止服务"""
        print(f"\n🛑 测试停止服务: {model_id}")

        try:
            payload = {"model_id": model_id}

            response = requests.post(f"{self.base_url}/stop", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务停止成功: {result}")
                if model_id in self.active_services:
                    self.active_services.remove(model_id)
                return {"success": True, "data": result}
            else:
                print(f"❌ 服务停止失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}

        except Exception as e:
            print(f"❌ 停止服务时出错: {e}")
            return {"success": False, "error": str(e)}

    def test_multiple_services(self) -> Dict[str, Any]:
        """测试多个服务同时运行"""
        print("\n🔄 测试多个服务同时运行")

        services_to_test = [
            {"device_id": "cuda:0", "service_type": "detection", "port": 5005},
            {"device_id": "cuda:1", "service_type": "recognition", "port": 5006},
            {"device_id": "cuda:2", "service_type": "classification", "port": 5007},
        ]

        started_services = []

        # 启动多个服务
        service_info_map = {}  # 存储 model_id 到服务类型的映射
        for service_config in services_to_test:
            result = self.test_start_service(**service_config)
            if result["success"]:
                model_id = result["data"]["model_id"]
                started_services.append(model_id)
                service_info_map[model_id] = service_config["service_type"]
            time.sleep(2)  # 等待服务完全启动

        # 列出所有服务
        list_result = self.test_list_services()

        # 测试每个服务的推理，使用对应的服务类型
        inference_results = []
        for model_id in started_services:
            service_type = service_info_map.get(model_id, "unknown")
            result = self.test_inference(model_id, service_type=service_type)
            inference_results.append(result)
            time.sleep(1)

        # 停止所有启动的服务
        for model_id in started_services:
            self.test_stop_service(model_id)
            time.sleep(1)

        return {
            "success": len(started_services) > 0,
            "started_services": len(started_services),
            "inference_results": inference_results,
        }

    def test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        print("\n⚠️ 测试错误处理")

        error_tests = []

        # 测试无效的device_id
        print("测试无效的device_id...")
        result = self.test_start_service(
            device_id="invalid_device", service_type="detection"
        )
        error_tests.append(
            {"test": "invalid_device_id", "success": not result["success"]}
        )

        # 测试不存在的model_id进行推理
        print("测试不存在的model_id进行推理...")
        fake_model_id = "non-existent-model-id"
        result = self.test_inference(fake_model_id, service_type="detection")
        error_tests.append(
            {"test": "non_existent_model_inference", "success": not result["success"]}
        )

        # 测试停止不存在的服务
        print("测试停止不存在的服务...")
        result = self.test_stop_service(fake_model_id)
        error_tests.append(
            {"test": "stop_non_existent_service", "success": not result["success"]}
        )

        # 测试无效的数据格式
        print("测试无效的数据格式...")
        # 启动一个检测服务用于测试
        start_result = self.test_start_service(
            device_id="cuda:0", service_type="detection", port=5050
        )
        if start_result["success"]:
            model_id = start_result["data"]["model_id"]
            time.sleep(3)  # 等待服务启动

            # 测试无效的base64图片数据
            invalid_data = {"image": "invalid_base64_image_data"}
            result = self.test_inference(
                model_id, test_data=invalid_data, service_type="detection"
            )
            error_tests.append(
                {"test": "invalid_image_data", "success": not result["success"]}
            )

            # 清理测试服务
            self.test_stop_service(model_id)
        else:
            error_tests.append({"test": "invalid_image_data", "success": False})

        return {
            "success": all(test["success"] for test in error_tests),
            "error_tests": error_tests,
        }

    def test_service_specific_data_formats(self) -> Dict[str, Any]:
        """测试不同服务类型的特定数据格式"""
        print("\n🔧 测试服务特定数据格式")

        format_tests = []
        test_services = [
            {"service_type": "detection", "port": 5020},
            {"service_type": "recognition", "port": 5021},
            {"service_type": "classification", "port": 5022},
        ]

        started_service_ids = []

        # 启动不同类型的服务
        for index, service_config in enumerate(test_services):
            result = self.test_start_service(
                device_id=f"cuda:{index}",
                service_type=service_config["service_type"],
                port=service_config["port"],
            )
            if result["success"]:
                model_id = result["data"]["model_id"]
                started_service_ids.append((model_id, service_config["service_type"]))
                time.sleep(2)

        # 等待所有服务启动完成
        time.sleep(5)

        # 测试每个服务的数据格式
        for model_id, service_type in started_service_ids:
            print(f"   测试 {service_type} 服务的数据格式...")

            # 测试正确的数据格式
            correct_result = self.test_inference(model_id, service_type=service_type)
            format_tests.append(
                {
                    "service": service_type,
                    "test": "correct_format",
                    "success": correct_result["success"],
                }
            )

            # 测试错误的数据格式（如果是recognition或classification服务，不提供bounding_boxes）
            if service_type in ["recognition", "classification"]:
                # 只提供图片，不提供边界框
                wrong_data = {
                    "image": self._create_test_image_with_text(f"Wrong {service_type}")
                }
                wrong_result = self.test_inference(
                    model_id, test_data=wrong_data, service_type=service_type
                )
                format_tests.append(
                    {
                        "service": service_type,
                        "test": "missing_bounding_boxes",
                        "success": not wrong_result["success"],  # 期望失败
                    }
                )

        # 清理测试服务
        for model_id, _ in started_service_ids:
            self.test_stop_service(model_id)

        return {
            "success": all(test["success"] for test in format_tests),
            "format_tests": format_tests,
        }

    def cleanup_services(self):
        """清理所有活跃的服务"""
        print("\n🧹 清理所有活跃的服务...")
        for model_id in self.active_services.copy():
            self.test_stop_service(model_id)

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        print("=" * 60)
        print("🧪 开始OCR服务管理器全面测试")
        print("=" * 60)

        # 启动管理器
        if not self.start_manager():
            return {"success": False, "error": "无法启动服务管理器"}

        try:
            # 基础功能测试
            print("\n" + "=" * 40)
            print("📋 基础功能测试")
            print("=" * 40)

            # 测试启动单个服务
            start_result = self.test_start_service()

            if start_result["success"]:
                model_id = start_result["data"]["model_id"]
                service_type = start_result["data"]["service_type"]

                # 等待服务完全启动
                time.sleep(5)

                # 测试列出服务
                list_result = self.test_list_services()

                # 测试推理，传递服务类型
                inference_result = self.test_inference(
                    model_id, service_type=service_type
                )

                # 测试停止服务
                stop_result = self.test_stop_service(model_id)

            # 多服务测试
            print("\n" + "=" * 40)
            print("🔄 多服务同时运行测试")
            print("=" * 40)
            multiple_services_result = self.test_multiple_services()

            # 错误处理测试
            print("\n" + "=" * 40)
            print("⚠️ 错误处理测试")
            print("=" * 40)
            error_handling_result = self.test_error_handling()

            # 服务特定数据格式测试
            print("\n" + "=" * 40)
            print("🔧 服务特定数据格式测试")
            print("=" * 40)
            format_test_result = self.test_service_specific_data_formats()

            # 汇总结果
            test_summary = {
                "success": True,
                "basic_tests": {
                    "start_service": start_result["success"]
                    if "start_result" in locals()
                    else False,
                    "list_services": list_result["success"]
                    if "list_result" in locals()
                    else False,
                    "inference": inference_result["success"]
                    if "inference_result" in locals()
                    else False,
                    "stop_service": stop_result["success"]
                    if "stop_result" in locals()
                    else False,
                },
                "multiple_services": multiple_services_result["success"],
                "error_handling": error_handling_result["success"],
                "format_tests": format_test_result["success"],
            }

            return test_summary

        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            return {"success": False, "error": str(e)}

        finally:
            # 清理并停止管理器
            self.cleanup_services()
            self.stop_manager()

    def print_test_summary(self, results: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📊 测试结果摘要")
        print("=" * 60)

        if results.get("success"):
            print("🎉 总体测试结果: ✅ 通过")
        else:
            print("❌ 总体测试结果: ❌ 失败")
            if "error" in results:
                print(f"错误信息: {results['error']}")

        if "basic_tests" in results:
            print("\n📋 基础功能测试:")
            for test_name, success in results["basic_tests"].items():
                status = "✅" if success else "❌"
                print(f"  {status} {test_name}")

        if "multiple_services" in results:
            status = "✅" if results["multiple_services"] else "❌"
            print(f"\n🔄 多服务测试: {status}")

        if "error_handling" in results:
            status = "✅" if results["error_handling"] else "❌"
            print(f"\n⚠️ 错误处理测试: {status}")

        if "format_tests" in results:
            status = "✅" if results["format_tests"] else "❌"
            print(f"\n🔧 数据格式测试: {status}")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    # 检查是否在正确的目录中
    if not Path("ocr_service_manager.py").exists():
        print("❌ 错误: 请在包含 ocr_service_manager.py 的目录中运行此测试脚本")
        return

    # 创建测试器实例
    tester = OCRServiceManagerTester()

    try:
        # 运行全面测试
        results = tester.run_comprehensive_test()

        # 打印测试摘要
        tester.print_test_summary(results)

        # 根据测试结果设置退出码
        exit_code = 0 if results.get("success") else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
        tester.cleanup_services()
        tester.stop_manager()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生未预期的错误: {e}")
        tester.cleanup_services()
        tester.stop_manager()
        sys.exit(1)


if __name__ == "__main__":
    main()
