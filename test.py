import requests
from datasets import load_dataset
import signal
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import aiofiles


PROFILER_HOST = "http://localhost"
PROFILER_PORT = 8091

# OCR服务配置
OCR_SERVICE_URL = "http://localhost:5005/ocr"


import asyncio

import httpx
from rich.progress import track
import io
import random
import base64


# Start profiler
def run_profiler():
    profiler_start_url = f"{PROFILER_HOST}:{PROFILER_PORT}/profiling/start"
    request_body = {
        "name": "ocr_benchmark",
    }
    profiler_start_response = requests.post(profiler_start_url, json=request_body)
    if profiler_start_response.status_code != 200:
        print("Failed to start profiler")
        exit(1)


def stop_profiler():
    profiler_stop_url = f"{PROFILER_HOST}:{PROFILER_PORT}/profiling/stop"
    request_body = {
        "name": "ocr_benchmark",
    }
    profiler_stop_response = requests.post(profiler_stop_url, json=request_body)
    if profiler_stop_response.status_code != 200:
        print("Failed to stop profiler")
        exit(1)


def submit_ocr_task(url, image_bytes):
    submit_time = time.time()

    # 将图像字节转换为base64编码
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    # 使用JSON格式发送请求，符合OCR服务的接口要求
    payload = {"image": image_base64}
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error submitting OCR task: {response.status_code} - {response.text}")
        return response, submit_time, None, None
    received_time = response.json().get("received", None)
    finish_time = time.time()
    return response, submit_time, finish_time, received_time


async def test_demo_image():
    """测试demo_text_ocr.jpg文件"""
    demo_image_path = "/home/yanweiye/Project/OnnxOCR-Ray/demo_text_ocr.jpg"

    try:
        # 读取demo图像文件
        with open(demo_image_path, "rb") as f:
            image_data = f.read()

        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # 准备请求数据
        payload = {"image": image_base64}

        print(f"正在测试demo图像: {demo_image_path}")
        start_time = time.time()

        # 发送OCR请求
        response = requests.post(OCR_SERVICE_URL, json=payload)

        end_time = time.time()
        processing_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"OCR识别成功！处理时间: {processing_time:.3f}秒")
            print(f"服务器处理时间: {result.get('processing_time', 'N/A')}秒")
            print(f"识别到 {len(result.get('results', []))} 个文本区域:")

            for i, text_result in enumerate(result.get("results", []), 1):
                print(f"  {i}. 文本: '{text_result.get('text', '')}'")
                print(f"     置信度: {text_result.get('confidence', 0):.4f}")
                print(f"     边界框: {text_result.get('bounding_box', [])}")
                print()
        else:
            print(f"OCR请求失败: {response.status_code} - {response.text}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {demo_image_path}")
    except Exception as e:
        print(f"测试demo图像时发生错误: {e}")


async def main():
    # 首先测试demo图像
    await test_demo_image()

    # 询问用户是否继续进行批量测试
    user_input = input("\n是否继续进行批量OCR测试？(y/n): ").strip().lower()
    if user_input != "y":
        print("退出程序")
        return

    responses = []

    failed = []
    image_byte_list = []
    ds = load_dataset("getomni-ai/ocr-benchmark", split="test")

    # Define helper function for processing a single image
    def process_image(index, sample):
        try:
            img_byte_arr = io.BytesIO()
            sample["image"].save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            return (index, img_byte_arr)
        except Exception as e:
            print(f"Error processing image {sample['image']}: {e}")
            return (index, None)

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        # Submit all tasks
        for index, sample in enumerate(ds):
            futures.append(executor.submit(process_image, index, sample))

        # Process results with progress tracking
        for future in track(futures, description="Processing images...", total=len(ds)):
            index, img_bytes = future.result()
            if img_bytes is None:
                failed.append(index)
            else:
                image_byte_list.append(img_bytes)

    # run_profiler()

    # 随机打乱图片顺序
    random.shuffle(image_byte_list)

    # 计算每个请求的发送时间间隔
    total_time = 5.0  # 总时间窗口（秒）
    if len(image_byte_list) > 0:
        interval = total_time / len(image_byte_list)
    else:
        interval = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        submit_timestamp = []
        finish_timestamp = []
        received_timestamp = []

        # 在指定时间窗口内提交所有任务
        for index, image_bytes in enumerate(image_byte_list):
            # 计算当前任务应该在什么时候发送
            target_time = start_time + (index * interval)
            current_time = time.time()

            # 如果还没到发送时间，等待
            if current_time < target_time:
                time.sleep(target_time - current_time)

            futures.append(executor.submit(submit_ocr_task, OCR_SERVICE_URL, image_bytes))

            # Process results with progress tracking
        for future in track(futures, description="Receiving OCR requests...", total=len(image_byte_list)):
            response, submit_time, finish_time, received_time = future.result()
            if response.status_code != 200:
                failed.append(index)
            else:
                responses.append(response.json())
                submit_timestamp.append(submit_time)
                finish_timestamp.append(finish_time)
                received_timestamp.append(received_time)

    # for image in track(image_byte_list, description="OCR images...", total=len(image_byte_list)):
    #   payload = {"image": image}
    #   requests.post(url, files=payload)
    # print("Status Code:", response.status_code)
    # stop_profiler()
    print("Failed images:", failed)
    async with aiofiles.open("timestamp.csv", "w") as f:
        await f.write("submit_time,finish_time,received_time\n")
        for submit, finish, receive in zip(submit_timestamp, finish_timestamp, received_timestamp):
            await f.write(f"{submit},{finish},{receive}\n")


def handler(signum, frame):
    print("\n收到 SIGINT 信号，执行清理操作...")
    stop_profiler()
    # 在这里做清理工作
    sys.exit(0)


if __name__ == "__main__":
    # 注册 SIGINT 的处理函数
    signal.signal(signal.SIGINT, handler)
    asyncio.run(main())
