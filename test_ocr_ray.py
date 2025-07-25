import requests
from datasets import load_dataset
import signal
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import aiofiles

# OCR服务配置
OCR_SERVICE_URL = "http://localhost:8000/ocr"



from rich.progress import track
import io
import random
import base64
import requests

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="OCR Benchmark Test")

    parser.add_argument("-t", "--total_time", type=float, default=2.0)
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of worker threads")

    args = parser.parse_args()
    return args


def submit_ocr_task(url, image_bytes):
    

    # 将图像字节转换为base64编码
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    # 使用JSON格式发送请求，符合OCR服务的接口要求
    payload = {"image": image_base64}
    # submit_time = time.time()
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error submitting OCR task: {response.status_code} - {response.text}")
        return response, None, None, None
    return response.json()


def main():
    # Send single file to serve first
    image_path = "draw_ocr.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = submit_ocr_task(OCR_SERVICE_URL, io.BytesIO(image_bytes))
    print(f"Single image OCR response: {response}")
    args = parse_args()
    print(f"Starting OCR benchmark with total time: {args.total_time} seconds and {args.workers} workers...")
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
    total_time = args.total_time  # 总时间窗口（秒）
    if len(image_byte_list) > 0:
        interval = total_time / len(image_byte_list)
    else:
        interval = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        submit_timestamp = []
        finish_timestamp = []
        received_timestamp = []
        rec_timestamp = []
        det_timestamp = []

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
            response = future.result()
            if response.status_code != 200:
                failed.append(index)
            else:
                responses.append(response.json())

    # # for image in track(image_byte_list, description="OCR images...", total=len(image_byte_list)):
    # #   payload = {"image": image}
    # #   requests.post(url, files=payload)
    # # print("Status Code:", response.status_code)
    # # stop_profiler()
    # print("Failed images:", failed)
    # with open("timestamp.csv", "w") as f:
    #     f.write("submit_time,finish_time,received_time,rec_time,det_time\n")
    #     for submit, finish, receive, rec, det in zip(submit_timestamp, finish_timestamp, received_timestamp, rec_timestamp, det_timestamp):
    #         f.write(f"{submit},{finish},{receive},{rec},{det}\n")


# def handler(signum, frame):
#     print("\n收到 SIGINT 信号，执行清理操作...")
#     # 在这里做清理工作
#     sys.exit(0)


if __name__ == "__main__":
    # 注册 SIGINT 的处理函数
    # signal.signal(signal.SIGINT, handler)
    main()