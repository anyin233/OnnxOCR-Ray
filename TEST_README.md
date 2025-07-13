# FastAPI OCR服务测试指南

本目录包含了用于测试FastAPI OCR服务的多个测试脚本。

## 服务启动

首先确保FastAPI服务正在运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app-service.py

# 或者使用uvicorn命令
uvicorn app-service:app --host 0.0.0.0 --port 5005

# 或者使用启动脚本
python start_fastapi.py
```

服务启动后可访问：
- 主页：http://localhost:5005/
- API文档：http://localhost:5005/docs
- 备用文档：http://localhost:5005/redoc

## 测试脚本说明

### 1. `test_service.py` - 简单测试脚本

最基础的测试脚本，对应原来的`test_ocr.py`功能：

```bash
python test_service.py
```

功能：
- 测试单个图像文件（默认：demo_text_ocr.jpg）
- 显示识别结果和性能统计
- 简单易用，适合快速验证

### 2. `test_fastapi_client.py` - 完整客户端测试

功能最全面的测试脚本：

```bash
python test_fastapi_client.py
```

功能：
- 支持多种图像输入方式（文件路径、OpenCV图像）
- 测试多个图像文件
- 详细的错误处理和状态检查
- 完整的结果显示

### 3. `batch_test.py` - 批量测试脚本

用于性能测试和批量验证：

```bash
python batch_test.py
```

功能：
- 自动发现测试图像文件
- 批量处理多个图像
- 生成详细的性能报告
- 统计成功率和平均耗时

## API使用示例

### Python requests示例

```python
import base64
import requests

# 读取图像并转换为base64
with open('demo_text_ocr.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# 发送请求
response = requests.post(
    'http://localhost:5005/ocr',
    json={'image': image_base64}
)

# 处理结果
if response.status_code == 200:
    result = response.json()
    print(f"处理时间: {result['processing_time']:.3f}秒")
    for item in result['results']:
        print(f"文本: {item['text']}")
        print(f"置信度: {item['confidence']:.4f}")
```

### curl命令示例

```bash
# 将图像转换为base64
IMAGE_BASE64=$(base64 -w 0 demo_text_ocr.jpg)

# 发送请求
curl -X POST "http://localhost:5005/ocr" \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"$IMAGE_BASE64\"}"
```

## 响应格式

### 成功响应

```json
{
  "processing_time": 0.123,
  "results": [
    {
      "text": "识别的文本",
      "confidence": 0.9876,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

### 错误响应

```json
{
  "detail": "错误信息"
}
```

## 性能对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| 直接调用 (test_ocr.py) | 速度最快，无网络开销 | 需要加载模型，内存占用大 |
| HTTP服务 | 模型只加载一次，支持并发 | 有网络开销，需要base64编码 |

## 故障排除

### 1. 连接失败
- 确保服务正在运行：`python app-service.py`
- 检查端口是否被占用：`netstat -an | grep 5005`

### 2. 图像识别失败
- 检查图像文件是否存在且可读
- 确认图像格式支持（jpg, png, bmp等）
- 查看服务日志中的错误信息

### 3. 性能问题
- 首次请求较慢是正常的（模型初始化）
- 大图像处理时间较长
- 可以调整图像尺寸来平衡速度和精度

## 扩展使用

这些测试脚本可以作为基础，扩展支持：
- 不同的图像预处理
- 批量文件处理
- 结果后处理和可视化
- 性能基准测试
- 自动化CI/CD测试
