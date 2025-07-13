import cv2
import time
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from fastapi import HTTPException
import copy
from onnxocr.utils import get_rotate_crop_image, get_minarea_rect_crop
import logging
import os
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 OCR 模型（使用串行版本，降低drop_score）
model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False, drop_score=0.1)

# Detection 相关模型
class OCRDetectionRequest(BaseModel):
    image: str  # base64 编码的图像字符串

class OCRDetectionResponse(BaseModel):
    processing_time: float  # 处理时间
    bounding_boxes: List[List[List[float]]]  # 文本框坐标，格式为 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]

# Recognition 相关模型
class OCRRecognitionBoundingBox(BaseModel):
    angle: float
    bounding_box: List[List[float]]  # 文本框坐标，格式为 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]

class OCRRecognitionRequest(BaseModel):
    image: str  # base64 编码的图像字符串
    bounding_boxes: List[OCRRecognitionBoundingBox]  # 文本框坐标，格式为 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
    
class OCRRecognitionResult(BaseModel):
    text: str  # 识别的文本
    confidence: float  # 置信度
    bounding_box: List[List[float]]  # 文本框坐标，格式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

class OCRRecognitionResponse(BaseModel):
    processing_time: float  # 处理时间
    results: List[OCRRecognitionResult]  # OCR识别结果列表

# Classification 相关模型
class OCRClassificationRequest(BaseModel):
    image: str  # base64 编码的图像字符串
    bounding_boxes: List[List[List[float]]]  # 文本框坐标，格式为 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]

class OCRClassificationResult(BaseModel):
    angle: float  # 旋转角度
    confidence: float  # 置信度
    bounding_box: List[List[float]]  # 文本框坐标，格式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

class OCRClassificationResponse(BaseModel):
    processing_time: float  # 处理时间
    results: List[OCRClassificationResult]  # 角度分类结果列表

# 完整OCR流程的模型（可选）
class OCRRequest(BaseModel):
    image: str  # base64 编码的图像字符串

class OCRResponse(BaseModel):
    processing_time: float  # 处理时间
    results: List[OCRRecognitionResult]  # OCR识别结果列表


def decode_base64_image(image_base64: str):
    """解码base64图像的辅助函数"""
    try:
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64.")
        return img
    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")

def order_points_properly(pts):
    """
    正确排序四个点为矩形的四个角点
    返回顺序：左上、右上、右下、左下
    """
    pts = np.array(pts, dtype=np.float32)
    
    # 计算所有点的重心
    center = np.mean(pts, axis=0)
    
    # 将点分为左侧和右侧
    left_points = pts[pts[:, 0] < center[0]]
    right_points = pts[pts[:, 0] >= center[0]]
    
    # 如果左右分布不均匀，按y坐标分为上下
    if len(left_points) != 2 or len(right_points) != 2:
        top_points = pts[pts[:, 1] < center[1]]
        bottom_points = pts[pts[:, 1] >= center[1]]
        
        if len(top_points) == 2 and len(bottom_points) == 2:
            # 按x坐标排序上下两组点
            top_left = top_points[np.argmin(top_points[:, 0])]
            top_right = top_points[np.argmax(top_points[:, 0])]
            bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
            bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
            
            return np.array([top_left, top_right, bottom_right, bottom_left])
    else:
        # 在左侧点中，y坐标小的是左上，大的是左下
        left_top = left_points[np.argmin(left_points[:, 1])]
        left_bottom = left_points[np.argmax(left_points[:, 1])]
        
        # 在右侧点中，y坐标小的是右上，大的是右下
        right_top = right_points[np.argmin(right_points[:, 1])]
        right_bottom = right_points[np.argmax(right_points[:, 1])]
        
        return np.array([left_top, right_top, right_bottom, left_bottom])
    
    # 如果上述方法都不能正确分组，使用极角排序
    # 计算每个点相对于重心的极角
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 按极角排序
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # 找到最左上的点作为起始点
    distances_to_origin = np.sum(sorted_pts**2, axis=1)
    start_idx = np.argmin(distances_to_origin)
    
    # 从起始点开始重新排列
    reordered_pts = np.roll(sorted_pts, -start_idx, axis=0)
    
    return reordered_pts

def sorted_boxes(dt_boxes):
    """对检测框进行排序并修复坐标顺序"""
    num_boxes = dt_boxes.shape[0]
    
    # 首先修复每个检测框的坐标顺序
    fixed_boxes = []
    for box in dt_boxes:
        fixed_box = order_points_properly(box)
        fixed_boxes.append(fixed_box)
    
    fixed_boxes = np.array(fixed_boxes)
    
    # 然后按位置排序（从上到下，从左到右）
    sorted_boxes = sorted(fixed_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
               (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

@app.post("/detection", response_model=OCRDetectionResponse)
def detection_service(request: OCRDetectionRequest):
    """文本检测服务 - 检测图像中的文本区域"""
    try:
        logger.info("开始文本检测服务")
        
        # 解码图像
        img = decode_base64_image(request.image)
        logger.info(f"图像解码成功，尺寸: {img.shape}")
        
        # 执行文本检测
        start_time = time.time()
        dt_boxes = model.detection_only(img)
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"检测完成，耗时: {processing_time:.4f}秒")
        logger.info(f"检测到的文本框数量: {len(dt_boxes) if dt_boxes is not None else 0}")
        
        # 格式化检测结果
        bounding_boxes = []
        if dt_boxes is not None and len(dt_boxes) > 0:
            logger.info("开始格式化检测结果")
            # 对检测框进行排序
            sorted_result = sorted_boxes(dt_boxes)
            for i, box in enumerate(sorted_result):
                # 将检测框转换为指定格式 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
                bounding_box = np.array(box).reshape(4, 2).tolist()
                bounding_boxes.append(bounding_box)
                logger.info(f"检测框 {i+1}: {bounding_box}")
            
            # 绘制检测框并保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_save_path = f"/home/yanweiye/Project/OnnxOCR-Ray/result_img/detection_{timestamp}.jpg"
            saved_path = draw_bounding_boxes(img, bounding_boxes, save_path=detection_save_path)
            if saved_path:
                logger.info(f"检测结果已绘制并保存到: {saved_path}")
            else:
                logger.warning("检测结果绘制失败")
        else:
            logger.warning("未检测到任何文本框")
        
        return OCRDetectionResponse(
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )
        
    except Exception as e:
        logger.error(f"检测服务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/classification", response_model=OCRClassificationResponse)
def classification_service(request: OCRClassificationRequest):
    """文本角度分类服务 - 对文本区域进行角度分类"""
    try:
        # 解码图像
        img = decode_base64_image(request.image)
        
        # 根据边界框裁剪图像
        img_crop_list = []
        for bounding_box in request.bounding_boxes:
            box = np.array(bounding_box, dtype=np.float32)  # 确保是float32类型
            if model.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(img, box)
            else:
                img_crop = get_minarea_rect_crop(img, box)
            img_crop_list.append(img_crop)
        
        # 保存裁切图像
        crop_dir = "/home/yanweiye/Project/OnnxOCR-Ray/result_img/cropped"
        os.makedirs(crop_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, img_crop in enumerate(img_crop_list):
            crop_save_path = os.path.join(crop_dir, f"crop_cls_{timestamp}_{i+1}.jpg")
            cv2.imwrite(crop_save_path, img_crop)
            logger.info(f"裁切图像 {i+1} 已保存到: {crop_save_path}")
        # 执行角度分类
        start_time = time.time()
        if model.use_angle_cls and len(img_crop_list) > 0:
            img_crop_list, cls_results = model.classification_only(img_crop_list)
        else:
            cls_results = [(0, 1.0) for _ in img_crop_list]  # 默认角度为0，置信度为1.0
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 格式化分类结果
        results = []
        for i, (bounding_box, cls_result) in enumerate(zip(request.bounding_boxes, cls_results)):
            if isinstance(cls_result, (list, tuple)) and len(cls_result) >= 2:
                # cls_result 格式: [label, score] 其中 label 可能是 "0" 或 "180"
                label, confidence = cls_result[0], cls_result[1]
                angle = 180.0 if "180" in str(label) else 0.0
            else:
                angle, confidence = 0.0, 1.0
            results.append(OCRClassificationResult(
                angle=float(angle),
                confidence=float(confidence),
                bounding_box=bounding_box
            ))
        
        return OCRClassificationResponse(
            processing_time=processing_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/recognition", response_model=OCRRecognitionResponse)
def recognition_service(request: OCRRecognitionRequest):
    """文本识别服务 - 识别文本区域中的文字内容"""
    try:
        logger.info("开始文本识别服务")
        
        # 解码图像
        img = decode_base64_image(request.image)
        logger.info(f"图像解码成功，尺寸: {img.shape}")
        logger.info(f"接收到 {len(request.bounding_boxes)} 个边界框")
        
        # 根据边界框裁剪图像
        img_crop_list = []
        # 创建裁切图像保存目录
        crop_dir = "/home/yanweiye/Project/OnnxOCR-Ray/result_img/cropped"
        os.makedirs(crop_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, bounding_box_obj in enumerate(request.bounding_boxes):
            logger.info(f"处理边界框 {i+1}: {bounding_box_obj}")
            # 从 OCRRecognitionBoundingBox 对象中提取 bounding_box
            bounding_box = bounding_box_obj.bounding_box
            angle = bounding_box_obj.angle
            logger.info(f"边界框坐标: {bounding_box}, 角度: {angle}")
            
            box = np.array(bounding_box, dtype=np.float32)  # 确保是float32类型
            if model.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(img, box)
            else:
                img_crop = get_minarea_rect_crop(img, box)
            
            # 如果提供了角度信息，可能需要对裁切的图像进行旋转
            if angle != 0:
                # 获取图像中心点
                (h, w) = img_crop.shape[:2]
                center = (w // 2, h // 2)
                # 创建旋转矩阵
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                # 执行旋转
                img_crop = cv2.warpAffine(img_crop, rotation_matrix, (w, h))
                logger.info(f"图像 {i+1} 已根据角度 {angle}° 进行旋转")
            
            img_crop_list.append(img_crop)
            logger.info(f"裁剪图像 {i+1} 尺寸: {img_crop.shape}")
            
            # 保存裁切后的图像
            crop_save_path = os.path.join(crop_dir, f"crop_rec_{timestamp}_{i+1}.jpg")
            cv2.imwrite(crop_save_path, img_crop)
            logger.info(f"裁切图像 {i+1} 已保存到: {crop_save_path}")
        
        # 执行文本识别
        start_time = time.time()
        if len(img_crop_list) > 0:
            rec_results = model.recognition_only(img_crop_list)
            logger.info(f"识别完成，结果数量: {len(rec_results)}")
        else:
            rec_results = []
            logger.warning("没有图像需要识别")
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"识别耗时: {processing_time:.4f}秒")
        
        # 格式化识别结果
        results = []
        valid_boxes = []  # 用于绘制的边界框
        valid_texts = []  # 用于绘制的文本
        
        for i, (bounding_box_obj, rec_result) in enumerate(zip(request.bounding_boxes, rec_results)):
            text, confidence = rec_result if isinstance(rec_result, tuple) else ("", 0.0)
            logger.info(f"识别结果 {i+1}: 文本='{text}', 置信度={confidence:.4f}")
            
            # 从 OCRRecognitionBoundingBox 对象中提取 bounding_box
            bounding_box = bounding_box_obj.bounding_box
            
            # 过滤低置信度的结果
            if confidence >= model.drop_score:
                logger.info(f"结果 {i+1} 通过置信度过滤 (>= {model.drop_score})")
                results.append(OCRRecognitionResult(
                    text=text,
                    confidence=float(confidence),
                    bounding_box=bounding_box
                ))
                # 收集用于绘制的数据
                valid_boxes.append(bounding_box)
                valid_texts.append(f"{text}({confidence:.2f})")
            else:
                logger.warning(f"结果 {i+1} 被置信度过滤 (< {model.drop_score})")
        
        # 绘制识别结果并保存
        if valid_boxes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recognition_save_path = f"/home/yanweiye/Project/OnnxOCR-Ray/result_img/recognition_{timestamp}.jpg"
            saved_path = draw_bounding_boxes(img, valid_boxes, valid_texts, save_path=recognition_save_path)
            if saved_path:
                logger.info(f"识别结果已绘制并保存到: {saved_path}")
            else:
                logger.warning("识别结果绘制失败")
        else:
            logger.info("没有通过置信度过滤的识别结果，跳过绘制")
        
        logger.info(f"最终返回 {len(results)} 个识别结果")
        return OCRRecognitionResponse(
            processing_time=processing_time,
            results=results
        )
    except Exception as e:
        logger.error(f"识别服务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.post("/inference", response_model=OCRResponse)
def ocr_service(request: OCRRequest):
    """完整的OCR服务 - 集成检测、分类和识别功能"""
    try:
        logger.info("开始完整OCR服务")
        
        # 解码图像
        img = decode_base64_image(request.image)
        logger.info(f"图像解码成功，尺寸: {img.shape}")
        
        # 执行完整的OCR流程
        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"OCR完成，耗时: {processing_time:.4f}秒")
        logger.info(f"OCR结果类型: {type(result)}")
        
        if result:
            logger.info(f"OCR结果长度: {len(result)}")
            if len(result) > 0 and result[0]:
                logger.info(f"第一层结果长度: {len(result[0])}")
        
        # 格式化结果
        ocr_results = []
        if result and len(result) > 0 and result[0]:
            logger.info("开始格式化OCR结果")
            for i, line in enumerate(result[0]):
                logger.info(f"处理第 {i+1} 个结果: {line}")
                
                # 确保 line[0] 是 NumPy 数组或列表
                if isinstance(line[0], (list, np.ndarray)):
                    # 将 bounding_box 转换为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 格式
                    bounding_box = np.array(line[0]).reshape(4, 2).tolist()
                else:
                    bounding_box = []
                    logger.warning(f"第 {i+1} 个结果的边界框格式异常: {line[0]}")

                text = line[1][0] if len(line[1]) > 0 else ""
                confidence = float(line[1][1]) if len(line[1]) > 1 else 0.0
                
                logger.info(f"结果 {i+1}: 文本='{text}', 置信度={confidence:.4f}")
                
                ocr_results.append(OCRRecognitionResult(
                    text=text,  # 识别文本
                    confidence=confidence,  # 置信度
                    bounding_box=bounding_box,  # 文本框坐标
                ))
        else:
            logger.warning("OCR未返回任何结果")

        logger.info(f"最终返回 {len(ocr_results)} 个OCR结果")
        
        # 返回结果
        return OCRResponse(
            processing_time=processing_time, 
            results=ocr_results
        )

    except Exception as e:
        logger.error(f"OCR服务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR service failed: {str(e)}")
        logger.error(f"OCR service failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR service failed: {str(e)}")


@app.post("/debug", response_model=dict)
def debug_service(request: OCRRequest):
    """调试服务 - 返回详细的处理信息"""
    try:
        logger.info("开始调试服务")
        
        # 解码图像
        img = decode_base64_image(request.image)
        logger.info(f"图像解码成功，尺寸: {img.shape}")
        
        debug_info = {
            "image_shape": img.shape,
            "model_config": {
                "drop_score": model.drop_score,
                "use_angle_cls": model.use_angle_cls,
                "det_box_type": model.args.det_box_type
            },
            "steps": {}
        }
        
        # 步骤1: 检测
        logger.info("执行检测步骤")
        dt_boxes = model.detection_only(img)
        debug_info["steps"]["detection"] = {
            "raw_boxes_count": len(dt_boxes) if dt_boxes is not None else 0,
            "raw_boxes": dt_boxes.tolist() if dt_boxes is not None else []
        }
        
        if dt_boxes is None or len(dt_boxes) == 0:
            debug_info["final_result"] = "检测阶段未找到文本框"
            return debug_info
        
        # 步骤2: 修复和排序坐标
        logger.info("修复和排序坐标")
        sorted_result = sorted_boxes(dt_boxes)
        debug_info["steps"]["sorted_boxes"] = [box.tolist() for box in sorted_result]
        
        # 步骤3: 裁剪图像
        logger.info("裁剪图像")
        img_crop_list = []
        crop_info = []
        
        for i, box in enumerate(sorted_result):
            box_array = np.array(box, dtype=np.float32)
            if model.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(img, box_array)
            else:
                img_crop = get_minarea_rect_crop(img, box_array)
            
            img_crop_list.append(img_crop)
            
            # 保存裁剪图像用于调试
            cv2.imwrite(f'/home/yanweiye/Project/OnnxOCR-Ray/debug_crop_{i+1}.jpg', img_crop)
            
            crop_info.append({
                "index": i+1,
                "crop_shape": img_crop.shape,
                "saved_as": f'debug_crop_{i+1}.jpg'
            })
        
        debug_info["steps"]["cropping"] = crop_info
        
        # 步骤4: 角度分类
        logger.info("执行角度分类")
        if model.use_angle_cls and len(img_crop_list) > 0:
            img_crop_list, cls_results = model.classification_only(img_crop_list)
            debug_info["steps"]["classification"] = [
                {"angle": cls[0], "confidence": float(cls[1])} 
                for cls in cls_results
            ]
        else:
            cls_results = [(0, 1.0) for _ in img_crop_list]
            debug_info["steps"]["classification"] = "跳过角度分类"
        
        # 步骤5: 文本识别
        logger.info("执行文本识别")
        if len(img_crop_list) > 0:
            rec_results = model.recognition_only(img_crop_list)
            
            recognition_info = []
            final_results = []
            
            for i, (box, rec_result) in enumerate(zip(sorted_result, rec_results)):
                text, confidence = rec_result if isinstance(rec_result, tuple) else ("", 0.0)
                
                rec_info = {
                    "index": i+1,
                    "text": text,
                    "confidence": float(confidence),
                    "passed_filter": confidence >= model.drop_score
                }
                recognition_info.append(rec_info)
                
                logger.info(f"识别结果 {i+1}: '{text}', 置信度={confidence:.4f}, 通过过滤={confidence >= model.drop_score}")
                
                if confidence >= model.drop_score:
                    final_results.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bounding_box": box.tolist()
                    })
            
            debug_info["steps"]["recognition"] = recognition_info
            debug_info["final_results_count"] = len(final_results)
            debug_info["final_results"] = final_results
        else:
            debug_info["steps"]["recognition"] = "没有图像需要识别"
            debug_info["final_results_count"] = 0
        
        return debug_info
        
    except Exception as e:
        logger.error(f"调试服务失败: {str(e)}", exc_info=True)
        return {"error": str(e)}


def draw_bounding_boxes(img, bounding_boxes, texts=None, save_path=None):
    """
    在图像上绘制边界框
    
    Args:
        img: 输入图像
        bounding_boxes: 边界框列表，格式为 [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
        texts: 可选的文本列表，用于在边界框上标注识别的文字
        save_path: 保存路径，如果不提供则自动生成
    
    Returns:
        保存的文件路径
    """
    try:
        # 复制原始图像以避免修改原图
        img_draw = img.copy()
        
        # 确保结果目录存在
        result_dir = "/home/yanweiye/Project/OnnxOCR-Ray/result_img"
        os.makedirs(result_dir, exist_ok=True)
        
        # 如果没有提供保存路径，自动生成
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(result_dir, f"draw_boxes_{timestamp}.jpg")
        
        # 绘制边界框
        for i, box in enumerate(bounding_boxes):
            # 转换为numpy数组并确保是整数坐标
            pts = np.array(box, dtype=np.int32)
            
            # 绘制边界框
            cv2.polylines(img_draw, [pts], True, (0, 255, 0), 2)
            
            # 如果提供了文本，在边界框上方标注
            if texts and i < len(texts):
                text = texts[i]
                # 获取文本框左上角坐标
                text_x, text_y = pts[0]
                
                # 绘制文本背景
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img_draw, 
                            (text_x, text_y - text_size[1] - 10), 
                            (text_x + text_size[0], text_y), 
                            (0, 255, 0), -1)
                
                # 绘制文本
                cv2.putText(img_draw, text, 
                          (text_x, text_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 保存图像
        cv2.imwrite(save_path, img_draw)
        logger.info(f"边界框绘制完成，保存到: {save_path}")
        
        return save_path
        
    except Exception as e:
        logger.error(f"绘制边界框失败: {str(e)}")
        return None


if __name__ == "__main__":
    import uvicorn
    # 启动 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=5005)
