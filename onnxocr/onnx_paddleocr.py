import time
import copy
import numpy as np
import cv2
from .utils import infer_args as init_args
from .utils import get_rotate_crop_image, get_minarea_rect_crop
from .predict_det import TextDetector
from .predict_cls import TextClassifier
from .predict_rec import TextRecognizer
import argparse


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


class ONNXPaddleOcr:
    def __init__(self, use_angle_cls=True, use_gpu=False, **kwargs):
        params = init_args()
        params.rec_image_shape = "3, 48, 320"
        params.use_angle_cls = use_angle_cls
        params.use_gpu = use_gpu
        
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)

        # 初始化模型组件
        self.text_detector = TextDetector(params)
        self.text_recognizer = TextRecognizer(params)
        if params.use_angle_cls:
            self.text_classifier = TextClassifier(params)
        else:
            self.text_classifier = None

        self.args = params
        self.use_angle_cls = params.use_angle_cls
        self.drop_score = params.drop_score
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        import os
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir, f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno],
            )
        self.crop_image_res_index += bbox_num

    def run(self, img, cls=True):
        """执行OCR流程（检测+分类+识别）"""
        ori_im = img
        # 文字检测
        dt_boxes, det_time = self.text_detector.run(img)

        if dt_boxes is None or len(dt_boxes) == 0:
            return None, None, det_time, 0

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)

        # 图片裁剪
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 方向分类
        rec_time = 0
        if self.use_angle_cls and cls and self.text_classifier is not None:
            img_crop_list, angle_list = self.text_classifier.run(img_crop_list)

        # 图像识别
        if len(img_crop_list) > 0:
            rec_res, rec_time = self.text_recognizer.run(img_crop_list)
        else:
            rec_res = []

        if hasattr(self.args, 'save_crop_res') and self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res, det_time, rec_time

    def ocr(self, img, cls=True):
        """完整的OCR流程"""
        if cls and not self.use_angle_cls:
            print("Since the angle classifier is not initialized, the angle classifier will not be used during the forward process")
        
        dt_boxes, rec_res, det_time, rec_time = self.run(img, cls)
        if dt_boxes is None:
            return [None]
        
        # 格式化输出结果
        result = []
        for box, res in zip(dt_boxes, rec_res):
            result.append([box, res])
        
        return [result]

    def detection_only(self, img):
        """仅执行文本检测"""
        dt_boxes, det_time = self.text_detector.run(img)
        return dt_boxes

    def classification_only(self, img_crop_list):
        """仅执行角度分类"""
        if self.use_angle_cls and self.text_classifier is not None:
            return self.text_classifier.run(img_crop_list)
        else:
            return img_crop_list, [(0, 1.0) for _ in img_crop_list]

    def recognition_only(self, img_crop_list):
        """仅执行文本识别"""
        if len(img_crop_list) > 0:
            rec_res, rec_time = self.text_recognizer.run(img_crop_list)
            return rec_res
        else:
            return []
