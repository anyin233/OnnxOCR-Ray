import time

# from .predict_system import TextSystem
from .utils import infer_args as init_args
from .utils import str2bool, draw_ocr
import argparse
import sys

import ray
import fastapi

from pydantic import BaseModel

import base64
import numpy as np

import cv2


import os
import copy
from . import predict_det
from . import predict_cls
from . import predict_rec
from .utils import get_rotate_crop_image, get_minarea_rect_crop



class OCRRequest(BaseModel):
    img: str  # Base64 encoded image
    det: bool = True  # Whether to perform detection
    rec: bool = True  # Whether to perform recognition
    cls: bool = True  # Whether to use angle classification


class OCRResponse(BaseModel):
    ocr_res: list  # List of OCR results
    det_res: list
    cls_res: list
    received_time: float
    finish_time: float
    rec_time: float = 0.0
    det_time: float = 0.0


app = fastapi.FastAPI()



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


parser = init_args()
inference_args_dict = {}
for action in parser._actions:
    inference_args_dict[action.dest] = action.default
params = argparse.Namespace(**inference_args_dict)
params.rec_image_shape = "3, 48, 320"


detector = predict_det.TextDetector.bind(params)
recognizer = predict_rec.TextRecognizer.bind(params)
if params.use_angle_cls:
    classifier = predict_cls.TextClassifier.bind(params)
else:
    classifier = None


@ray.serve.deployment(
    name="onnx_paddle_ocr",
    ray_actor_options={"num_cpus": 0, "num_gpus": 0},
)
@ray.serve.ingress(app)
class ONNXPaddleOcr:
    def __init__(self, params, detector=None, recognizer=None, classifier=None, **kwargs):
        # 初始化模型
        self.use_angle_cls = params.use_angle_cls
        self.drop_score = params.drop_score
        self.text_detector = detector
        self.text_recognizer = recognizer
        if params.use_angle_cls:
            self.text_classifier = classifier

        self.args = params
        self.crop_image_res_index = 0

        self.sorted_boxes = sorted_boxes

    async def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir, f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno],
            )

        self.crop_image_res_index += bbox_num

    async def run(self, img, cls=True):
        ori_im = img
        # 文字检测
        dt_boxes, det_time = await self.text_detector.run.remote(img)

        if dt_boxes is None:
            return None, None

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
        if self.use_angle_cls and cls:
            img_crop_list, angle_list = await self.text_classifier.run.remote(img_crop_list)

        # 图像识别
        rec_res, rec_time = await self.text_recognizer.run.remote(img_crop_list)

        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res, det_time, rec_time

    @app.post("/ocr")
    async def ocr(self, request: OCRRequest) -> OCRResponse:
        cls = request.cls
        det = request.det
        rec = request.rec
        img = request.img

        response = OCRResponse(ocr_res=[], det_res=[], cls_res=[], received_time=time.time(), finish_time=0)
        # Decode base64 image to cv2 format
        img_data = base64.b64decode(img)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if cls and not self.use_angle_cls:
            print(
                "Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process"
            )
        
        if det and rec:
            dt_boxes, rec_res, det_time, rec_time = await self.run(img, cls)
            tmp_res = [{"boxes": box.tolist(), "res": res} for box, res in zip(dt_boxes, rec_res)]
            response.ocr_res.append(tmp_res)
            response.det_time = det_time
            response.rec_time = rec_time
        elif det and not rec:
            dt_boxes, det_time = await self.text_detector.run.remote(img)
            tmp_res = [{"boxes": box.tolist()} for box in dt_boxes]
            response.det_res.append(tmp_res)
            response.det_time = det_time
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = await self.text_classifier.run.remote(img)
                if not rec:
                    response.cls_res.append(cls_res_tmp)
            rec_res, rec_time = await self.text_recognizer.run.remote(img)
            response.ocr_res.append(rec_res)
            
            response.rec_time = rec_time

        response.finish_time = time.time()
        return response


ocr_app = ONNXPaddleOcr.bind(
    params, detector=detector, recognizer=recognizer, classifier=classifier if params.use_angle_cls else None
)
