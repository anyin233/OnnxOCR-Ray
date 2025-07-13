import numpy as np
from .imaug import transform, create_operators
from .db_postprocess import DBPostProcess
from .predict_base import PredictBase
import time


class TextDetector(PredictBase):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode
        postprocess_params["box_type"] = args.det_box_type

        # 实例化预处理操作类
        self.preprocess_op = create_operators(pre_process_list)
        # 实例化后处理操作类
        self.postprocess_op = DBPostProcess(**postprocess_params)

        # 初始化模型
        self.det_onnx_session = self.get_onnx_session(args.det_model_dir, args.use_gpu)
        self.det_input_name = self.get_input_name(self.det_onnx_session)
        self.det_output_name = self.get_output_name(self.det_onnx_session)

    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = np.linalg.norm(np.cross(rightMost[0] - tl, rightMost[1] - tl)) / np.linalg.norm(rightMost[1] - rightMost[0])
        (br, tr) = rightMost[np.argsort(rightMost[:, 1]), :] if D > 0 else rightMost[np.argsort(rightMost[:, 1])[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def run(self, img):
        """串行执行文本检测"""
        start_time = time.time()
        ori_im = img.copy()
        data = {"image": img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        input_feed = self.get_input_feed(self.det_input_name, img)
        outputs = self.det_onnx_session.run(self.det_output_name, input_feed=input_feed)

        preds = {}
        preds["maps"] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]

        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        processing_time = time.time() - start_time
        return dt_boxes, processing_time

    def __call__(self, img):
        """支持直接调用"""
        dt_boxes, _ = self.run(img)
        return dt_boxes
