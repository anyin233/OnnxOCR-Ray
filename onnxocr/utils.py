import numpy as np
import cv2
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 获取当前文件所在的目录
module_dir = Path(__file__).resolve().parent


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string

    count_zh = count_pu = 0
    s_len = len(str(s))
    en_dg_count = 0
    for c in str(s):
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(
    texts,
    scores,
    img_h=400,
    img_w=600,
    threshold=0.0,
    font_path=str(module_dir / "fonts/simfang.ttf"),
):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1 :] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[: img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ": " + txt
                first_line = False
            else:
                new_txt = "    " + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4 :]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ": " + txt + "   " + "%.3f" % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + "%.3f" % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def draw_ocr(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=str(module_dir / "fonts/simfang.ttf"),
):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path,
        )
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def base64_to_cv2(b64str):
    import base64

    data = base64.b64decode(b64str.encode("utf8"))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def str2bool(v):
    return v.lower() in ("true", "t", "1")


import yaml
from pathlib import Path
from types import SimpleNamespace
import os

# 假设 module_dir 在某处定义
module_dir = Path(__file__).parent

def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def load_config_from_yaml(config_path="config.yaml"):
    """
    从YAML文件加载配置
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        配置对象，包含所有参数
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 扁平化配置字典，将嵌套的配置展开为单层
    flat_config = {}
    
    # 处理预测引擎参数
    if 'prediction_engine' in config_dict:
        flat_config.update(config_dict['prediction_engine'])
    
    # 处理文本检测参数
    if 'text_detector' in config_dict:
        flat_config.update(config_dict['text_detector'])
    
    # 处理各种算法参数
    for section in ['db_params', 'east_params', 'sast_params', 'pse_params', 'fce_params']:
        if section in config_dict:
            flat_config.update(config_dict[section])
    
    # 处理文本识别参数
    if 'text_recognizer' in config_dict:
        flat_config.update(config_dict['text_recognizer'])
    
    # 处理E2E参数
    if 'e2e' in config_dict:
        flat_config.update(config_dict['e2e'])
    
    # 处理PGNet参数
    if 'pgnet_params' in config_dict:
        flat_config.update(config_dict['pgnet_params'])
    
    # 处理文本分类参数
    if 'text_classifier' in config_dict:
        flat_config.update(config_dict['text_classifier'])
    
    # 处理性能参数
    if 'performance' in config_dict:
        flat_config.update(config_dict['performance'])
    
    # 处理SR参数
    if 'sr_params' in config_dict:
        flat_config.update(config_dict['sr_params'])
    
    # 处理输出参数
    if 'output' in config_dict:
        flat_config.update(config_dict['output'])
    
    # 处理多进程参数
    if 'multiprocess' in config_dict:
        flat_config.update(config_dict['multiprocess'])
    
    # 处理日志参数
    if 'logging' in config_dict:
        flat_config.update(config_dict['logging'])
    
    # 处理路径，将相对路径转换为绝对路径
    path_params = [
        'det_model_dir', 'rec_model_dir', 'cls_model_dir', 'sr_model_dir',
        'rec_char_dict_path', 'vis_font_path', 'e2e_char_dict_path',
        'draw_img_save_dir', 'crop_res_save_dir', 'save_log_path'
    ]
    
    for param in path_params:
        if param in flat_config and flat_config[param] is not None:
            if not os.path.isabs(flat_config[param]):
                flat_config[param] = str(module_dir / flat_config[param])
    
    # 转换为对象形式，便于访问
    args = SimpleNamespace(**flat_config)
    
    return args

def infer_args(config_path="config.yaml"):
    """
    加载推理配置参数
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        配置参数对象
    """
    try:
        args = load_config_from_yaml(config_path)
        return args
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        raise

