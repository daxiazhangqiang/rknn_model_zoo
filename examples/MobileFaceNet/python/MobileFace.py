# 帮我把这个脚本做以下修改：
# 1.输入可以指定一个图像文件夹， 然后对文件夹下的所有图像进行人脸检测和关键点标注，输出结果图像到指定的输出文件夹用于保存人脸检测的结果。
# 2.对检测到的人脸，使用5个关键点做仿射变换对齐，并保存对齐后的人脸图像到指定的输出文件夹，该文件夹的设计目的是为了保存112x112的对齐人脸图像。
# 3.在检测到的人脸上，使用MobileFaceNet模型提取人脸特征向量，并将特征向量保存到指定的输出文件夹中，每个特征向量保存为一个.bin文件，bin文件按照0,1,2,3
# 的顺序不断叠加，上限为20个，超过20个就循环覆盖，该文件夹内保存一个全局json，json的信息用于保存人脸特征向量和人的一一对应关系，JSON的键值对中键用图像名，值为其对应的bin文件，后续该json可以用于人脸识别的比对。
# 4. 我当前实现的是注册这套机制，用来是从图像中提取人脸特征向量并保存到本地，后续可以使用这些特征向量做比对。这个过程中存在的问题是bin文件会一直增大，
# 而循环覆盖又会导致历史注册的人丢失，我的使用场景是家庭场景，最多不会超过100个人， 所以考虑到100个bin文件的大小，我是否应该在板端运行时做覆盖机制。


import os
import json
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2
from math import ceil
from itertools import product as product

from rknn.api import RKNN

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def PriorBox(image_size): #image_size Support (320,320) and (640,640)
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors).reshape(-1, 4)
    print("image_size:",image_size," num_priors=",output.shape[0])
    return output

def box_decode(loc, priors):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    variances = [0.1, 0.2]
    landmarks = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    ), axis=1)
    return landmarks

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# face recognition demo
# 标准112x112人脸的5个关键点参考坐标（InsightFace标准）
REFERENCE_KPS = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041]   # 右嘴角
], dtype=np.float32)

def align_face(img, kps, output_size=(112, 112)):
    """根据5个关键点做仿射变换对齐"""
    kps = np.array(kps, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(kps, REFERENCE_KPS, method=cv2.RANSAC)
    aligned = cv2.warpAffine(img, M, output_size, borderValue=0.0)
    return aligned

def recognition_preprocess(aligned_face_bgr):
    """对齐后人脸 -> 识别模型输入。

    归一化 (mean=127.5, std=127.5) 已在转换时烧进 mobilefacenet.rknn，
    这里只需 BGR -> RGB，保持 uint8 输入即可。
    """
    return cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)


def l2_normalize(embedding):
    """输出特征向量 -> L2 归一化"""
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(feat1, feat2):
    """计算两个 L2 归一化特征的余弦相似度"""
    return float(np.dot(feat1, feat2))

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def list_images(input_dir):
    """列出目录下所有图像文件（按文件名排序）"""
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(input_dir, name))
    return files


def detect_faces(rknn, img_bgr, model_size=(320, 320)):
    """RetinaFace 检测，返回 dets: (N, 15)
    每行: [x1, y1, x2, y2, score, kp1_x, kp1_y, ..., kp5_x, kp5_y]
    """
    img_height, img_width = img_bgr.shape[:2]
    model_height, model_width = model_size

    letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(
        img_bgr, (model_width, model_height), 114)
    infer_img = letterbox_img[..., ::-1]  # BGR2RGB

    outputs = rknn.inference(inputs=[infer_img])
    loc, conf, landmarks = outputs

    priors = PriorBox(image_size=(model_height, model_width))
    boxes = box_decode(loc.squeeze(0), priors)
    scale = np.array([model_width, model_height, model_width, model_height])
    boxes = boxes * scale // 1
    boxes[..., 0::2] = np.clip((boxes[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
    boxes[..., 1::2] = np.clip((boxes[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)

    scores = conf.squeeze(0)[:, 1]
    landmarks = decode_landm(landmarks.squeeze(0), priors)
    scale_landmarks = np.array([model_width, model_height] * 5)
    landmarks = landmarks * scale_landmarks // 1
    landmarks[..., 0::2] = np.clip((landmarks[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
    landmarks[..., 1::2] = np.clip((landmarks[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)

    inds = np.where(scores > 0.02)[0]
    if inds.size == 0:
        return np.empty((0, 15), dtype=np.float32)

    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, 0.5)
    dets = dets[keep, :]
    landmarks = landmarks[keep]
    return np.concatenate((dets, landmarks), axis=1)


def load_registry(feat_dir):
    """加载注册表: {图像名/人名: bin文件名}"""
    reg_path = os.path.join(feat_dir, 'registry.json')
    if os.path.exists(reg_path):
        with open(reg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_registry(feat_dir, registry):
    """保存注册表到 registry.json"""
    reg_path = os.path.join(feat_dir, 'registry.json')
    with open(reg_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return reg_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸检测 + 注册(特征提取)脚本', add_help=True)
    parser.add_argument('input_dir', type=str, help='输入图像文件夹')
    parser.add_argument('--det_model', type=str, default='../model/RetinaFace.rknn', help='人脸检测 RKNN 模型路径')
    parser.add_argument('--rec_model', type=str, default='../model/mobilefacenet.rknn', help='人脸识别(特征提取) RKNN 模型路径')
    parser.add_argument('--target', type=str, default='rk3588', help='目标 RKNPU 平台')
    parser.add_argument('--device_id', type=str, default=None, help='设备 ID')
    parser.add_argument('--det_out', type=str, default='./det_result', help='检测标注结果输出目录')
    parser.add_argument('--face_out', type=str, default='./aligned_faces', help='112x112 对齐人脸输出目录')
    parser.add_argument('--feat_out', type=str, default='./features', help='特征 bin 与注册表输出目录')
    parser.add_argument('--max_bins', type=int, default=100, help='特征 bin 文件数量上限')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='人脸置信度阈值')
    args = parser.parse_args()

    os.makedirs(args.det_out, exist_ok=True)
    os.makedirs(args.face_out, exist_ok=True)
    os.makedirs(args.feat_out, exist_ok=True)

    images = list_images(args.input_dir)
    if not images:
        print('未在目录 {} 中找到图像'.format(args.input_dir))
        exit(1)

    # 加载注册表，计算下一个可用 bin 编号
    registry = load_registry(args.feat_out)
    used_ids = []
    for v in registry.values():
        stem = os.path.splitext(os.path.basename(str(v)))[0]
        if stem.isdigit():
            used_ids.append(int(stem))
    next_id = (max(used_ids) + 1) if used_ids else 0

    # 初始化检测模型
    rknn_det = RKNN(verbose=False)
    ret = rknn_det.load_rknn(args.det_model)
    if ret != 0:
        print('Load detection model "{}" failed!'.format(args.det_model))
        exit(ret)
    ret = rknn_det.init_runtime(target=args.target, device_id=args.device_id)
    if ret != 0:
        print('Init detection runtime failed!')
        exit(ret)

    # 初始化识别模型
    rknn_rec = RKNN(verbose=False)
    ret = rknn_rec.load_rknn(args.rec_model)
    if ret != 0:
        print('Load recognition model "{}" failed!'.format(args.rec_model))
        exit(ret)
    ret = rknn_rec.init_runtime(target=args.target, device_id=args.device_id)
    if ret != 0:
        print('Init recognition runtime failed!')
        exit(ret)

    print('共 {} 张图像待处理'.format(len(images)))
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            print('[{}] 读取失败，跳过'.format(img_path))
            continue

        dets = detect_faces(rknn_det, img)
        faces = [d for d in dets if d[4] >= args.score_thresh]
        print('[{}] 检测到 {} 张人脸'.format(stem, len(faces)))

        annotated = img.copy()   # 用于绘制检测标注
        img_clean = img.copy()   # 保持干净，用于对齐抠图

        for i, data in enumerate(faces):
            print('  face {} @ ({},{},{},{}) score={:.4f}'.format(
                i, int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4]))
            kps = data[5:15].astype(np.float32).reshape(5, 2)
            d = list(map(int, data))

            # 1. 绘制检测标注
            cv2.rectangle(annotated, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 2)
            cv2.putText(annotated, '{:.4f}'.format(data[4]), (d[0], d[1] + 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            for (x, y), color in zip(((d[5], d[6]), (d[7], d[8]), (d[9], d[10]),
                                      (d[11], d[12]), (d[13], d[14])),
                                     ((0, 0, 255), (0, 255, 255), (255, 0, 255),
                                      (0, 255, 0), (255, 0, 0))):
                cv2.circle(annotated, (x, y), 1, color, 5)

            # 2. 5 点关键点对齐并保存 112x112 人脸, img_clean是BGR格式的
            aligned = align_face(img_clean, kps, output_size=(112, 112))
            face_name = '{}.jpg'.format(stem) if len(faces) == 1 else '{}_{}.jpg'.format(stem, i)
            cv2.imwrite(os.path.join(args.face_out, face_name), aligned)

            # 3. 特征提取并保存 bin
            key = stem if len(faces) == 1 else '{}_{}'.format(stem, i)
            rec_input = recognition_preprocess(aligned)   # uint8 RGB，归一化已在模型内
            out = rknn_rec.inference(inputs=[rec_input])
            embedding = l2_normalize(out[0])

            if key in registry:
                bin_name = registry[key]
                print('  -> {} 已存在，更新特征到 {}'.format(key, bin_name))
            else:
                if next_id >= args.max_bins:
                    print('  -> 特征数量已达上限 {}，跳过注册: {}'.format(args.max_bins, key))
                    continue
                bin_name = '{}.bin'.format(next_id)
                next_id += 1
                registry[key] = bin_name
                print('  -> 新注册 {} -> {}'.format(key, bin_name))

            embedding.astype(np.float32).tofile(os.path.join(args.feat_out, bin_name))

        # 保存检测标注图
        cv2.imwrite(os.path.join(args.det_out, '{}_det.jpg'.format(stem)), annotated)

    reg_path = save_registry(args.feat_out, registry)
    print('注册表已保存: {}'.format(reg_path))
    print('当前注册人数: {}'.format(len(registry)))

    rknn_det.release()
    rknn_rec.release()

