import os
import cv2
import sys
import argparse
from typing import List, Tuple
import argparse
import glob
import os
from typing import List, Tuple
import numpy as np

# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

# WIDER Face 5-point colors (BGR format for cv2)
FACE_KPT_COLORS_BGR = [
    (255, 0, 255),    # B=255,G=0,R=255   → 紫色 magenta
    (0, 255, 0),      # B=0,G=255,R=0     → 绿色 green
    (0, 0, 255),      # B=0,G=0,R=255     → 红色 red
    (255, 255, 255),  # B=255,G=255,R=255 → 白色 white
    (0, 255, 255),    # B=0,G=255,R=255   → 黄色 yellow
]

COCO_CLASSES=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def _preprocess_image(
    img_path: str,
    input_size: Tuple[int, int] = (640, 640),
    to_rgb: bool = True,
):
    """Load image, optionally convert to RGB, resize, normalize.

    Returns:
        tuple: (nchw_array, img_meta) where img_meta contains
            'scale_factor': (w_scale, h_scale) for bbox rescaling,
            'img_shape': (original_h, original_w),
            'pad_shape': (padded_h, padded_w).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    orig_h, orig_w = img.shape[:2]
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Keep aspect ratio: scale long side to target long side,
    # then pad the short side with zeros to reach `input_size`.
    target_w, target_h = input_size
    target_long = max(target_w, target_h)
    h, w = img.shape[:2]
    scale = float(target_long) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    w_scale = new_w / orig_w
    h_scale = new_h / orig_h
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # right-bottom padding (consistent with mmengine stack_batch)
    padded = np.zeros((target_h, target_w, resized.shape[2]), dtype=resized.dtype)
    padded[:new_h, :new_w] = resized
    img = padded.astype(np.uint8)
    img_meta = {
        'scale_factor': (w_scale, h_scale),
        'img_shape': (orig_h, orig_w),
        'pad_shape': (target_h, target_w),
    }
    return img, img_meta


def _flatten_outputs(outputs) -> List[np.ndarray]:
    flat = []
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            flat.extend(_flatten_outputs(item))
    else:
        flat.append(np.asarray(outputs))
    return flat


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_grid_priors(
    featmap_sizes: List[Tuple[int, int]],
    strides: Tuple[int, ...] = (8, 16, 32),
    offset: float = 0.0,
) -> np.ndarray:
    """Generate YOLOX-style center priors for each FPN level (numpy version).

    Each prior is (cx, cy, stride_w, stride_h).
    """
    mlvl_priors = []
    for stride, (fh, fw) in zip(strides, featmap_sizes):
        xs = (np.arange(fw) + offset) * stride
        ys = (np.arange(fh) + offset) * stride
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        strides_arr = np.full_like(xs, stride)
        level_priors = np.stack([xs, ys, strides_arr, strides_arr], axis=-1)
        mlvl_priors.append(level_priors)
    return np.concatenate(mlvl_priors, axis=0)


def _bbox_decode_numpy(priors: np.ndarray, bbox_preds: np.ndarray) -> np.ndarray:
    """Decode YOLOX bbox predictions (delta_x, delta_y, log_w, log_h) to
    (x1, y1, x2, y2).

    Args:
        priors: (N, 4) in (cx, cy, stride_w, stride_h).
        bbox_preds: (1, N, 4) or (N, 4).
    Returns:
        decoded_bboxes: (N, 4) in (x1, y1, x2, y2).
    """
    if bbox_preds.ndim == 3:
        bbox_preds = bbox_preds[0]
    xys = bbox_preds[:, :2] * priors[:, 2:] + priors[:, :2]
    whs = np.exp(bbox_preds[:, 2:]) * priors[:, 2:]
    x1 = xys[:, 0] - whs[:, 0] / 2.0
    y1 = xys[:, 1] - whs[:, 1] / 2.0
    x2 = xys[:, 0] + whs[:, 0] / 2.0
    y2 = xys[:, 1] + whs[:, 1] / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def _decode_obstacle_outputs(
    raw_outputs: List[np.ndarray],
    img_meta: dict,
    strides: Tuple[int, ...] = (8, 16, 32),
    score_thr: float = 0.3,
):
    """Decode YOLOX obstacle outputs following predict_by_feat logic.

    raw_outputs layout (9 elements from 3 FPN levels):
        [cls_p3, cls_p4, cls_p5,
         bbox_p3, bbox_p4, bbox_p5,
         obj_p3,  obj_p4,  obj_p5]

    Args:
        raw_outputs: List of 9 numpy arrays from ONNX output.
        img_meta: dict with 'scale_factor' (w_scale, h_scale) and 'img_shape'.
        strides: strides per FPN level.
        score_thr: score threshold.

    Returns:
        list of dicts with 'bbox' [x1,y1,x2,y2] and 'score'.
    """
    if len(raw_outputs) < 9:
        return []

    num_levels = len(strides)
    cls_scores = [raw_outputs[i] for i in range(num_levels)]
    bbox_preds = [raw_outputs[i + num_levels] for i in range(num_levels)]
    objectnesses = [raw_outputs[i + 2 * num_levels] for i in range(num_levels)]

    w_scale, h_scale = img_meta['scale_factor']

    featmap_sizes = [(arr.shape[2], arr.shape[3]) for arr in cls_scores]
    mlvl_priors = _generate_grid_priors(featmap_sizes, strides=strides, offset=0.0)

    # flatten per level: permute (1,C,H,W) -> (1,H,W,C) -> (1, N, C)
    flatten_cls = []
    flatten_bbox = []
    flatten_obj = []
    num_classes = cls_scores[0].shape[1]

    for cls_s, bbox_s, obj_s in zip(cls_scores, bbox_preds, objectnesses):
        # (1, C, H, W) -> (1, H*W, C)
        flatten_cls.append(cls_s.transpose(0, 2, 3, 1).reshape(1, -1, num_classes))
        flatten_bbox.append(bbox_s.transpose(0, 2, 3, 1).reshape(1, -1, 4))
        flatten_obj.append(obj_s.transpose(0, 2, 3, 1).reshape(1, -1))

    flatten_cls_scores = _sigmoid(np.concatenate(flatten_cls, axis=1))
    flatten_bbox_preds = np.concatenate(flatten_bbox, axis=1)
    flatten_objectness = _sigmoid(np.concatenate(flatten_obj, axis=1))

    # decode bboxes
    decoded_bboxes = _bbox_decode_numpy(mlvl_priors, flatten_bbox_preds)

    # score = max(cls) * obj, filter
    max_scores = np.max(flatten_cls_scores[0], axis=1)
    labels = np.argmax(flatten_cls_scores[0], axis=1)
    scores = max_scores * flatten_objectness[0]
    valid = scores >= score_thr

    bboxes = decoded_bboxes[valid]
    scores = scores[valid]
    labels = labels[valid]

    if len(scores) == 0:
        return []

    # Rescale bboxes to original image space:
    # bboxes /= scale_factor (repeated for x1y1x2y2)
    sf = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    bboxes = bboxes / sf

    # clip to original image
    orig_h, orig_w = img_meta['img_shape']
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, orig_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, orig_h)

    # greedy NMS per class
    results = []
    keep_mask = np.ones(len(scores), dtype=bool)
    for cls_id in np.unique(labels):
        inds = np.where((labels == cls_id) & keep_mask)[0]
        order = inds[np.argsort(scores[inds])[::-1]]
        while len(order) > 0:
            best = order[0]
            results.append({
                'bbox': bboxes[best].tolist(),
                'score': float(scores[best]),
                'label': int(cls_id),
            })
            # suppress overlapping boxes of same class
            rest = order[1:]
            if len(rest) == 0:
                break
            ious = np.array([_box_iou(bboxes[best], bboxes[j]) for j in rest])
            suppressed = rest[ious > 0.5]
            keep_mask[suppressed] = False
            order = rest[ious <= 0.5]

    return results


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter) / max(float(area_a + area_b - inter), 1e-8)


def _decode_kpt_numpy(
    priors: np.ndarray,
    kpt_offsets: np.ndarray,
    num_keypoints: int,
) -> np.ndarray:
    """Decode keypoint offsets (delta_x, delta_y) → (x, y).

    Args:
        priors: (N, 4) in (cx, cy, stride_w, stride_h).
        kpt_offsets: (1, N, K*2) or (N, K*2).
        num_keypoints: number of keypoints K.
    Returns:
        decoded_kpts: (N, K, 2).
    """
    if kpt_offsets.ndim == 3:
        kpt_offsets = kpt_offsets[0]  # (N, K*2)
    kpt_offsets = kpt_offsets.reshape(-1, num_keypoints, 2)  # (N, K, 2)
    stride = priors[:, 2:3]  # (N, 1)
    priors_xy = priors[:, :2]  # (N, 2)
    return kpt_offsets * stride[:, np.newaxis, :] + priors_xy[:, np.newaxis, :]


def _class_agnostic_nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.65,
) -> np.ndarray:
    """Class-agnostic NMS returning keep indices (like YOLOXPose nms_torch)."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([_box_iou(bboxes[best], bboxes[j]) for j in rest])
        order = rest[ious <= iou_thr]
    return np.array(keep, dtype=int)


def _decode_pose_outputs(
    raw_outputs: List[np.ndarray],
    img_meta: dict,
    strides: Tuple[int, ...] = (8, 16, 32),
    score_thr: float = 0.3,
    nms_thr: float = 0.65,
):
    """Decode YOLOXPose outputs following YOLOXPoseHead.predict logic.

    raw_outputs layout (15 elements = 5 groups × 3 FPN levels):
        [cls_p3, cls_p4, cls_p5,
         obj_p3, obj_p4, obj_p5,
         bbox_p3, bbox_p4, bbox_p5,
         kpt_p3, kpt_p4, kpt_p5,
         vis_p3, vis_p4, vis_p5]

    Args:
        raw_outputs: List of 15 numpy arrays from ONNX output.
        img_meta: dict with 'scale_factor', 'img_shape'.
        strides: strides per FPN level.
        score_thr: score threshold.
        nms_thr: NMS IoU threshold (class-agnostic).

    Returns:
        list of dicts with 'bbox', 'score', 'keypoints', 'keypoint_scores'.
    """
    if len(raw_outputs) < 15:
        return []

    num_levels = len(strides)
    # Output order: cls, obj, bbox, kpt, vis (YOLOXPoseHead.forward)
    cls_scores = [raw_outputs[i] for i in range(num_levels)]
    objectnesses = [raw_outputs[i + num_levels] for i in range(num_levels)]
    bbox_preds = [raw_outputs[i + 2 * num_levels] for i in range(num_levels)]
    kpt_offsets = [raw_outputs[i + 3 * num_levels] for i in range(num_levels)]
    kpt_vis = [raw_outputs[i + 4 * num_levels] for i in range(num_levels)]

    w_scale, h_scale = img_meta['scale_factor']
    num_keypoints = kpt_vis[0].shape[1]  # auto-detect K from (1, K, H, W)
    num_classes = cls_scores[0].shape[1]

    featmap_sizes = [(arr.shape[2], arr.shape[3]) for arr in cls_scores]
    mlvl_priors = _generate_grid_priors(featmap_sizes, strides=strides, offset=0.0)

    # flatten per level
    flatten_cls = []
    flatten_obj = []
    flatten_bbox = []
    flatten_kpt = []
    flatten_vis = []

    for i in range(num_levels):
        flatten_cls.append(cls_scores[i].transpose(0, 2, 3, 1).reshape(1, -1, num_classes))
        flatten_obj.append(objectnesses[i].transpose(0, 2, 3, 1).reshape(1, -1))
        flatten_bbox.append(bbox_preds[i].transpose(0, 2, 3, 1).reshape(1, -1, 4))
        flatten_kpt.append(kpt_offsets[i].transpose(0, 2, 3, 1).reshape(1, -1, num_keypoints * 2))
        flatten_vis.append(kpt_vis[i].transpose(0, 2, 3, 1).reshape(1, -1, num_keypoints))

    flatten_cls_scores = _sigmoid(np.concatenate(flatten_cls, axis=1))      # (1, N, C)
    flatten_bbox_preds = np.concatenate(flatten_bbox, axis=1)               # (1, N, 4)
    flatten_objectness = _sigmoid(np.concatenate(flatten_obj, axis=1))       # (1, N)
    flatten_kpt_offsets = np.concatenate(flatten_kpt, axis=1)               # (1, N, K*2)
    flatten_kpt_vis = _sigmoid(np.concatenate(flatten_vis, axis=1))          # (1, N, K)

    # decode bboxes and keypoints
    decoded_bboxes = _bbox_decode_numpy(mlvl_priors, flatten_bbox_preds)    # (N, 4)
    decoded_kpts = _decode_kpt_numpy(mlvl_priors, flatten_kpt_offsets, num_keypoints)  # (N, K, 2)

    # scores = max(sigmoid(cls)) * sigmoid(obj)
    max_scores = np.max(flatten_cls_scores[0], axis=1)                      # (N,)
    labels = np.argmax(flatten_cls_scores[0], axis=1)                       # (N,)
    scores = max_scores * flatten_objectness[0]                              # (N,)

    if score_thr > 0:
        keep = np.where(scores >= score_thr)[0]
        decoded_bboxes = decoded_bboxes[keep]
        decoded_kpts = decoded_kpts[keep]
        flatten_kpt_vis = flatten_kpt_vis[0][keep]
        scores = scores[keep]
        labels = labels[keep]

    if len(scores) == 0:
        return []

    # class-agnostic NMS
    keep_nms = _class_agnostic_nms(decoded_bboxes, scores, iou_thr=nms_thr)
    bboxes = decoded_bboxes[keep_nms]
    keypoints = decoded_kpts[keep_nms]
    kpt_scores = flatten_kpt_vis[keep_nms]
    scores = scores[keep_nms]
    labels = labels[keep_nms]

    # Rescale to original image space
    sf = np.array([w_scale, h_scale], dtype=np.float32)
    bboxes = bboxes / np.tile(sf, 2)
    keypoints = keypoints / sf.reshape(1, 1, 2)

    # clip to original image
    orig_h, orig_w = img_meta['img_shape']
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, orig_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, orig_h)
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, orig_w)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, orig_h)

    results = []
    for i in range(len(scores)):
        results.append({
            'bbox': bboxes[i].tolist(),
            'score': float(scores[i]),
            'label': int(labels[i]),
            'keypoints': keypoints[i],
            'keypoint_scores': kpt_scores[i],
        })
    return results


def _decode_face_outputs(
    raw_outputs: List[np.ndarray],
    img_meta: dict,
    strides: Tuple[int, ...] = (8, 16, 32),
    score_thr: float = 0.3,
    nms_thr: float = 0.65,
):
    """Face decode is identical to pose decode."""
    return _decode_pose_outputs(raw_outputs, img_meta, strides=strides,
                                score_thr=score_thr, nms_thr=nms_thr)


def _draw_coco_skeleton(img: np.ndarray,
                        kpts: np.ndarray,
                        kpts_vis: np.ndarray = None,
                        bbox_color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw COCO 17-keypoint skeleton on image (in-place).

    Args:
        img: Image in HWC BGR format (modified in-place).
        kpts: (N_person, 17, 2).
        kpts_vis: (N_person, 17). vis>0 means visible.
        bbox_color: fallback color for keypoints.
    """
    skeleton_color = (0, 255, 255)  # yellow skeleton lines
    for person_idx, person_kpts in enumerate(kpts):
        # Draw skeleton connections
        for i, j in COCO_SKELETON:
            ki, kj = person_kpts[i], person_kpts[j]
            vi = kpts_vis[person_idx, i] if kpts_vis is not None else 1
            vj = kpts_vis[person_idx, j] if kpts_vis is not None else 1
            if vi > 0 and vj > 0 and ki[0] > 0 and ki[1] > 0 and kj[0] > 0 and kj[1] > 0:
                cv2.line(img, (int(ki[0]), int(ki[1])), (int(kj[0]), int(kj[1])),
                         skeleton_color, 1, cv2.LINE_AA)
        # Draw keypoints
        for kp_idx, kp in enumerate(person_kpts):
            vis = kpts_vis[person_idx, kp_idx] if kpts_vis is not None else 1
            if vis > 0 and kp[0] > 0 and kp[1] > 0:
                cv2.circle(img, (int(kp[0]), int(kp[1])), 3, bbox_color, -1)


def _draw_face_keypoints(img: np.ndarray,
                         kpts: np.ndarray,
                         kpts_vis: np.ndarray = None) -> None:
    """Draw WIDER Face 5-point landmarks on image (in-place).

    Visible keypoints: solid circle. Invisible: hollow.

    Args:
        img: Image in HWC BGR format (modified in-place).
        kpts: (N_face, 5, 2).
        kpts_vis: (N_face, 5). vis>0 means visible.
    """
    for face_idx, face_kpts in enumerate(kpts):
        for kp_idx, kp in enumerate(face_kpts):
            if kp[0] <= 0 or kp[1] <= 0:
                continue
            vis = kpts_vis[face_idx, kp_idx] if kpts_vis is not None else 1
            color = (FACE_KPT_COLORS_BGR[kp_idx]
                     if kp_idx < len(FACE_KPT_COLORS_BGR) else (255, 255, 255))
            pt = (int(kp[0]), int(kp[1]))
            if vis > 0:
                cv2.circle(img, pt, 3, color, -1)   # solid
            else:
                cv2.circle(img, pt, 3, color, 1)    # hollow


def _visualize_single_task(
    img: np.ndarray,
    results: list,
    out_path: str,
    color: Tuple[int, int, int],
    label: str,
):
    """Draw results on a copy of img and save to out_path."""
    vis = img.copy()

    # collect keypoints for skeleton/landmark drawing
    all_kpts = []
    all_kpt_scores = []

    for res in results:
        x1, y1, x2, y2 = [int(v) for v in res['bbox']]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # build display text: "[class_name] [score]"
        parts = []
        if 'label' in res and isinstance(res['label'], (int, np.integer)):
            cls_id = int(res['label'])
            if 0 <= cls_id < len(COCO_CLASSES):
                parts.append(COCO_CLASSES[cls_id])
        if 'score' in res:
            parts.append(f'{res["score"]:.2f}')
        text_str = ' '.join(parts)

        cv2.putText(vis, text_str, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # collect keypoints for batch drawing
        if 'keypoints' in res:
            kpts = np.asarray(res['keypoints'], dtype=np.float32)
            kpt_scores = np.asarray(
                res.get('keypoint_scores', np.ones(kpts.shape[0])),
                dtype=np.float32)
            all_kpts.append(kpts)
            all_kpt_scores.append(kpt_scores)

    # batch draw keypoints based on keypoint count
    if all_kpts:
        kpts_stack = np.stack(all_kpts, axis=0)          # (N_inst, K, 2)
        vis_stack = np.stack(all_kpt_scores, axis=0)     # (N_inst, K)
        num_kpts = kpts_stack.shape[1]
        if num_kpts == 17:
            _draw_coco_skeleton(vis, kpts_stack, vis_stack, bbox_color=color)
        elif num_kpts == 5:
            _draw_face_keypoints(vis, kpts_stack, vis_stack)
        else:
            # fallback: simple circles
            for kpts, scores_arr in zip(all_kpts, all_kpt_scores):
                for kp, ks in zip(kpts, scores_arr):
                    if float(ks) > 0.3:
                        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, color, -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f'[INFO] Visualization saved to {out_path}')


def run_rknn_inference(args):
    # init input and output dir
    img_dir = args.img_folder
    out_dir = args.out_dir
    # list images    
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    img_paths = sorted(set(img_paths))

    if not img_paths:
        print(f'[WARN] No images found in {img_dir}')
        return []

    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    # init model
    model, platform = setup_model(args)

    for img_path in img_paths:
        input_data, img_meta = _preprocess_image(img_path)
        outputs = model.run([input_data])
        flat_outputs = _flatten_outputs(outputs)

        obt_results = _decode_obstacle_outputs(flat_outputs[:9], img_meta)
        pose_results = _decode_pose_outputs(flat_outputs[9:24], img_meta)
        face_results = _decode_face_outputs(flat_outputs[24:], img_meta)

        # save per-task visualization images
        img = cv2.imread(img_path)
        if img is not None:
            name, ext = os.path.splitext(os.path.basename(img_path))
            _visualize_single_task(img, obt_results,
                                   os.path.join(out_dir, f'{name}_obt{ext}'),
                                   color=(0, 255, 0), label='obt')
            _visualize_single_task(img, pose_results,
                                   os.path.join(out_dir, f'{name}_pose{ext}'),
                                   color=(255, 0, 0), label='pose')
            _visualize_single_task(img, face_results,
                                   os.path.join(out_dir, f'{name}_face{ext}'),
                                   color=(0, 0, 255), label='face')

        all_results.append({
            'image': os.path.basename(img_path),
            'obt_results': obt_results,
            'pose_results': pose_results,
            'face_results': face_results,
        })
    
    # release
    model.release()
    print(f'[INFO] Processed {len(img_paths)} images, results saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--img_folder', type=str, default='../model/imgs', help='img folder path')
    parser.add_argument('--out_dir', type=str, default='../model/imgs_out', help='output results')

    args = parser.parse_args()
    # inference
    run_rknn_inference(args)


    
