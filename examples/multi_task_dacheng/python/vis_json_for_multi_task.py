#!/usr/bin/env python3
"""Visualize multi-task JSON results (from C++ inference) as three separate images.

Usage:
    python vis_json_for_multi_task.py --image bus.jpg --json bus.json --out_dir ./vis_out
"""

import argparse
import json
import os
from typing import Tuple

import cv2
import numpy as np

# ============================================================
# Constants & drawing helpers (self-contained, one-file)
# ============================================================

# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11),  (6, 12),  (5, 6),   (5, 7),   (6, 8),
    (7, 9),   (8, 10),  (1, 2),   (0, 1),   (0, 2),
    (1, 3),   (2, 4),   (3, 5),   (4, 6),
]

# WIDER Face 5-point colors (BGR format for cv2)
FACE_KPT_COLORS_BGR = [
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (255, 255, 255),  # white
    (0, 255, 255),    # yellow
]


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

# ============================================================


def visualize_single_task(img, results, out_path, color):
    """Draw detection results on a copy of *img* and save to *out_path*.

    Each element in *results* is a dict:
        { 'bbox': [x1,y1,x2,y2], 'score': float, 'class_name': str,
          'keypoints': np.ndarray (K,2), 'keypoint_scores': np.ndarray (K,) }  # optional
    """
    vis = img.copy()
    all_kpts = []
    all_kpt_scores = []

    for res in results:
        x1, y1, x2, y2 = [int(v) for v in res['bbox']]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # label text
        parts = []
        if res.get('class_name'):
            parts.append(res['class_name'])
        if 'score' in res:
            parts.append(f'{res["score"]:.2f}')
        if parts:
            cv2.putText(vis, ' '.join(parts), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # collect keypoints
        if 'keypoints' in res:
            all_kpts.append(np.asarray(res['keypoints'], dtype=np.float32))
            all_kpt_scores.append(np.asarray(res['keypoint_scores'], dtype=np.float32))

    # batch-draw keypoints
    if all_kpts:
        kpts_stack = np.stack(all_kpts, axis=0)       # (N, K, 2)
        vis_stack = np.stack(all_kpt_scores, axis=0)   # (N, K)
        num_kpts = kpts_stack.shape[1]
        if num_kpts == 17:
            _draw_coco_skeleton(vis, kpts_stack, vis_stack, bbox_color=color)
        elif num_kpts == 5:
            _draw_face_keypoints(vis, kpts_stack, vis_stack)
        else:
            for kpts, scores in zip(all_kpts, all_kpt_scores):
                for kp, ks in zip(kpts, scores):
                    if float(ks) > 0.3:
                        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, color, -1)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f'[INFO] Saved: {out_path}')


def _convert_kpt_dict(d, num_kpts):
    """Convert JSON keypoints dict {kpt0:{x,y,score}, ...} -> (num_kpts,2), (num_kpts,)."""
    kpts = np.zeros((num_kpts, 2), dtype=np.float32)
    kpt_scores = np.zeros(num_kpts, dtype=np.float32)
    for k in range(num_kpts):
        kdata = d.get(f'kpt{k}', {})
        kpts[k, 0] = kdata.get('x', 0)
        kpts[k, 1] = kdata.get('y', 0)
        kpt_scores[k] = kdata.get('score', 0)
    return kpts, kpt_scores


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-task JSON results')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON result file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f'Image not found: {args.image}')

    with open(args.json, 'r') as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    mr = data.get('model_results', {})

    # ---- object (green boxes) ----
    obj_vis = [{'bbox': r['bbox'], 'score': r['score'],
                'class_name': r.get('class_name', '')}
               for r in mr.get('object', [])]
    visualize_single_task(img, obj_vis,
                          os.path.join(args.out_dir, f'{basename}_od.jpg'),
                          color=(0, 255, 0))

    # ---- pose (red boxes + 17-kpt skeleton) ----
    pose_vis = []
    for r in mr.get('pose', []):
        kpts, kpt_scores = _convert_kpt_dict(r.get('keypoints', {}), 17)
        pose_vis.append({'bbox': r['bbox'], 'score': r['score'],
                         'class_name': r.get('class_name', 'person'),
                         'keypoints': kpts, 'keypoint_scores': kpt_scores})
    visualize_single_task(img, pose_vis,
                          os.path.join(args.out_dir, f'{basename}_pose.jpg'),
                          color=(255, 0, 0))

    # ---- face (red/blue boxes + 5-point landmarks) ----
    face_vis = []
    for r in mr.get('face', []):
        kpts, kpt_scores = _convert_kpt_dict(r.get('keypoints', {}), 5)
        face_vis.append({'bbox': r['bbox'], 'score': r['score'],
                         'class_name': r.get('class_name', 'face'),
                         'keypoints': kpts, 'keypoint_scores': kpt_scores})
    visualize_single_task(img, face_vis,
                          os.path.join(args.out_dir, f'{basename}_face.jpg'),
                          color=(0, 0, 255))

    print(f'[INFO] All visualizations saved to {args.out_dir}')


if __name__ == '__main__':
    main()