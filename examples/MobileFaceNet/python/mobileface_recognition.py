import os
import json
import argparse
import numpy as np
import cv2

from rknn.api import RKNN

import mobileface_registry as mf


def match_one(registry, feat_dir, query_embedding):
    """在注册库中查找与查询特征最相似的人。

    Args:
        registry: {名字: bin文件名}
        feat_dir: bin 文件所在目录
        query_embedding: 已 L2 归一化的查询特征

    Returns:
        (best_name, best_sim)，若库为空返回 (None, 0.0)
    """
    best_name = None
    best_sim = 0.0
    for name, bin_file in registry.items():
        bin_path = os.path.join(feat_dir, bin_file)
        if not os.path.exists(bin_path):
            continue
        gal_feat = mf.l2_normalize(np.fromfile(bin_path, dtype=np.float32))
        sim = mf.cosine_similarity(query_embedding, gal_feat)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, best_sim


def main():
    parser = argparse.ArgumentParser(description='人脸识别比对脚本', add_help=True)
    parser.add_argument('input_path', type=str, help='单张图像路径或图像文件夹')
    parser.add_argument('--det_model', type=str, default='../model/RetinaFace.rknn', help='人脸检测 RKNN 模型路径')
    parser.add_argument('--rec_model', type=str, default='../model/mobilefacenet.rknn', help='人脸识别 RKNN 模型路径')
    parser.add_argument('--feat_dir', type=str, default='../model/features', help='注册特征目录(内含 registry.json 与 *.bin)')
    parser.add_argument('--target', type=str, default='rk3588', help='目标 RKNPU 平台')
    parser.add_argument('--device_id', type=str, default=None, help='设备 ID')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='人脸检测置信度阈值')
    parser.add_argument('--sim_thresh', type=float, default=0.5, help='识别匹配的余弦相似度阈值')
    args = parser.parse_args()

    registry = mf.load_registry(args.feat_dir)
    if not registry:
        print('警告: 注册库 {} 为空，无法比对'.format(args.feat_dir))
    else:
        print('注册库共 {} 人'.format(len(registry)))

    if os.path.isdir(args.input_path):
        images = mf.list_images(args.input_path)
    else:
        images = [args.input_path]
    if not images:
        print('未找到待识别图像: {}'.format(args.input_path))
        exit(1)

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

    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            print('[{}] 读取失败，跳过'.format(img_path))
            continue

        dets = mf.detect_faces(rknn_det, img)
        faces = [d for d in dets if d[4] >= args.score_thresh]

        # 没有人脸：提醒
        if len(faces) == 0:
            print('[{}] 未检测到人脸'.format(stem))
            continue

        img_clean = img.copy()
        for i, data in enumerate(faces):
            kps = data[5:15].astype(np.float32).reshape(5, 2)
            aligned = mf.align_face(img_clean, kps, output_size=(112, 112))
            rec_input = mf.recognition_preprocess(aligned)
            out = rknn_rec.inference(inputs=[rec_input])
            embedding = mf.l2_normalize(out[0])

            best_name, best_sim = match_one(registry, args.feat_dir, embedding)
            if best_name is not None and best_sim >= args.sim_thresh:
                print('[{}] face {} -> 识别为: {} (相似度 {:.4f})'.format(stem, i, best_name, best_sim))
            else:
                print('[{}] face {} -> 未注册 (最高相似度 {:.4f})'.format(stem, i, best_sim))

    rknn_det.release()
    rknn_rec.release()


if __name__ == '__main__':
    main()
