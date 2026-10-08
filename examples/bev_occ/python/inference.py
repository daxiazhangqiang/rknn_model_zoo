#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-OCC 感知模型 RKNN 板端推理 + 可视化

模型 (examples/bev_occ/model/bevocc_96_lossweight2.rknn):
  input            NHWC uint8 RGB [0,255]  (1, 384, 640, 3)
                   mean=[123.675,116.28,103.53] / std=[58.395,57.12,57.375]
                   已在 rknn 内部固化, 所以直接喂 RAW 图像即可 (不要自己做归一化)
  static_seg_out   (1, 10, 48, 80)   静态障碍物语义分割 logits
  dynamic_seg_out  (1,  4, 48, 80)   动态障碍物语义分割 logits
  bev_occ_out      (1, 11, 64, 64)   BEV 占据栅格 logits

用法:
  python inference.py <model.rknn> <img_or_dir> [out_dir]

  python inference.py model/bevocc_96_lossweight2.rknn imgs/000017.jpg
  python inference.py model/bevocc_96_lossweight2.rknn imgs/ bevocc_out

输出:
  <out_dir>/<图像名>/<图像名>_static.jpg    静态分割叠加到输入图像
  <out_dir>/<图像名>/<图像名>_dynamic.jpg   动态分割叠加到输入图像
  <out_dir>/<图像名>/<图像名>_bevocc.jpg    BEV 输出上色 (64x64 放大, 不叠加)
  <out_dir> 缺省为 bevocc_out
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------- 模型输入尺寸 ----------------
MODEL_W = 640
MODEL_H = 384

# 可视化参数
OVERLAY_ALPHA = 0.55     # 叠加透明度
BEV_VIEW_SIZE = 384      # BEV 输出放大后的显示边长
JPEG_QUALITY = 95

# ---------------- 类别名 / RGB 调色板 (与训练侧可视化一致) ----------------
# 注意: 类别数与模型输出通道严格对应, 改模型时需同步修改
STATIC_CLASSES = 10
DYNAMIC_CLASSES = 4
BEV_CLASSES = 11

STATIC_NAMES = ('floor', 'wall', 'door', 'window', 'hanging',
                'table', 'chair', 'sofa', 'bed', 'cabinet')
STATIC_PALETTE_RGB = [
    (0, 160, 0), (180, 255, 0), (139, 90, 43), (0, 170, 220),
    (200, 160, 230), (255, 128, 0), (220, 60, 60), (200, 0, 160),
    (120, 60, 200), (30, 70, 200)]

DYNAMIC_NAMES = ('background', 'person', 'cat', 'dog')
DYNAMIC_PALETTE_RGB = [
    (0, 0, 0),        # 0 background
    (235, 220, 0),    # 1 person
    (0, 220, 150),    # 2 cat
    (222, 184, 135)]  # 3 dog

BEV_NAMES = ('floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table',
             'tvs', 'furn', 'objs', 'unknown')
BEV_PALETTE_RGB = [
    (0, 160, 0), (180, 255, 0), (0, 170, 220), (220, 60, 60),
    (120, 60, 200), (200, 0, 160), (255, 128, 0), (30, 70, 200),
    (139, 90, 43), (205, 205, 205), (150, 150, 150)]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')


def check_class_tables():
    """校验调色板长度与模型类别数一致"""
    for name, table, num in (('STATIC', STATIC_PALETTE_RGB, STATIC_CLASSES),
                             ('DYNAMIC', DYNAMIC_PALETTE_RGB, DYNAMIC_CLASSES),
                             ('BEV', BEV_PALETTE_RGB, BEV_CLASSES)):
        if len(table) < num:
            raise ValueError('%s 调色板只有 %d 项, 少于类别数 %d' %
                             (name, len(table), num))


def is_image_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1] in IMAGE_EXTS


def collect_images(path):
    """输入是图像 -> [图像]; 输入是目录 -> 目录下所有图像 (按名称排序)"""
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
        return [f for f in files if is_image_file(f)]
    if is_image_file(path):
        return [path]
    return []


# ----------------------------------------------------------------------
# 预处理: 短边对齐 + 长边居中裁剪 -> 640x384 RGB uint8 NHWC
# ----------------------------------------------------------------------
def preprocess(bgr, mode='cover'):
    """BGR 原图 -> (NHWC uint8 RGB 输入, 几何信息 geom)

    mode='cover'  : 等比例缩放, 短边对齐, 长边居中裁掉 (默认, 不变形)
    mode='stretch' : 直接拉伸到 640x384 (与 onnx 实时脚本 normalize_frame 一致)
    geom 记录裁剪/缩放信息, 用于把分割结果映射回原图尺寸
    """
    h, w = bgr.shape[:2]

    if mode == 'cover':
        scale = max(MODEL_H / h, MODEL_W / w)
        crop_h = min(h, max(1, int(round(MODEL_H / scale))))
        crop_w = min(w, max(1, int(round(MODEL_W / scale))))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        roi = bgr[top:top + crop_h, left:left + crop_w]
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    else:
        top, left = 0, 0
        crop_h, crop_w = h, w
        roi = bgr
        interp = cv2.INTER_AREA if (w > MODEL_W or h > MODEL_H) else cv2.INTER_LINEAR

    resized = cv2.resize(roi, (MODEL_W, MODEL_H), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    geom = {'mode': mode, 'top': top, 'left': left, 'crop_h': crop_h,
            'crop_w': crop_w, 'src_w': w, 'src_h': h}
    return np.ascontiguousarray(rgb[None]), geom


def label_map_to_source(argmax, geom):
    """模型分辨率 (Hm,Wm) 的类别图 -> 原图分辨率 (H,W) 的类别图

    裁剪区域之外的像素填 -1 (表示不参与叠加)
    """
    mask_model = cv2.resize(argmax.astype(np.uint8), (MODEL_W, MODEL_H),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
    full = np.full((geom['src_h'], geom['src_w']), -1, np.int32)

    if geom['mode'] == 'cover':
        crop_map = cv2.resize(mask_model, (geom['crop_w'], geom['crop_h']),
                              interpolation=cv2.INTER_NEAREST)
        full[geom['top']:geom['top'] + geom['crop_h'],
             geom['left']:geom['left'] + geom['crop_w']] = crop_map
    else:
        full = cv2.resize(mask_model, (geom['src_w'], geom['src_h']),
                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return full


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def palette_to_bgr(pal_rgb):
    return np.asarray([(c[2], c[1], c[0]) for c in pal_rgb], np.uint8)


def label_to_bgr(argmax, pal_rgb):
    """类别图 -> BGR 彩色图 (调色板存 RGB, 输出前翻到 BGR)"""
    pal = palette_to_bgr(pal_rgb)
    return pal[np.mod(argmax, len(pal))]


def overlay_label(base_bgr, label_full, pal_rgb, skip_class=None,
                  alpha=OVERLAY_ALPHA):
    """把类别图半透明叠加到原图; skip_class 指定的类不叠加 (如动态背景)"""
    out = base_bgr.copy()
    mask = label_full >= 0
    if skip_class is not None:
        mask = mask & (label_full != skip_class)
    pal = palette_to_bgr(pal_rgb)
    color = pal[np.mod(np.where(mask, label_full, 0), len(pal))]
    mask3 = mask[..., None]
    out = np.where(mask3,
                   out.astype(np.float32) * (1 - alpha) +
                   color.astype(np.float32) * alpha,
                   out.astype(np.float32)).astype(np.uint8)
    return out


def split_outputs(outs):
    """按通道数把 3 个输出拆成 static / dynamic / bev"""
    found = {}
    for i, o in enumerate(outs):
        a = np.asarray(o)
        if a.ndim == 4:
            a = a[0]
        if a.ndim != 3:
            print('[WARN] 跳过无法识别的输出 %d: shape=%s' % (i, a.shape))
            continue
        c = a.shape[0]
        if c == STATIC_CLASSES and 'static' not in found:
            found['static'] = a
        elif c == DYNAMIC_CLASSES and 'dynamic' not in found:
            found['dynamic'] = a
        elif c == BEV_CLASSES and 'bev' not in found:
            found['bev'] = a
        else:
            print('[WARN] 无法归类的输出 %d: shape=%s' % (i, a.shape))
    missing = [k for k in ('static', 'dynamic', 'bev') if k not in found]
    if missing:
        raise RuntimeError('模型输出缺少 %s, 实际输出 shape=%s'
                           % (missing, [np.asarray(o).shape for o in outs]))
    return found


def save_image(path, bgr):
    ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('写图像失败: %s' % path)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_image(model, bgr, out_dir, stem, alpha=OVERLAY_ALPHA,
              geom_verbose=False):
    """单张图像预处理 + 推理 + 可视化, 返回 (推理耗时ms, 结果dict)"""
    inp, geom = preprocess(bgr)
    if geom_verbose:
        print('    预处理: %s crop=(top=%d,left=%d,%dx%d) <- %dx%d'
              % (geom['mode'], geom['top'], geom['left'],
                 geom['crop_h'], geom['crop_w'], geom['src_w'], geom['src_h']))

    t0 = time.perf_counter()
    outs = model.inference([inp])
    infer_ms = (time.perf_counter() - t0) * 1000.0

    heads = split_outputs(outs)
    static_label = label_map_to_source(heads['static'].argmax(0), geom)
    dynamic_label = label_map_to_source(heads['dynamic'].argmax(0), geom)
    bev_label = heads['bev'].argmax(0).astype(np.uint8)

    img_dir = os.path.join(out_dir, stem)
    os.makedirs(img_dir, exist_ok=True)

    static_vis = overlay_label(bgr, static_label, STATIC_PALETTE_RGB,
                               alpha=alpha)
    dynamic_vis = overlay_label(bgr, dynamic_label, DYNAMIC_PALETTE_RGB,
                                skip_class=0, alpha=alpha)
    bev_vis = cv2.resize(label_to_bgr(bev_label, BEV_PALETTE_RGB),
                         (BEV_VIEW_SIZE, BEV_VIEW_SIZE),
                         interpolation=cv2.INTER_NEAREST)

    save_image(os.path.join(img_dir, '%s_static.jpg' % stem), static_vis)
    save_image(os.path.join(img_dir, '%s_dynamic.jpg' % stem), dynamic_vis)
    save_image(os.path.join(img_dir, '%s_bevocc.jpg' % stem), bev_vis)

    result = {'geom': geom,
              'static': static_label,
              'dynamic': dynamic_label,
              'bev': bev_label}
    return infer_ms, result


def print_label_stats(result):
    """打印各 head 的类别占比 (便于快速判断结果是否合理)"""
    for key, names in (('static', STATIC_NAMES), ('dynamic', DYNAMIC_NAMES),
                       ('bev', BEV_NAMES)):
        label = result[key]
        label = label[label >= 0]
        total = label.size
        if total == 0:
            continue
        counts = np.bincount(label.ravel(), minlength=len(names))
        top = np.argsort(counts)[::-1][:3]
        txt = ', '.join('%s %.1f%%' % (names[i] if i < len(names) else str(i),
                                       100.0 * counts[i] / total)
                        for i in top if counts[i] > 0)
        print('    %-8s %s' % (key, txt))


def main():
    parser = argparse.ArgumentParser(
        description='BEV-OCC RKNN 推理 + 可视化',
        usage='%(prog)s <model.rknn> <img_or_dir> [out_dir]')
    parser.add_argument('model', help='rknn 模型路径')
    parser.add_argument('input', help='输入图像或图像文件夹')
    parser.add_argument('out_dir', nargs='?', default='bevocc_out',
                        help='输出文件夹 (缺省 bevocc_out)')
    parser.add_argument('--resize', choices=('cover', 'stretch'),
                        default='cover',
                        help='预处理缩放方式: cover=短边对齐+长边居中裁剪(默认), '
                             'stretch=直接拉伸')
    parser.add_argument('--alpha', type=float, default=OVERLAY_ALPHA,
                        help='叠加透明度 (默认 %.2f)' % OVERLAY_ALPHA)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print('[ERROR] 模型文件不存在: %s' % args.model)
        return 1
    images = collect_images(args.input)
    if not images:
        print('[ERROR] 找不到输入图像: %s' % args.input)
        return 1

    check_class_tables()

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print('[ERROR] 未安装 rknnlite, 请使用 rknn-toolkit2 的环境, 例如:')
        print('  /home/robot/miniconda3/envs/rknn/bin/python %s' % sys.argv[0])
        return 1

    print('[INFO] model   : %s' % args.model)
    print('[INFO] input   : %s (%d 张图像)' % (args.input, len(images)))
    print('[INFO] out_dir : %s' % args.out_dir)
    print('[INFO] resize  : %s' % args.resize)

    model = RKNNLite(verbose=False)
    if model.load_rknn(args.model) != 0:
        print('[ERROR] load_rknn 失败')
        return 1
    if model.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL) != 0:
        print('[ERROR] init_runtime 失败')
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    total_ms = 0.0
    fail = 0
    for idx, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        print('[%d/%d] %s' % (idx + 1, len(images), os.path.basename(path)))
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print('    [ERROR] 读取图像失败, 跳过')
            fail += 1
            continue

        infer_ms, result = run_image(model, bgr, args.out_dir, stem,
                                     alpha=args.alpha, geom_verbose=True)
        total_ms += infer_ms
        print('    推理 %.1f ms -> %s/' % (infer_ms, os.path.join(args.out_dir, stem)))
        print_label_stats(result)

    done = len(images) - fail
    if done > 0:
        print('[INFO] 完成 %d/%d 张, 平均推理 %.1f ms (%.2f FPS)'
              % (done, len(images), total_ms / done, 1000.0 * done / total_ms))
    if fail:
        print('[WARN] 失败 %d 张' % fail)

    model.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
