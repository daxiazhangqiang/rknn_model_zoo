#!/usr/bin/env python3
"""
quantize_bev_occ.py  (整合自 compare_rknn_onnx.py + 旧版 quantize_bev_occ.py)
---------------------------------------------------------------------------
把指定的 ONNX 转换成 RKNN (默认 INT8 量化) -> 在 rk3588 板端 NPU 推理 ->
与 ONNX 逐输出对比精度 (cosine/mse) -> 测板端帧率。

默认输入图像: [H,W] = 384x640  (RAW [0,255], NCHW float32)。

归一化策略 (默认 --norm):
  - 这类导出模型 input 直接进第一个 Conv (内部不做归一化), 期望归一化 RGB。
  - 因此 rknn.config 里设 mean_values/std_values (ImageNet),
    RKNN 会在量化标定与推理时对 RAW [0,255] 输入自行做 (x-mean)/std。
  - onnx 对比时喂 (x-mean)/std 归一化数据 (与 RKNN 内部归一化等价)。
  - 标定数据 = RAW [0,255] 图像。
若模型本身不需要归一化 / 已在内部归一化, 用 --no-norm (旧 compare_rknn_onnx.py
行为): 不设 mean/std, onnx 与 rknn 喂同一份输入。

用法 (在 examples/bev_occ 目录):
  python quantize_bev_occ.py --onnx sepheads_fpn128_untrained.onnx
  python quantize_bev_occ.py --onnx sepheads_fpn128_untrained.onnx --no-quant   # fp16
  python quantize_bev_occ.py --onnx a.onnx --no-norm --data-format nhwc
  python quantize_bev_occ.py --onnx a.onnx --build-only                          # 只导出
  python quantize_bev_occ.py --onnx a.onnx --dataset my_ds.txt --input img.npy

注意: 若 ONNX 含 IsNaN/IsInf (板端 librknnrt.so 不支持, 推理会段错误),
      请先 fix_nan_inf_generic.py 处理后再转换。
"""
import argparse
import glob
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN

# ---------------- 默认输入图像 [H,W] = 384x640 (RAW [0,255]) ----------------
MEAN = [123.675, 116.28, 103.53]   # ImageNet 均值 (RGB)
STD = [58.395, 57.12, 57.375]      # ImageNet 方差 (RGB)
# 历史 5D 标定形状 (1,1,3,H,W) -> 其 [0] 即 4D (1,3,H,W), 可重排复用
RAW_INPUT_NPY = 'bench_input_640_384.npy'   # 默认 384x640 RAW 输入(供 benchmark)


def get_input_meta(onnx_path):
    m = onnx.load(onnx_path, load_external_data=False)
    inp = m.graph.input[0]
    shape = [d.dim_value if d.HasField('dim_value') else None
             for d in inp.type.tensor_type.shape.dim]
    return inp.name, tuple(shape)


def has_nan_inf_ops(onnx_path):
    m = onnx.load(onnx_path, load_external_data=False)
    return any(n.op_type in ('IsNaN', 'IsInf') for n in m.graph.node)


def _match_shape(a_shape, b_shape):
    """a 为 npy 实际 shape, b 为模型输入 shape (可能含 None 动态维)."""
    if len(a_shape) != len(b_shape):
        return False
    return all(d is None or s == d for s, d in zip(a_shape, b_shape))


def find_matching_npy(calib_dir, input_shape):
    files = []
    for p in sorted(glob.glob(os.path.join(calib_dir, '*.npy'))):
        try:
            s = np.load(p, mmap_mode='r').shape
        except Exception:
            continue
        if _match_shape(s, input_shape):
            files.append(p)
    return files


def write_dataset(txt_path, files):
    with open(txt_path, 'w') as f:
        for p in files:
            f.write(os.path.abspath(p) + '\n')
    print(f'[calib] dataset {txt_path} ({len(files)} samples)')


def default_calib_dir(onnx_path, input_shape):
    """默认标定目录名: calib_data_<W>_<H>_<ndim>d (与现有 384x640 命名一致)."""
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    dims = [d if d is not None else 1 for d in input_shape]
    ndim = len(dims)
    if ndim == 4 and dims[0] == 1 and dims[1] == 3:      # [1,3,H,W]
        h, w = dims[2], dims[3]
        return os.path.join(onnx_dir, f'calib_data_{w}_{h}_4d')
    if ndim == 4:                                        # 其它 4D
        _, _, h, w = dims
        return os.path.join(onnx_dir, f'calib_data_{w}_{h}_4d')
    if ndim == 5 and dims[0] == 1 and dims[1] == 1:      # [1,1,3,H,W]
        _, _, _, h, w = dims
        return os.path.join(onnx_dir, f'calib_data_{w}_{h}_5d')
    if ndim >= 4:
        h, w = dims[-2], dims[-1]
        return os.path.join(onnx_dir, f'calib_data_{w}_{h}_{ndim}d')
    return os.path.join(onnx_dir, 'calib_data')


def prepare_calib(onnx_path, input_shape, calib_dir=None, num_default=2):
    """确保存在 shape 匹配的 RAW [0,255] 标定 npy 并生成 dataset txt.
    返回 dataset txt 路径 (绝对路径列表).
    优先级:
      1) 复用已存在的默认目录 (如 calib_data_640_384_4d) 中 shape 匹配的 npy;
      2) 复用历史 5D (1,1,3,H,W) 标定 (取 [0] 重排为 4D);
      3) 按模型输入 shape 生成 num_default 个随机 RAW [0,255] 样本.
    """
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    calib_dir = calib_dir or default_calib_dir(onnx_path, input_shape)
    os.makedirs(calib_dir, exist_ok=True)
    dims = [d if d is not None else 1 for d in input_shape]

    files = find_matching_npy(calib_dir, input_shape)

    # 复用历史 5D 标定 (1,1,3,H,W) -> 4D
    if not files and len(dims) == 4 and dims[0] == 1 and dims[1] == 3:
        h, w = dims[2], dims[3]
        legacy5d = os.path.join(onnx_dir, f'calib_data_{w}_{h}')
        legacy_shape = (1, 1, 3, h, w)
        for p in sorted(glob.glob(os.path.join(legacy5d, '*.npy'))):
            try:
                a = np.load(p)
            except Exception:
                continue
            if a.shape == legacy_shape:
                out = os.path.join(calib_dir,
                                   os.path.basename(p).replace('.npy', '_4d.npy'))
                if not os.path.exists(out):
                    np.save(out, np.ascontiguousarray(a[0], dtype=np.float32))
                    print(f'[calib] reused {p} -> {out}')
                files.append(out)

    # 生成随机 RAW [0,255] 样本
    if not files:
        rng = np.random.default_rng(0)
        shape = tuple(dims)
        for i in range(num_default):
            out = os.path.join(calib_dir, f'calib_rand_{i}.npy')
            if not os.path.exists(out):
                np.save(out, rng.integers(0, 256, size=shape).astype(np.float32))
            files.append(out)
        print(f'[calib] generated {num_default} random RAW samples shape={shape}')

    txt = os.path.join(onnx_dir, os.path.splitext(os.path.basename(calib_dir))[0] + '.txt')
    write_dataset(txt, files)
    return txt


def build_rknn(onnx_path, dataset, rknn_out, do_quant=True, norm=True):
    rknn = RKNN(verbose=False)
    cfg = dict(target_platform='rk3588')
    if norm:
        # 归一化由 mean/std 决定 (rknn-toolkit2 2.3.2 config 无 input_type);
        # 标定/推理输入均为 RAW [0,255].
        cfg['mean_values'] = [MEAN]
        cfg['std_values'] = [STD]
    rknn.config(**cfg)
    print(f'[rknn ] config: norm={norm} mean={MEAN if norm else None} '
          f'std={STD if norm else None}')
    assert rknn.load_onnx(model=onnx_path) == 0, 'load_onnx failed'
    print(f'[rknn ] building (do_quantization={do_quant}) ...')
    t0 = time.time()
    ret = rknn.build(do_quantization=do_quant, dataset=dataset if do_quant else None)
    print(f'[rknn ] build ret={ret} in {time.time()-t0:.1f}s')
    assert ret == 0, 'build failed'
    rknn.export_rknn(rknn_out)
    print(f'[rknn ] exported -> {rknn_out}')
    rknn.release()
    return rknn_out


def run_onnx(onnx_path, x):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    t0 = time.time()
    outs = sess.run(None, {input_name: x})
    print(f'[onnx ] {len(outs)} outputs, inference {time.time()-t0:.2f}s (CPU)')
    return outs


def run_rknn(rknn_path, x, device, data_format='nchw'):
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0, 'load_rknn failed'
    assert rknn.init_runtime(target=device) == 0, 'init_runtime failed'
    t0 = time.time()
    outs = rknn.inference(inputs=[x], data_format=data_format)
    print(f'[rknn ] inference {time.time()-t0:.2f}s ({device}, data_format={data_format})')
    rknn.release()
    return outs


def compare_outputs(onnx_outs, rknn_outs):
    assert len(onnx_outs) == len(rknn_outs), \
        f'output count mismatch: onnx={len(onnx_outs)} vs rknn={len(rknn_outs)}'
    print('\n========== 输出对比 (onnx vs rknn) ==========')
    for i, (a, b) in enumerate(zip(onnx_outs, rknn_outs)):
        b = np.asarray(b, dtype=np.float32)
        note = ''
        if a.shape != b.shape:
            if b.size == a.size:
                note = f'  (rknn shape {b.shape} -> reshape {a.shape})'
                b = b.reshape(a.shape)
            else:
                print(f'[out{i}] SHAPE MISMATCH: onnx={a.shape} rknn={b.shape}')
                continue
        diff = np.abs(a - b)
        denom = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12
        cos = float(np.dot(a.ravel(), b.ravel()) / denom)
        rel = float((diff / (np.abs(a) + 1e-6)).mean())
        print(f'[out{i}] shape={a.shape}  cosine={cos:.6f}  '
              f'max_abs_diff={diff.max():.6e}  mean_abs_diff={diff.mean():.6e}  '
              f'mse={np.mean(diff**2):.6e}  mean_rel_err={rel:.4%}{note}')


def measure_fps(rknn_path, x, device, data_format='nchw', iters=20, warmup=5):
    """板端测帧率. 与根目录 benchmark_rknn.py 对齐: core_mask=NPU_CORE_ALL (3 核全开);
    否则默认(auto/0)约慢 30%."""
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0, 'load_rknn failed'
    assert rknn.init_runtime(target=device, core_mask=RKNN.NPU_CORE_ALL) == 0, \
        'init_runtime failed'
    print(f'\n[fps  ] warm-up {warmup} runs ...')
    for _ in range(warmup):
        rknn.inference(inputs=[x], data_format=data_format)
    print(f'[fps  ] running {iters} timed iterations ...')
    start = time.perf_counter()
    for _ in range(iters):
        rknn.inference(inputs=[x], data_format=data_format)
    total = time.perf_counter() - start
    rknn.release()
    avg_ms = total / iters * 1000.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
    print(f'[fps  ] total {total:.3f}s for {iters} runs ({device})')
    print(f'[fps  ] average {avg_ms:.3f} ms/run')
    print(f'[fps  ] FPS = {fps:.2f}')
    return fps


def main():
    ap = argparse.ArgumentParser(description='ONNX -> RKNN(INT8) 量化 + 板端验证')
    ap.add_argument('--onnx', required=True, help='ONNX 模型路径')
    ap.add_argument('--device', default='rk3588', help='RKNN 目标设备 (默认 rk3588)')
    ap.add_argument('--rknn', default=None,
                    help='导出 .rknn 路径 (默认 ONNX 同级 + 同名, 仅替换后缀; '
                         'fp16 时为 <stem>_fp16.rknn)')
    ap.add_argument('--dataset', default=None,
                    help='标定 dataset txt 或目录 (默认自动复用/生成 384x640 RAW 标定)')
    ap.add_argument('--calib-dir', default=None,
                    help='标定 npy 目录 (默认按输入 shape 取名, 如 calib_data_640_384_4d)')
    ap.add_argument('--input', default=None,
                    help='对比用的输入 npy (默认取标定第一个; 与模型输入 shape 须一致)')
    ap.add_argument('--data-format', default=None,
                    help='rknn.inference data_format (默认 norm 时 nchw; '
                         'no-norm 时 >=5 维 nchw, 否则 nhwc)')
    ap.add_argument('--fps-iters', type=int, default=100, help='测帧率推理次数')
    ap.add_argument('--fps-warmup', type=int, default=10, help='测帧率预热次数')
    ap.add_argument('--build-only', action='store_true',
                    help='只构建/导出 rknn, 不做板端验证')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--quant', dest='quant', action='store_true', default=True,
                     help='构建时做 INT8 量化 (默认)')
    grp.add_argument('--no-quant', dest='quant', action='store_false',
                     help='不量化 (fp16)')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--norm', dest='norm', action='store_true', default=True,
                     help='RKNN 内部做 ImageNet mean/std 归一化, 喂 RAW [0,255] (默认)')
    grp.add_argument('--no-norm', dest='norm', action='store_false',
                     help='不设 mean/std, onnx 与 rknn 喂同一份输入')
    ap.add_argument('--no-fps', dest='fps', action='store_false', default=True,
                    help='跳过最后的帧率测试')
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        ap.error(f'onnx not found: {args.onnx}')
    if has_nan_inf_ops(args.onnx):
        print('WARNING: onnx contains IsNaN/IsInf -> board librknnrt will SEGFAULT!')
        print('         run first: python fix_nan_inf_generic.py <onnx> <fixed.onnx>')

    input_name, input_shape = get_input_meta(args.onnx)
    out_names = [o.name for o in onnx.load(args.onnx, load_external_data=False).graph.output]
    print(f'[onnx ] input "{input_name}" shape={input_shape}')
    print(f'[onnx ] outputs = {out_names}')

    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    onnx_dir = os.path.dirname(os.path.abspath(args.onnx))
    rknn_out = args.rknn or os.path.join(
        onnx_dir, f'{stem}{"" if args.quant else "_fp16"}.rknn')
    print(f'[rknn ] output path = {rknn_out}')

    # ---- 标定数据 ----
    dataset = args.dataset
    if dataset is None:
        dataset = prepare_calib(args.onnx, input_shape, args.calib_dir)
    else:
        if not os.path.exists(dataset) and os.path.exists(dataset + '.txt'):
            print(f'[dataset] "{dataset}" not found, fallback to "{dataset}.txt"')
            dataset = dataset + '.txt'
        if os.path.isdir(dataset):
            print(f'[dataset] "{dataset}" is a directory, scanning *.npy ...')
            files = find_matching_npy(dataset, input_shape)
            if not files:
                ap.error(f'no shape-matched npy in {dataset} for {input_shape}')
            dataset = os.path.join(onnx_dir, 'dataset_calib_user.txt')
            write_dataset(dataset, files)
    with open(dataset) as f:
        calib_files = [l.strip() for l in f if l.strip()]

    build_rknn(args.onnx, dataset, rknn_out, do_quant=args.quant, norm=args.norm)

    if args.build_only:
        print(f'\n[build ] done: {rknn_out}')
        return

    # ---- 输入与 data_format ----
    data_format = args.data_format or ('nchw' if (args.norm or len(input_shape) >= 5)
                                       else 'nhwc')
    print(f'[auto ] data_format = {data_format}, norm = {args.norm}')

    if args.input:
        x = np.load(args.input)
    else:
        x = np.load(calib_files[0]) if calib_files else None
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x is None:
        ap.error('no input npy available; pass --input')
    if x.shape != tuple(d if d is not None else x.shape[i]
                        for i, d in enumerate(input_shape)):
        print(f'WARNING: input {x.shape} != model input {input_shape}')

    # ---- onnx / rknn 推理对比 ----
    if args.norm:
        # 保存 RAW 输入供 benchmark 复用 (默认 384x640)
        np.save(os.path.join(onnx_dir, RAW_INPUT_NPY), x)
        mean = np.array(MEAN, np.float32).reshape(-1, 1, 1)
        std = np.array(STD, np.float32).reshape(-1, 1, 1)
        x_onnx = (x - mean) / std          # onnx 用归一化输入
    else:
        x_onnx = x                        # no-norm: 两边同输入

    onnx_outs = run_onnx(args.onnx, x_onnx)
    rknn_outs = run_rknn(rknn_out, x, args.device, data_format=data_format)
    compare_outputs(onnx_outs, rknn_outs)

    if args.fps:
        measure_fps(rknn_out, x, args.device, data_format=data_format,
                    iters=args.fps_iters, warmup=args.fps_warmup)


if __name__ == '__main__':
    main()
