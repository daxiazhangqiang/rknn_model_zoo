#!/usr/bin/env python3
"""Batch inference: run rknn_multi_task_demo on an image folder.

Usage:
    python process_img_folder.py \
        --demo ./rknn_multi_task_demo \
        --model ./model/multi_task_obt_pose_face.rknn \
        --img_dir ./model/imgs \
        --json_dir ./output_json
"""

import argparse
import glob
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Batch multi-task inference')
    parser.add_argument('--demo', type=str, required=True,
                        help='Path to rknn_multi_task_demo executable')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to RKNN model file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Input image folder')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Output JSON folder')
    args = parser.parse_args()

    # resolve to absolute path (handles "./demo" and "demo" uniformly)
    demo_path = os.path.abspath(args.demo)
    if not os.path.isfile(demo_path):
        print(f'[ERROR] executable not found: {demo_path}')
        sys.exit(1)
    if not os.access(demo_path, os.X_OK):
        print(f'[ERROR] executable not runnable: {demo_path}')
        sys.exit(1)

    # collect images
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
    img_paths = sorted(set(img_paths))

    if not img_paths:
        print(f'[ERROR] No images found in {args.img_dir}')
        sys.exit(1)

    os.makedirs(args.json_dir, exist_ok=True)

    print(f'Found {len(img_paths)} image(s) in {args.img_dir}')
    print(f'Model : {args.model}')
    print(f'JSON  : {args.json_dir}')
    print('-' * 50)

    total_time = 0.0
    success = 0
    crashed = 0

    for idx, img_path in enumerate(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(args.json_dir, f'{basename}.json')

        cmd = [demo_path, args.model, img_path, json_path]
        print(f'[{idx+1}/{len(img_paths)}] {os.path.basename(img_path)} ... ', end='', flush=True)

        t0 = time.time()
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - t0

        if ret.returncode == 0:
            print(f'OK  ({elapsed:.2f}s)')
            success += 1
            total_time += elapsed
        else:
            print(f'CRASH (exit={ret.returncode}, {elapsed:.2f}s)')
            crashed += 1
            # print stderr for debugging
            if ret.stderr:
                print(f'    stderr: {ret.stderr.strip()}')

    print('-' * 50)
    print(f'Total  : {len(img_paths)} images')
    print(f'Success: {success}')
    print(f'Crashed: {crashed}')
    if success > 0:
        print(f'Average time per image: {total_time / success:.3f}s')


if __name__ == '__main__':
    main()


