"""图像左右拼接脚本：将两个文件夹中同名图像水平拼接，方便对比查看。"""

import os
import cv2
import argparse

def concat_images(dir1, dir2, out_dir):
    """
    遍历 dir1 中所有图像，在 dir2 中查找同名文件，水平拼接后保存到 out_dir。

    Args:
        dir1: 第一个文件夹（左侧图像）
        dir2: 第二个文件夹（右侧图像）
        out_dir: 拼接结果保存文件夹
    """
    os.makedirs(out_dir, exist_ok=True)

    # 支持的图像扩展名
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 递归收集 dir1 中所有图像文件
    img1_paths = []
    for root, _, files in os.walk(dir1):
        for f in files:
            if os.path.splitext(f.lower())[1] in img_exts:
                img1_paths.append(os.path.join(root, f))

    print(f'dir1 图像数量: {len(img1_paths)}')
    matched = 0
    skipped = 0

    for path1 in img1_paths:
        fname = os.path.basename(path1)

        # 在 dir2 中递归查找同名文件
        path2 = None
        for root2, _, files2 in os.walk(dir2):
            if fname in files2:
                path2 = os.path.join(root2, fname)
                break

        if path2 is None:
            print(f'[跳过] 未找到同名文件: {fname}')
            skipped += 1
            continue

        # 读取并拼接
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None:
            print(f'[跳过] 读取失败: {path1}')
            skipped += 1
            continue
        if img2 is None:
            print(f'[跳过] 读取失败: {path2}')
            skipped += 1
            continue

        # 统一高度（以较高的为准，较矮的 padding）
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)

        if h1 < max_h:
            pad = max_h - h1
            img1 = cv2.copyMakeBorder(img1, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if h2 < max_h:
            pad = max_h - h2
            img2 = cv2.copyMakeBorder(img2, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        concat = cv2.hconcat([img1, img2])
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, concat)
        matched += 1

    print(f'拼接完成: 成功 {matched} 张, 跳过 {skipped} 张, 保存至 {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将两个文件夹中同名图像水平拼接')
    parser.add_argument('dir1', help='左侧图像文件夹')
    parser.add_argument('dir2', help='右侧图像文件夹')
    parser.add_argument('-o', '--out-dir', default=None, help='输出文件夹（默认: dir1 同级目录 _concat）')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.dir1.rstrip('/') + '_concat'
    concat_images(args.dir1, args.dir2, args.out_dir)