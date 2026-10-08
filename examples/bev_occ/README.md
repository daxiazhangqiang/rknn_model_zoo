# bev_occ

BEV-OCC 感知模型的板端推理示例，一个模型三个输出头：

| 输出 | shape | 含义 |
| --- | --- | --- |
| `static_seg_out` | `(1, 10, 48, 80)` | 静态障碍物语义分割 |
| `dynamic_seg_out` | `(1, 4, 48, 80)` | 动态障碍物语义分割 |
| `bev_occ_out` | `(1, 11, 64, 64)` | BEV 占据栅格 |

## 1. 模型信息

- 模型文件：`model/bevocc_96_lossweight2.rknn`（量化脚本：`python/quantize_bev_occ.py`）
- 输入：`input`，NHWC **uint8 RGB [0,255]**，`(1, 384, 640, 3)`
- 归一化 `mean=[123.675,116.28,103.53]`、`std=[58.395,57.12,57.375]`
  已经固化在 rknn 内部，**推理时直接喂 RAW 图像，不要再自己减均值除方差**
- 相机内外参已经固化进模型，无需用户输入

## 2. 预处理 / 后处理

- 预处理：等比例缩放，**短边对齐，长边居中裁掉**，再缩放到 640x384；
  可视化时按同样的几何关系把结果逆映射回输入图像的原始尺寸
- 后处理：三个输出按通道取 `argmax` 得到类别图，叠加上色

## 3. Python 推理

```bash
# 单张图像
/home/robot/miniconda3/envs/rknn/bin/python python/inference.py \
    model/bevocc_96_lossweight2.rknn imgs/000017_resize.jpg

# 图像文件夹 (输出目录缺省为 bevocc_out)
/home/robot/miniconda3/envs/rknn/bin/python python/inference.py \
    model/bevocc_96_lossweight2.rknn imgs/ bevocc_out
```

输出目录结构：

```
bevocc_out/000017_resize/000017_resize_static.jpg    静态分割叠加到输入图像
bevocc_out/000017_resize/000017_resize_dynamic.jpg   动态分割叠加到输入图像
bevocc_out/000017_resize/000017_resize_bevocc.jpg    BEV 输出上色 (64x64 放大)
```

可选参数：`--resize cover|stretch`（缺省 `cover`）、`--alpha`（叠加透明度）。

## 4. CPP 推理

编译（在仓库根目录）：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d bev_occ
```

运行：

```bash
cd install/rk3588_linux_aarch64/rknn_bev_occ_demo
./rknn_bev_occ_demo model/bevocc_96_lossweight2.rknn <image_or_dir> [out_dir]
```

可视化由宏 `ENABLE_VISUALIZATION` 控制，正式发布不需要可视化时编译关闭：

```bash
cd build && mkdir -p build_bev_occ_noviz && cd build_bev_occ_noviz
cmake ../../examples/bev_occ/cpp -DTARGET_SOC=rk3588 \
      -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DENABLE_VISUALIZATION=OFF && make -j4
```

## 5. 测试

```bash
/home/robot/miniconda3/envs/rknn/bin/python python/inference.py \
    model/bevocc_96_lossweight2.rknn imgs/
```

结果（rk3588 三核，单张约 70~80 ms）：

- `imgs/raw_20260904_151309_f0023.png`：static floor 52.5% / wall 31.5% / cabinet 12.1%，
  dynamic background 92.1% / person 7.9%
- `imgs/000017_resize.jpg`：static sofa 31.0% / wall 22.8% / floor 22.6%，dynamic 全背景
- `bev_occ_out` 当前基本全为 `unknown` 类，属于模型本身（内参暂时不准）的表现
