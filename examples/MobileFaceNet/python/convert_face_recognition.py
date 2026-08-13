from rknn.api import RKNN

RECOGNITION_ONNX_PATH='../model/w600k_mbf_bs1.onnx'
DATASET_TXT='../model/dataset_recognition.txt'
def convert_onnx_to_rknn(onnx_path, rknn_path, target='rk3588',
                         quantize=True, dataset_txt=None):
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],   # (x - 127.5) / 127.5 -> [-1, 1]
        std_values=[[127.5, 127.5, 127.5]],    # ArcFace(w600k_mbf) 112x112: std=127.5
        target_platform=target
    )
    # w600k_mbf (buffalo_s 的 ArcFace) 要求 RGB 输入，归一化 (x-127.5)/127.5。
    # 归一化参数会烧进 RKNN，板端推理只需喂原始 RGB 图像（uint8, 0~255）。
    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=quantize, dataset=dataset_txt)
    rknn.export_rknn(rknn_path)
    rknn.release()
    print(f"转换完成: {rknn_path}")

if __name__ == '__main__':
    # MobileFaceNet
    convert_onnx_to_rknn(
        onnx_path=RECOGNITION_ONNX_PATH,
        rknn_path='../model/mobilefacenet.rknn',
        target='rk3588',
        quantize=True,
        dataset_txt=DATASET_TXT
    )