import os
import sys
import json
import cv2

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


# 为每个类别生成固定颜色的调色板，不同类别颜色不同，且每次运行颜色一致
COLOR_PALETTE = []
for i in range(len(CLASSES)):
    # 使用 HSV 色相环均匀分布，保证颜色差异明显
    hue = (i * 137.508) % 360  # 黄金角度 137.508°，确保颜色分布均匀且不重复
    sat = 200
    val = 220
    # HSV -> BGR (OpenCV 使用 BGR)
    h = hue / 60.0
    s = sat / 255.0
    v = val / 255.0
    hi = int(h) % 6
    f = h - int(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    COLOR_PALETTE.append((int(b * 255), int(g * 255), int(r * 255)))


if __name__ == "__main__":
    # 解析命令行参数，有两个参数，第一个参数是一个图片路径，第二个参数是一个json文件路径
    # 加一个参数，第三个参数为画上检测框的图片保存路径
    if len(sys.argv) != 4:
        print("Usage: python validate_json_det.py <image_path> <json_path> <output_image_path>")
        exit(1)     

    image_path = sys.argv[1]
    json_path = sys.argv[2]
    output_image_path = sys.argv[3]

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        exit(1)

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found - {json_path}")
        exit(1)
    # 读取image_path的图片和json_path的json文件，将json文件中的检测框画在图片上，并保存为第三个参数的图片路径
    # 
    image = cv2.imread(image_path)    
    with open(json_path, 'r') as f:
        detections = json.load(f)
        for detection in detections:
            bbox = detection.get("bbox", [])
            score = detection.get("score", 0)
            cls_id = detection.get("class", 0)
            if len(bbox) == 4:
                # 判断deteciton中class_id跟class_name小写字母字符串是否完全相同，不相同则警示打印，跳过当前的框的可视化
                if 0 <= cls_id < len(CLASSES):
                    class_name = CLASSES[cls_id].strip().lower()
                    if class_name != detection.get("class_name", "").strip().lower():
                        print(f"Warning: class_id {cls_id} does not match class_name '{detection.get('class_name', '')}'")
                        continue
                else:
                    print(f"Warning: class_id {cls_id} is out of range for CLASSES")
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                # 把不同class_id的框画成不同颜色，使用CLASSES中的类别名作为标签
                color = (0, 255, 0)  # 默认绿色
                if 0 <= cls_id < len(CLASSES):
                    color = COLOR_PALETTE[cls_id]  # 根据类别ID获取调色板中的颜色
                print(f"cls_id={cls_id},color={color},bbox={bbox},score={score}")
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 根据类别ID绘制标签
                if 0 <= cls_id < len(CLASSES):
                    cv2.putText(image, CLASSES[cls_id], (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_image_path, image)

