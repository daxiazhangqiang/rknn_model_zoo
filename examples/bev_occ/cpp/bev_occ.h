// BEV-OCC 感知模型 (静态分割 / 动态分割 / BEV 占据) RKNN 推理接口
#ifndef _RKNN_DEMO_BEV_OCC_H_
#define _RKNN_DEMO_BEV_OCC_H_

#include "rknn_api.h"
#include "common.h"

/*
 * 可视化开关
 *   1 : 输出三张可视化图像 (static / dynamic / bevocc)
 *   0 : 只跑推理, 不写任何图像 (正式发布用)
 * 编译时可通过 -DENABLE_VISUALIZATION=0 覆盖
 */
#ifndef ENABLE_VISUALIZATION
#define ENABLE_VISUALIZATION 1
#endif

// 三个 head 的类别数 (与可视化调色板一一对应)
#define BEV_OCC_STATIC_CLASSES  10
#define BEV_OCC_DYNAMIC_CLASSES 4
#define BEV_OCC_BEV_CLASSES     11

// 动态分割里 background 类别不参与叠加
#define BEV_OCC_DYNAMIC_BACKGROUND 0

// 可视化参数
#define BEV_OCC_OVERLAY_ALPHA   0.55f
#define BEV_OCC_BEV_VIEW_SIZE   384

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

/*
 * 预处理几何关系 (原图 -> 模型输入)
 * 预处理策略: 等比例缩放, 短边对齐, 长边居中裁掉
 */
typedef struct {
    int src_left;      // 裁剪区域左上角 x (原图坐标)
    int src_top;       // 裁剪区域左上角 y
    int src_w;         // 裁剪区域宽
    int src_h;         // 裁剪区域高
    int src_img_w;     // 原图宽
    int src_img_h;     // 原图高
} bev_occ_transform_t;

// 三个 head 的 argmax 类别图 (尺寸均取自模型输出)
typedef struct {
    int static_w, static_h;
    int dynamic_w, dynamic_h;
    int bev_w, bev_h;
    unsigned char* static_label;
    unsigned char* dynamic_label;
    unsigned char* bev_label;
    bev_occ_transform_t transform;
} bev_occ_result_t;

int init_bev_occ_model(const char* model_path, rknn_app_context_t* app_ctx);

int release_bev_occ_model(rknn_app_context_t* app_ctx);

int inference_bev_occ_model(rknn_app_context_t* app_ctx, image_buffer_t* img,
                            bev_occ_result_t* result);

void release_bev_occ_result(bev_occ_result_t* result);

// 类别名, 供日志输出使用
const char* bev_occ_static_name(int id);
const char* bev_occ_dynamic_name(int id);
const char* bev_occ_bev_name(int id);

#if ENABLE_VISUALIZATION
/*
 * 保存三张可视化图像
 *   static_path  : 静态分割叠加到输入图像
 *   dynamic_path : 动态分割叠加到输入图像 (background 不叠加)
 *   bev_path     : BEV 占据输出上色并放大 (不叠加)
 */
int save_bev_occ_visualization(const image_buffer_t* src_image,
                               const bev_occ_result_t* result,
                               const char* static_path,
                               const char* dynamic_path,
                               const char* bev_path);
#endif

#endif  // _RKNN_DEMO_BEV_OCC_H_
