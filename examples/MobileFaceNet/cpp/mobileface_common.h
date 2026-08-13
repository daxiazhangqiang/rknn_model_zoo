#ifndef RKNN_MOBILEFACE_COMMON_H_
#define RKNN_MOBILEFACE_COMMON_H_

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "common.h"

// 5 个关键点：左眼，右眼，鼻子，左嘴角，右嘴角
typedef struct {
    float x;
    float y;
    float score;
} face_kpt_t;

typedef struct {
    float left;
    float top;
    float right;
    float bottom;
    float score;
    face_kpt_t kpts[5];
    bool valid[5];  // 关键点 score > 0.5 视为有效
} face_info_t;

// 识别模型上下文
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_width;
    int model_height;
    int model_channel;
    int feat_dim;  // 输出特征维度
} rec_context_t;

// 初始化 / 释放识别模型
int init_recognition_model(const char *model_path, rec_context_t *ctx);
int release_recognition_model(rec_context_t *ctx);

// 输入 112x112 RGB888，输出 L2 归一化后的特征（长度 ctx->feat_dim）
int inference_recognition_model(rec_context_t *ctx, const unsigned char *rgb_data, int w, int h,
                                std::vector<float> &feature);

// 从检测结果 JSON 中解析人脸（bbox + 5 关键点）
int parse_detection_faces(const std::string &json_text, std::vector<face_info_t> &faces);

// 用 5 点关键点做相似变换，把人脸对齐到 112x112，输出 RGB888（112*112*3）
int align_face_to_112(const image_buffer_t *src_img, const face_info_t &face, unsigned char *out_rgb);

// 注册表读写
int load_registry(const std::string &feat_dir, std::map<std::string, std::string> &registry);
int save_registry(const std::string &feat_dir, const std::map<std::string, std::string> &registry);

// 在注册库中查找与查询特征最相似的人
int recognize_best(const std::string &feat_dir, const std::map<std::string, std::string> &registry,
                   int dim, const std::vector<float> &query, std::string &best_uid, float &best_sim);

// 生成单条识别结果 JSON 对象字符串
std::string make_result_object(const std::string &image_name, const std::string &uid, float similarity,
                               bool matched);

// 写二进制文件
int write_bin_file(const std::string &path, const void *data, size_t size);

// 递归创建目录
int make_dirs(const std::string &path);

#endif  // RKNN_MOBILEFACE_COMMON_H_
