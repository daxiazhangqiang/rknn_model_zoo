// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                人脸识别（比对）可执行文件
    用法: rknn_mobileface_recognize <model_path> <image_path> <det_json> <output_dir> [--sim_thresh 0.5]

    输入的人脸检测结果(det_json)来自上一级检测环节，本程序不再做检测，
    只从 JSON 中读取 bbox 和 5 个关键点，完成对齐、特征提取与注册库比对。
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "file_utils.h"
#include "image_utils.h"
#include "mobileface_common.h"

static void print_usage(const char *prog) {
    printf("%s <model_path> <image_path> <det_json> <output_dir> [--sim_thresh <val>]\n", prog);
}

static std::string basename_of(const std::string &path) {
    size_t p = path.find_last_of('/');
    if (p == std::string::npos) {
        return path;
    }
    return path.substr(p + 1);
}

static std::string stem_of(const std::string &path) {
    std::string base = basename_of(path);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        return base.substr(0, dot);
    }
    return base;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const char *det_json = argv[3];
    const char *output_dir = argv[4];

    float sim_thresh = 0.5f;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--sim_thresh") == 0 && i + 1 < argc) {
            sim_thresh = (float)atof(argv[++i]);
        }
    }

    std::string face_dir = std::string(output_dir) + "/face_det";
    std::string feat_dir = std::string(output_dir) + "/face_features";
    make_dirs(face_dir);
    make_dirs(feat_dir);

    // 1. 读图
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(src_img));
    if (read_image(image_path, &src_img) != 0) {
        printf("read image fail! image_path=%s\n", image_path);
        return -1;
    }

    // 2. 解析检测结果 JSON
    char *jdata = NULL;
    int jsize = read_data_from_file(det_json, &jdata);
    if (jsize <= 0 || jdata == NULL) {
        printf("read detect json fail! json=%s\n", det_json);
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }
    std::string jtext(jdata, jsize);
    free(jdata);

    std::vector<face_info_t> faces;
    if (parse_detection_faces(jtext, faces) != 0) {
        printf("parse detect json fail! json=%s\n", det_json);
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }

    std::string image_name = basename_of(image_path);
    std::string stem = stem_of(image_path);

    // 没有人脸：提醒后直接退出
    if (faces.empty()) {
        printf("[%s] 未检测到人脸\n", image_name.c_str());
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return 0;
    }
    printf("[%s] 检测到 %d 张人脸\n", image_name.c_str(), (int)faces.size());

    // 3. 初始化识别模型
    rec_context_t rec_ctx;
    memset(&rec_ctx, 0, sizeof(rec_ctx));
    if (init_recognition_model(model_path, &rec_ctx) != 0) {
        printf("init recognition model fail!\n");
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }

    // 4. 加载注册库
    std::map<std::string, std::string> registry;
    if (load_registry(feat_dir, registry) != 0 || registry.empty()) {
        printf("警告: 注册库 %s 为空或不存在，无法比对\n", feat_dir.c_str());
    } else {
        printf("注册库共 %d 人\n", (int)registry.size());
    }

    // 5. 逐个人脸：对齐 -> 特征 -> 比对
    std::string result_json;
    if (faces.size() == 1) {
        result_json = "";  // 单个人脸输出单个对象
    } else {
        result_json = "[\n";  // 多个人脸输出数组
    }

    std::vector<unsigned char> aligned(112 * 112 * 3);
    for (size_t i = 0; i < faces.size(); i++) {
        align_face_to_112(&src_img, faces[i], aligned.data());

        // 保存 112x112 对齐人脸
        char face_name[256];
        snprintf(face_name, sizeof(face_name), "%s_%d.jpg", stem.c_str(), (int)i);
        image_buffer_t face_img;
        memset(&face_img, 0, sizeof(face_img));
        face_img.width = 112;
        face_img.height = 112;
        face_img.format = IMAGE_FORMAT_RGB888;
        face_img.virt_addr = aligned.data();
        face_img.size = 112 * 112 * 3;
        write_image((face_dir + "/" + face_name).c_str(), &face_img);

        // 特征提取
        std::vector<float> feature;
        if (inference_recognition_model(&rec_ctx, aligned.data(), 112, 112, feature) != 0) {
            printf("inference fail!\n");
            continue;
        }

        // 与注册库比对
        std::string best_uid;
        float best_sim = -1.0f;
        recognize_best(feat_dir, registry, rec_ctx.feat_dim, feature, best_uid, best_sim);

        bool matched = (!best_uid.empty() && best_sim >= sim_thresh);
        std::string uid = matched ? best_uid : "unknown";
        float sim = matched ? best_sim : 0.0f;
        std::string obj = make_result_object(image_name, uid, sim, matched);

        if (matched) {
            printf("[%s] face %d -> 识别为: %s (相似度 %.4f)\n", image_name.c_str(), (int)i,
                   uid.c_str(), best_sim);
        } else {
            printf("[%s] face %d -> 未注册 (最高相似度 %.4f)\n", image_name.c_str(), (int)i,
                   best_sim < 0 ? 0.0f : best_sim);
        }

        if (faces.size() == 1) {
            result_json = obj;
        } else {
            if (i > 0) result_json += ",\n";
            result_json += "  " + obj;
        }
    }

    if (faces.size() > 1) {
        result_json += "\n]\n";
    }

    // 6. 写识别结果 JSON
    std::string result_path = std::string(output_dir) + "/face_recognition_result.json";
    FILE *fp = fopen(result_path.c_str(), "w");
    if (fp != NULL) {
        fwrite(result_json.data(), 1, result_json.size(), fp);
        fclose(fp);
        printf("识别结果已保存: %s\n", result_path.c_str());
    } else {
        printf("写识别结果失败: %s\n", result_path.c_str());
    }

    release_recognition_model(&rec_ctx);
    if (src_img.virt_addr != NULL) {
        free(src_img.virt_addr);
    }
    return 0;
}
