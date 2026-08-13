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
                人脸注册可执行文件
    用法: rknn_mobileface_register <model_path> <image_path> <det_json> <output_dir> --uid <UID>

    从检测结果 JSON 中读取 bbox 和 5 个关键点，对齐到 112x112 后提取特征，
    保存到 output_dir/face_features/<N>.bin，并更新 registry.json。
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

static const int MAX_REGISTER = 100;

static void print_usage(const char *prog) {
    printf("%s <model_path> <image_path> <det_json> <output_dir> --uid <UID>\n", prog);
}

static std::string stem_of(const std::string &path) {
    size_t p = path.find_last_of('/');
    std::string base = (p == std::string::npos) ? path : path.substr(p + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        return base.substr(0, dot);
    }
    return base;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        print_usage(argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const char *det_json = argv[3];
    const char *output_dir = argv[4];

    std::string uid;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--uid") == 0 && i + 1 < argc) {
            uid = argv[++i];
        }
    }
    if (uid.empty()) {
        printf("缺少 --uid 参数\n");
        print_usage(argv[0]);
        return -1;
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

    if (faces.empty()) {
        printf("图像中未检测到人脸\n");
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }

    // 3. 初始化识别模型
    rec_context_t rec_ctx;
    memset(&rec_ctx, 0, sizeof(rec_ctx));
    if (init_recognition_model(model_path, &rec_ctx) != 0) {
        printf("init recognition model fail!\n");
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }

    // 4. 选取最大人脸进行注册
    int best = 0;
    float best_area = -1.0f;
    for (size_t i = 0; i < faces.size(); i++) {
        float area = (faces[i].right - faces[i].left) * (faces[i].bottom - faces[i].top);
        if (area > best_area) {
            best_area = area;
            best = (int)i;
        }
    }
    printf("选取最大人脸进行注册 (共 %d 张人脸)\n", (int)faces.size());

    // 5. 对齐 + 特征提取
    std::vector<unsigned char> aligned(112 * 112 * 3);
    align_face_to_112(&src_img, faces[best], aligned.data());

    image_buffer_t face_img;
    memset(&face_img, 0, sizeof(face_img));
    face_img.width = 112;
    face_img.height = 112;
    face_img.format = IMAGE_FORMAT_RGB888;
    face_img.virt_addr = aligned.data();
    face_img.size = 112 * 112 * 3;
    std::string face_path = face_dir + "/" + uid + ".jpg";
    write_image(face_path.c_str(), &face_img);
    printf("对齐人脸已保存: %s\n", face_path.c_str());

    std::vector<float> feature;
    if (inference_recognition_model(&rec_ctx, aligned.data(), 112, 112, feature) != 0) {
        printf("inference fail!\n");
        release_recognition_model(&rec_ctx);
        if (src_img.virt_addr != NULL) free(src_img.virt_addr);
        return -1;
    }

    // 6. 注册：更新或新增
    std::map<std::string, std::string> registry;
    load_registry(feat_dir, registry);

    std::string bin_name;
    std::map<std::string, std::string>::iterator it = registry.find(uid);
    if (it != registry.end()) {
        bin_name = it->second;
        printf("UID %s 已存在，更新特征到 %s\n", uid.c_str(), bin_name.c_str());
    } else {
        if ((int)registry.size() >= MAX_REGISTER) {
            printf("注册已满(%d 人)，无法注册新用户 %s\n", MAX_REGISTER, uid.c_str());
            release_recognition_model(&rec_ctx);
            if (src_img.virt_addr != NULL) free(src_img.virt_addr);
            return -1;
        }
        // 计算下一个 bin 编号：取现有最大编号 + 1
        int next_id = 0;
        for (it = registry.begin(); it != registry.end(); ++it) {
            int id = atoi(it->second.c_str());
            if (id >= next_id) {
                next_id = id + 1;
            }
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%d.bin", next_id);
        bin_name = buf;
        registry[uid] = bin_name;
        printf("新注册 UID %s -> %s\n", uid.c_str(), bin_name.c_str());
    }

    write_bin_file(feat_dir + "/" + bin_name, feature.data(), feature.size() * sizeof(float));
    save_registry(feat_dir, registry);
    printf("特征已保存: %s/%s\n", feat_dir.c_str(), bin_name.c_str());

    release_recognition_model(&rec_ctx);
    if (src_img.virt_addr != NULL) {
        free(src_img.virt_addr);
    }
    return 0;
}
