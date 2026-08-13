#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <string>
#include <vector>

#include "json.h"
#include "mobileface_common.h"
#include "file_utils.h"

// 标准 112x112 人脸的 5 个关键点参考坐标（InsightFace/ArcFace 标准）
// 顺序：左眼，右眼，鼻尖，左嘴角，右嘴角
static const float REFERENCE_KPS[5][2] = {
    {38.2946f, 51.6963f},  // 左眼
    {73.5318f, 51.5014f},  // 右眼
    {56.0252f, 71.7366f},  // 鼻尖
    {41.5493f, 92.3655f},  // 左嘴角
    {70.7299f, 92.2041f}   // 右嘴角
};

static const int ALIGN_SIZE = 112;
static const int MAX_REGISTER = 100;

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static std::string dir_join(const std::string &dir, const std::string &name) {
    return dir + "/" + name;
}

static int write_text_file(const std::string &path, const std::string &content) {
    FILE *fp = fopen(path.c_str(), "w");
    if (fp == NULL) {
        printf("open %s fail!\n", path.c_str());
        return -1;
    }
    fwrite(content.data(), 1, content.size(), fp);
    fclose(fp);
    return 0;
}

int write_bin_file(const std::string &path, const void *data, size_t size) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp == NULL) {
        printf("open %s fail!\n", path.c_str());
        return -1;
    }
    fwrite(data, 1, size, fp);
    fclose(fp);
    return 0;
}

int make_dirs(const std::string &path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' && !cur.empty()) {
            mkdir(cur.c_str(), 0755);
        }
    }
    if (!path.empty()) {
        mkdir(path.c_str(), 0755);
    }
    return 0;
}

int init_recognition_model(const char *model_path, rec_context_t *ctx) {
    int ret;
    int model_len = 0;
    char *model = NULL;
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    rknn_context rknn_ctx = 0;
    ret = rknn_init(&rknn_ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query IO num fail! ret=%d\n", ret);
        rknn_destroy(rknn_ctx);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    memset(input_attrs.data(), 0, input_attrs.size() * sizeof(rknn_tensor_attr));
    for (int i = 0; i < (int)io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query input attr fail! ret=%d\n", ret);
            rknn_destroy(rknn_ctx);
            return -1;
        }
    }

    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    memset(output_attrs.data(), 0, output_attrs.size() * sizeof(rknn_tensor_attr));
    for (int i = 0; i < (int)io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query output attr fail! ret=%d\n", ret);
            rknn_destroy(rknn_ctx);
            return -1;
        }
    }

    ctx->rknn_ctx = rknn_ctx;
    ctx->io_num = io_num;
    ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx->input_attrs, input_attrs.data(), io_num.n_input * sizeof(rknn_tensor_attr));
    ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(ctx->output_attrs, output_attrs.data(), io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        ctx->model_channel = input_attrs[0].dims[1];
        ctx->model_height = input_attrs[0].dims[2];
        ctx->model_width = input_attrs[0].dims[3];
    } else {
        ctx->model_height = input_attrs[0].dims[1];
        ctx->model_width = input_attrs[0].dims[2];
        ctx->model_channel = input_attrs[0].dims[3];
    }
    ctx->feat_dim = output_attrs[0].n_elems;
    printf("model input: %dx%dx%d, feature dim: %d\n", ctx->model_width, ctx->model_height,
           ctx->model_channel, ctx->feat_dim);
    return 0;
}

int release_recognition_model(rec_context_t *ctx) {
    if (ctx->input_attrs != NULL) {
        free(ctx->input_attrs);
        ctx->input_attrs = NULL;
    }
    if (ctx->output_attrs != NULL) {
        free(ctx->output_attrs);
        ctx->output_attrs = NULL;
    }
    if (ctx->rknn_ctx != 0) {
        rknn_destroy(ctx->rknn_ctx);
        ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_recognition_model(rec_context_t *ctx, const unsigned char *rgb_data, int w, int h,
                                std::vector<float> &feature) {
    int ret;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = (uint32_t)(w * h * ctx->model_channel);
    inputs[0].buf = (void *)rgb_data;

    ret = rknn_inputs_set(ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(ctx->rknn_ctx, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    std::vector<rknn_output> outputs(ctx->io_num.n_output);
    memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
    for (int i = 0; i < (int)ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(ctx->rknn_ctx, ctx->io_num.n_output, outputs.data(), NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    int dim = ctx->feat_dim;
    const float *out = (const float *)outputs[0].buf;
    feature.resize(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        feature[i] = out[i];
        norm += out[i] * out[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        for (int i = 0; i < dim; i++) {
            feature[i] /= norm;
        }
    }

    rknn_outputs_release(ctx->rknn_ctx, ctx->io_num.n_output, outputs.data());
    return 0;
}

static int parse_one_face(const jsonmini::Value &v, face_info_t &f) {
    const jsonmini::Value &bbox = v.get("bbox");
    if (!bbox.is_array() || bbox.size() < 4) {
        return -1;
    }
    f.left = (float)bbox.at(0).as_number();
    f.top = (float)bbox.at(1).as_number();
    f.right = (float)bbox.at(2).as_number();
    f.bottom = (float)bbox.at(3).as_number();
    f.score = (float)v.get("score").as_number();

    const jsonmini::Value &kps = v.get("keypoints");
    if (!kps.is_object()) {
        return -1;
    }
    for (int i = 0; i < 5; i++) {
        char key[16];
        snprintf(key, sizeof(key), "kpt%d", i);
        const jsonmini::Value &k = kps.get(key);
        float x = (float)k.get("x").as_number();
        float y = (float)k.get("y").as_number();
        float s = (float)k.get("score").as_number();
        // 容错：关键点裁剪到 bbox 范围内
        x = clampf(x, f.left, f.right);
        y = clampf(y, f.top, f.bottom);
        f.kpts[i].x = x;
        f.kpts[i].y = y;
        f.kpts[i].score = s;
        f.valid[i] = (s > 0.5f);  // 容错：只使用高置信度关键点
    }
    return 0;
}

int parse_detection_faces(const std::string &json_text, std::vector<face_info_t> &faces) {
    faces.clear();
    jsonmini::Value root = jsonmini::parse(json_text);
    const jsonmini::Value &model_results = root.get("model_results");
    if (!model_results.is_object()) {
        return -1;
    }
    const jsonmini::Value &face_val = model_results.get("face");
    if (face_val.is_array()) {
        for (size_t i = 0; i < face_val.size(); i++) {
            face_info_t f;
            if (parse_one_face(face_val.at(i), f) == 0) {
                faces.push_back(f);
            }
        }
    } else if (face_val.is_object()) {
        face_info_t f;
        if (parse_one_face(face_val, f) == 0) {
            faces.push_back(f);
        }
    } else {
        return -1;
    }
    return 0;
}

int align_face_to_112(const image_buffer_t *src_img, const face_info_t &face, unsigned char *out_rgb) {
    // 收集有效关键点
    float sx[5], sy[5], tx[5], ty[5];
    int n = 0;
    for (int i = 0; i < 5; i++) {
        if (face.valid[i]) {
            sx[n] = face.kpts[i].x;
            sy[n] = face.kpts[i].y;
            tx[n] = REFERENCE_KPS[i][0];
            ty[n] = REFERENCE_KPS[i][1];
            n++;
        }
    }

    // 相似变换矩阵 M（源图 -> 112x112 目标）
    float m00, m01, m10, m11, mtx, mty;
    bool use_bbox_fallback = false;

    if (n >= 2) {
        float mean_sx = 0, mean_sy = 0, mean_tx = 0, mean_ty = 0;
        for (int i = 0; i < n; i++) {
            mean_sx += sx[i];
            mean_sy += sy[i];
            mean_tx += tx[i];
            mean_ty += ty[i];
        }
        mean_sx /= n;
        mean_sy /= n;
        mean_tx /= n;
        mean_ty /= n;

        float A = 0, B = 0, D = 0;
        for (int i = 0; i < n; i++) {
            float ux = sx[i] - mean_sx;
            float uy = sy[i] - mean_sy;
            float vx = tx[i] - mean_tx;
            float vy = ty[i] - mean_ty;
            A += (vx * ux + vy * uy);
            B += (vy * ux - vx * uy);
            D += (ux * ux + uy * uy);
        }
        if (D <= 1e-6f) {
            use_bbox_fallback = true;
        } else {
            // 相似变换：c = (A + iB) / D
            float creal = A / D;
            float cimag = B / D;
            m00 = creal;
            m01 = -cimag;
            m10 = cimag;
            m11 = creal;
            mtx = mean_tx - (creal * mean_sx - cimag * mean_sy);
            mty = mean_ty - (cimag * mean_sx + creal * mean_sy);
        }
    } else {
        use_bbox_fallback = true;
    }

    if (use_bbox_fallback) {
        // 关键点不足时退化：按 bbox 等比缩放并居中到 112x112（无旋转）
        float box_w = face.right - face.left;
        float box_h = face.bottom - face.top;
        if (box_w <= 0) box_w = 112.0f;
        if (box_h <= 0) box_h = 112.0f;
        float s = 112.0f / (box_w > box_h ? box_w : box_h);
        float cx = (face.left + face.right) / 2.0f;
        float cy = (face.top + face.bottom) / 2.0f;
        m00 = s;
        m01 = 0;
        m10 = 0;
        m11 = s;
        mtx = 56.0f - s * cx;
        mty = 56.0f - s * cy;
    }

    // 逆变换：目标像素 -> 源图坐标
    float det = m00 * m11 - m01 * m10;
    if (fabsf(det) < 1e-9f) det = 1.0f;
    float im00 = m11 / det;
    float im01 = -m01 / det;
    float im10 = -m10 / det;
    float im11 = m00 / det;

    int sw = src_img->width;
    int sh = src_img->height;
    const unsigned char *sdata = src_img->virt_addr;

    for (int y = 0; y < ALIGN_SIZE; y++) {
        for (int x = 0; x < ALIGN_SIZE; x++) {
            float dx = x - mtx;
            float dy = y - mty;
            float src_x = im00 * dx + im01 * dy;
            float src_y = im10 * dx + im11 * dy;

            unsigned char *dst = out_rgb + (y * ALIGN_SIZE + x) * 3;
            if (src_x >= 0 && src_x <= sw - 1 && src_y >= 0 && src_y <= sh - 1) {
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                if (x1 >= sw) x1 = sw - 1;
                if (y1 >= sh) y1 = sh - 1;
                float fx = src_x - x0;
                float fy = src_y - y0;

                const unsigned char *p00 = sdata + (y0 * sw + x0) * 3;
                const unsigned char *p01 = sdata + (y0 * sw + x1) * 3;
                const unsigned char *p10 = sdata + (y1 * sw + x0) * 3;
                const unsigned char *p11 = sdata + (y1 * sw + x1) * 3;
                for (int c = 0; c < 3; c++) {
                    float top = p00[c] * (1 - fx) + p01[c] * fx;
                    float bot = p10[c] * (1 - fx) + p11[c] * fx;
                    float val = top * (1 - fy) + bot * fy;
                    dst[c] = (unsigned char)(val + 0.5f);
                }
            } else {
                dst[0] = 0;
                dst[1] = 0;
                dst[2] = 0;
            }
        }
    }
    return 0;
}

int load_registry(const std::string &feat_dir, std::map<std::string, std::string> &registry) {
    registry.clear();
    std::string path = dir_join(feat_dir, "registry.json");
    char *data = NULL;
    int size = read_data_from_file(path.c_str(), &data);
    if (size <= 0 || data == NULL) {
        return -1;  // 注册表不存在或为空
    }
    std::string text(data, size);
    free(data);

    jsonmini::Value root = jsonmini::parse(text);
    if (!root.is_object()) {
        return -1;
    }
    for (std::map<std::string, jsonmini::Value>::const_iterator it = root.obj.begin();
         it != root.obj.end(); ++it) {
        registry[it->first] = it->second.as_string();
    }
    return 0;
}

int save_registry(const std::string &feat_dir, const std::map<std::string, std::string> &registry) {
    std::string path = dir_join(feat_dir, "registry.json");
    std::string out = "{\n";
    int idx = 0;
    for (std::map<std::string, std::string>::const_iterator it = registry.begin(); it != registry.end(); ++it) {
        if (idx > 0) {
            out += ",\n";
        }
        out += "  " + jsonmini::escape_string(it->first) + ": " + jsonmini::escape_string(it->second);
        idx++;
    }
    out += "\n}\n";
    return write_text_file(path, out);
}

int recognize_best(const std::string &feat_dir, const std::map<std::string, std::string> &registry,
                   int dim, const std::vector<float> &query, std::string &best_uid, float &best_sim) {
    best_uid = "";
    best_sim = -1.0f;

    for (std::map<std::string, std::string>::const_iterator it = registry.begin(); it != registry.end(); ++it) {
        std::string bin_path = dir_join(feat_dir, it->second);
        char *data = NULL;
        int size = read_data_from_file(bin_path.c_str(), &data);
        if (size < (int)(dim * sizeof(float)) || data == NULL) {
            if (data != NULL) {
                free(data);
            }
            continue;
        }
        const float *gal = (const float *)data;
        float norm = 0.0f;
        std::vector<float> feat(dim);
        for (int i = 0; i < dim; i++) {
            feat[i] = gal[i];
            norm += gal[i] * gal[i];
        }
        norm = sqrtf(norm);
        if (norm > 1e-8f) {
            for (int i = 0; i < dim; i++) {
                feat[i] /= norm;
            }
        }
        free(data);

        float sim = 0.0f;
        for (int i = 0; i < dim; i++) {
            sim += query[i] * feat[i];
        }
        if (sim > best_sim) {
            best_sim = sim;
            best_uid = it->first;
        }
    }
    return 0;
}

std::string make_result_object(const std::string &image_name, const std::string &uid, float similarity,
                               bool matched) {
    std::string out = "{\"image_name\": " + jsonmini::escape_string(image_name) + ", \"UID\": " +
                      jsonmini::escape_string(uid) + ", \"similarity\": ";
    if (!matched) {
        out += "0.0";
    } else {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.6f", similarity);
        out += buf;
    }
    out += "}";
    return out;
}
