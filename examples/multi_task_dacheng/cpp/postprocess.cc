// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "multi_task.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

#include "image_drawing.h"
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process_u8(uint8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t thres_u8 = qnt_f32_to_affine_u8(threshold, zp, scale);

    for (int i = 0; i < grid_h; ++i)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            uint8_t box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence >= thres_u8)
            {
                int offset = i * grid_w + j;
                uint8_t *in_ptr = input + offset;

                uint8_t maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    uint8_t prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_u8)
                {
                    float box_x = (deqnt_affine_u8_to_f32(*in_ptr, zp, scale));
                    float box_y = (deqnt_affine_u8_to_f32(in_ptr[grid_len], zp, scale));
                    float box_w = (deqnt_affine_u8_to_f32(in_ptr[2 * grid_len], zp, scale));
                    float box_h = (deqnt_affine_u8_to_f32(in_ptr[3 * grid_len], zp, scale));
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    objProbs.push_back((deqnt_affine_u8_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_u8_to_f32(box_confidence, zp, scale)));
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

static int process_i8(int8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            int8_t box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence >= thres_i8) {
                int offset = i * grid_w + j;
                int8_t *in_ptr = input + offset;

                int8_t maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    int8_t prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_i8)
                {
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale));
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale));
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale));
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale));
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

static int process_i8_rv1106(int8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            int8_t box_confidence = input[4 + (i * grid_w + j) * PROP_BOX_SIZE];
            if (box_confidence >= thres_i8) {
                int offset = (i * grid_w + j)*PROP_BOX_SIZE;
                int8_t *in_ptr = input + offset;

                int8_t maxClassProbs = input[5 + (i * grid_w + j) * PROP_BOX_SIZE];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    int8_t prob = input[(5 + k) + (i * grid_w + j) * PROP_BOX_SIZE];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_i8)
                {
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale));
                    float box_y = (deqnt_affine_to_f32(in_ptr[1], zp, scale));
                    float box_w = (deqnt_affine_to_f32(in_ptr[2], zp, scale));
                    float box_h = (deqnt_affine_to_f32(in_ptr[3], zp, scale));
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

static int process_fp32(float *input, int grid_h, int grid_w, int height, int width, int stride,
                        std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            float box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence >= threshold)
            {
                int offset = i * grid_w + j;
                float *in_ptr = input + offset;
                float box_x = *in_ptr;
                float box_y = in_ptr[grid_len];
                float box_w = in_ptr[2 * grid_len];
                float box_h = in_ptr[3 * grid_len];
                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = exp(box_w) * stride;
                box_h = exp(box_h) * stride;
                box_x -= (box_w / 2.0);
                box_y -= (box_h / 2.0);

                float maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    float prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }
                if (maxClassProbs > threshold)
                {
                    objProbs.push_back(maxClassProbs * box_confidence);
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

// ===================================================================
// Multi-task split-tensor decoders (for BBOX / POSE / FACE tasks)
// Each task has its cls/bbox/obj (and kpt/vis) in separate tensors
// rather than packed into one tensor like the original YOLOX.
// ===================================================================

// ---------- BBOX level decoder (int8, split tensors) ----------
// cls:  [1, 80, H, W]  80-class classification logits
// bbox: [1,  4, H, W]  dx,dy,dw,dh regression
// obj:  [1,  1, H, W]  objectness logit
static int process_bbox_level_i8(
    int8_t *cls_buf, int8_t *bbox_buf, int8_t *obj_buf,
    int grid_h, int grid_w, int stride,
    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
    float threshold,
    int32_t cls_zp, float cls_scale,
    int32_t bbox_zp, float bbox_scale,
    int32_t obj_zp, float obj_scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // Quantize threshold for fast int8-domain comparison
    int8_t thres_obj_i8 = qnt_f32_to_affine(threshold, obj_zp, obj_scale);
    int8_t thres_cls_i8 = qnt_f32_to_affine(threshold, cls_zp, cls_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int8_t box_confidence = obj_buf[offset];

            if (box_confidence >= thres_obj_i8) {
                // Find max class score
                int8_t maxClassProbs = cls_buf[0 * grid_len + offset];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; k++) {
                    int8_t prob = cls_buf[k * grid_len + offset];
                    if (prob > maxClassProbs) {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_cls_i8) {
                    // Dequantize + sigmoid for scores
                    float cls_prob = sigmoid(deqnt_affine_to_f32(maxClassProbs, cls_zp, cls_scale));
                    float obj_prob = sigmoid(deqnt_affine_to_f32(box_confidence, obj_zp, obj_scale));

                    // Dequantize bbox (NO sigmoid - regression values)
                    float box_x = deqnt_affine_to_f32(bbox_buf[0 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_y = deqnt_affine_to_f32(bbox_buf[1 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_w = deqnt_affine_to_f32(bbox_buf[2 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_h = deqnt_affine_to_f32(bbox_buf[3 * grid_len + offset], bbox_zp, bbox_scale);

                    // YOLOX decode: center + offset → pixel coords
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    objProbs.push_back(cls_prob * obj_prob);
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

// ---------- BBOX level decoder (fp32, split tensors) ----------
static int process_bbox_level_fp32(
    float *cls_buf, float *bbox_buf, float *obj_buf,
    int grid_h, int grid_w, int stride,
    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
    float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            float box_confidence = sigmoid(obj_buf[offset]);

            if (box_confidence >= threshold) {
                float maxClassProbs = sigmoid(cls_buf[0 * grid_len + offset]);
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; k++) {
                    float prob = sigmoid(cls_buf[k * grid_len + offset]);
                    if (prob > maxClassProbs) {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > threshold) {
                    float box_x = bbox_buf[0 * grid_len + offset];
                    float box_y = bbox_buf[1 * grid_len + offset];
                    float box_w = bbox_buf[2 * grid_len + offset];
                    float box_h = bbox_buf[3 * grid_len + offset];

                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    objProbs.push_back(maxClassProbs * box_confidence);
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

// ---------- Keypoint level decoder (int8, split tensors) ----------
// Shared by POSE (17 kpts) and FACE (5 kpts).
// cls:  [1, 1, H, W]     single-class logit
// obj:  [1, 1, H, W]     objectness logit
// bbox: [1, 4, H, W]     dx,dy,dw,dh
// kpt:  [1, K*2, H, W]   keypoint offsets (x,y interleaved per channel)
// vis:  [1, K, H, W]     keypoint visibility logits
static int process_kpt_level_i8(
    int8_t *cls_buf, int8_t *obj_buf, int8_t *bbox_buf,
    int8_t *kpt_buf, int8_t *vis_buf,
    int grid_h, int grid_w, int stride, int num_keypoints,
    std::vector<float> &boxes, std::vector<float> &scores,
    std::vector<float> &kpt_flat,
    float threshold,
    int32_t cls_zp, float cls_scale,
    int32_t obj_zp, float obj_scale,
    int32_t bbox_zp, float bbox_scale,
    int32_t kpt_zp, float kpt_scale,
    int32_t vis_zp, float vis_scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    int8_t thres_obj_i8 = qnt_f32_to_affine(threshold, obj_zp, obj_scale);
    int8_t thres_cls_i8 = qnt_f32_to_affine(threshold, cls_zp, cls_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int8_t box_confidence = obj_buf[offset];

            if (box_confidence >= thres_obj_i8) {
                int8_t cls_score = cls_buf[offset];

                if (cls_score > thres_cls_i8) {
                    float cls_prob = sigmoid(deqnt_affine_to_f32(cls_score, cls_zp, cls_scale));
                    float obj_prob = sigmoid(deqnt_affine_to_f32(box_confidence, obj_zp, obj_scale));

                    // Bbox decode
                    float box_x = deqnt_affine_to_f32(bbox_buf[0 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_y = deqnt_affine_to_f32(bbox_buf[1 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_w = deqnt_affine_to_f32(bbox_buf[2 * grid_len + offset], bbox_zp, bbox_scale);
                    float box_h = deqnt_affine_to_f32(bbox_buf[3 * grid_len + offset], bbox_zp, bbox_scale);

                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    // Keypoint decode: offset * stride + prior_center
                    float cx = j * (float)stride;
                    float cy = i * (float)stride;
                    for (int k = 0; k < num_keypoints; k++) {
                        float kx = deqnt_affine_to_f32(kpt_buf[(k * 2 + 0) * grid_len + offset], kpt_zp, kpt_scale) * stride + cx;
                        float ky = deqnt_affine_to_f32(kpt_buf[(k * 2 + 1) * grid_len + offset], kpt_zp, kpt_scale) * stride + cy;
                        float kv = sigmoid(deqnt_affine_to_f32(vis_buf[k * grid_len + offset], vis_zp, vis_scale));
                        kpt_flat.push_back(kx);
                        kpt_flat.push_back(ky);
                        kpt_flat.push_back(kv);
                    }

                    scores.push_back(cls_prob * obj_prob);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

// ---------- Keypoint level decoder (fp32, split tensors) ----------
static int process_kpt_level_fp32(
    float *cls_buf, float *obj_buf, float *bbox_buf,
    float *kpt_buf, float *vis_buf,
    int grid_h, int grid_w, int stride, int num_keypoints,
    std::vector<float> &boxes, std::vector<float> &scores,
    std::vector<float> &kpt_flat,
    float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            float box_confidence = sigmoid(obj_buf[offset]);

            if (box_confidence >= threshold) {
                float cls_score = sigmoid(cls_buf[offset]);

                if (cls_score > threshold) {
                    float box_x = bbox_buf[0 * grid_len + offset];
                    float box_y = bbox_buf[1 * grid_len + offset];
                    float box_w = bbox_buf[2 * grid_len + offset];
                    float box_h = bbox_buf[3 * grid_len + offset];

                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    float cx = j * (float)stride;
                    float cy = i * (float)stride;
                    for (int k = 0; k < num_keypoints; k++) {
                        float kx = kpt_buf[(k * 2 + 0) * grid_len + offset] * stride + cx;
                        float ky = kpt_buf[(k * 2 + 1) * grid_len + offset] * stride + cy;
                        float kv = sigmoid(vis_buf[k * grid_len + offset]);
                        kpt_flat.push_back(kx);
                        kpt_flat.push_back(ky);
                        kpt_flat.push_back(kv);
                    }

                    scores.push_back(cls_score * box_confidence);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

// ---------- Class-agnostic NMS (used by POSE / FACE single-class tasks) ----------
static int nms_class_agnostic(int validCount, std::vector<float> &outputLocations,
                              std::vector<int> &order, float threshold)
{
    for (int i = 0; i < validCount; i++) {
        int n = order[i];
        if (n == -1) continue;
        for (int j = i + 1; j < validCount; j++) {
            int m = order[j];
            if (m == -1) continue;

            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

// ===================================================================
// Visualization helpers (controlled by DRAW_IMAGE macro)
// ===================================================================
#if DRAW_IMAGE

#include <sys/stat.h>
#include <string>

// COCO 17-keypoint skeleton connections (0-based keypoint indices)
static const int COCO_SKELETON[19][2] = {
    {15,13}, {13,11}, {16,14}, {14,12}, {11,12},
    {5,11},  {6,12},  {5,6},   {5,7},   {6,8},
    {7,9},   {8,10},  {1,2},   {0,1},   {0,2},
    {1,3},   {2,4},   {3,5},   {4,6}
};

// WIDER Face 5-point landmark colours (ARGB)
static const unsigned int FACE_KPT_COLORS[5] = {
    0xFFFF00FF,  // magenta
    0xFF00FF00,  // green
    0xFF0000FF,  // red
    0xFFFFFFFF,  // white
    0xFFFFFF00,  // yellow
};

static void draw_coco_skeleton(image_buffer_t *image,
                               const pose_detect_result_list *pose_results)
{
    for (int p = 0; p < pose_results->count; p++) {
        const pose_detect_result *res = &pose_results->results[p];

        // skeleton lines (yellow)
        for (int s = 0; s < 19; s++) {
            int i = COCO_SKELETON[s][0];
            int j = COCO_SKELETON[s][1];
            float vi = res->keypoints[i][2];
            float vj = res->keypoints[j][2];
            if (vi > 0 && vj > 0) {
                int x0 = (int)res->keypoints[i][0];
                int y0 = (int)res->keypoints[i][1];
                int x1 = (int)res->keypoints[j][0];
                int y1 = (int)res->keypoints[j][1];
                if (x0 > 0 && y0 > 0 && x1 > 0 && y1 > 0) {
                    draw_line(image, x0, y0, x1, y1, COLOR_YELLOW, 1);
                }
            }
        }

        // keypoints (filled red circle)
        for (int k = 0; k < 17; k++) {
            float vis = res->keypoints[k][2];
            if (vis > 0 && res->keypoints[k][0] > 0 && res->keypoints[k][1] > 0) {
                draw_circle(image, (int)res->keypoints[k][0], (int)res->keypoints[k][1],
                            3, COLOR_RED, -1);
            }
        }
    }
}

static void draw_face_keypoints(image_buffer_t *image,
                                const face_detect_result_list *face_results)
{
    for (int f = 0; f < face_results->count; f++) {
        const face_detect_result *res = &face_results->results[f];
        for (int k = 0; k < 5; k++) {
            if (res->keypoints[k][0] <= 0 || res->keypoints[k][1] <= 0)
                continue;

            float vis = res->keypoints[k][2];
            unsigned int color = FACE_KPT_COLORS[k];
            int cx = (int)res->keypoints[k][0];
            int cy = (int)res->keypoints[k][1];

            if (vis > 0) {
                draw_circle(image, cx, cy, 3, color, -1);  // filled
            } else {
                draw_circle(image, cx, cy, 3, color, 1);   // hollow outline
            }
        }
    }
}

static std::string get_basename_without_ext(const char *path)
{
    std::string s(path);
    // strip directory
    size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos)
        s = s.substr(slash + 1);
    // strip extension (guard against dotfiles like ".hidden")
    size_t dot = s.find_last_of('.');
    if (dot != std::string::npos && dot > 0)
        s = s.substr(0, dot);
    return s;
}

static std::string get_dirname(const char *path)
{
    std::string s(path);
    size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) return s.substr(0, slash);
    return ".";
}

static image_buffer_t copy_image_buffer(const image_buffer_t *src)
{
    image_buffer_t dst;
    memset(&dst, 0, sizeof(dst));
    dst.width        = src->width;
    dst.height       = src->height;
    dst.width_stride  = src->width_stride;
    dst.height_stride = src->height_stride;
    dst.format       = src->format;
    dst.size         = src->size;
    dst.virt_addr    = (unsigned char *)malloc(src->size);
    if (dst.virt_addr) {
        memcpy(dst.virt_addr, src->virt_addr, src->size);
    }
    return dst;
}

static void free_image_buffer_viz(image_buffer_t *img)
{
    if (img->virt_addr) {
        free(img->virt_addr);
        img->virt_addr = NULL;
    }
}

int write_multi_task_results(
    const char *image_path,
    image_buffer_t *src_image,
    object_detect_result_list *od_results,
    pose_detect_result_list *pose_results,
    face_detect_result_list *face_results)
{
    std::string dir      = get_dirname(image_path);
    std::string basename = get_basename_without_ext(image_path);
    std::string out_dir  = dir + "/" + basename;

    mkdir(out_dir.c_str(), 0755);

    char out_path[512];
    char text[256];

    // ---- BBOX (green boxes, 80-class labels) ----
    {
        image_buffer_t vis = copy_image_buffer(src_image);
        if (!vis.virt_addr) return -1;

        for (int i = 0; i < od_results->count; i++) {
            const object_detect_result *det = &od_results->results[i];
            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            draw_rectangle(&vis, x1, y1, x2 - x1, y2 - y1, COLOR_GREEN, 2);

            snprintf(text, sizeof(text), "%s %.2f",
                     coco_cls_to_name(det->cls_id), det->prop);
            draw_text(&vis, text, x1, (y1 - 8 > 0) ? y1 - 8 : 0,
                      COLOR_GREEN, 10);
        }

        snprintf(out_path, sizeof(out_path), "%s/%s_od.jpg",
                 out_dir.c_str(), basename.c_str());
        write_image(out_path, &vis);
        printf("  saved: %s\n", out_path);
        free_image_buffer_viz(&vis);
    }

    // ---- POSE (red boxes + COCO 17-kpt skeleton) ----
    {
        image_buffer_t vis = copy_image_buffer(src_image);
        if (!vis.virt_addr) return -1;

        for (int i = 0; i < pose_results->count; i++) {
            const pose_detect_result *res = &pose_results->results[i];
            int x1 = res->box.left;
            int y1 = res->box.top;
            int x2 = res->box.right;
            int y2 = res->box.bottom;

            draw_rectangle(&vis, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 2);

            snprintf(text, sizeof(text), "person %.2f", res->prop);
            draw_text(&vis, text, x1, (y1 - 8 > 0) ? y1 - 8 : 0,
                      COLOR_RED, 10);
        }

        draw_coco_skeleton(&vis, pose_results);

        snprintf(out_path, sizeof(out_path), "%s/%s_pose.jpg",
                 out_dir.c_str(), basename.c_str());
        write_image(out_path, &vis);
        printf("  saved: %s\n", out_path);
        free_image_buffer_viz(&vis);
    }

    // ---- FACE (red boxes + 5-point landmarks) ----
    {
        image_buffer_t vis = copy_image_buffer(src_image);
        if (!vis.virt_addr) return -1;

        for (int i = 0; i < face_results->count; i++) {
            const face_detect_result *res = &face_results->results[i];
            int x1 = res->box.left;
            int y1 = res->box.top;
            int x2 = res->box.right;
            int y2 = res->box.bottom;

            draw_rectangle(&vis, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 2);

            snprintf(text, sizeof(text), "face %.2f", res->prop);
            draw_text(&vis, text, x1, (y1 - 8 > 0) ? y1 - 8 : 0,
                      COLOR_RED, 10);
        }

        draw_face_keypoints(&vis, face_results);

        snprintf(out_path, sizeof(out_path), "%s/%s_face.jpg",
                 out_dir.c_str(), basename.c_str());
        write_image(out_path, &vis);
        printf("  saved: %s\n", out_path);
        free_image_buffer_viz(&vis);
    }

    return 0;
}

#endif // DRAW_IMAGE

// 该文件所属项目"examples/multi_task_dacheng"是我从examples/yolox拷贝而来，打算在这个文件夹内的CPP文件夹内实现"examples/multi_task_dacheng/model/multi_task_obt_pose_face.rknn"模型的C++部署，
// 这个模型的python版本的推理脚本我已经验证完了，该python脚本的后处理可以正确在开发板上推理该rknn模型。后续你可以参考该python脚本来考虑实现其python版本，
// 我目前已经完成了多任务模型的数据结构的定义，"examples/multi_task_dacheng/cpp/postprocess.h"中#sym:object_detect_result_list 保存BBOX检测的结果，#sym:pose_detect_result_list 保存人体关键点的结果。#sym:face_detect_result_list 保存人脸关键点的结果。
// 该模型用于检测三个任务，BBOX，POSE和FACE。
// 1.BBOX是常见的BBOX检测任务，基于COCO 80 个分类，bbox_results=outputs[:9]， 每个tensor分别是[cls_p3, cls_p4, cls_p5, bbox_p3, bbox_p4, bbox_p5, obj_p3,  obj_p4,  obj_p5]
// 2.POSE是COCO person的人体17个关键点的检测任务，输出人的bobox和关键点。pose_results=outputs[9:24]
// 每个tensor分别是['pose_cls_score_p3','pose_cls_score_p4','pose_cls_score_p5',
//                   'pose_objectnesses_p3','pose_objectnesses_p4','pose_objectnesses_p5',
//                   'pose_bbox_preds_p3','pose_bbox_preds_p4','pose_bbox_preds_p5',
//                   'pose_kpt_offsets_p3','pose_kpt_offsets_p4','pose_kpt_offsets_p5',
//                   'pose_kpt_vis_p3','pose_kpt_vis_p4','pose_kpt_vis_p5']
// 3.FACE是人脸检测任务，解码方法跟POSE完全一致，只是人脸的关键点是5个，而不是17个。face_results=outputs[24:]
// //                   'face_cls_score_p3','face_cls_score_p4','face_cls_score_p5',
//                   'face_objectnesses_p3','face_objectnesses_p4','face_objectnesses_p5',
//                   'face_bbox_preds_p3','face_bbox_preds_p4','face_bbox_preds_p5',
//                   'face_kpt_offsets_p3','face_kpt_offsets_p4','face_kpt_offsets_p5',
//                   'face_kpt_vis_p3','face_kpt_vis_p4','face_kpt_vis_p5',
// 总的output输出是39个， len(outputs)=39。
// 后处理的接口是下面的post_process，这个post_process是RKNN提供的yolox的bbox的C++解码后处理，我对其解码逻辑并不是很清晰，
// 关键点是两处 1.int8和fp32两个函数接口的逻辑该如何分别实现， 我调试下来发现其走的是process_i8的函数  2.内存排列顺序不是很清楚，numpy和tensor
// 输出的排列见到那，可是RKNN嵌入式端的数据访问涉及内存排列问题。
// 你先根据我提供的这些信息，理解下下main的post_process函数接口，如果你想知道demo入口的话，那就"examples/multi_task_dacheng/cpp/main.cc",
// 最终会将rknn板端的输出送到outputs，传入post_process函数的outputs是rknn_output outputs[39]的数组首地址，你先理解下yolox的代码后处理逻辑，
// 或者你基于整个项目工程，了解RKNN的数据排布后，跟我解释下post_process函数的是如何解码的，每个主要的代码段的功能，
// 我们通过交互式的方式，先确认清楚我们对于RKNN的内存数据访问排布和int8/fp32的量化与反量化部分的理解是正确的，再来完成多任务模型的解码。
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, 
    object_detect_result_list *od_results,
    pose_detect_result_list *pose_results,
    face_detect_result_list *face_results)
{
#if defined(RV1106_1103) 
    // RV1106: not yet adapted for multi-task 39-output split-tensor model
    memset(od_results, 0, sizeof(object_detect_result_list));
    memset(pose_results, 0, sizeof(pose_detect_result_list));
    memset(face_results, 0, sizeof(face_detect_result_list));
    return 0;
#elif defined(RKNPU1)
    // RKNPU1: not yet adapted for multi-task 39-output split-tensor model
    memset(od_results, 0, sizeof(object_detect_result_list));
    memset(pose_results, 0, sizeof(pose_detect_result_list));
    memset(face_results, 0, sizeof(face_detect_result_list));
    return 0;
#else
    // RK3588 / RK3576 / RK356x: multi-task split-tensor decoding (39 outputs)
    rknn_output *_outputs = (rknn_output *)outputs;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));
    memset(pose_results, 0, sizeof(pose_detect_result_list));
    memset(face_results, 0, sizeof(face_detect_result_list));

    // ================================================================
    // BBOX Task: outputs[0:9]
    //   outputs[0:3] = cls_p3, cls_p4, cls_p5    (1×80×H×W)
    //   outputs[3:6] = bbox_p3, bbox_p4, bbox_p5 (1×4×H×W)
    //   outputs[6:9] = obj_p3, obj_p4, obj_p5    (1×1×H×W)
    // ================================================================
    {
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;
        int validCount = 0;

        for (int level = 0; level < 3; level++) {
            int cls_idx  = level;       // 0, 1, 2
            int bbox_idx = 3 + level;   // 3, 4, 5
            int obj_idx  = 6 + level;   // 6, 7, 8

            int grid_h = app_ctx->output_attrs[cls_idx].dims[2];
            int grid_w = app_ctx->output_attrs[cls_idx].dims[3];
            int stride = model_in_h / grid_h;

            if (app_ctx->is_quant) {
                validCount += process_bbox_level_i8(
                    (int8_t *)_outputs[cls_idx].buf,
                    (int8_t *)_outputs[bbox_idx].buf,
                    (int8_t *)_outputs[obj_idx].buf,
                    grid_h, grid_w, stride,
                    filterBoxes, objProbs, classId, conf_threshold,
                    app_ctx->output_attrs[cls_idx].zp,  app_ctx->output_attrs[cls_idx].scale,
                    app_ctx->output_attrs[bbox_idx].zp, app_ctx->output_attrs[bbox_idx].scale,
                    app_ctx->output_attrs[obj_idx].zp,  app_ctx->output_attrs[obj_idx].scale);
            } else {
                validCount += process_bbox_level_fp32(
                    (float *)_outputs[cls_idx].buf,
                    (float *)_outputs[bbox_idx].buf,
                    (float *)_outputs[obj_idx].buf,
                    grid_h, grid_w, stride,
                    filterBoxes, objProbs, classId, conf_threshold);
            }
        }

        // per-class NMS → fill od_results
        if (validCount > 0) {
            std::vector<int> indexArray(validCount);
            for (int i = 0; i < validCount; i++) indexArray[i] = i;
            quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

            std::set<int> class_set(classId.begin(), classId.end());
            for (auto c : class_set) {
                nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
            }

            int last_count = 0;
            for (int i = 0; i < validCount; i++) {
                if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
                int n = indexArray[i];

                float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
                float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
                float x2 = x1 + filterBoxes[n * 4 + 2];
                float y2 = y1 + filterBoxes[n * 4 + 3];

                od_results->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
                od_results->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
                od_results->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
                od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
                od_results->results[last_count].prop   = objProbs[i];
                od_results->results[last_count].cls_id = classId[n];
                last_count++;
            }
            od_results->count = last_count;
        }
    }

    // ================================================================
    // POSE Task: outputs[9:24] (17 keypoints)
    //   outputs[ 9:12] = pose_cls_score_p3/4/5     (1×1×H×W)
    //   outputs[12:15] = pose_objectnesses_p3/4/5   (1×1×H×W)
    //   outputs[15:18] = pose_bbox_preds_p3/4/5     (1×4×H×W)
    //   outputs[18:21] = pose_kpt_offsets_p3/4/5    (1×34×H×W)
    //   outputs[21:24] = pose_kpt_vis_p3/4/5        (1×17×H×W)
    // ================================================================
    {
        const int NUM_KPTS = 17;
        const int BASE = 9;
        std::vector<float> filterBoxes;
        std::vector<float> scores;
        std::vector<float> kpt_flat;  // per detection: NUM_KPTS×(x,y,conf)
        int validCount = 0;

        for (int level = 0; level < 3; level++) {
            int cls_idx  = BASE + level;           // 9, 10, 11
            int obj_idx  = BASE + 3 + level;       // 12, 13, 14
            int bbox_idx = BASE + 6 + level;       // 15, 16, 17
            int kpt_idx  = BASE + 9 + level;       // 18, 19, 20
            int vis_idx  = BASE + 12 + level;      // 21, 22, 23

            int grid_h = app_ctx->output_attrs[cls_idx].dims[2];
            int grid_w = app_ctx->output_attrs[cls_idx].dims[3];
            int stride = model_in_h / grid_h;

            if (app_ctx->is_quant) {
                validCount += process_kpt_level_i8(
                    (int8_t *)_outputs[cls_idx].buf,
                    (int8_t *)_outputs[obj_idx].buf,
                    (int8_t *)_outputs[bbox_idx].buf,
                    (int8_t *)_outputs[kpt_idx].buf,
                    (int8_t *)_outputs[vis_idx].buf,
                    grid_h, grid_w, stride, NUM_KPTS,
                    filterBoxes, scores, kpt_flat,
                    conf_threshold,
                    app_ctx->output_attrs[cls_idx].zp,  app_ctx->output_attrs[cls_idx].scale,
                    app_ctx->output_attrs[obj_idx].zp,  app_ctx->output_attrs[obj_idx].scale,
                    app_ctx->output_attrs[bbox_idx].zp, app_ctx->output_attrs[bbox_idx].scale,
                    app_ctx->output_attrs[kpt_idx].zp,  app_ctx->output_attrs[kpt_idx].scale,
                    app_ctx->output_attrs[vis_idx].zp,  app_ctx->output_attrs[vis_idx].scale);
            } else {
                validCount += process_kpt_level_fp32(
                    (float *)_outputs[cls_idx].buf,
                    (float *)_outputs[obj_idx].buf,
                    (float *)_outputs[bbox_idx].buf,
                    (float *)_outputs[kpt_idx].buf,
                    (float *)_outputs[vis_idx].buf,
                    grid_h, grid_w, stride, NUM_KPTS,
                    filterBoxes, scores, kpt_flat,
                    conf_threshold);
            }
        }

        // class-agnostic NMS → fill pose_results
        if (validCount > 0) {
            std::vector<int> indexArray(validCount);
            for (int i = 0; i < validCount; i++) indexArray[i] = i;
            quick_sort_indice_inverse(scores, 0, validCount - 1, indexArray);
            nms_class_agnostic(validCount, filterBoxes, indexArray, nms_threshold);

            int last_count = 0;
            for (int i = 0; i < validCount; i++) {
                if (indexArray[i] == -1 || last_count >= POSE_NUMB_MAX_SIZE) continue;
                int n = indexArray[i];

                float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
                float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
                float x2 = x1 + filterBoxes[n * 4 + 2];
                float y2 = y1 + filterBoxes[n * 4 + 3];

                pose_results->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
                pose_results->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
                pose_results->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
                pose_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
                pose_results->results[last_count].prop   = scores[i];
                pose_results->results[last_count].cls_id = 0;  // person

                float *kpt_src = &kpt_flat[n * NUM_KPTS * 3];
                for (int k = 0; k < NUM_KPTS; k++) {
                    pose_results->results[last_count].keypoints[k][0] = (kpt_src[k * 3 + 0] - letter_box->x_pad) / letter_box->scale;
                    pose_results->results[last_count].keypoints[k][1] = (kpt_src[k * 3 + 1] - letter_box->y_pad) / letter_box->scale;
                    pose_results->results[last_count].keypoints[k][2] = kpt_src[k * 3 + 2];
                }
                last_count++;
            }
            pose_results->count = last_count;
        }
    }

    // ================================================================
    // FACE Task: outputs[24:39] (5 keypoints, decode identical to POSE)
    //   outputs[24:27] = face_cls_score_p3/4/5     (1×1×H×W)
    //   outputs[27:30] = face_objectnesses_p3/4/5   (1×1×H×W)
    //   outputs[30:33] = face_bbox_preds_p3/4/5     (1×4×H×W)
    //   outputs[33:36] = face_kpt_offsets_p3/4/5    (1×10×H×W)
    //   outputs[36:39] = face_kpt_vis_p3/4/5        (1×5×H×W)
    // ================================================================
    {
        const int NUM_KPTS = 5;
        const int BASE = 24;
        std::vector<float> filterBoxes;
        std::vector<float> scores;
        std::vector<float> kpt_flat;
        int validCount = 0;

        for (int level = 0; level < 3; level++) {
            int cls_idx  = BASE + level;           // 24, 25, 26
            int obj_idx  = BASE + 3 + level;       // 27, 28, 29
            int bbox_idx = BASE + 6 + level;       // 30, 31, 32
            int kpt_idx  = BASE + 9 + level;       // 33, 34, 35
            int vis_idx  = BASE + 12 + level;      // 36, 37, 38

            int grid_h = app_ctx->output_attrs[cls_idx].dims[2];
            int grid_w = app_ctx->output_attrs[cls_idx].dims[3];
            int stride = model_in_h / grid_h;

            if (app_ctx->is_quant) {
                validCount += process_kpt_level_i8(
                    (int8_t *)_outputs[cls_idx].buf,
                    (int8_t *)_outputs[obj_idx].buf,
                    (int8_t *)_outputs[bbox_idx].buf,
                    (int8_t *)_outputs[kpt_idx].buf,
                    (int8_t *)_outputs[vis_idx].buf,
                    grid_h, grid_w, stride, NUM_KPTS,
                    filterBoxes, scores, kpt_flat,
                    conf_threshold,
                    app_ctx->output_attrs[cls_idx].zp,  app_ctx->output_attrs[cls_idx].scale,
                    app_ctx->output_attrs[obj_idx].zp,  app_ctx->output_attrs[obj_idx].scale,
                    app_ctx->output_attrs[bbox_idx].zp, app_ctx->output_attrs[bbox_idx].scale,
                    app_ctx->output_attrs[kpt_idx].zp,  app_ctx->output_attrs[kpt_idx].scale,
                    app_ctx->output_attrs[vis_idx].zp,  app_ctx->output_attrs[vis_idx].scale);
            } else {
                validCount += process_kpt_level_fp32(
                    (float *)_outputs[cls_idx].buf,
                    (float *)_outputs[obj_idx].buf,
                    (float *)_outputs[bbox_idx].buf,
                    (float *)_outputs[kpt_idx].buf,
                    (float *)_outputs[vis_idx].buf,
                    grid_h, grid_w, stride, NUM_KPTS,
                    filterBoxes, scores, kpt_flat,
                    conf_threshold);
            }
        }

        // class-agnostic NMS → fill face_results
        if (validCount > 0) {
            std::vector<int> indexArray(validCount);
            for (int i = 0; i < validCount; i++) indexArray[i] = i;
            quick_sort_indice_inverse(scores, 0, validCount - 1, indexArray);
            nms_class_agnostic(validCount, filterBoxes, indexArray, nms_threshold);

            int last_count = 0;
            for (int i = 0; i < validCount; i++) {
                if (indexArray[i] == -1 || last_count >= FACE_NUMB_MAX_SIZE) continue;
                int n = indexArray[i];

                float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
                float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
                float x2 = x1 + filterBoxes[n * 4 + 2];
                float y2 = y1 + filterBoxes[n * 4 + 3];

                face_results->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
                face_results->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
                face_results->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
                face_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
                face_results->results[last_count].prop   = scores[i];
                face_results->results[last_count].cls_id = 0;  // face

                float *kpt_src = &kpt_flat[n * NUM_KPTS * 3];
                for (int k = 0; k < NUM_KPTS; k++) {
                    face_results->results[last_count].keypoints[k][0] = (kpt_src[k * 3 + 0] - letter_box->x_pad) / letter_box->scale;
                    face_results->results[last_count].keypoints[k][1] = (kpt_src[k * 3 + 1] - letter_box->y_pad) / letter_box->scale;
                    face_results->results[last_count].keypoints[k][2] = kpt_src[k * 3 + 2];
                }
                last_count++;
            }
            face_results->count = last_count;
        }
    }

    return 0;
#endif
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
