// BEV-OCC 结果后处理与可视化
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bev_occ.h"
#include "image_utils.h"

// ---------------- 类别名 (与训练侧一致) ----------------
static const char* STATIC_NAMES[BEV_OCC_STATIC_CLASSES] = {
    "floor", "wall", "door", "window", "hanging",
    "table", "chair", "sofa", "bed", "cabinet"};

static const char* DYNAMIC_NAMES[BEV_OCC_DYNAMIC_CLASSES] = {
    "background", "person", "cat", "dog"};

static const char* BEV_NAMES[BEV_OCC_BEV_CLASSES] = {
    "floor", "wall", "window", "chair", "bed", "sofa",
    "table", "tvs", "furn", "objs", "unknown"};

const char* bev_occ_static_name(int id)
{
    if (id < 0 || id >= BEV_OCC_STATIC_CLASSES) {
        return "unknown";
    }
    return STATIC_NAMES[id];
}

const char* bev_occ_dynamic_name(int id)
{
    if (id < 0 || id >= BEV_OCC_DYNAMIC_CLASSES) {
        return "unknown";
    }
    return DYNAMIC_NAMES[id];
}

const char* bev_occ_bev_name(int id)
{
    if (id < 0 || id >= BEV_OCC_BEV_CLASSES) {
        return "unknown";
    }
    return BEV_NAMES[id];
}

void release_bev_occ_result(bev_occ_result_t* result)
{
    if (result == NULL) {
        return;
    }
    if (result->static_label != NULL) {
        free(result->static_label);
        result->static_label = NULL;
    }
    if (result->dynamic_label != NULL) {
        free(result->dynamic_label);
        result->dynamic_label = NULL;
    }
    if (result->bev_label != NULL) {
        free(result->bev_label);
        result->bev_label = NULL;
    }
}

#if ENABLE_VISUALIZATION

// ---------------- RGB 调色板 (与训练侧可视化一致) ----------------
static const unsigned char STATIC_PALETTE[BEV_OCC_STATIC_CLASSES][3] = {
    {0, 160, 0}, {180, 255, 0}, {139, 90, 43}, {0, 170, 220},
    {200, 160, 230}, {255, 128, 0}, {220, 60, 60}, {200, 0, 160},
    {120, 60, 200}, {30, 70, 200}};

static const unsigned char DYNAMIC_PALETTE[BEV_OCC_DYNAMIC_CLASSES][3] = {
    {0, 0, 0},        // 0 background
    {235, 220, 0},    // 1 person
    {0, 220, 150},    // 2 cat
    {222, 184, 135}}; // 3 dog

static const unsigned char BEV_PALETTE[BEV_OCC_BEV_CLASSES][3] = {
    {0, 160, 0}, {180, 255, 0}, {0, 170, 220}, {220, 60, 60},
    {120, 60, 200}, {200, 0, 160}, {255, 128, 0}, {30, 70, 200},
    {139, 90, 43}, {205, 205, 205}, {150, 150, 150}};

/*
 * 把模型分辨率的类别图逆映射回原图, 并半透明叠加到 src 上
 * dst 为 src 的副本 (RGB888), 调色板为 RGB, skip_class < 0 表示全部叠加
 */
static void overlay_label_image(image_buffer_t* dst, const unsigned char* label,
                                int label_w, int label_h,
                                const bev_occ_transform_t* transform,
                                const unsigned char* palette, int num_classes,
                                int skip_class, float alpha)
{
    int x, y, c;

    for (y = transform->src_top; y < transform->src_top + transform->src_h; y++) {
        int my = (y - transform->src_top) * label_h / transform->src_h;
        if (my >= label_h) {
            my = label_h - 1;
        }
        unsigned char* row = dst->virt_addr + (size_t)y * dst->width * 3;
        const unsigned char* label_row = label + (size_t)my * label_w;

        for (x = transform->src_left; x < transform->src_left + transform->src_w; x++) {
            int mx = (x - transform->src_left) * label_w / transform->src_w;
            int cls;
            unsigned char* px;

            if (mx >= label_w) {
                mx = label_w - 1;
            }
            cls = label_row[mx];
            if (cls == skip_class || cls >= num_classes) {
                continue;
            }
            px = row + (size_t)x * 3;
            for (c = 0; c < 3; c++) {
                px[c] = (unsigned char)(px[c] * (1.0f - alpha) +
                                        palette[cls * 3 + c] * alpha + 0.5f);
            }
        }
    }
}

// 类别图上色并最近邻放大 (BEV 输出)
static void colorize_label_view(const unsigned char* label, int label_w, int label_h,
                                const unsigned char* palette, int num_classes,
                                int view_size, image_buffer_t* dst)
{
    int x, y, c;

    for (y = 0; y < view_size; y++) {
        int my = y * label_h / view_size;
        unsigned char* row = dst->virt_addr + (size_t)y * dst->width * 3;

        for (x = 0; x < view_size; x++) {
            int mx = x * label_w / view_size;
            int cls = label[(size_t)my * label_w + mx];
            unsigned char* px = row + (size_t)x * 3;
            if (cls >= num_classes) {
                cls = 0;
            }
            for (c = 0; c < 3; c++) {
                px[c] = palette[cls * 3 + c];
            }
        }
    }
}

// 申请一张 RGB888 图像缓冲, 并拷贝 src 内容
static int alloc_image_like(const image_buffer_t* src, image_buffer_t* dst)
{
    memset(dst, 0, sizeof(image_buffer_t));
    dst->width = src->width;
    dst->height = src->height;
    dst->format = IMAGE_FORMAT_RGB888;
    dst->size = get_image_size(dst);
    dst->virt_addr = (unsigned char*)malloc(dst->size);
    if (dst->virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", dst->size);
        return -1;
    }
    memcpy(dst->virt_addr, src->virt_addr, dst->size);
    return 0;
}

int save_bev_occ_visualization(const image_buffer_t* src_image,
                               const bev_occ_result_t* result,
                               const char* static_path,
                               const char* dynamic_path,
                               const char* bev_path)
{
    image_buffer_t vis_image;
    image_buffer_t bev_view;
    int ret = 0;

    if (src_image == NULL || result == NULL) {
        return -1;
    }

    // 静态分割叠加
    if (static_path != NULL) {
        if (alloc_image_like(src_image, &vis_image) != 0) {
            return -1;
        }
        overlay_label_image(&vis_image, result->static_label,
                            result->static_w, result->static_h,
                            &result->transform,
                            (const unsigned char*)STATIC_PALETTE,
                            BEV_OCC_STATIC_CLASSES, -1, BEV_OCC_OVERLAY_ALPHA);
        ret |= write_image(static_path, &vis_image);
        free(vis_image.virt_addr);
    }

    // 动态分割叠加 (背景不叠加)
    if (dynamic_path != NULL) {
        if (alloc_image_like(src_image, &vis_image) != 0) {
            return -1;
        }
        overlay_label_image(&vis_image, result->dynamic_label,
                            result->dynamic_w, result->dynamic_h,
                            &result->transform,
                            (const unsigned char*)DYNAMIC_PALETTE,
                            BEV_OCC_DYNAMIC_CLASSES, BEV_OCC_DYNAMIC_BACKGROUND,
                            BEV_OCC_OVERLAY_ALPHA);
        ret |= write_image(dynamic_path, &vis_image);
        free(vis_image.virt_addr);
    }

    // BEV 占据输出上色 (不叠加到输入图像)
    if (bev_path != NULL) {
        memset(&bev_view, 0, sizeof(image_buffer_t));
        bev_view.width = BEV_OCC_BEV_VIEW_SIZE;
        bev_view.height = BEV_OCC_BEV_VIEW_SIZE;
        bev_view.format = IMAGE_FORMAT_RGB888;
        bev_view.size = get_image_size(&bev_view);
        bev_view.virt_addr = (unsigned char*)malloc(bev_view.size);
        if (bev_view.virt_addr == NULL) {
            printf("malloc buffer size:%d fail!\n", bev_view.size);
            return -1;
        }
        colorize_label_view(result->bev_label, result->bev_w, result->bev_h,
                            (const unsigned char*)BEV_PALETTE, BEV_OCC_BEV_CLASSES,
                            BEV_OCC_BEV_VIEW_SIZE, &bev_view);
        ret |= write_image(bev_path, &bev_view);
        free(bev_view.virt_addr);
    }

    return ret;
}

#endif  // ENABLE_VISUALIZATION
