// BEV-OCC RKNN 板端推理 (rk3588 / rk3576 / rk356x, RKNPU2)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bev_occ.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// 从 tensor attr 中取出 (channel, height, width), 兼容 NCHW / NHWC
static void get_tensor_chw(const rknn_tensor_attr* attr, int* channel, int* height, int* width)
{
    if (attr->fmt == RKNN_TENSOR_NCHW) {
        if (attr->n_dims == 4) {
            *channel = attr->dims[1];
            *height = attr->dims[2];
            *width = attr->dims[3];
        } else {
            *channel = attr->dims[0];
            *height = attr->dims[1];
            *width = attr->dims[2];
        }
    } else {
        if (attr->n_dims == 4) {
            *height = attr->dims[1];
            *width = attr->dims[2];
            *channel = attr->dims[3];
        } else {
            *height = attr->dims[0];
            *width = attr->dims[1];
            *channel = attr->dims[2];
        }
    }
}

// float 输出 -> 类别图 (按通道取 argmax)
static void argmax_label(const float* buf, const rknn_tensor_attr* attr, unsigned char* label)
{
    int channel = 0, height = 0, width = 0;
    int is_nhwc = (attr->fmt != RKNN_TENSOR_NCHW);
    int x, y, k;

    get_tensor_chw(attr, &channel, &height, &width);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float best_val = -INFINITY;
            int best_id = 0;
            for (k = 0; k < channel; k++) {
                float val;
                if (is_nhwc) {
                    val = buf[((size_t)y * width + x) * channel + k];
                } else {
                    val = buf[(size_t)k * height * width + (size_t)y * width + x];
                }
                if (val > best_val) {
                    best_val = val;
                    best_id = k;
                }
            }
            label[(size_t)y * width + x] = (unsigned char)best_id;
        }
    }
}

int init_bev_occ_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;

    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // rk3588 使用三核
    rknn_set_core_mask(ctx, RKNN_NPU_CORE_ALL);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr* input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            free(input_attrs);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr* output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            free(input_attrs);
            free(output_attrs);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = input_attrs;
    app_ctx->output_attrs = output_attrs;
    app_ctx->is_quant = true;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_bev_occ_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_bev_occ_model(rknn_app_context_t* app_ctx, image_buffer_t* img,
                            bev_occ_result_t* result)
{
    int ret;
    image_buffer_t dst_img;
    image_rect_t src_box;
    image_rect_t dst_box;
    rknn_input inputs[1];
    rknn_output outputs[3];

    if ((!app_ctx) || (!img) || (!result) || app_ctx->io_num.n_output != 3) {
        printf("invalid argument or unexpected output num!\n");
        return -1;
    }

    memset(result, 0, sizeof(*result));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    /*
     * 预处理: 等比例缩放, 短边对齐, 长边居中裁掉
     * 先算出原图上的裁剪区域, 再一步 crop+resize 到模型输入尺寸
     */
    float scale = fmaxf((float)app_ctx->model_height / img->height,
                        (float)app_ctx->model_width / img->width);
    int crop_w = (int)(app_ctx->model_width / scale + 0.5f);
    int crop_h = (int)(app_ctx->model_height / scale + 0.5f);
    if (crop_w > img->width) {
        crop_w = img->width;
    }
    if (crop_h > img->height) {
        crop_h = img->height;
    }
    int crop_left = (img->width - crop_w) / 2;
    int crop_top = (img->height - crop_h) / 2;

    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    src_box.left = crop_left;
    src_box.top = crop_top;
    src_box.right = crop_left + crop_w - 1;
    src_box.bottom = crop_top + crop_h - 1;
    dst_box.left = 0;
    dst_box.top = 0;
    dst_box.right = app_ctx->model_width - 1;
    dst_box.bottom = app_ctx->model_height - 1;

    ret = convert_image(img, &dst_img, &src_box, &dst_box, 0);
    if (ret != 0) {
        printf("convert_image fail! ret=%d\n", ret);
        free(dst_img.virt_addr);
        return -1;
    }

    // mean/std 已固化在 rknn 内部, 这里直接喂 RAW RGB [0,255] 的 uint8 数据
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = dst_img.size;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        free(dst_img.virt_addr);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        free(dst_img.virt_addr);
        return -1;
    }

    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;  // 直接取反量化后的 float, 输出很小, 无性能压力
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        free(dst_img.virt_addr);
        return -1;
    }

    // 按通道数区分三个 head
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        int channel = 0, height = 0, width = 0;
        const float* buf = (const float*)outputs[i].buf;
        unsigned char* label;

        get_tensor_chw(&(app_ctx->output_attrs[i]), &channel, &height, &width);
        label = (unsigned char*)malloc((size_t)height * width);
        if (label == NULL) {
            printf("malloc label map fail!\n");
            ret = -1;
            goto out;
        }
        argmax_label(buf, &(app_ctx->output_attrs[i]), label);

        if (channel == BEV_OCC_STATIC_CLASSES && result->static_label == NULL) {
            result->static_label = label;
            result->static_w = width;
            result->static_h = height;
        } else if (channel == BEV_OCC_DYNAMIC_CLASSES && result->dynamic_label == NULL) {
            result->dynamic_label = label;
            result->dynamic_w = width;
            result->dynamic_h = height;
        } else if (channel == BEV_OCC_BEV_CLASSES && result->bev_label == NULL) {
            result->bev_label = label;
            result->bev_w = width;
            result->bev_h = height;
        } else {
            printf("unexpected output %d: channel=%d, size=%dx%d\n", i, channel, height, width);
            free(label);
            ret = -1;
            goto out;
        }
    }

    if (result->static_label == NULL || result->dynamic_label == NULL ||
        result->bev_label == NULL) {
        printf("missing head output!\n");
        ret = -1;
        goto out;
    }

    // 记录几何关系, 供可视化逆映射回原图
    result->transform.src_left = crop_left;
    result->transform.src_top = crop_top;
    result->transform.src_w = crop_w;
    result->transform.src_h = crop_h;
    result->transform.src_img_w = img->width;
    result->transform.src_img_h = img->height;

out:
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    free(dst_img.virt_addr);
    if (ret != 0) {
        release_bev_occ_result(result);
    }
    return ret;
}
