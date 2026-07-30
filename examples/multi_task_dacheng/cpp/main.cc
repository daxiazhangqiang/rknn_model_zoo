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
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#include "multi_task.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "easy_timer.h"
#include "json.hpp"

using json = nlohmann::json;

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

#define DRAW_IMAGE 0

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("%s <model_path> <image_path> <json_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const char *json_path = argv[3];

    int ret;
    TIMER timer;

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    timer.indent_set("");
    init_post_process();

    timer.tik();
    ret = init_yolox_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolox_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    timer.tok();
    timer.print_time("init_yolox_model");

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

#if defined(RV1106_1103) 
    //RV1106 rga requires that input and output bufs are memory allocated by dma
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                       (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif

    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;
    pose_detect_result_list pose_results;
    face_detect_result_list face_results;

    timer.tik();
    ret = inference_yolox_model(&rknn_app_ctx, &src_image, &od_results, &pose_results, &face_results);
    if (ret != 0)
    {
        printf("init_yolox_model fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("inference_yolox_model");

    timer.tik();
    // print detection results to console
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
    }
    for (int i = 0; i < pose_results.count; i++)
    {
        printf("pose[%d] @ (%d %d %d %d) %.3f\n", i,
               pose_results.results[i].box.left, pose_results.results[i].box.top,
               pose_results.results[i].box.right, pose_results.results[i].box.bottom,
               pose_results.results[i].prop);
    }
    for (int i = 0; i < face_results.count; i++)
    {
        printf("face[%d] @ (%d %d %d %d) %.3f\n", i,
               face_results.results[i].box.left, face_results.results[i].box.top,
               face_results.results[i].box.right, face_results.results[i].box.bottom,
               face_results.results[i].prop);
    }
    // ===== write results to JSON =====
    {
        json j;
        j["orign_img_path"] = image_path;

        json model_results;

        // ---- object (BBOX) ----
        json obj_arr = json::array();
        for (int i = 0; i < od_results.count; i++) {
            json obj;
            obj["bbox"] = {od_results.results[i].box.left,
                           od_results.results[i].box.top,
                           od_results.results[i].box.right,
                           od_results.results[i].box.bottom};
            obj["class_name"] = coco_cls_to_name(od_results.results[i].cls_id);
            obj["score"] = od_results.results[i].prop;
            obj_arr.push_back(obj);
        }
        model_results["object"] = obj_arr;

        // ---- pose (17 keypoints) ----
        json pose_arr = json::array();
        for (int i = 0; i < pose_results.count; i++) {
            json pose;
            pose["bbox"] = {pose_results.results[i].box.left,
                            pose_results.results[i].box.top,
                            pose_results.results[i].box.right,
                            pose_results.results[i].box.bottom};
            pose["class_name"] = "person";
            pose["score"] = pose_results.results[i].prop;

            json kpts;
            for (int k = 0; k < 17; k++) {
                char kname[16];
                snprintf(kname, sizeof(kname), "kpt%d", k);
                kpts[kname] = {{"x", pose_results.results[i].keypoints[k][0]},
                               {"y", pose_results.results[i].keypoints[k][1]},
                               {"score", pose_results.results[i].keypoints[k][2]}};
            }
            pose["keypoints"] = kpts;
            pose_arr.push_back(pose);
        }
        model_results["pose"] = pose_arr;

        // ---- face (5 keypoints) ----
        json face_arr = json::array();
        for (int i = 0; i < face_results.count; i++) {
            json face;
            face["bbox"] = {face_results.results[i].box.left,
                            face_results.results[i].box.top,
                            face_results.results[i].box.right,
                            face_results.results[i].box.bottom};
            face["class_name"] = "face";
            face["score"] = face_results.results[i].prop;

            json kpts;
            for (int k = 0; k < 5; k++) {
                char kname[16];
                snprintf(kname, sizeof(kname), "kpt%d", k);
                kpts[kname] = {{"x", face_results.results[i].keypoints[k][0]},
                               {"y", face_results.results[i].keypoints[k][1]},
                               {"score", face_results.results[i].keypoints[k][2]}};
            }
            face["keypoints"] = kpts;
            face_arr.push_back(face);
        }
        model_results["face"] = face_arr;

        j["model_results"] = model_results;

        std::ofstream ofs(json_path);
        ofs << j.dump(4);
        ofs.close();
        printf("Results saved to %s\n", json_path);
    }
    timer.tok();
    timer.print_time("write out to json");
#if DRAW_IMAGE
    write_multi_task_results(image_path, &src_image, &od_results, &pose_results, &face_results);
#endif

out:
    deinit_post_process();

    ret = release_yolox_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolox_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103) 
        dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }

    return 0;
}
