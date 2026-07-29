#ifndef _RKNN_YOLOX_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOX_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define POSE_NUMB_MAX_SIZE 32
#define FACE_NUMB_MAX_SIZE 32
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

// class rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    image_rect_t box;
    float keypoints[17][3];//keypoints x,y,conf
    float prop;
    int cls_id;
} pose_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[POSE_NUMB_MAX_SIZE];
} pose_detect_result_list;


typedef struct {
    image_rect_t box;
    float keypoints[5][3];//keypoints x,y,conf
    float prop;
    int cls_id;
} face_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[FACE_NUMB_MAX_SIZE];
} face_detect_result_list;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, 
    void *outputs, 
    letterbox_t *letter_box, 
    float conf_threshold, 
    float nms_threshold, 
    object_detect_result_list *od_results,
    pose_detect_result_list *pose_results,
    face_detect_result_list *face_results);

void deinitPostProcess();
#endif //_RKNN_YOLOX_DEMO_POSTPROCESS_H_
