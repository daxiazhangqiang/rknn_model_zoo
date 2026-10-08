// BEV-OCC 感知模型推理示例 (静态分割 / 动态分割 / BEV 占据)
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <algorithm>
#include <string>
#include <vector>

#include "bev_occ.h"
#include "file_utils.h"
#include "image_utils.h"

// 常用图像后缀
static const char* IMAGE_EXTS[] = {".jpg", ".jpeg", ".png", ".bmp",
                                   ".JPG", ".JPEG", ".PNG", ".BMP"};

// 取当前时间, 单位 ms
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// 取文件名去掉后缀
static std::string file_stem(const std::string& path)
{
    size_t slash = path.find_last_of('/');
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    return (dot == std::string::npos) ? name : name.substr(0, dot);
}

// 判断是否图像文件
static bool is_image_file(const std::string& path)
{
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string ext = path.substr(dot);
    for (size_t i = 0; i < sizeof(IMAGE_EXTS) / sizeof(IMAGE_EXTS[0]); i++) {
        if (ext == IMAGE_EXTS[i]) {
            return true;
        }
    }
    return false;
}

// 判断是否目录
static bool is_directory(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

// 递归创建目录
static int mkdir_p(const std::string& path)
{
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur.empty() || cur == "/") {
                continue;
            }
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                return -1;
            }
        }
    }
    return 0;
}

// 输入是图像则返回单张, 是目录则返回目录下所有图像 (按名称排序)
static std::vector<std::string> collect_images(const std::string& input)
{
    std::vector<std::string> images;

    if (is_directory(input)) {
        DIR* dir = opendir(input.c_str());
        if (dir == NULL) {
            printf("opendir %s fail!\n", input.c_str());
            return images;
        }
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            if (name == "." || name == "..") {
                continue;
            }
            std::string path = input + "/" + name;
            if (is_image_file(path)) {
                images.push_back(path);
            }
        }
        closedir(dir);
        std::sort(images.begin(), images.end());
    } else if (is_image_file(input)) {
        images.push_back(input);
    }

    return images;
}

// 打印某个 head 的类别占比 (取占比最高的前 3 类)
static void print_label_stats(const char* tag, const unsigned char* label,
                              int label_w, int label_h, int num_classes,
                              const char* (*name_of)(int))
{
    if (label == NULL || label_w <= 0 || label_h <= 0) {
        return;
    }
    std::vector<int> counts(num_classes, 0);
    int total = label_w * label_h;

    for (int i = 0; i < total; i++) {
        if (label[i] < num_classes) {
            counts[label[i]]++;
        }
    }

    std::vector<int> order(num_classes);
    for (int i = 0; i < num_classes; i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&counts](int a, int b) { return counts[a] > counts[b]; });

    printf("    %-8s ", tag);
    for (int i = 0; i < num_classes && i < 3; i++) {
        int id = order[i];
        if (counts[id] == 0) {
            break;
        }
        printf("%s %.1f%%  ", name_of(id), 100.0 * counts[id] / total);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        printf("%s <model_path> <image_or_dir> [out_dir]\n", argv[0]);
        printf("  example: %s model/bevocc_96_lossweight2.rknn imgs/ bevocc_out\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    std::string input = argv[2];
    std::string out_dir = (argc == 4) ? argv[3] : "bevocc_out";

    std::vector<std::string> images = collect_images(input);
    if (images.empty()) {
        printf("no image found in %s\n", input.c_str());
        return -1;
    }

    printf("model   : %s\n", model_path);
    printf("input   : %s (%zu images)\n", input.c_str(), images.size());
#if ENABLE_VISUALIZATION
    printf("out_dir : %s\n", out_dir.c_str());
#else
    printf("visualization: disabled (ENABLE_VISUALIZATION=0)\n");
#endif

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_bev_occ_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_bev_occ_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

#if ENABLE_VISUALIZATION
    if (mkdir_p(out_dir) != 0) {
        printf("mkdir %s fail!\n", out_dir.c_str());
        release_bev_occ_model(&rknn_app_ctx);
        return -1;
    }
#endif

    double total_ms = 0.0;
    int done = 0;

    for (size_t i = 0; i < images.size(); i++) {
        std::string stem = file_stem(images[i]);
        image_buffer_t src_image;
        bev_occ_result_t result;
        memset(&src_image, 0, sizeof(image_buffer_t));

        printf("[%zu/%zu] %s\n", i + 1, images.size(), images[i].c_str());
        ret = read_image(images[i].c_str(), &src_image);
        if (ret != 0) {
            printf("    read image fail! ret=%d\n", ret);
            continue;
        }

#if !ENABLE_VISUALIZATION
        printf("    src image: %d x %d\n", src_image.width, src_image.height);
#endif

        double t0 = now_ms();
        ret = inference_bev_occ_model(&rknn_app_ctx, &src_image, &result);
        double cost_ms = now_ms() - t0;
        if (ret != 0) {
            printf("    inference fail! ret=%d\n", ret);
            free(src_image.virt_addr);
            continue;
        }
        total_ms += cost_ms;
        done++;

        print_label_stats("static", result.static_label, result.static_w,
                          result.static_h, BEV_OCC_STATIC_CLASSES, bev_occ_static_name);
        print_label_stats("dynamic", result.dynamic_label, result.dynamic_w,
                          result.dynamic_h, BEV_OCC_DYNAMIC_CLASSES, bev_occ_dynamic_name);
        print_label_stats("bev", result.bev_label, result.bev_w,
                          result.bev_h, BEV_OCC_BEV_CLASSES, bev_occ_bev_name);
        printf("    inference %.1f ms\n", cost_ms);

#if ENABLE_VISUALIZATION
        std::string img_dir = out_dir + "/" + stem;
        if (mkdir_p(img_dir) == 0) {
            std::string static_path = img_dir + "/" + stem + "_static.jpg";
            std::string dynamic_path = img_dir + "/" + stem + "_dynamic.jpg";
            std::string bev_path = img_dir + "/" + stem + "_bevocc.jpg";
            if (save_bev_occ_visualization(&src_image, &result, static_path.c_str(),
                                           dynamic_path.c_str(), bev_path.c_str()) != 0) {
                printf("    save visualization fail!\n");
            } else {
                printf("    saved to %s/\n", img_dir.c_str());
            }
        } else {
            printf("    mkdir %s fail!\n", img_dir.c_str());
        }
#endif

        release_bev_occ_result(&result);
        free(src_image.virt_addr);
    }

    if (done > 0) {
        printf("done %d/%zu images, average inference %.1f ms (%.2f FPS)\n",
               done, images.size(), total_ms / done, 1000.0 * done / total_ms);
    }

    release_bev_occ_model(&rknn_app_ctx);
    return 0;
}
