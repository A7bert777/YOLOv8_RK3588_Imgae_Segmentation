#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h> 
#include "yolov8_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <chrono> 
#include <iostream>
#include <opencv2/opencv.hpp>

/*-------------------------------------------
                  Main Function
-------------------------------------------*/


//去除文件地址&后缀
std::string extractFileNameWithoutExtension(const std::string& path) 
{  
    auto pos = path.find_last_of("/\\");  
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);  
      
    // 查找并去除文件后缀  
    pos = filename.find_last_of(".");  
    if (pos != std::string::npos) {  
        filename = filename.substr(0, pos);  
    }  
      
    return filename;  
}


// 处理一个文件夹中的所有图像文件  
void processImagesInFolder(const std::string& folderPath, rknn_app_context_t* rknn_app_ctx, const std::string& outputFolderPath) {  
    DIR *dir = opendir(folderPath.c_str());  
    if (dir == nullptr) 
    {  
        perror("opendir");  
        return;  
    }  
    
    unsigned char class_colors[][3] = 
    {
        //{255, 56, 56},   // 'FF3838'
        {128, 0, 128},
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

    struct dirent *entry;  
    while ((entry = readdir(dir)) != nullptr) 
    {  
        std::string fileName = entry->d_name;  
        std::string fullPath = folderPath + "/" + fileName;  
         // 检查文件扩展名  
        if ((fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".jpg") == 0) ||  
            (fileName.size() >= 5 && strcmp(fileName.c_str() + fileName.size() - 5, ".jpeg") == 0) ||  
            (fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".png") == 0)) 
            {  
  
            std::string outputFileName = outputFolderPath + "/" + extractFileNameWithoutExtension(fullPath) + "_out.png";  
  
            int ret;  
            image_buffer_t src_image;  
            memset(&src_image, 0, sizeof(image_buffer_t));  
            ret = read_image(fullPath.c_str(), &src_image);  

  
            if (ret != 0) {  
                printf("read image fail! ret=%d image_path=%s\n", ret, fullPath.c_str());  
                continue;  
            }  
  
            object_detect_result_list od_results;  
            
            ret = inference_yolov8_seg_model(rknn_app_ctx, &src_image, &od_results);  
            if (ret != 0) {  
                printf("inference_yolov8_model fail! ret=%d\n", ret);  
                if (src_image.virt_addr != NULL) {  
                    free(src_image.virt_addr);  
                }  
                continue;  
            } 


            if (od_results.count >= 1)
            {
                int width = src_image.width;
                int height = src_image.height;
                char *ori_img = (char *)src_image.virt_addr;
                int cls_id = od_results.results[0].cls_id;
                uint8_t *seg_mask = od_results.results_seg[0].seg_mask;
                float alpha = 0.5f; // opacity
                for (int j = 0; j < height; j++)
                {
                    for (int k = 0; k < width; k++)
                    {
                        int pixel_offset = 3 * (j * width + k);
                        if (seg_mask[j * width + k] != 0)
                        {
                            ori_img[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255); // r
                            ori_img[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255); // g
                            ori_img[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % N_CLASS_COLORS][2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255); // b
                        }
                    }
                }
                free(seg_mask);
            }

            // draw boxes
            char text[256];
            for (int i = 0; i < od_results.count; i++)
            {
                object_detect_result *det_result = &(od_results.results[i]);
                printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                    det_result->box.left, det_result->box.top,
                    det_result->box.right, det_result->box.bottom,
                    det_result->prop);
                int x1 = det_result->box.left;
                int y1 = det_result->box.top;
                int x2 = det_result->box.right;
                int y2 = det_result->box.bottom;

                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
                sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
            }
            write_image(outputFileName.c_str(), &src_image);

            if (src_image.virt_addr != NULL) 
            {  
                free(src_image.virt_addr);  
            }  
        }
    }
    closedir(dir);  
}   

int main(int argc, char **argv)
{
    const std::string modelPath = "/home/firefly/yolov8seg_github/model/3_28carpetseg.rknn";  
    const std::string imageFolder = "/home/firefly/yolov8seg_github/inputimage";  
    const std::string outputFolder = "/home/firefly/yolov8seg_github/outputimage";

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov8_seg_model(modelPath.c_str(), &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_seg_model fail! ret=%d model_path=%s\n", ret, modelPath.c_str());
        return -1;
    }

    std::cout<<"start processImagesInFolder"<<std::endl;
    processImagesInFolder(imageFolder, &rknn_app_ctx, outputFolder); 

    std::cout<<"start release_yolov8_seg_model"<<std::endl;
    ret = release_yolov8_seg_model(&rknn_app_ctx);

    if (ret != 0) 
    {  
        printf("release_yolov8_model fail! ret=%d\n", ret);  
    }  

    deinit_post_process();

    return 0;
}
