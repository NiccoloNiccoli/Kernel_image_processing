//
// Created by user on 13/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEQUENTIAL_H
#define KERNEL_IMAGE_PROCESSING_SEQUENTIAL_H
#include "opencv2/opencv.hpp"
double applyFilter_seq(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel);
#endif //KERNEL_IMAGE_PROCESSING_SEQUENTIAL_H
