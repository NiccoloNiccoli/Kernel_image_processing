//
// Created by user on 13/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEQUENTIALOMP_H
#define KERNEL_IMAGE_PROCESSING_SEQUENTIALOMP_H
#include "opencv2/opencv.hpp"
void applyFilter_seq(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel);
#endif //KERNEL_IMAGE_PROCESSING_SEQUENTIALOMP_H
