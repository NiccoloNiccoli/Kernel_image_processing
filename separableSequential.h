//
// Created by user on 14/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIAL_H
#define KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIAL_H
#include "opencv2/opencv.hpp"
double applyFilter_sepSeq(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel_col, std::vector<double> filter_row);
#endif //KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIAL_H
