//
// Created by user on 14/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIALOMP_H
#define KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIALOMP_H
#include "opencv2/opencv.hpp"
void applyFilter_sepSeq(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel_col, std::vector<double> filter_row);
#endif //KERNEL_IMAGE_PROCESSING_SEPARABLESEQUENTIALOMP_H
