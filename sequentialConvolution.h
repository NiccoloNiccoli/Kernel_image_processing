//
// Created by Niccolò Niccoli on 23/12/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEQUENTIALCONVOLUTION_H
#define KERNEL_IMAGE_PROCESSING_SEQUENTIALCONVOLUTION_H
#include "opencv2/opencv.hpp"
void applyFilter(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel);
#endif //KERNEL_IMAGE_PROCESSING_SEQUENTIALCONVOLUTION_H
