//
// Created by user on 13/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_PARALLELOMP_H
#define KERNEL_IMAGE_PROCESSING_PARALLELOMP_H
#include "opencv2/opencv.hpp"
void filter(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel, int y0, int y1);
void applyFilter_parOMP(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel);
#endif //KERNEL_IMAGE_PROCESSING_PARALLELOMP_H
