//
// Created by user on 14/01/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEPARABLEPARALLELOMP_H
#define KERNEL_IMAGE_PROCESSING_SEPARABLEPARALLELOMP_H
#include "opencv2/opencv.hpp"
void applyColumnFilter(cv::Mat* src, cv::Mat* intermediate, std::vector<double> kernel_col, int kernelSize, int offset, int y0, int y1);
void applyRowFilter(cv::Mat* intermediate, cv::Mat* dst, std::vector<double> kernel_row, int kernelSize, int offset, int y0, int y1);
void applyFilter_sepPar(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel_col, std::vector<double> filter_row);
#endif //KERNEL_IMAGE_PROCESSING_SEPARABLEPARALLELOMP_H
