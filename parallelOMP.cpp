//
// Created by user on 13/01/2023.
//
#include <iostream>
#include <chrono>
#include "parallelOMP.h"
void filter(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel, int kernelSize, int offset, int y0, int y1){
    int start = (((kernelSize-1)/2) > y0) ? ((kernelSize-1)/2) : y0;
    int end = ((src->rows - (kernelSize - 1)/2) < y1) ? (src->rows - (kernelSize - 1)/2) : y1;
    for(int y = start; y < end; y++){
        for(int x = (kernelSize-1)/2; x < src->cols - (kernelSize - 1)/2; x++){
            for(int channel = 0; channel < src->channels(); channel++) {
                double convolutedValue = 0;
                for(int i = 0; i<kernelSize; i++){
                    for(int j = 0; j<kernelSize; j++){
                        convolutedValue += src->data[(y+i - offset) * src->step + (x+j - offset) * src->channels() + channel] * kernel[i*kernelSize + j];
                    }
                }
                dst->data[y * dst->step + x * dst->channels() + channel] = static_cast<uchar>(convolutedValue);
            }
        }
    }
};

void applyFilter_parOMP(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel){
    auto begin = std::chrono::high_resolution_clock::now();
    int kernelSize = static_cast<int>(std::sqrt(kernel.size()));
    int offset = (kernelSize - 1)/2;
    int tileHeight = std::ceil(src->rows / numProcs);
#pragma omp parallel default(none) shared (src, dst, kernel, tileHeight, numProcs, kernelSize, offset)
#pragma omp for
    for (int threadIdx = 0; threadIdx < numProcs; threadIdx++) {
        filter(src, dst, kernel, kernelSize, offset, tileHeight * threadIdx, tileHeight * (threadIdx + 1));
    }
#pragma omp barrier
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Par Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
};


