//
// Created by user on 14/01/2023.
//

#include "separableParallelOMP.h"
#include "chrono"
void applyColumnFilter(cv::Mat* src, cv::Mat* intermediate, std::vector<double> kernel_col, int kernelSize, int offset, int y0, int y1){
    int start = (((kernelSize-1)/2) > y0) ? ((kernelSize-1)/2) : y0;
    int end = ((src->rows - (kernelSize - 1)/2) < y1) ? (src->rows - (kernelSize - 1)/2) : y1;
    for(int y = start; y < end; y++){
        for(int x = 0; x < src->cols; x++){
            for(int channel = 0; channel < src->channels(); channel++) {
                double convolutedValue = 0;
                for(int i = 0; i < kernelSize; i++){
                    convolutedValue += src->data[(y + i - offset) * src->step + (x) * src->channels() + channel] * kernel_col[i];
                }
                intermediate->data[y * intermediate->step + x * intermediate->channels() + channel] = static_cast<uchar>(convolutedValue);
            }
        }
    }
};

void applyRowFilter(cv::Mat* intermediate, cv::Mat* dst, std::vector<double> kernel_row, int kernelSize, int offset, int y0, int y1){
    int start = (((kernelSize-1)/2) > y0) ? ((kernelSize-1)/2) : y0;
    int end = ((dst->rows - (kernelSize - 1)/2) < y1) ? (dst->rows - (kernelSize - 1)/2) : y1;
    for(int y = start; y < end; y++){
        for(int x = (kernelSize - 1)/2; x < dst->cols - (kernelSize - 1)/2; x++){
            for(int channel = 0; channel < dst->channels(); channel++) {
                double convolutedValue = 0;
                for(int j = 0; j < kernelSize; j++){
                    convolutedValue += intermediate->data[(y - offset) * dst->step + (x + j - offset) * dst->channels() + channel] * kernel_row[j];
                }
                dst->data[y * dst->step + x * dst->channels() + channel] = static_cast<uchar>(convolutedValue);
            }
        }
    }
};

void applyFilter_sepPar(int numProcs, cv::Mat* src, cv::Mat* dst, std::vector<double> kernel_col, std::vector<double> kernel_row){
    auto begin = std::chrono::high_resolution_clock::now();
    int kernelSize = kernel_col.size();
    int offset = (kernelSize - 1)/2;
    cv::Mat intermediate = cv::Mat::zeros(cv::Size(src->cols, src->rows), CV_8UC3);
    int tileHeight = std::ceil(src->rows / numProcs);
#pragma omp parallel default(none) shared (src, dst, kernel_col, kernelSize, tileHeight, numProcs, offset, intermediate)
#pragma omp for
    for (int threadIdx = 0; threadIdx < numProcs; threadIdx++) {
        applyColumnFilter(src, &intermediate, kernel_col, kernelSize, offset, tileHeight * threadIdx, tileHeight * (threadIdx + 1));
    }
#pragma omp barrier
#pragma omp for
    for (int threadIdx = 0; threadIdx < numProcs; threadIdx++) {
        applyRowFilter(&intermediate, dst, kernel_row, kernelSize, offset, tileHeight * threadIdx, tileHeight * (threadIdx + 1));
    }
#pragma omp barrier
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Seq Sep Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
};