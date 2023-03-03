//
// Created by user on 13/01/2023.
//
#include <iostream>
#include "chrono"
#include "sequential.h"
double applyFilter_seq(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel){
    auto begin = std::chrono::high_resolution_clock::now();
    int kernelSize = static_cast<int>(std::sqrt(kernel.size()));
    int offset = (kernelSize - 1)/2;
    /*for(int y = (kernelSize-1)/2; y < src->rows - (kernelSize - 1)/2; y++){
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
    }*/
    for(int y = 0; y < src->rows; y++){
        for(int x = 0; x < src->cols; x++){
            for(int channel = 0; channel < src->channels(); channel++) {
                double convolutedValue = 0;
                    for (int i = 0; i < kernelSize; i++) {
                        for (int j = 0; j < kernelSize; j++) {
                            if(x + j >= offset && x + j < src->cols + offset && y + i >= offset && y + i < src->rows + offset) {
                                convolutedValue +=src->data[(y + i - offset) * src->step + (x + j - offset) * src->channels() +
                                                            channel] * kernel[i * kernelSize + j];
                            }/*else{
                                convolutedValue+=src->data[(y) * src->step + (x) * src->channels() +
                                                           channel] * kernel[i * kernelSize + j];
                            }*/
                        }
                    }
                dst->data[y * dst->step + x * dst->channels() + channel] = static_cast<uchar>(convolutedValue);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Seq Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    return elapsed.count() * 1e-9;
};
