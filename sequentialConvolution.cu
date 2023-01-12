//
// Created by Niccolò Niccoli on 23/12/2022.
//
#include <iostream>
#include "sequentialConvolution.h"
void applyFilter(cv::Mat* src, cv::Mat* dst, std::vector<double> kernel){
    int kernelSize = static_cast<int>(std::sqrt(kernel.size()));
    std::cout<<src->channels()<<std::endl;
    int offset = (kernelSize - 1)/2;
    std::cout<<offset<<std::endl;
    for(int y = (kernelSize-1)/2; y < src->rows - (kernelSize - 1)/2; y++){
        for(int x = (kernelSize-1)/2; x < src->cols - (kernelSize - 1)/2; x++){
            for(int channel = 0; channel < src->channels(); channel++) {
                double convolutedValue = 0;
                for(int i = 0; i<kernelSize; i++){
                    for(int j = 0; j<kernelSize; j++){
                        convolutedValue += src->data[(y+i - offset) * src->step + (x+j - offset) * src->channels() + channel] * kernel[i*kernelSize + j];
                    }
                }
                //std::cout<<convolutedValue<<" vs "<<static_cast<double>(src->data[y*src->step + x * src->channels() + channel])<<std::endl;
                dst->data[y * dst->step + x * dst->channels() + channel] = static_cast<uchar>(convolutedValue);
            }
        }
    }
};