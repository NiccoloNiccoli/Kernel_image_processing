#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/cudaarithm.hpp>
#include "chrono"
#include "sequential.h"
#include "separableSequential.h"
#include "definitions.h"
#include "utils.h"


__global__ void convolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel, int kernelWidth, int kernelHeight, uchar* dst){

    __shared__ uchar Ns[BLOCK_WIDTH][BLOCK_WIDTH];
    int mask_radius_w = kernelWidth/2;
    int mask_radius_h = kernelHeight/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - mask_radius_h;
    int col_i = col_o - mask_radius_w;

    if((row_i >= 0) && (row_i < srcHeight) && (col_i >= 0) && (col_i < srcWidth)){
        Ns[ty][tx] = src[(row_i * srcWidth + col_i) * srcChannels + blockIdx.z];
    }else{
        Ns[ty][tx]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
        for(int i = 0; i < kernelHeight; i++){
            for(int j = 0; j < kernelWidth; j++){
                output += convKernel[i * kernelWidth + j] * (float)Ns[i+ty][j+tx];
            }
        }

        if(row_o < srcHeight && col_o < srcWidth){
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
    __syncthreads();
}

__global__ void sepRowConvolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel_row, int kernelWidth, uchar* dst){

    __shared__ uchar Ns[BLOCK_WIDTH * BLOCK_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o;
    int col_i = col_o - MASK_WIDTH/2;

    if((col_i >= 0) && (col_i < srcWidth)){
        Ns[tx + ty * BLOCK_WIDTH] = src[(row_i * srcWidth + col_i) * srcChannels + blockIdx.z];
    }else{
        Ns[tx + ty * BLOCK_WIDTH]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
            for(int j = 0; j < kernelWidth; j++){
                output += convKernel_row[j] * (float)Ns[j + tx + ty * BLOCK_WIDTH];
            }

        if(row_o < srcHeight && col_o < srcWidth){
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
}
__global__ void sepColConvolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel_col, int kernelHeight, uchar* dst){

    __shared__ uchar Ns[BLOCK_WIDTH * BLOCK_WIDTH];
    int mask_radius_h = kernelHeight/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - mask_radius_h;
    int col_i = col_o;

    if((row_i >= 0) && (row_i < srcHeight)){
        Ns[tx + ty * BLOCK_WIDTH] = src[(row_i * srcWidth + col_i) * srcChannels + blockIdx.z];
    }else{
        Ns[tx + ty * BLOCK_WIDTH]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
        for(int i = 0; i < kernelHeight; i++){
                output += convKernel_col[i] * (float)Ns[tx + (ty+i) * BLOCK_WIDTH];
        }

        if(row_o < srcHeight && col_o < srcWidth){
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
    __syncthreads();
}
int main() {
    //Data structures to save execution times
    std::vector<double> t_cuda, t_cuda_sep, t_seq, t_seq_sep;
    std::string imagesName[] = {"cat512.jpg", "cat1024.jpg", "cat2048.jpg", "cat4096.jpg", "cat8192.jpg"};
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("CC %d.%d\n", deviceProp.major, deviceProp.minor);
    for(const std::string& imageName : imagesName) {
        cv::Mat src = cv::imread("../images/"+imageName);
        cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
        if(RUN_CUDA){
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 dimGrid((src.cols - 1) / TILE_WIDTH + 1, (src.rows - 1) / TILE_WIDTH + 1, src.channels());
        const int imgSize = src.cols * src.rows * src.channels() * sizeof(uchar);
        const int kernelSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);
        const int kernelSizeSep = MASK_WIDTH * sizeof(float);
        //Defining the filter
        float filter[MASK_WIDTH * MASK_WIDTH];
        float filterCol[MASK_WIDTH];
        float filterRow[MASK_WIDTH];
        for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++)
            filter[i] = 1 / (float) (MASK_WIDTH * MASK_WIDTH);
        for (int i = 0; i < MASK_WIDTH; i++){
            filterCol[i] = 1 / (float) (MASK_WIDTH);
            filterRow[i] = 1 / (float) (MASK_WIDTH);
        }
        //Defining "device" variables
        uchar* d_src,*d_mid_dst, *d_dst;
        float* d_filter, *d_filterCol, *d_filterRow;
        //Allocating memory
        cudaMalloc((void**) &d_src, imgSize);
        cudaMalloc((void**) &d_mid_dst, imgSize);
        cudaMalloc((void**) &d_dst, imgSize);
        cudaMalloc((void**) &d_filter, kernelSize);
        cudaMalloc((void**) &d_filterCol, kernelSizeSep);
        cudaMalloc((void**) &d_filterRow, kernelSizeSep);
        //Copying data inside device variables
        cudaMemcpy(d_src, src.data, imgSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, filter, kernelSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filterCol, filterCol, kernelSizeSep, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filterRow, filterRow, kernelSizeSep, cudaMemcpyHostToDevice);
        //Launching the first convolution
        auto begin = std::chrono::high_resolution_clock::now();
        convolution<<<dimGrid, dimBlock>>>(d_src, src.cols, src.rows, src.channels(), d_filter, MASK_WIDTH, MASK_WIDTH, d_dst);
        cudaMemcpy(dst.data, d_dst, imgSize, cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //Saving the result
        cv::imwrite("../results/MW"+ std::to_string(MASK_WIDTH)+"/cuda_normal_"+imageName,dst);
        printf("Normal %.6f sec\n", elapsed.count() * 1e-9);
        t_cuda.push_back(elapsed.count() * 1e-9);
        //Launching the second convolution
        begin = std::chrono::high_resolution_clock::now();
        sepColConvolution<<<dimGrid, dimBlock>>>(d_src, src.cols, src.rows, src.channels(), d_filterCol, MASK_WIDTH, d_mid_dst);
        sepRowConvolution<<<dimGrid, dimBlock>>>(d_mid_dst, src.cols, src.rows, src.channels(), d_filterRow, MASK_WIDTH, d_dst);
        //Copying back data
        cudaMemcpy(dst.data, d_dst, imgSize, cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        //Saving the result
        cv::imwrite("../results/MW"+ std::to_string(MASK_WIDTH)+"/cuda_sep_"+imageName,dst);
        printf("Separable %.6f sec\n", elapsed.count() * 1e-9);
        t_cuda_sep.push_back(elapsed.count() * 1e-9);
        //De-allocating memory
        cudaFree(d_src);
        cudaFree(d_mid_dst);
        cudaFree(d_dst);
        cudaFree(d_filter);
        cudaFree(d_filterCol);
        cudaFree(d_filterRow);
        }
        if(RUN_SEQ){
            //Creating the filters
            std::vector<double> filter, filterCol, filterRow;
            const int filterElements = MASK_WIDTH * MASK_WIDTH;
            for(int i = 0; i < filterElements; i++){
                filter.push_back(1.0/(filterElements));
            }
            for(int i = 0; i < MASK_WIDTH; i++){
                filterCol.push_back(1.0/MASK_WIDTH);
                filterRow.push_back(1.0/MASK_WIDTH);
            }
            //Doing the "normal" convolution and saving the result
            double t = applyFilter_seq(&src, &dst, filter);
            t_seq.push_back(t);
            cv::imwrite("../results/MW"+ std::to_string(MASK_WIDTH)+"/seq_normal_"+imageName,dst);
            //Doing the separable convolution
            t = applyFilter_sepSeq(&src, &dst, filterCol, filterRow);
            t_seq_sep.push_back(t);
            cv::imwrite("../results/MW"+ std::to_string(MASK_WIDTH)+"/seq_sep_"+imageName,dst);
        }
    }
    if(RUN_SEQ && RUN_CUDA){
        save("../results/MW"+std::to_string(MASK_WIDTH)+"/cuda.csv", t_seq, t_cuda, t_seq_sep, t_cuda_sep);
    }
    return 0;
}
