#include <iostream>
#include "opencv2/opencv.hpp"
#include "sequentialConvolution.cu"
#include "parallelConvolution.h"
#include <opencv2/cudaarithm.hpp>
#include "chrono"

#define TILE_WIDTH 16
#define MASK_WIDTH 3
#define MASK_HEIGHT MASK_WIDTH
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
__global__ void conv(uchar* src, int srcWidth, int srcHeight, int srcChannels, double* convKernel, int kernelWidth, int kernelHeight, uchar* dst = nullptr){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double convValue = 0.0;
    if(index < kernelWidth * kernelHeight)
        printf("%f", convKernel[index]);
    int startingCol = -(kernelWidth - 1) / 2;
    int startingRow = -(kernelHeight - 1) / 2;
    for(int j = 0; j < kernelHeight; j++){
        for(int i = 0; i < kernelWidth; i++){
            convValue += src[index + ((j + startingRow) * kernelWidth + (i + startingRow)) * srcChannels] * convKernel[j*kernelWidth + i];
        }
    }

    dst[index] = static_cast<uchar>(convValue);

}
/*__global__ void convolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel, int kernelWidth, int kernelHeight, uchar* dst){

    __shared__ uchar Ns[BLOCK_WIDTH][BLOCK_WIDTH];
   // printf("%d", srcChannels);
    int mask_radius_w = kernelWidth/2;
    int mask_radius_h = kernelHeight/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - mask_radius_h;
    int col_i = col_o - mask_radius_w;

    if((row_i >= 0) && (row_i < srcHeight) && (col_i >= 0) && (col_i < srcWidth)){
        Ns[ty][tx] = src[row_i * srcWidth + col_i];
    }else{
        Ns[ty][tx]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
       // printf("Con %d siamo dentro...\n", threadIdx.x);
        for(int i = 0; i < kernelHeight; i++){
            for(int j = 0; j < kernelWidth; j++){
               // printf("dentro, %f\n", (float)Ns[i+ty][j+tx]);
                //printf("%f %d", convKernel[i * kernelWidth +j], Ns[i+ty][j+tx]);
                output += convKernel[i * kernelWidth + j] * (float)Ns[i+ty][j+tx]; //fixme 11-02 kernelWidth e MASK_WIDTH sono la stessa cosa
                //printf("%f", output);
            }
        }
        //printf("Con %d siamo fuori...\n", threadIdx.x);

        if(row_o < srcHeight && col_o < srcWidth){
            //printf("%d sta aggiornando dst...\n", threadIdx.x);
            dst[row_o * srcWidth + col_o] = static_cast<uchar>(output);
        }
    }
}*/
__global__ void convolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel, int kernelWidth, int kernelHeight, uchar* dst){

    __shared__ uchar Ns[BLOCK_WIDTH][BLOCK_WIDTH];
    // printf("%d", srcChannels);
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
        // printf("Con %d siamo dentro...\n", threadIdx.x);
        for(int i = 0; i < kernelHeight; i++){
            for(int j = 0; j < kernelWidth; j++){
                // printf("dentro, %f\n", (float)Ns[i+ty][j+tx]);
                //printf("%f %d", convKernel[i * kernelWidth +j], Ns[i+ty][j+tx]);
                output += convKernel[i * kernelWidth + j] * (float)Ns[i+ty][j+tx]; //fixme 11-02 kernelWidth e MASK_WIDTH sono la stessa cosa
                //printf("%f", output);
            }
        }
        //printf("Con %d siamo fuori...\n", threadIdx.x);

        if(row_o < srcHeight && col_o < srcWidth){
            //printf("%d sta aggiornando dst...\n", threadIdx.x);
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
}

__global__ void sepRowConvolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel_row, int kernelWidth, uchar* dst){

    __shared__ uchar Ns[TILE_WIDTH][BLOCK_WIDTH];
    // printf("%d", srcChannels);
    int mask_radius_w = kernelWidth/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o;
    int col_i = col_o - mask_radius_w;

    if((col_i >= 0) && (col_i < srcWidth)){
        Ns[tx][ty] = src[(row_i * srcWidth + col_i) * srcChannels + blockIdx.z];
    }else{
        Ns[tx][ty]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
            for(int j = 0; j < kernelWidth; j++){
                output += convKernel_row[j] * (float)Ns[j+tx][ty]; //fixme 11-02 kernelWidth e MASK_WIDTH sono la stessa cosa
            }

        if(row_o < srcHeight && col_o < srcWidth){
            //printf("%d sta aggiornando dst...\n", threadIdx.x);
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
}
__global__ void sepColConvolution(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel_col, int kernelHeight, uchar* dst){

    __shared__ uchar Ns[TILE_WIDTH][BLOCK_WIDTH];
    // printf("%d", srcChannels);
    int mask_radius_h = kernelHeight/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - mask_radius_h;
    int col_i = col_o;

    if((row_i >= 0) && (row_i < srcHeight)){
        Ns[tx][ty] = src[(row_i * srcWidth + col_i) * srcChannels + blockIdx.z];
    }else{
        Ns[tx][ty]=0.0f;
    }
    __syncthreads();
    float output = 0.0f;
    if(ty < TILE_WIDTH && tx < TILE_WIDTH){
        for(int i = 0; i < kernelHeight; i++){
                output += convKernel_col[i] * (float)Ns[tx][i+ty]; //fixme 11-02 kernelWidth e MASK_WIDTH sono la stessa cosa
        }

        if(row_o < srcHeight && col_o < srcWidth){
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = static_cast<uchar>(output);
        }
    }
}

__global__ void sepColConvolution2(const uchar* __restrict__ src, int srcWidth, int srcHeight, int srcChannels,const float* __restrict__ convKernel_col, int kernelHeight, uchar* dst, bool fC = true){
    int mask_radius_h = kernelHeight/2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - mask_radius_h;
    int col_i = col_o - mask_radius_h;

    if(fC){
        if(row_i >= 0 && row_i < srcHeight && col_i >= 0 && col_i < srcWidth) {
            float sum = 0.0f;
            float filterCol[] = {1, 2, 1};
            for (int i = 0; i < MASK_WIDTH; i++) {
                sum += src[(col_o  + (row_i+i) * srcWidth) * srcChannels + blockIdx.z] * filterCol[i];
            }
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = sum;
        }
    }else {
        if (row_i >= 0 && row_i < srcHeight && col_i >= 0 && col_i < srcWidth) {
            float sum = 0.0f;
            float filterRow[] = {1, 0, -1};
            for (int i = 0; i < MASK_WIDTH; i++) {
                sum += src[(col_i + i + row_o * srcWidth) * srcChannels + blockIdx.z] * filterRow[i];
            }
            dst[(row_o * srcWidth + col_o) * srcChannels + blockIdx.z] = sum;
        }
    }

}
int main() {
    if(false){
    cv::Mat src = cv::imread("../images/cat01.jpg", cv::IMREAD_COLOR);
    cv::Mat dst = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC3);
   /* std::vector<double> filter {1, 2, 1,
                                2, 4, 2,
                                1, 2, 1};
    for(double & i : filter)
        i/=16;*/
    /*std::vector<double> filter {0, 0, 1, 2, 1, 0, 0,
                                0, 3, 13,22,13,3,0,
                                1,13,59,97,59,13,1,
                                2,22,97,159,97,22,2,
                                1,13,59,97,59,13,1,
                                0, 3, 13,22,13,3,0,
                                0, 0, 1, 2, 1, 0, 0};
    for(double & i : filter)
        i/=1003;
        */

    std::vector<double> filter {1,0,-1,1,0,-1,1,0,-1};
    applyFilter(&src, &dst, filter);
    cv::imshow("el gatito", dst);
    /*cv::Mat dst2;
    cv::Mat kernel(3,3, CV_32F);
    float sum = 16.0f;
    kernel.at<float>(0,0) = 1.0f/sum;
    kernel.at<float>(0,1) = 2.0f/sum;
    kernel.at<float>(0,2) = 1.0f/sum;
    kernel.at<float>(1,0) = 2.0f/sum;
    kernel.at<float>(1,1) = 4.0f/sum;
    kernel.at<float>(1,2) = 2.0f/sum;
    kernel.at<float>(2,0) = 1.0f/sum;
    kernel.at<float>(2,1) = 2.0f/sum;
    kernel.at<float>(2,2) = 1.0f/sum;
    cv::filter2D(src, dst2, -1, kernel);
    cv::imshow("el gaton", dst2);*/
    cv::waitKey();}
    /*
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    int cuda_device_number = cv::cuda::getCudaEnabledDeviceCount();
    std::cout<<"CUDA DEVICE(S) NUMBER: "<<cuda_device_number<<std::endl;
    cv::Mat src = cv::imread("../images/cat01.jpg", cv::IMREAD_COLOR);
    std::cout<<src.cols<<"x"<<src.rows<<std::endl;
    int count = 0;
    for(int j = 0; j < src.rows; j++){
        for(int i = 0; i<src.cols; i++){
            for(int c = 0; c < src.channels(); c++) {
                src.data[(j * src.cols + i) * src.channels() + c] =
                        255 - src.data[(j * src.cols + i) * src.channels() + c];
            }
            count++;
        }
    }
    cv::imshow("Ciao", src);
    printf("%d", count);
    cv::waitKey();
     */
    /*cv::Mat src = cv::imread("../images/cat01.jpg", cv::IMREAD_COLOR);
    std::cout<<src.cols<<"x"<<src.rows<<"x"<<src.channels()<<std::endl;
    //uchar dataOnHost[src.cols * src.rows * src.channels()];
    uchar* dataOnDevice;
    uchar* dst;
    const int kernelWidth = 3;
    const int kernelHeight = 3;
    double convKernel[kernelWidth * kernelHeight] = {1,0,-1,1,0,-1,1,0,-1};
    for(int i = 0; i < kernelWidth * kernelHeight; i++){
        convKernel[i]/= 1.0;
    }
    double* d_convKernel;
    const int kernelSize = kernelHeight * kernelWidth * sizeof(double);
    const int size = src.cols * src.rows * src.channels() * sizeof(uchar);
    cudaMalloc((void**)&d_convKernel, kernelSize);
    cudaMalloc((void**)&dataOnDevice, size);
    cudaMalloc((void**)&dst, size);
    cudaMemcpy(d_convKernel, convKernel, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dataOnDevice, src.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dst, src.data, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(16*16*3);
    dim3 dimGrid(5120);
    conv<<<512 * 16,256>>>(dataOnDevice,src.cols, src.rows,
                               src.channels(), d_convKernel, kernelWidth, kernelHeight, dst);
    cudaDeviceSynchronize();
    cudaMemcpy(src.data, dst, size, cudaMemcpyDeviceToHost);
    cudaFree(dataOnDevice);
    cudaFree(d_convKernel);
    cudaFree(dst);
    cv::imshow("ciao", src);
    cv::imwrite("pog.jpg", src);
    cv::waitKey();*/
    cv::Mat src = cv::imread("../images/cat01.jpg");
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    cv::Mat dst2 = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    dim3 dimBlock (BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid ((src.cols - 1)/TILE_WIDTH +1, (src.rows -1)/TILE_WIDTH+1,src.channels());
    uchar* d_src;
    uchar* d_dst;
    uchar* d_mid_dst;
    float convKernel[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
 /*   for(int i = 0; i<MASK_HEIGHT*MASK_WIDTH; i++){
        convKernel[i] /= 16.0f;
    }*/
    float convKernel_row[] = {-1,0,1};
    float convKernel_col[] = {1,2,1};
    /*for(int i= 0; i< MASK_WIDTH; i++){
        convKernel_col[i] /= 4.0f;
        convKernel_row[i] /= 4.0f;
    }*/
    for(float f : convKernel){
        std::cout<<f<<std::endl;
    }
    float* d_convKernel;
    float* d_convKernel_row;
    float* d_convKernel_col;
    const int srcSize = src.cols * src.rows * src.channels() * sizeof(uchar);
    const int kernelSize = MASK_WIDTH * MASK_HEIGHT * sizeof(float);
    const int kernelSize_sep = MASK_WIDTH * sizeof(float);
    cudaMalloc((void**)&d_convKernel, kernelSize);
    cudaMalloc((void**)&d_convKernel_row, kernelSize_sep);
    cudaMalloc((void**)&d_convKernel_col, kernelSize_sep);
    cudaMalloc((void**)&d_src, srcSize);
    cudaMalloc((void**)&d_dst, srcSize);
    cudaMalloc((void**)&d_mid_dst, srcSize);
    cudaMemcpy(d_convKernel, convKernel, kernelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_convKernel_row, convKernel_row, kernelSize_sep, cudaMemcpyHostToDevice);
    cudaMemcpy(d_convKernel_col, convKernel_col, kernelSize_sep, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src.data, srcSize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dst, nullptr, srcSize, cudaMemcpyHostToDevice);

    auto tic = std::chrono::high_resolution_clock::now();
    convolution<<<dimGrid, dimBlock>>>(d_src, src.cols, src.rows, src.channels(), d_convKernel, MASK_WIDTH, MASK_HEIGHT, d_dst);
    cudaDeviceSynchronize();
    auto toc = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    cudaMemcpy(src.data, d_dst, srcSize, cudaMemcpyDeviceToHost);
    cv::imshow("ciao", src);
    printf("Classic Conv,Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    cv::waitKey();
    cudaFree(d_dst);
    cudaMalloc((void**)&d_dst, srcSize);
    //cudaMemcpy(d_dst, nullptr, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(src.data, d_dst, srcSize, cudaMemcpyDeviceToHost);
    cv::imshow("Blank", src);
    cv::waitKey();
    tic = std::chrono::high_resolution_clock::now();
    sepRowConvolution<<<dimGrid, dimBlock>>>(d_src, src.cols, src.rows, src.channels(), d_convKernel_row, MASK_WIDTH, d_mid_dst);
    cudaDeviceSynchronize();
    cudaMemcpy(src.data, d_mid_dst, srcSize, cudaMemcpyDeviceToHost);
cv::imshow("ciao intermedio", src);
    cv::imwrite("pog_nel_mezzo.png", src);
cv::waitKey();
    sepColConvolution<<<dimGrid, dimBlock>>>(d_mid_dst, src.cols, src.rows, src.channels(), d_convKernel_col, MASK_HEIGHT, d_dst);

    //prima filtro colonna poi filtro riga
  /* sepColConvolution2<<<dimGrid, dimBlock>>>(d_src, src.cols, src.rows, src.channels(), d_convKernel_col, MASK_HEIGHT, d_mid_dst, false);
    cudaDeviceSynchronize();
    sepColConvolution2<<<dimGrid, dimBlock>>>(d_mid_dst, src.cols, src.rows, src.channels(), d_convKernel_col, MASK_HEIGHT, d_dst, true);
    */
    toc = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    cudaMemcpy(src.data, d_dst, srcSize, cudaMemcpyDeviceToHost);
    cv::imshow("ciao ma 20", src);
    cv::imwrite("pog.png", src);
    printf("Sep Conv,Time measured: %.4f seconds.\n", elapsed.count() * 1e-9);
    cv::waitKey();
    cudaFree(d_src);
    cudaFree(d_convKernel);
    cudaFree(d_convKernel_row);
    cudaFree(d_convKernel_col);
    cudaFree(d_dst);
    return 0;
}
