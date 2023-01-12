#include <iostream>
#include "opencv2/opencv.hpp"
#include "sequentialConvolution.cu"

int main() {
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
    cv::waitKey();
    return 0;
}
