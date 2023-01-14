//
// Created by user on 13/01/2023.
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/imgproc/imgproc.hpp>
#include "sequentialOMP.cpp"
#include "parallelOMP.cpp"
#include "separableSequentialOMP.cpp"
#include "separableParallelOMP.cpp"
#include <chrono>

int main() {
    int numProcs = 4;
    cv::Mat src = cv::imread("../images/cat01.jpg", cv::IMREAD_COLOR);
    cv::Mat dst_seq = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC3);
    cv::Mat dst_par = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC3);
    cv::Mat dst_sepSeq = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC3);
    cv::Mat dst_sepPar = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC3);
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
    std::vector<double> filter_col {1,1,1};
    std::vector<double> filter_row {1,0,-1};
    applyFilter_seq(numProcs, &src, &dst_seq, filter);
    applyFilter_parOMP(numProcs, &src, &dst_par, filter);
    applyFilter_sepSeq(numProcs, &src, &dst_sepSeq, filter_col, filter_row);
    applyFilter_sepPar(numProcs, &src, &dst_sepPar, filter_col, filter_row);
    cv::imshow("el gatito seq", dst_seq);
    cv::imshow("el gatito par", dst_par);
    cv::imshow("el gatito sepSeq", dst_sepSeq);
    cv::imshow("el gatito sepPar", dst_sepPar);
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