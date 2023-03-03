//
// Created by user on 13/01/2023.
//
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <omp.h>
#include "definitions.h"
#include "parallelOMP.h"
#include "separableParallelOMP.h"
#include "sequential.h"
#include "separableSequential.h"
#include "utils.h"

int main() {
#ifdef _OPENMP
    printf("_OPENMP defined\nNum procs: %d\n", omp_get_num_procs());
#endif
    std::vector<double> t_seq, t_seq_sep, t_omp, t_omp_sep;
    std::string imagesName[] = {"cat512.jpg", "cat1024.jpg", "cat2048.jpg", "cat4096.jpg", "cat8192.jpg"};
    int nProcs[] = {2, 4, 8, 16, 32};
    for(const std::string& imageName : imagesName){
        cv::Mat src = cv::imread("../images/"+imageName);
        cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
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
        if(RUN_OMP) {
            for (int nProc: nProcs) {
                //Doing the "normal" convolution and saving the result
                double t = applyFilter_parOMP(nProc, &src, &dst, filter);
                t_omp.push_back(t);
                cv::imwrite("../results/MW" + std::to_string(MASK_WIDTH) + "/omp_normal_" + imageName, dst);
                //Doing the separable convolution
                t = applyFilter_sepPar(nProc, &src, &dst, filterCol, filterRow);
                t_omp_sep.push_back(t);
                cv::imwrite("../results/MW" + std::to_string(MASK_WIDTH) + "/omp_sep_" + imageName, dst);
            }
        }
        if(RUN_SEQ){
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
    if(RUN_SEQ && RUN_OMP){
        save("../results/MW" + std::to_string(MASK_WIDTH) + "/omp.csv", t_seq, t_omp, t_seq_sep, t_omp_sep, nProcs);
    }
    return 0;
}