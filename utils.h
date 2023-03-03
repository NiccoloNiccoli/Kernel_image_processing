//
// Created by Niccolò Niccoli on 19/02/2023.
//

#ifndef KERNEL_IMAGE_PROCESSING_UTILS_H
#define KERNEL_IMAGE_PROCESSING_UTILS_H
#include <iostream>
#include <vector>
#include <fstream>

void save(const std::string& path, std::vector<double> seq, std::vector<double> par, std::vector<double> seqSep,
          std::vector<double> parSep, int nProcs[5]);
void save(const std::string& path, std::vector<double> seq, std::vector<double> par, std::vector<double> seqSep,
          std::vector<double> parSep);

#endif //KERNEL_IMAGE_PROCESSING_UTILS_H
