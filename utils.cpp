//
// Created by Niccolò Niccoli on 19/02/2023.
//

#include "utils.h"

void save(std::string path, std::vector<double> seq, std::vector<double> par, std::vector<double> seqSep, std::vector<double> parSep){
    std::ofstream outputFile;
    outputFile.open(path, std::ios::out | std::ios::app);
    if(outputFile.is_open()){
        outputFile << "Sequential;Sequential separable;Parallel;Parallel separable\n";
        for(int i = 0; i < seq.size(); i++){
            outputFile << seq[i] << ";" << seqSep[i] << ";" << par[i] << ";" << parSep[i] << "\n";
        }
    }
}