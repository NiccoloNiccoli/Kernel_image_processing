//
// Created by Niccolò Niccoli on 19/02/2023.
//

#include "utils.h"
#include <string>

void save(const std::string& path, std::vector<double> seq, std::vector<double> par, std::vector<double> seqSep, std::vector<double> parSep){
    std::ofstream outputFile;
    outputFile.open(path, std::ios::out | std::ios::app);
    if(outputFile.is_open()){
        outputFile << "Sequential;Sequential separable;Parallel;Parallel separable\n";
        for(int i = 0; i < seq.size(); i++){
            outputFile << seq[i] << ";" << seqSep[i] << ";" << par[i] << ";" << parSep[i] << "\n";
        }
    }
}
void save(const std::string& path, std::vector<double> seq, std::vector<double> par, std::vector<double> seqSep, std::vector<double> parSep, int nProcs[5]){
    std::ofstream outputFile;
    outputFile.open(path, std::ios::out | std::ios::app);
    if(outputFile.is_open()){
        outputFile << "Sequential;Sequential separable;Parallel (with "+std::to_string(nProcs[0])+" cores);Parallel separable (with "+ std::to_string(nProcs[0]) +" cores);Parallel (with "+std::to_string(nProcs[1])+" cores);Parallel separable (with "+std::to_string(nProcs[1])+" cores);Parallel (with "+std::to_string(nProcs[2])+" cores);Parallel separable (with "+std::to_string(nProcs[2])+" cores);Parallel (with "+std::to_string(nProcs[3])+" cores);Parallel separable (with "+std::to_string(nProcs[3])+" cores);Parallel (with "+std::to_string(nProcs[4])+" cores);Parallel separable (with "+std::to_string(nProcs[4])+" cores)\n";
        int j = 0;
        for(int i = 0; i < seq.size(); i++){
            double t2, t2sep, t4, t4sep, t8, t8sep, t16, t16sep, t32, t32sep;
            t2 = par[j];
            t2sep = parSep[j++];
            t4 = par[j];
            t4sep = parSep[j++];
            t8 = par[j];
            t8sep = parSep[j++];
            t16 = par[j];
            t16sep = parSep[j++];
            t32 = par[j];
            t32sep = parSep[j++];
            outputFile << seq[i] << ";" << seqSep[i] << ";" << t2 << ";" << t2sep << ";" << t4 << ";" << t4sep<< ";" << t8
            << ";" << t8sep << ";" << t16 << ";" << t16sep << ";" << t32 << ";" << t32sep << "\n";
        }
    }
}
