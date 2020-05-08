#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "Feature.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
    // open structure file
    
    string fname;
    for(int i = 0; i < argc - 1; i++){
        if(strncmp(argv[i], "-f", 2) == 0 || strncmp(argv[i], "--file", 2) == 0){
            fname = argv[i + 1];
        }
    }

    ifstream in;
    in.open("structure.txt");

    int windowSize, haarNum;
    in >> windowSize >> haarNum;

    // generate features

    vector<Feature> features;
    for(int i = 0; i < windowSize; i++){
        for(int j = 0; j < windowSize; j++){
            for(int k = 1; k < windowSize - i; k++){
                for(int l = 1; l < windowSize - j; l++){
                    for(int m = 0; m < 5; m++){
                        if((m == 0 && k % 2 != 0) || (m == 1 && l % 2 != 0) || (m == 2 && k % 3 != 0) || (m == 3 && l % 3 != 0) || (m == 4 && (k % 2 != 0 || l % 2 != 0))) { continue; }
                        features.push_back(Feature(i, j, k, l, m));
                    }
                }
            }
        }
    }

    vector<Feature> bestfeatures;
    int thresholds[haarNum];
    bool gt[haarNum];
    double featureweights[haarNum];
    for(int i = 0; i < haarNum; i++){
        int featuresel;
        in >> featuresel;
        bestfeatures.push_back(features.at(featuresel));
    }
    for(int i = 0; i < haarNum; i++){
        in >> thresholds[i];
    }
    for(int i = 0; i < haarNum; i++){
        in >> gt[i];
    }
    for(int i = 0; i < haarNum; i++){
        in >> featureweights[i];
    }
    // open image
    Mat rawimg = imread(fname, IMREAD_GRAYSCALE);
    auto start = chrono::high_resolution_clock::now();
    Mat resized, intimg;
    resize(rawimg, resized, Size(windowSize, windowSize));
    integral(resized, intimg);
    double ret = 0;
    for(int i = 0; i < haarNum; i++){
        if(gt[i]){
            ret += (bestfeatures.at(i).evaluate(intimg) < thresholds[i]) * featureweights[i];
        }else{
            ret += (bestfeatures.at(i).evaluate(intimg) >= thresholds[i]) * featureweights[i];
        }
    }
    auto end = chrono::high_resolution_clock::now();
    if(round(ret)){
        cout << "Face with confidence " << ret << endl;
    }else{
        cout << "Not face with confidence" << 1 - ret << endl;
    }
    cout << "Execution time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " micros" << endl;
}
