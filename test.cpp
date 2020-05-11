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

    vector<Feature> bestfeatures;
    int thresholds[haarNum];
    bool gt[haarNum];
    double featureweights[haarNum];
    for(int i = 0; i < haarNum; i++){
        int x, y, dx, dy, type;
        in >> x >> y >> dx >> dy >> type;
        bestfeatures.push_back(Feature(x, y, dx, dy, type));
        in >> thresholds[i] >> gt[i] >> featureweights[i];
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
