#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <fstream>
#include "Feature.hpp"

using namespace cv;

int main(int argc, char* argv[]){
    // important constants
    int trainSize = 1000;
    int epochs = 5;
    int windowSize = 24;
    int haarNum = 5000;
    int testSize = 500;

    for(int i = 0; i < argc; i++){
        if(argv[i][0] == '-' && i < argc - 1){
            if(strncmp(argv[i], "--epochs", 8) == 0 || strncmp(argv[i], "-e", 2) == 0){
                epochs = atoi(argv[i + 1]);
            }else if(strncmp(argv[i], "--train-size", 12) == 0 || strncmp(argv[i], "-t", 2) == 0){
                trainSize = atoi(argv[i + 1]);
            }else if(strncmp(argv[i], "--window-size", 13) == 0 || strncmp(argv[i], "-w", 2) == 0){
                windowSize = atoi(argv[i + 1]);
            }else if(strncmp(argv[i], "--learners", 10) == 0 || strncmp(argv[i], "-l", 2) == 0){
                haarNum = atoi(argv[i + 1]);
            }else if(strncmp(argv[i], "--test-size", 11) == 0 || strncmp(argv[i], "-s", 2) == 0){
                testSize = atoi(argv[i + 1]);
            }
        }
    }

    cout << "Starting..." << endl;
    // get all files in UTKFace
    vector<string> facenames, notnames;
    DIR* dir = opendir("./UTKFace");
    struct dirent* ent;
    while((ent = readdir(dir)) != NULL){
        if(ent->d_name[0] == '.'){continue;}
        facenames.push_back(ent->d_name);
    }
    closedir(dir);

    dir = opendir("./Images");
    while((ent = readdir(dir)) != NULL){
        if(ent->d_name[0] == '.'){continue;}
        notnames.push_back(ent->d_name);
    }
    closedir(dir);

    cout << "Files read" << endl;

    // populate testing set
    vector<string> testfaces, testnots;
    for(int i = 0; i < 0.2 * facenames.size(); i++){
        int idx = rand() % facenames.size();
        testfaces.push_back("./UTKFace/" + facenames.at(idx));
        facenames.erase(facenames.begin() + idx);
    }
    for(int i = 0; i < 0.2 * notnames.size(); i++){
        int idx = rand() % notnames.size();
        testnots.push_back("./Images/" + notnames.at(idx));
        notnames.erase(notnames.begin() + idx);
    }

    int numFaces = facenames.size();
    int numNots = notnames.size();
    int numSamples = numFaces + numNots;

    // initialize computational resources
    double sampleweights[numSamples];
    fill(sampleweights, sampleweights + numSamples, 1);
    double cumweights[numSamples];
    for(int i = 0; i < numSamples; i++){
        cumweights[i] = i + 1;
    }
    double total = numSamples;

    cout << "Sample weights initialized" << endl;

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
    int numFeatures = features.size();
    int thresholds[numFeatures];
    fill(thresholds, thresholds + numFeatures, 0.5);
    bool gt[numFeatures];

    cout << "Features initialized" << endl;

    double featureweights[numFeatures];
    fill(featureweights, featureweights + numFeatures, 1/numFeatures);

    int bestidxs[haarNum];
    double bestweights[haarNum];

    cout << "Feature weights initialized" << endl;

    cout << "================= TRAINING SUMMARY ====================" << endl;
    cout << "Training size: " << trainSize << endl;
    cout << "Number of epochs: " << epochs << endl;
    cout << "Window size: " << windowSize << endl;
    cout << "Total feature count: " << numFeatures << endl;
    cout << "Total training count: " << numSamples << endl;
    cout << "Total testing count: " << testfaces.size() + testnots.size() << endl;
    cout << "    - Face training count: " << facenames.size() << endl;
    cout << "    - Face testing count: " << testfaces.size() << endl;
    cout << "    - Other training count: " << notnames.size() << endl;
    cout << "    - Other testing count: " << testnots.size() << endl;

    for(int epoch = 0; epoch < epochs; epoch++){
        // construct training set
        cout << "================ EPOCH " << epoch + 1 << " ===================" << endl;
        vector<Mat> dataset;
        int labels[trainSize];
        int keys[trainSize];

        for(int i = 0; i < trainSize; i++){
            // generate random number
            int sel = rand() % (int) cumweights[numSamples - 1];
            // binary search for closest number
            int idx = lower_bound(cumweights, cumweights + numSamples, sel) - cumweights;
            if(idx >= facenames.size()){
                // if it's not a face, add it to the set, and negate its key
                string fname = "./Images/" + notnames.at(idx - facenames.size());
                Mat rawimg = imread(fname, IMREAD_GRAYSCALE);
                if(rawimg.cols == 0){
                    cout << "fname: " << fname << "rawimg: " << rawimg << endl;
                }
                Mat resizedimg, intimg;
                resize(rawimg, resizedimg, Size(windowSize, windowSize));
                integral(resizedimg, intimg);
                dataset.push_back(intimg);
                keys[i] = idx;
                labels[i] = sampleweights[idx] * -1;
            }else{
                // if it is a face, add it to the set, and add the key
                string fname = "./UTKFace/" + facenames.at(idx);
                Mat rawimg = imread(fname, IMREAD_GRAYSCALE);
                if(rawimg.cols == 0){
                    cout << "fname: " << fname << "rawimg: " << rawimg << endl;
                }
                Mat resizedimg, intimg;
                resize(rawimg, resizedimg, Size(windowSize, windowSize));
                integral(resizedimg, intimg);
                dataset.push_back(intimg);
                keys[i] = idx;
                labels[i] = sampleweights[idx];
            }
            
        }
        cout << "Epoch dataset loaded..." << endl;
        // iterate through each feature
        cout << "Feature training sequence initiated" << endl;
        vector<pair<int, int> > errorpairs;
        for(int i = 0; i < numFeatures; i++){
            // print progress
            cout << "[";
            int pos = 100 * i/numFeatures;
            for(int j = 0; j < 100; j++){
                if(j < pos) cout << "=";
                else if (j == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << 100 * (i + 1)/numFeatures << "% " << (i + 1) << "/" << numFeatures << " \r";
            cout.flush();
            
            Feature f = features.at(i);
            int tplus = 0;
            int tminus = 0;
            vector<pair<int, int> >vals;
            // iterate through each image to determine thresholds
            for(int j = 0; j < trainSize; j++){
                // open the image
                Mat intimg = dataset.at(j);
                vals.push_back(pair<int, int>(f.evaluate(intimg), labels[j]));
                if(labels[j] >= 0){
                    tplus += labels[j];
                }else{
                    tminus -= labels[j];
                }
            }
            sort(vals.begin(), vals.end());
            
            // analyze parity and threshold 
            int  minerr, minthresh = 0;
            int gterr = tplus, lterr = tminus;
            bool mingt;
            if(tplus < tminus){
                mingt = true;
                minerr = tplus;
            }else{
                mingt = false;
                minerr = tminus;
            }
            // iterate through all examples 
            for(int j = 1; j < trainSize; j++){
                int weight = vals.at(j).second;
                gterr -= weight;
                lterr += weight;
                if(lterr < minerr){
                    minerr = lterr;
                    mingt = false;
                    minthresh = vals.at(j).first;
                    //cout << "Update on sample " << j << "! minerr: " << minerr << " mingt: false minthresh: " << minthresh << endl;
                }else if(gterr < minerr){
                    minerr = gterr;
                    mingt = true;
                    minthresh = vals.at(j).first;
                    //cout << "Update on sample " << j << "! minerr: " << minerr << " mingt: true minthresh: " << minthresh << endl;
                }
            }
            thresholds[i] = minthresh;
            gt[i] = mingt;

            // change feature weights
            double errorratio = (double) minerr / trainSize;
            featureweights[i] = log((1 - errorratio)/errorratio);
            // change example weights
            for(int j = 0; j < trainSize; j++){
                int y = (vals.at(j).second >= 0) ? 1 : -1;
                int h;
                if(mingt){
                    h = (vals.at(j).first > minthresh) ? 1 : -1;
                }else{
                    h = (vals.at(j).first <= minthresh) ? 1 : -1;
                }
                int w = abs(labels[j]);
                double neww = w * exp(-1 * y * featureweights[i] * h);
                sampleweights[keys[j]] = neww; 
                total += (neww-w);
            }
            errorpairs.push_back(pair<int, int>(minerr, i));
        }
        double totalweight = 0;
        // select top features
        sort(errorpairs.begin(), errorpairs.end());
        for(int i = 0; i < haarNum; i++){
            bestidxs[i] = errorpairs.at(i).second;
            bestweights[i] = featureweights[bestidxs[i]];
            totalweight += bestweights[i];
        }
        for(int i = 0; i < haarNum; i++){
            bestweights[i] /= totalweight;
        }
        cumweights[0] = sampleweights[0];
        for(int j = 1; j < trainSize; j++){
            cumweights[keys[j]] = cumweights[keys[j] - 1] + sampleweights[keys[j]];
        } 
        cout << endl;
        cout << "Testing model..." << endl;
        // run tests
        int correct = 0;
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for(int i = 0; i < 500; i++){
            // load dataset
            Mat pic;
            int sel = rand() % 2;
            if(sel){
                string fname = testfaces.at(rand() % testfaces.size()); 
                pic = imread(fname, IMREAD_GRAYSCALE);
            }else{
                string fname = testnots.at(rand() % testnots.size());
                pic = imread(fname, IMREAD_GRAYSCALE);
            }
            Mat resized, intimg;
            resize(pic, resized, Size(windowSize, windowSize));
            integral(resized, intimg);
            double ret = 0;
            for(int j = 0; j < haarNum; j++){
                if(gt[j]){
                    ret += bestweights[j] * (features.at(bestidxs[j]).evaluate(intimg) < thresholds[bestidxs[j]]); 
                }else{
                    ret += bestweights[j] * (features.at(bestidxs[j]).evaluate(intimg) >= thresholds[bestidxs[j]]);
                }
            }
            int res = round(ret);
            if(res == sel){
                correct++;
                if(res == 1) { tp++; }
                if(res == 0) { tn++; }
            }else{
                if(res == 1) { fp++; }
                if(res == 0) { fn++; }
            }
            cout << "Current accuracy: " << 100 * (double) correct/(i + 1) << "% " << correct << "/" << (i+1) << " Confidence level: " << ret << " \r";
        }
        cout << endl;
        cout << "Confusion matrix: " << endl;
        cout << "tp: " << tp << " fp: " << fp << endl;
        cout << "fn: " << fn << " tn: " << tn << endl;
    }

    
    // write data to file
    ofstream data;
    data.open("structure.txt");
    data << "winSize: " << windowSize << endl;
    data << "bestidxs: ";
    for(int i = 0; i < haarNum; i++){
        data << bestidxs[i] << " ";
    }
    data << "threshs: ";
    for(int i = 0; i < haarNum; i++){
        data << thresholds[bestidxs[i]] << " ";
    } 
    data << endl;
    data << "gt: ";
    for(int i = 0; i < haarNum; i++){
        data << gt[bestidxs[i]] << " ";
    }
    data << endl;
    data << "fweights: ";
    double totalfweights = 0, maxweights = 0;
    for(int i = 0; i < haarNum; i++){
        data << featureweights[bestidxs[i]] << " ";
        totalfweights += featureweights[bestidxs[i]];
        maxweights = max(featureweights[bestidxs[i]], maxweights);
    }
    data << endl;
    return 0;
}

