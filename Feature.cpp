#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "Feature.hpp"

int Feature::rectSum(cv::Mat img, int x, int y, int dx, int dy){
    cv::Mat mat = img;
    return (mat.at<int>(x, y) + mat.at<int>(x + dx, y + dy) - mat.at<int>(x, y + dy) - mat.at<int>(x + dx, y));
}

int Feature::evaluate(cv::Mat img){
    int ret;
    switch(type){
        case 0:
            ret = rectSum(img, x, y, dx/2, dy) 
                - rectSum(img, x + dx/2, y, dx/2, dy);
            break;
        case 1:
            ret = rectSum(img, x, y, dx, dy/2) 
                - rectSum(img, x, y + dy/2, dx, dy/2);
            break;
        case 2:
            ret = rectSum(img, x, y, dx/3, dy) 
                - rectSum(img, x + dx/3, y, dx/3, dy)
                + rectSum(img, x + dx/3*2, y, dx/3, dy);
            break;
        case 3:
            ret = rectSum(img, x, y, dx, dy/3)
                - rectSum(img, x, y + dy/3, dx, dy/3)
                + rectSum(img, x, y + dy/3*2, dx, dy/3);
            break;
        case 4:
            ret = rectSum(img, x, y, dx/2, dy/2)
                - rectSum(img, x, y + dy/2, dx/2, dy/2)
                - rectSum(img, x + dx/2, y, dx/2, dy/2)
                + rectSum(img, x + dx/2, y + dy/2, dx/2, dy/2);
            break;
        default:
            ret = rectSum(img, x, y, dx/2, dy) 
                - rectSum(img, x + dx/2, y, dx/2, dy);
    }
    return ret;
}

int Feature::getX(){
    return x;
}

int Feature::getY(){
    return y;
}

int Feature::getDx(){
    return dx;
}

int Feature::getDy(){
    return dy;
}

int Feature::getType(){
    return type;
}

Feature::Feature(){
    this->x = 0;
    this->y = 0;
    this->dx = 0;
    this->dy = 0;
    this->type = 0;
}

Feature::Feature(int x, int y, int dx, int dy, int type){
    assert(x >= 0 && y >= 0); 
    this->x = x;
    this->y = y;
    this->dx = dx;
    this->dy = dy;
    this->type = type;
}
