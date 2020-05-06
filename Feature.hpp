#include <opencv2/opencv.hpp>

using namespace std;

class Feature{
    private:
        int x, y, dx, dy;
        int type;
        int rectSum(cv::Mat img, int x, int y, int dx, int dy);
    public:
        int getX();
        int getY();
        int getDx();
        int getDy();
        int getType();
        int getSum();
        int evaluate(cv::Mat img);
        Feature();
        Feature(int x, int y, int dx, int dy, int type);
};
