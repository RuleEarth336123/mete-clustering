#ifndef FEATURE_H
#define FEATURE_H

#include "bt_compute.h"
#include <cmath>

using std::vector;

class FeatureComputer{
public:
    explicit FeatureComputer() = default;
    ~FeatureComputer() = default;

    void DtwCompute(const vector<vector<Point>>& trajectorys,vector<vector<float>>& output_marix);
    void CosCompute(const vector<vector<Point>>& trajectorys,vector<vector<float>>& output_marix);
    void DotCompute(const vector<vector<float>>& input_marix1,const vector<vector<float>>& input_marix2,vector<vector<float>>& output_marix);
    void NormalizeFeatures(const vector<vector<float>>& input_matrix,vector<vector<float>>& feature_matrix);

    float dtw_distance(const vector<Point>& trajectory1, const vector<Point>& trajectory2);
    float cos_distance(const vector<Point>& trajectory1, const vector<Point>& trajectory2);



private:
    inline float euclidean_distance(const Point& p1, const Point& p2) {
        return sqrt((p1.latitude - p2.latitude) * (p1.latitude - p2.latitude) +
                    (p1.longitude - p2.longitude) * (p1.longitude - p2.longitude));
    }

};


#endif
