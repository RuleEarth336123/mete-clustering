#include "features.h"
#include <limits>
#include <algorithm>

void FeatureComputer::DtwCompute(const vector<vector<Point>>& trajectorys, vector<vector<float>> &output_matrix)
{
    size_t N = trajectorys.size();
    output_matrix.resize(N, std::vector<float>(N, 0.0));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                output_matrix[i][j] = dtw_distance(trajectorys[i], trajectorys[j]);
            } else {
                output_matrix[i][j] = 0.0f; // 自身与自身的距离为0
            }
        }
    } 
}
void FeatureComputer::NormalizeFeatures(const vector<vector<float>> &input_matrix, vector<vector<float>> &feature_matrix)
{
    int numRows = input_matrix.size();
    int numCols = input_matrix[0].size();
    feature_matrix.resize(numRows, std::vector<float>(numCols));

    for (int i = 0; i < numRows; ++i) {
        float minElem = *std::min_element(input_matrix[i].begin(), input_matrix[i].end());
        float maxElem = *std::max_element(input_matrix[i].begin(), input_matrix[i].end());
        float range = maxElem - minElem;

        for (int j = 0; j < numCols; ++j) {
            if (range == 0) {
                feature_matrix[i][j] = 0; // 避免除以零
            } else {
                feature_matrix[i][j] = static_cast<float>((input_matrix[i][j] - minElem) / range);
            }
        }
    }
}
float FeatureComputer::dtw_distance(const std::vector<Point> &trajectory1, const std::vector<Point> &trajectory2)
{
    int n = trajectory1.size();
    int m = trajectory2.size();
    std::vector<std::vector<float>> cost(n + 1, std::vector<float>(m + 1, std::numeric_limits<float>::max()));

    cost[0][0] = 0;

    for (int i = 1; i <= n; ++i) {
        cost[i][0] = std::numeric_limits<float>::max();
    }
    for (int j = 1; j <= m; ++j) {
        cost[0][j] = std::numeric_limits<float>::max();
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float costDiag = cost[i-1][j-1] + euclidean_distance(trajectory1[i-1], trajectory2[j-1]);
            float costLeft = cost[i][j-1];
            float costUp = cost[i-1][j];
            cost[i][j] = std::min({costDiag, costLeft, costUp});
        }
    }

    return cost[n][m];
}

float FeatureComputer::cos_distance(const vector<Point> &trajectory1, const vector<Point> &trajectory2)
{
    return 0.0f;
}
