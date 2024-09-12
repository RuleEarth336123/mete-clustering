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
void FeatureComputer::CosCompute(const vector<vector<Point>> &trajectorys, vector<vector<float>> &output_matrix)
{
    size_t N = trajectorys.size();
    output_matrix.resize(N, std::vector<float>(N, 0.0));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                output_matrix[i][j] = cos_distance(trajectorys[i], trajectorys[j]);
            } else {
                output_matrix[i][j] = 0.0f; // 自身与自身的距离为0
            }
        }
    } 
}
void FeatureComputer::DotCompute(const vector<vector<float>> &input_marix1, const vector<vector<float>> &input_marix2, vector<vector<float>> &output_marix)
{
    int M = input_marix1.size();
    int N = input_marix1[0].size();
    output_marix.resize(M, std::vector<float>(N, 0.0));
    if(M != input_marix2.size() || N != input_marix2[0].size()){
        std::cerr << "the two matrix's shape is different." <<std::endl;
    }
    for(int i = 0;i<M;i++){
        for(int j=0;j<N;j++){
            output_marix[i][j] = input_marix1[i][j] * input_marix2[i][j];
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
        for (int j = 1; j <= m; ++j) {
            float eudis = euclidean_distance(trajectory1[i-1], trajectory2[j-1]);
            //std::cout << "euclidean_distance : "<<eudis <<std::endl;
            float costDiag = cost[i-1][j-1] + eudis;
            float costLeft = cost[i][j-1] + eudis;
            float costUp = cost[i-1][j] + eudis;
            cost[i][j] = std::min({costDiag, costLeft, costUp});
        }
    }

    return cost[n][m];
}

float FeatureComputer::cos_distance(const vector<Point> &trajectory1, const vector<Point> &trajectory2)
{

    if (trajectory1.empty() || trajectory2.empty()) {
        return 0.0f; // 如果任一轨迹为空，则相似度为0
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < trajectory1.size() && i < trajectory2.size(); ++i) {
        // 计算点积
        dot_product += trajectory1[i].latitude * trajectory2[i].latitude + trajectory1[i].longitude * trajectory2[i].longitude;

        // 计算第一个轨迹的向量范数
        norm1 += std::pow(trajectory1[i].latitude, 2) + std::pow(trajectory1[i].longitude, 2);

        // 计算第二个轨迹的向量范数
        norm2 += std::pow(trajectory2[i].latitude, 2) + std::pow(trajectory2[i].longitude, 2);
    }

    // 计算范数
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    // 计算余弦相似度
    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f; // 如果任一轨迹的范数为0，则相似度为0
    }

    return dot_product / (norm1 * norm2);
}
