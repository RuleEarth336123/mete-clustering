#include <iostream>
#include <ostream>
#include <fstream>
#include <string>
#include <vector>
#include <dirent.h>
#include "stdlib.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <functional>
#include "time.h"
#include <execution>

struct TrajectoryPoint {
    double latitude;
    double longitude;
};

using TimeCallback = std::function<void(double duration)>;


std::vector<TrajectoryPoint> readTrajectoryBinary(const std::string& filename) {
    std::vector<TrajectoryPoint> trajectory;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "unable to open the file!: " << filename << std::endl;
        return trajectory;
    }
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t numPoints = fileSize / sizeof(TrajectoryPoint);
    trajectory.resize(numPoints);
    file.read(reinterpret_cast<char*>(trajectory.data()), fileSize);

    file.close();
    return trajectory;
}

std::vector<std::vector<TrajectoryPoint>> readAllTrajectoryBinaries(const std::string& folderPath) {
    DIR* dir = opendir(folderPath.c_str());
    if (dir == nullptr) {
        std::cerr << "无法打开目录: " << folderPath << std::endl;
        exit(-1);
    }
    std::vector<std::string> filenames;
    struct dirent* entry;
    int step=0;
    while ((entry = readdir(dir)) != nullptr) {
        if(step < 2){
            step++;
            continue;
        }
        filenames.push_back(entry->d_name);
    }
    
    closedir(dir);
    std::sort(filenames.begin(), filenames.end());

    static std::vector<std::vector<TrajectoryPoint>> TrajHash;

    for(auto const& it : filenames){
        std::string filePath = folderPath + "/" + it;
        std::vector<TrajectoryPoint> trajectory = readTrajectoryBinary(filePath);
        TrajHash.push_back(trajectory);
    }
    return TrajHash;
}

double euclideanDistance(const TrajectoryPoint& p1, const TrajectoryPoint& p2) {
    double latDiff = p1.latitude - p2.latitude;
    double lonDiff = p1.longitude - p2.longitude;
    return std::sqrt(latDiff * latDiff + lonDiff * lonDiff);
}

double dtwDistance(const std::vector<TrajectoryPoint>& traj1, const std::vector<TrajectoryPoint>& traj2) {
    size_t n = traj1.size();
    size_t m = traj2.size();
    std::vector<std::vector<double>> dtw(n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));

    dtw[0][0] = 0;
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            double cost = euclideanDistance(traj1[i - 1], traj2[j - 1]);
            dtw[i][j] = cost + std::min({dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]});
        }
    }
    return dtw[n][m];
}

void seqComputerdtw(const std::vector<std::vector<TrajectoryPoint>> &TrajMap,\
std::vector<std::vector<double>> &dtwgroup,std::function<void(int i)> cb){
    for(int i = 0;i<TrajMap.size();i++){
        std::vector<double> dist;
        for(int j=1;j<TrajMap.size();j++){
            double dtwdistance = dtwDistance(TrajMap[i],TrajMap[j]);
            dist.push_back(dtwdistance);
        }
        dtwgroup.push_back(dist);
        if(cb){
            cb(i);
        }
    }
}

// void seqComputerdtwVec(const std::vector<std::vector<TrajectoryPoint>> &TrajMap,\
// std::vector<std::vector<double>> &dtwgroup,std::function<void(int i)> cb){

//     for(int i = 0;i<TrajMap.size();i++){
//         dtwgroup.push_back(std::vector<double>());
//     }

//     std::for_each(std::execution::par, TrajMap.begin(), TrajMap.end(),
//      [&](const auto &sublist) {
//       int i = &sublist - &TrajMap[0];
//       for(int j = 1; j < TrajMap.size();j++){
//           double dtwdistance = dtwDistance(TrajMap[i], TrajMap[j]);
//           dtwgroup[i].push_back(dtwdistance);
//       }

//       if(cb) {
//          cb(i);
//       }
//     });
// }


int main() {
    auto printProgress = [](int i) {
        std::cout << "处理到第 " << i << " 个轨迹" << std::endl;
    };
    clock_t start = clock();
    std::string folderPath = "cpp/binary"; // 替换为您的文件夹路径
    std::vector<std::vector<TrajectoryPoint>> TrajMap = readAllTrajectoryBinaries(folderPath);
    std::vector<std::vector<double>> dtwgroup;
    seqComputerdtw(TrajMap,dtwgroup,printProgress);
    clock_t end = clock();
    std::cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    return 0;
}


