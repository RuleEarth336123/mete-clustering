#pragma

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

class DtwComputer{
public:
    DtwComputer(){}
    DtwComputer(std::string fname) : filename(fname){
        inputbinlist = "/home/hikaka/mete-clustering/cpp/binary";
        outputbinlist = "/home/hikaka/mete-clustering/cpp/out";
    }
    ~DtwComputer(){

    }
    std::vector<TrajectoryPoint> readTrajectoryBinary(const std::string& filename);
    std::vector<std::vector<TrajectoryPoint>> readAllTrajectoryBinaries(const std::string& folderPath);

    void seqComputerdtw(const std::vector<std::vector<TrajectoryPoint>> &TrajMap,\
        std::vector<std::vector<double>> &dtwgroup,std::function<void(int i)> cb);
    void seqComputerdtwVec(const std::vector<std::vector<TrajectoryPoint>> &TrajMap,\
        std::vector<std::vector<double>> &dtwgroup,std::function<void(int i)> cb);

    // inline void printProgress(int i) {
    //     std::cout << "处理到第 " << i << " 个轨迹" << std::endl;
    // };
    void writeDTWResultsToFile(const std::vector<std::vector<double>>& dtwgroup, const std::string& outputPath);
    double euclideanDistance(const TrajectoryPoint& p1, const TrajectoryPoint& p2);
    double dtwDistance(const std::vector<TrajectoryPoint>& traj1, const std::vector<TrajectoryPoint>& traj2);

private:
    std::string filename;
    std::string inputbinlist,outputbinlist;
    std::vector<std::vector<TrajectoryPoint>> TrajDistMap;

};