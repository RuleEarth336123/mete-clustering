#include "dtw.h"

std::vector<TrajectoryPoint> DtwComputer::readTrajectoryBinary(const std::string &filename)
{
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

std::vector<std::vector<TrajectoryPoint>> DtwComputer::readAllTrajectoryBinaries(const std::string &folderPath)
{
    DIR* dir = opendir(folderPath.c_str());
    if (dir == nullptr) {
        std::cerr << "can't open list: readAllTrajectoryBinaries()" << folderPath << std::endl;
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

// void DtwComputer::writeDTWResultsToFile(const std::vector<std::vector<double>> &dtwgroup, const std::string &outputPath)
// {

// }
void DtwComputer::writeDTWResultsToFile(const std::vector<std::vector<double>> &dtwgroup, const std::string &outputPath)
{
    std::ofstream outfile(outputPath);
    if (!outfile) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return;
    }

    // 假设dtwgroup是一个矩阵，其中每一行代表一个轨迹与其他所有轨迹的DTW距离
    for (const auto &row : dtwgroup) {
        for (double dist : row) {
            outfile << dist << " ";
        }
        outfile << "\n"; // 每行结束后换行
    }

    outfile.close();
}
double DtwComputer::euclideanDistance(const TrajectoryPoint &p1, const TrajectoryPoint &p2)
{
    double latDiff = p1.latitude - p2.latitude;
    double lonDiff = p1.longitude - p2.longitude;
    return std::sqrt(latDiff * latDiff + lonDiff * lonDiff);
}

double DtwComputer::dtwDistance(const std::vector<TrajectoryPoint> &traj1, const std::vector<TrajectoryPoint> &traj2)
{
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

void DtwComputer::seqComputerdtw(const std::vector<std::vector<TrajectoryPoint>> &TrajMap, std::vector<std::vector<double>> &dtwgroup, std::function<void(int i)> cb)
{
    for(int i = 0;i<TrajMap.size();i++){
        std::vector<double> dist;
        outputbinlist = "/home/hikaka/mete-clustering/cpp/out";
        std::string filename = outputbinlist + "/dtw_distance_" + std::to_string(i) + ".bin";
        std::ofstream outfile(filename);
        if(!outfile){
            std::cerr << "can't output file!" <<std::endl;
            return;
        }
        outfile << "start point: " << i << TrajMap[i][0].latitude << " " << TrajMap[i][0].longitude << "\n";
        for(int j=1;j<TrajMap.size();j++){
            double dtwdistance = dtwDistance(TrajMap[i],TrajMap[j]);
            dist.push_back(dtwdistance);
            outfile << dtwdistance << "\n";
        }
        dtwgroup.push_back(dist);
        if(cb){
            cb(i);
        }
        outfile.close();
    }
}

