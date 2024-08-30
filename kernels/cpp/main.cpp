#include "dtw.h"
#include <emmintrin.h>

int main(int argc,char* argv[]){
    auto printProgress = [](int i) {
        std::cout << "处理到第 " << i << " 个轨迹" << "\n";
    };
    clock_t start = clock();
    std::string folderPath = "/home/hikaka/mete-clustering/cpp/binary"; // 替换为您的文件夹路径
    DtwComputer *dtw = new DtwComputer();
    std::vector<std::vector<TrajectoryPoint>> TrajMap = dtw->readAllTrajectoryBinaries(folderPath);
    std::vector<std::vector<double>> dtwgroup;
    dtw->seqComputerdtw(TrajMap,dtwgroup,printProgress);
    clock_t end = clock();
    std::cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    delete dtw;
    return 0;
}



