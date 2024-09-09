#include "dtw.h"
#include "bt_compute.h"
// #include "era5_reader.h"
#include <emmintrin.h>
#include <memory>
#include <vector>
#include <tuple>
#include <cstdlib>
#include "http_handler.h"

using namespace std;

int initServer(int port = 12123)
{
    httplib::Server svr;

    svr.Post("/compute/1h", handleSingleCompute);
    svr.Post("/compute/6h", handleComputePer6h);

    if (svr.listen("0.0.0.0", port)) {
        std::cout << "Server is running at http://localhost:12123" << std::endl;
    } else {
        std::cerr << "Failed to start server." << std::endl;
        return 1;
    }
}


int main1(int argc,char* argv[]){
#if 0   

    vector<vector<float>> tmp{{1,2,3}};
    vector<float>* aa = tmp.data();
    float* bb = tmp[0].data();
    float* cc = &tmp[0][0];
    
    

    // auto printProgress = [](int i) {
    //     std::cout << "处理到第 " << i << " 个轨迹" << "\n";
    // };
    // clock_t start = clock();
    // std::string folderPath = "/home/hikaka/mete-clustering/cpp/binary"; // 替换为您的文件夹路径
    // DtwComputer *dtw = new DtwComputer();
    // std::vector<std::vector<TrajectoryPoint>> TrajMap = dtw->readAllTrajectoryBinaries(folderPath);
    // std::vector<std::vector<float>> dtwgroup;
    // dtw->seqComputerdtw(TrajMap,dtwgroup,printProgress);
    // clock_t end = clock();
    // std::cout<<"time = "<<float(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    // delete dtw;
    // return 0;
#endif
#if 0
    std::unique_ptr<NetCDFReader>  reader = std::make_unique<NetCDFReader>(
        "/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc",24,5,161,301
    );

    std::vector<std::vector<std::vector<std::vector<float>>>> value_4d(
        24,
        std::vector<std::vector<std::vector<float>>>(
            5,
            std::vector<std::vector<float>>(
                161,
                std::vector<float>(301)
            )
        )
    );
    std::vector<float> value_1d;
    //reader->Nc1dReader("time",value_1d);
    reader->Nc4dReader("w",value_4d);

#endif
#if 0
    std::unique_ptr<NetCDFReader>  reader = std::make_unique<NetCDFReader>(
        "/mnt/d/linux/MERRA2_400.tavgM_2d_aer_Nx.201501.nc4",1,361,576,0
    );

    std::vector<std::vector<std::vector<float>>> value_3d(1,std::vector<std::vector<float>>(361,std::vector<float>(576)));

    reader->Nc3dReader("SO2CMASS",value_3d);

#endif
#if 0
    std::unique_ptr<NetCDFReader>  reader = std::make_unique<NetCDFReader>(
        "/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc",24,5,161,301
    );
    reader->InitIndex();
    auto [time_index,levIndex,latIndex,lonIndex,flag] = reader->getIndex(
        0,500.0,39.5,-28.1
    );
    if(flag == false){
        std::cerr << "get indexs error : [ " <<  time_index << " " <<
        levIndex << " " << latIndex << " " << lonIndex << " " << std::endl;
    }
    auto indexs = std::make_tuple(time_index,levIndex,latIndex,lonIndex);//0 2 82 188
    float value = reader->getVar(indexs,"u");
    std::cout << "the indexs is :" << time_index << " " << levIndex << " "<< latIndex << " "<< lonIndex << " and the value is : "<< value << std::endl;;
#endif

#if 1




    std::unique_ptr bt = std::make_unique<BackTraj>();
    const char* file = "/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc";
    bt->InitNc(1,file);
    Point cur_loc(39.5,-28.1,500);
    std::vector<Point> trajs;
    bt->ComputeSingle(file,23,cur_loc,3600.f,trajs);


#endif
    return 0;
}
//./main --file "/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc" --time 23 --lat 39.5 --lon 28.1 --lev 500.0
int main2(int argc,char* argv[]){
    std::string file_path;
    int hour = 0;
    float latitude = 0.0f;
    float longitude = 0.0f;
    float level = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            file_path = argv[++i];
        } else if (arg == "--hour" && i + 1 < argc) {
            hour = std::atoi(argv[++i]);
        } else if (arg == "--lat" && i + 1 < argc) {
            latitude = std::atof(argv[++i]);
        } else if (arg == "--lon" && i + 1 < argc) {
            longitude = std::atof(argv[++i]);
        } else if (arg == "--lev" && i + 1 < argc) {
            level = std::atof(argv[++i]);
        } else {
            std::cerr << "Unknown option or missing argument for " << arg << std::endl;
            return 1;
        }
    }

    if (file_path.empty() || time == 0 || latitude == 0.0f || longitude == 0.0f || level == 0.0f) {
        std::cerr << "Usage: " << argv[0]
                  << " --file <file_path> --hour <hour> --lat <latitude> --lon <longitude> --lev <level>" << std::endl;
        return 1;
    }

    std::unique_ptr<BackTraj> bt = std::make_unique<BackTraj>();

    bt->InitNc(1, file_path.c_str());

    // 创建初始位置
    Point cur_loc(latitude, longitude, level);

    // 计算轨迹
    std::vector<Point> trajs;
    bt->ComputeSingle(file_path.c_str(), hour, cur_loc, 3600.0f,trajs);

    // // 打印轨迹（示例）
    // for (const auto& point : trajs) {
    //     std::cout << "Latitude: " << point.latitude << ", Longitude: " << point.longitude << ", Level: " << point.level << std::endl;
    // }
}

int main(int argc,char* argv[]){

    int port;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc){
            port = std::atoi(argv[++i]);
        }
            
    }

    initServer(port);
    
    return 0;
}