#include <iostream>
#include "stdio.h"
#include <chrono>
#include <cmath>
#include <list>
#include <queue>
#include <map>
#include "Python.h"
using namespace std;

#define earth_radius 6371000 // 地球半径，单位：米

//class PythonCaller;

struct Wind{
    float u_wind;   //向东/经度
    float v_wind;   //向北/纬度
    float w_wind;   //向上/海拔
    float sh;
};

struct Point{
    float latitude;
    float longitude;
    float level;
};

struct CurPointData{
    int time;
    Point point;
    Wind wind;
    CurPointData *next;
    CurPointData():next(nullptr){
        time = 0;
        point = {};
        wind = {};
    }
};

class PythonCaller {
public:
    static PythonCaller* GetInstance(){
        static PythonCaller py_caller;
        return &py_caller;
    }

    Wind Call(std::string ncpath,double time, double lev,double lat, double lon) {
        Wind wind_mes;

        pFunc = PyDict_GetItemString(pDict, "Merra2Parser");
        if (PyCallable_Check(pFunc))
        {
            PyObject *pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(ncpath.c_str()));

            // Create an instance of the class
            PyObject* pInstance = PyObject_CallObject(pFunc, pArgs);
            if (pInstance == NULL) {
                PyErr_Print();
                throw std::runtime_error("Failed to create Python object.");
            }

            // Prepare the argument list for the call
            pFunc = PyObject_GetAttrString(pInstance, "parse_uvw2");
            if (pFunc == NULL || !PyCallable_Check(pFunc)) {
                PyErr_Print();
                throw std::runtime_error("Cannot find method \"parse_uvw\"");
            }

            PyObject *pArgs2 = PyTuple_New(4);
            PyTuple_SetItem(pArgs2, 0, PyFloat_FromDouble(time));
            PyTuple_SetItem(pArgs2, 1, PyFloat_FromDouble(lev));
            PyTuple_SetItem(pArgs2, 2, PyFloat_FromDouble(lat));
            PyTuple_SetItem(pArgs2, 3, PyFloat_FromDouble(lon));
            //PyTuple_SetItem(pArgs2, 4, PyUnicode_FromString("U"));
            try
            {
                PyObject* pValue = PyObject_CallObject(pFunc, pArgs2);
                float rValue1 = 0.0,rValue2=0.0,rValue3=0.0;
                if (pValue != NULL)
                {
                    PyObject* pValue1 = PyTuple_GetItem(pValue, 0);
                    PyObject* pValue2 = PyTuple_GetItem(pValue, 1);
                    PyObject* pValue3 = PyTuple_GetItem(pValue, 2);
                    rValue1 = static_cast<float>(PyFloat_AsDouble(pValue1));
                    std::cout << "Return of call u_wind:" << rValue1 <<std::endl;
                    wind_mes.u_wind  = rValue1;
                    rValue2 = static_cast<float>(PyFloat_AsDouble(pValue2));
                    std::cout << "Return of call v_wind:" << rValue2 <<std::endl;
                    wind_mes.v_wind  = rValue2;
                    rValue3 = static_cast<float>(PyFloat_AsDouble(pValue3));
                    std::cout << "Return of call w_wind:" << rValue3 <<std::endl;
                    wind_mes.w_wind  = rValue3;
                    Py_DECREF(pValue1);
                    Py_DECREF(pValue2);
                    Py_DECREF(pValue3);

                    pFunc = PyObject_GetAttrString(pInstance, "__del__");
                    PyObject* pValueDel = PyObject_CallObject(pFunc, NULL);
                    if (pValueDel == NULL) {
                        PyErr_Print();  // 打印错误信息
                    }
                    Py_DECREF(pValueDel);  // 减少引用计数
                    
                }
                else
                {
                    PyErr_Print();
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                exit(-1);
            }
            
            
            // Py_DECREF(pValue);
            // Py_DECREF(pArgs);
            // Py_DECREF(pArgs2);
            // Py_DECREF(pInstance);
            
            return wind_mes;
        }
        // Py_DECREF(pFunc);

    }

private:
    PythonCaller() {
        Py_Initialize();
        PyRun_SimpleString("import sys; sys.path.append('/home/hikaka/mete-clustering/cpp/test/bt_compute')");
        pName = PyUnicode_FromString("merra2_parser");
        pModule = PyImport_Import(pName);
        if (pModule == NULL) {
            PyErr_Print();
            throw std::runtime_error("Failed to load Python module.");
        }
        pDict = PyModule_GetDict(pModule);
    }

    ~PythonCaller() {
        Py_DECREF(pModule);
        Py_DECREF(pName);
        Py_DECREF(pFunc);
        Py_Finalize();
    }

private:
    PyObject *pName, *pModule, *pDict,*pFunc;
};

class BackTrajectory{
public:
    static BackTrajectory* GetInstance(){
        static BackTrajectory bt_obj;
        return &bt_obj;
    }

    void SingleComputeBackTraj(Point cur_loacation,float cur_time){
        Point next_location;
        for(int i=0;i<24;i++){
            single_traj_[i] = cur_loacation;
            printf("%.5f %.5f %.5f\n", cur_loacation.latitude, cur_loacation.longitude, cur_loacation.level);
            next_location = this->find_uvw_and_compute_location(cur_time,cur_loacation,next_location);
             cur_loacation = next_location;
            cur_time -= 180.0f;
        }
    }
private:
    BackTrajectory(){
        py_caller = PythonCaller::GetInstance();
    };
    BackTrajectory(CurPointData* start_point_data){
        //single_traj_single_traj_.push(start_point_data);
    };
    ~BackTrajectory() = default;

private:
    // CurPointData get_wind_data(const Point& point, int hour) {
    //     // 这里是从ERA5数据集中获取风速数据的实现
    //     return {0.0, 0.0, 0.0, 0.0};
    // }

    Point compute_new_location(const Point& cur_point,const Wind& wind_data,Point& next_point,float delta_t){

        const float degrees_to_radians = M_PI / 180.0f;
        
        // 将垂直风速转换为压力变化 (假设：1 Pa/s 垂直风速对应 1 hPa 的压力变化)
        float pressure_change = wind_data.w_wind * delta_t;
        next_point.level =  cur_point.level - pressure_change; // 更新压力水平

        // 后向，取反
        float dx = -wind_data.u_wind * delta_t * cos(cur_point.latitude * degrees_to_radians); // 将纬度转换为弧度
        float dy = -wind_data.v_wind * delta_t;

        next_point.longitude = cur_point.longitude + dx / (earth_radius * cos(cur_point.latitude * degrees_to_radians))/degrees_to_radians;
        next_point.latitude = cur_point.latitude - dy / earth_radius * degrees_to_radians;

        // 调整经度范围
        if (next_point.longitude > 180) next_point.longitude -= 360;
        if (next_point.longitude < -180) next_point.longitude += 360;

        // 调整纬度范围
        if (next_point.latitude > 90) next_point.latitude = 90;
        if (next_point.latitude < -90) next_point.latitude = -90;

        return next_point;
    }

    Point find_uvw_and_compute_location(double time,const Point& point,Point& next_point){
        Wind wind_data;
        try
        {
            wind_data = py_caller->Call("/home/hikaka/mete-clustering/cpp/merra2/MERRA2_400.inst3_3d_asm_Np.20230101.nc4",time,point.level,point.latitude,point.longitude);
            return this->compute_new_location(point,wind_data,next_point,3600);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << "error time : " << time<<std::endl;
            std::cerr <<"error point : "<<point.level << " " <<point.latitude << " "<<point.longitude << std::endl;
            exit(-1);
        }
        return Point{};
    }

private:
    PythonCaller* py_caller;
    CurPointData cur_point_data_;
    CurPointData next_point_data_;
    //std::queue<CurPointData*> single_traj_;
    std::map<int, Point> single_traj_;
};

int main(int argc,char* argv[]) {
#if 0
    PythonCaller caller;
    auto start_time = std::chrono::high_resolution_clock::now();
    for(int i=0;i<1000;i++){
        caller.Call(1069152.0, 23.75, -109.25, "v");
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Execution time: " << duration << " milliseconds" << std::endl;
    //printf("Return of call : %f\n", result);
#endif

    // PythonCaller* caller = PythonCaller::GetInstance();
#if 1
    CurPointData start_point_data={};
    auto obj = BackTrajectory::GetInstance();
    float start_time = 1260.0;
    Point cur_loacation = {23.75,-28.1,500.0};
    obj->SingleComputeBackTraj(cur_loacation,start_time);
    
    // for(int i=0;i<10;i++){
    // obj->compute_new_location(point,wind_data,delta_t);
    // cout << point.latitude <<" "<< point.` <<" "<< point.level << endl;
    // }
#endif
    return 0;
}










// int main(){
    // CurPointData start_point_data={};
    // auto obj = BackTrajectory::GetInstance();
    // Point point = {-29,31,500};
    // Wind wind_data = {100.0, 100.0, 0.0, 0.0};
    // float delta_t = 1.0;
    // for(int i=0;i<10;i++){
    // obj->compute_new_location(point,wind_data,delta_t);
    // cout << point.latitude <<" "<< point.longitude <<" "<< point.level << endl;
    // }

//     return 0;
// }

/*
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <stdexcept>

// 假设我们有一个函数来获取风速数据
struct WindData {
    float u; // 东向风速分量 (m/s)
    float v; // 北向风速分量 (m/s)
    float w; // 垂直风速分量 (Pa/s)
    float sh; // 比湿 (kg/kg)
};

WindData get_wind_data(float latitude, float longitude, float level, int hour) {
    // 这里应该是从ERA5数据集中获取风速数据的实现
    // 为了示例，我们只是返回一些假数据
    return {0.0, 0.0, 0.0, 0.0};
}

// 计算新位置的函数
void compute_new_location(float& latitude, float& longitude, float& level,
                           const WindData& wind_data, float delta_t) {
    const float earth_radius = 6378000; // 地球半径，单位：米
    const float degrees_to_radians = M_PI / 180.0;
    
    // 将垂直风速转换为压力变化 (假设：1 Pa/s 垂直风速对应 1 hPa 的压力变化)
    float pressure_change = wind_data.w * delta_t;
    level -= pressure_change; // 更新压力水平

    // 计算新位置
    float dx = wind_data.u * delta_t * 1000 * cos(latitude * degrees_to_radians); // 将纬度转换为弧度
    float dy = wind_data.v * delta_t * 1000;

    latitude -= dy / earth_radius * degrees_to_radians;
    longitude -= dx / (earth_radius * cos(latitude * degrees_to_radians)) * degrees_to_radians;

    // 调整经度范围
    if (longitude > 180) longitude -= 360;
    else if (longitude < -180) longitude += 360;
}

// 后向轨迹计算的主函数
void compute_backward_trajectory(float start_latitude, float start_longitude,
                                 float start_level, int start_hour,
                                 float delta_t, int days) {
    std::vector<std::tuple<float, float, float>> trajectory;

    float current_latitude = start_latitude;
    float current_longitude = start_longitude;
    float current_level = start_level;

    for (int day = 0; day < days; ++day) {
        for (int hour = 24 - 1; hour >= 0; --hour) { // 从24小时前开始逆向计算
            WindData wind_data = get_wind_data(current_latitude, current_longitude, current_level, start_hour - hour);
            compute_new_location(current_latitude, current_longitude, current_level, wind_data, delta_t);
            trajectory.push_back(std::make_tuple(current_longitude, current_latitude, current_level));
        }
    }

    // 打印轨迹
    for (const auto& [longitude, latitude, level] : trajectory) {
        std::cout << "Longitude: " << longitude << ", Latitude: " << latitude << ", Level: " << level << std::endl;
    }
}

int main() {
    try {
        // 初始条件
        float start_latitude = 6.25184; // 初始纬度
        float start_longitude = -75.56359; // 初始经度
        float start_level = 825; // 初始压力水平 (hPa)
        int start_hour = 0; // 起始时间 (小时)

        // 计算参数
        float delta_t = 1.0; // 时间步长 (小时)
        int days = 10; // 计算天数

        compute_backward_trajectory(start_latitude, start_longitude, start_level, start_hour, delta_t, days);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0; 
}
*/