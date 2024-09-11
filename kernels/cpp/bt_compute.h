#ifndef BT_H
#define BT_H

#include "era5_reader.h"
#include <cmath>
#include <string>
#include <map>
#include <memory>
#include <cstdarg>
#include "json11.hpp"
#include <mutex>

using std::string;
using std::map;
using json11::Json;

struct Point {
    float latitude;
    float longitude;
    float level;
    Point() : latitude(0.f), longitude(0.f), level(0.f) {}
    Point(float lat, float lon, float lvl) : latitude(lat), longitude(lon), level(lvl) {}

    Point copy() const {
        return *this;
    }
    Json to_json() const {
        std::vector<json11::Json> json_vec = {latitude, longitude, level};
        return Json::array(json_vec);
    }
};

struct Wind {
    float u_wind;
    float v_wind;
    float w_wind;
    Wind() : u_wind(0.f), v_wind(0.f), w_wind(0.f) {}
    Wind(float u, float v, float w) : u_wind(u), v_wind(v), w_wind(w) {}
};

class BackTraj{
private:
    string nc_path_;
    map<string, std::unique_ptr<NetCDFReader>> nc_ptrs_map_;
    static std::unique_ptr<BackTraj> instance;
    static std::mutex mutex;

public:

    BackTraj(){}

    BackTraj(const BackTraj&) = delete;
    BackTraj& operator=(const BackTraj&) = delete;
    static BackTraj* getInstance() {
        if (!instance) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!instance) {
                instance.reset(new BackTraj());
            }
        }
        return instance.get();
    }

    ~BackTraj() = default;
    explicit BackTraj(const std::string& path, const std::string& time, const Point& point)
            : nc_path_(path) {}

    //传入n个文件路径
    void InitNc(int num,...);

    int ComputeSingle(string nc,int cur_time, Point cur_location, float delta_t,vector<Point>& single_traj);

    int ComuteSinglePer6h(string yesterday_nc,string today_nc,int cur_time, Point cur_location, float delta_t,vector<vector<Point>>& trajlist);

private:
    int parse_uvw(NetCDFReader* reader,int time, float level, float latitude, float longitude,Wind& wind);
    Point compute_new_location(const Point& cur_point, const Wind& wind_data, float delta_t);
    float deg2rad(float degrees);



};

#endif