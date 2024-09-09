#include "bt_compute.h"

void BackTraj::InitNc(int num,...)
{
    va_list args;
    va_start(args, num); 

    for(int i=0; i<num; i++){
        const char* file_path = va_arg(args,const char*);
        string file(file_path);
        
        if(nc_ptrs_map_.find(file) != nc_ptrs_map_.end()){
            continue;
        }
        
        std::unique_ptr<NetCDFReader> reader = std::make_unique<NetCDFReader>(
            file,24,5,161,301);
        reader->InitIndex();
        nc_ptrs_map_[file] = std::move(reader);
    }

    va_end(args);
}

int BackTraj::ComputeSingle(string nc,int cur_time, Point cur_location, float delta_t,vector<Point>& single_traj)
{
    auto it = nc_ptrs_map_.find(nc);
    if(it == nc_ptrs_map_.end()){
        std::cerr << "not init the nc_ptr : "<< nc <<std::endl;
        return -1;
    }

    NetCDFReader* reader = it->second.get();

    int ret;

    //vector<Point> single_traj;
    
    // std::vector<Point> single_traj(24);
    for (int i = 0; i < 23; ++i) {

        Wind wind_data(0.0f,0.0f,0.0f);
        ret = parse_uvw(reader,cur_time, cur_location.level, cur_location.latitude, cur_location.longitude,wind_data);
        if(ret < 0){
            std::cerr << "parse uvw failed." <<std::endl;
            return -1;
        }
        
        Point forest_location = compute_new_location(cur_location, wind_data, delta_t);

        Wind forest_wind_data(0.0f,0.0f,0.0f);
        ret = parse_uvw(reader,cur_time - 1, forest_location.level, forest_location.latitude, forest_location.longitude,forest_wind_data);
        if(ret < 0){
            std::cerr << "parse uvw failed." <<std::endl;
            return -1;
        }       
        
        Wind avg_wind_data = Wind((wind_data.u_wind + forest_wind_data.u_wind) / 2.0f,
                                  (wind_data.v_wind + forest_wind_data.v_wind) / 2.0f,
                                  (wind_data.w_wind + forest_wind_data.w_wind) / 2.0f);

        Point next_location = compute_new_location(cur_location, avg_wind_data, delta_t);
        
        single_traj.push_back(next_location);
        cur_location = next_location;
        cur_time -= 1;
        std::cout << "Step " << i + 1 << ": Latitude = " << next_location.latitude << ", Longitude = " <<
            next_location.longitude << ", Level = " << next_location.level << std::endl;
    }

    single_traj.push_back(cur_location);

    return 0;
}

int BackTraj::ComuteSinglePer6h(string yesterday_nc, string today_nc, int cur_time, Point cur_location, float delta_t, vector<vector<Point>>& trajlist)
{
    auto it = nc_ptrs_map_.find(yesterday_nc);
    if(it == nc_ptrs_map_.end()){
        std::cerr << "not init the yesterday_nc_ptr : "<< yesterday_nc <<std::endl;
        return -1;
    }

    NetCDFReader* yesterday_reader = it->second.get();

    auto it2 = nc_ptrs_map_.find(today_nc);
    if(it2 == nc_ptrs_map_.end()){
        std::cerr << "not init the today_nc_ptr : "<< today_nc <<std::endl;
        return -1;
    }

    NetCDFReader* today_reader = it->second.get();
    
    int ret;
    vector<Point> single_traj;
    Point cur_location_tmp = cur_location;
    int cur_time_tmp = cur_time;
    
    for(int j = 0;j < 4;j++){

        auto reader = today_reader;
        cur_location = cur_location_tmp;//每次从源点发射
        single_traj.clear();
        single_traj.push_back(cur_location);
        for (int i = 0; i < 23; ++i) {
            if((j == 1 && i == 17) || ((j == 2 && i == 11)) || ((j == 3 && i == 5))){
                reader = yesterday_reader;
                cur_time = cur_time_tmp;
            }

            Wind wind_data(0.0f,0.0f,0.0f);
            ret = parse_uvw(reader,cur_time, cur_location.level, cur_location.latitude, cur_location.longitude,wind_data);
            if(ret < 0){
                std::cerr << "parse uvw failed." <<std::endl;
                return -1;
            }
            
            Point forest_location = compute_new_location(cur_location, wind_data, delta_t);

            Wind forest_wind_data(0.0f,0.0f,0.0f);
            ret = parse_uvw(reader,cur_time - 1, forest_location.level, forest_location.latitude, forest_location.longitude,forest_wind_data);
            if(ret < 0){
                std::cerr << "parse uvw failed." <<std::endl;
                return -1;
            }       
            
            Wind avg_wind_data = Wind((wind_data.u_wind + forest_wind_data.u_wind) / 2.0f,
                                    (wind_data.v_wind + forest_wind_data.v_wind) / 2.0f,
                                    (wind_data.w_wind + forest_wind_data.w_wind) / 2.0f);

            Point next_location = compute_new_location(cur_location, avg_wind_data, delta_t);
            
            single_traj.push_back(next_location);
            cur_location = next_location;
            cur_time -= 1;
            std::cout << "Step " << i + 1 << ": Latitude = " << next_location.latitude << ", Longitude = " <<
                next_location.longitude << ", Level = " << next_location.level << std::endl;
        }

        trajlist.push_back(single_traj);

        cur_time = cur_time_tmp - 6;//下一次发射开始时间
    }


    

    return 0;
}

int BackTraj::parse_uvw(NetCDFReader* reader,int time, float level, float latitude, float longitude,Wind& wind)
{
    auto [time_index,levIndex,latIndex,lonIndex,flag] = reader->getIndex(
        time,level,latitude,longitude
    );
    if(flag == false){
        std::cerr << "get indexs error : [ " <<  time_index << " " <<
        levIndex << " " << latIndex << " " << lonIndex << " " << std::endl;
        return -1;
    }
    auto indexs = std::make_tuple(time_index,levIndex,latIndex,lonIndex);//0 2 82 188
    float u = reader->getVar(indexs,"u");
    std::cout << "the indexs is :" << time_index << " " << levIndex << " "<< latIndex << " "<< lonIndex << " and the u is : "<< u << std::endl;
    float v = reader->getVar(indexs,"v");
    std::cout << "the indexs is :" << time_index << " " << levIndex << " "<< latIndex << " "<< lonIndex << " and the v is : "<< v << std::endl;
    float w = reader->getVar(indexs,"w");
    std::cout << "the indexs is :" << time_index << " " << levIndex << " "<< latIndex << " "<< lonIndex << " and the w is : "<< w << std::endl;

    wind.u_wind = u;
    wind.v_wind = v;
    wind.w_wind = w;

    return 0;
}

Point BackTraj::compute_new_location(const Point &cur_point, const Wind &wind_data, float delta_t)
{
    const float earth_radius = 6371.0f;
    float pressure_change = wind_data.w_wind * delta_t / 100.0f;
    Point next_point = cur_point.copy();
    next_point.level = cur_point.level - pressure_change;

    float dx = -wind_data.u_wind * delta_t / 1000.0f;
    float dy = -wind_data.v_wind * delta_t / 1000.0f;

    float delta_lon = (dx / (earth_radius * cosf(deg2rad(cur_point.latitude)))) * (180.0f / static_cast<float>(M_PI));
    float delta_lat = (dy / earth_radius) * (180.0f / static_cast<float>(M_PI));

    next_point.longitude += delta_lon;
    next_point.latitude += delta_lat;

    if (next_point.longitude > 180.0f)
        next_point.longitude -= 360.0f;
    if (next_point.longitude < -180.0f)
        next_point.longitude += 360.0f;

    if (next_point.latitude > 90.0f)
        next_point.latitude = 90.0f;
    if (next_point.latitude < -90.0f)
        next_point.latitude = -90.0f;

    return next_point;
}

float BackTraj::deg2rad(float degrees)
{
    return degrees * static_cast<float>(M_PI) / 180.0f;
}
