input:
u_wind :11.0839
v_wind :-2.37963
omega :0.022431
sh :0.00133718
input_point = {23.75,-109.25,500};

output = {2061790373369840133224766294196224.00 -0.00 499.98}

    Point compute_new_location(const Point& cur_point,const Wind& wind_data,Point& next_point,float delta_t){

        
        const float degrees_to_radians = M_PI / 180.0;
        
        // 将垂直风速转换为压力变化 (假设：1 Pa/s 垂直风速对应 1 hPa 的压力变化)
        float pressure_change = wind_data.w_wind * delta_t;
        next_point.level =  cur_point.level - pressure_change; // 更新压力水平

        // 计算新位置
        float dx = wind_data.u_wind * delta_t * 1000 * cos(cur_point.latitude * degrees_to_radians); // 将纬度转换为弧度
        float dy = wind_data.v_wind * delta_t * 1000;

        next_point.latitude = cur_point.latitude - dy / earth_radius * degrees_to_radians;
        next_point.longitude = cur_point.longitude - dx / (earth_radius * cos(cur_point.latitude * degrees_to_radians)) * degrees_to_radians;

        // 调整经度范围
        if (next_point.longitude > 180) next_point.longitude -= 360;
        else if (next_point.longitude < -180) next_point.longitude += 360;

        return next_point;
    }

        for(int i=0;i<24;i++){
            single_traj_[i] = cur_loacation;
            printf("%.5f %.5f %.5f\n", cur_loacation.latitude, cur_loacation.longitude, cur_loacation.level);
            next_location = this->find_uvw_and_compute_location(time,cur_loacation,next_location);
            cur_loacation = next_location;
        }
1000.00
975.000
950.000
925.000
900.000
875.000
850.000
825.000
800.000
775.000
750.000
725.000
700.000
650.000
600.000
550.000
500.000
450.000
400.000
350.000
300.000
250.000
200.000
150.000
100.000
70.0000
50.0000
40.0000
30.0000
20.0000
10.0000
7.00000
5.00000
4.00000
3.00000
2.00000
1.00000
0.700000
0.500000
0.400000
0.300000
0.100000