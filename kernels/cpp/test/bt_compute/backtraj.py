from merra2_parser import Merra2Parser
import math
import os
import numpy as np
from scipy.interpolate import interp1d
class Point:
    def __init__(self, latitude, longitude, level):
        self.longitude = longitude
        self.latitude = latitude
        self.level = level
    
    def copy(self):
        return Point(self.longitude, self.latitude, self.level)

class Wind:
    def __init__(self, u_wind, v_wind, w_wind):
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.w_wind = w_wind

class BackTraj:
    def __init__(self,nc_path:str,start_time,start_point:Point) -> None:
        self.nc_path = nc_path
        self.start_time = start_time
        self.start_point = start_point
        pass

    """
        24h=>23h
        forest_location = cur_location*cur_wind_speed
        forest_wind_speed = SELECT(forest_location)
        next_location = cur_location*(forest_wind_speed+cur_wind_speed)/2
    """

    def compute_multi_traj(self,folder_path,cur_time,cur_location:Point,delta_t=3600*3) -> list:

        file_list = []
        TrajGroup = []
        # for item in os.listdir(folder_path):
        #     item_path = os.path.join(folder_path, item)
        #     file_list.append(item_path)
        # for nc in file_list:
        #     self.nc_path = nc
        #     Traj:list = self.compute_single_traj(cur_time,cur_location,delta_t)
        #     TrajGroup.append(Traj)
        # np.save('cpp/test/bt_compute/out/TrajGroup.npy', TrajGroup)   
        TrajGroup = np.load('cpp/test/bt_compute/out/TrajGroup.npy',allow_pickle=True)

        def interpolate_traj_group(TrajGroup, num_points=4800):
            interpolated_traj_group = []
            for Traj in TrajGroup:
                # Extract longitude, latitude, and level from Traj
                lons = [point.longitude for point in Traj]
                lats = [point.latitude for point in Traj]
                levels = [point.level for point in Traj]

                # Calculate the ratio step for interpolation
                step = (len(lons) - 1) / (num_points - 1)

                # Create interpolation functions
                lon_interp = interp1d(range(len(lons)), lons, kind='quadratic')
                lat_interp = interp1d(range(len(lats)), lats, kind='quadratic')
                level_interp = interp1d(range(len(levels)), levels, kind='quadratic')

                # Generate new points
                new_traj = []
                for i in range(num_points):
                    index = min(int(i * step), len(lons) - 1)  # Ensure not to exceed the original trajectory length
                    lon = lon_interp(index)
                    lat = lat_interp(index)
                    level = level_interp(index)
                    new_traj.append(Point(lat, lon, level))

                interpolated_traj_group.append(new_traj)

            return interpolated_traj_group

        
        interpolated_traj_group = interpolate_traj_group(TrajGroup)
        interpolated_traj_group = np.array(interpolated_traj_group)
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        # 绘图部分
        lat_min, lat_max= 20, 60
        lon_min, lon_max = -75, 0
        map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
        map.drawcoastlines()
        map.drawcountries()

        colors = 'black'  # 可以添加更多颜色以区分更多轨迹

        # 绘制每个轨迹
        for i,Traj in enumerate(interpolated_traj_group):
            # 提取轨迹中所有点的经度和纬度
            lons = [point.longitude for point in Traj]
            lats = [point.latitude for point in Traj]

            concatenated_array = np.concatenate([arr.flatten() for arr in lats])
            lats = concatenated_array.tolist()

            concatenated_array = np.concatenate([arr.flatten() for arr in lons])
            lons = concatenated_array.tolist()
            
            # 将经纬度转换为地图坐标
            x, y = map(lons, lats)
            
            map.plot(x, y, color=colors, alpha=1,linewidth=0.2)
        
            #map.plot(x, y, color=colors[i % len(colors)], linewidth=1, linestyle='-', marker='o', markersize=3)
        
        plt.title('Back Trajectories')
        plt.savefig("pics/bt.pdf")      
        return

    def compute_single_traj(
            self,
            cur_time,
            cur_location:Point,
            delta_t=3600*3) -> list:

        single_traj = [cur_location.copy() for _ in range(24)]
        obj = Merra2Parser(self.nc_path)
        Traj = [] 
        Traj.append(cur_location)
        for i in range(7):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            #*********************************************************************************
            # single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            # print(f"{cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            
            # u_wind,v_wind,w_wind = Merra2Parser(self.nc_path).parse_uvw2(\
            #             cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            # print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            # wind_data = Wind(u_wind, v_wind, w_wind)
            # next_location:Point = self.__compute_new_location__(cur_location,wind_data,delta_t)
            # print(f"we get next point's  is = {next_location.level} -- {next_location.latitude} -- {next_location.longitude}")
            cur_location = next_location  # 更新当前位置
            cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
        
        # file_path = 'points.txt'
        # with open(file_path, 'w') as file:
        #     for point in Traj:
        #         file.write(f'{longitude}, {latitude}\n')
        
        return Traj

    def __compute_new_location__(self,cur_point, wind_data, delta_t=3600*3):

        # 地球半径，单位为千米
        earth_radius = 6371.0
        
        # 将垂直风速转换为压力变化 (假设：1 Pa/s 垂直风速对应 1 hPa 的压力变化)
        pressure_change = wind_data.w_wind * delta_t/100
        next_point = cur_point.copy()  # 创建当前点的副本以进行修改
        next_point.level = cur_point.level - pressure_change  # 更新压力水平
        

        # 后向，取反
        dx = -wind_data.u_wind * delta_t / 1000  # m/s to km/3h
        dy = -wind_data.v_wind * delta_t / 1000  # m/s to km/3h

        # 使用更精确的球面距离公式
        delta_lon = (dx / (earth_radius * math.cos(math.radians(cur_point.latitude)))) * (180.0 / math.pi)
        delta_lat = (dy / earth_radius) * (180.0 / math.pi)

        next_point.longitude = cur_point.longitude + delta_lon
        next_point.latitude = cur_point.latitude + delta_lat
        
        # 调整经度范围
        if next_point.longitude > 180:
            next_point.longitude -= 360
        if next_point.longitude < -180:
            next_point.longitude += 360
        
        # 调整纬度范围
        if next_point.latitude > 90:
            next_point.latitude = 90
        if next_point.latitude < -90:
            next_point.latitude = -90
        
        return next_point
import re
class BackTrajMuti(BackTraj):
    def __init__(self, nc_path: str, nc_paths:list,start_time, start_point: Point,time_chip = 6 ) -> None:
        super().__init__(nc_path, start_time, start_point)
        self.nc_paths = sorted(nc_paths, key=lambda x: int(re.search(r'(\d{8})\.nc4$', x).group(1)) if re.search(r'(\d{8})\.nc4$', x) else 0)
        return

    def compute_multi_traj2(self,cur_time,cur_location:Point,delta_t=3600*3) -> list:

        file_list = []
        TrajGroup = []
        file_list = self.nc_paths
        i=0
        for nc in file_list:
            if i<2:
                i += 1
                continue
            self.nc_path = nc
            Trajs:list = self.compute_single_traj_all(cur_time,cur_location,delta_t)
            TrajGroup += Trajs
        np.save('cpp/test/bt_compute/out/TrajGroup.npy', TrajGroup)   
        TrajGroup = np.load('cpp/test/bt_compute/out/TrajGroup.npy',allow_pickle=True)

        def interpolate_traj_group(TrajGroup, num_points=24):
            interpolated_traj_group = []
            for Traj in TrajGroup:
                # Extract longitude, latitude, and level from Traj
                lons = [point.longitude for point in Traj]
                lats = [point.latitude for point in Traj]
                levels = [point.level for point in Traj]

                # Calculate the ratio step for interpolation
                step = (len(lons) - 1) / (num_points - 1)

                # Create interpolation functions
                lon_interp = interp1d(range(len(lons)), lons, kind='quadratic')
                lat_interp = interp1d(range(len(lats)), lats, kind='quadratic')
                level_interp = interp1d(range(len(levels)), levels, kind='quadratic')

                # Generate new points
                new_traj = []
                for i in range(num_points):
                    index = min(int(i * step), len(lons) - 1)  # Ensure not to exceed the original trajectory length
                    lon = lon_interp(index)
                    lat = lat_interp(index)
                    level = level_interp(index)
                    new_traj.append(Point(lat, lon, level))

                interpolated_traj_group.append(new_traj)

            return interpolated_traj_group

        interpolated_traj_group = interpolate_traj_group(TrajGroup)
        interpolated_traj_group = np.array(interpolated_traj_group)
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        # 绘图部分
        lat_min, lat_max= 20, 60
        lon_min, lon_max = -75, 0
        map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
        map.drawcoastlines()
        map.drawcountries()

        colors = 'black'  # 可以添加更多颜色以区分更多轨迹

        # 绘制每个轨迹
        for i,Traj in enumerate(interpolated_traj_group):
            # 提取轨迹中所有点的经度和纬度
            lons = [point.longitude for point in Traj]
            lats = [point.latitude for point in Traj]

            concatenated_array = np.concatenate([arr.flatten() for arr in lats])
            lats = concatenated_array.tolist()

            concatenated_array = np.concatenate([arr.flatten() for arr in lons])
            lons = concatenated_array.tolist()
            
            # 将经纬度转换为地图坐标
            x, y = map(lons, lats)
            
            map.plot(x, y, color=colors, alpha=1,linewidth=0.2)
        
            #map.plot(x, y, color=colors[i % len(colors)], linewidth=1, linestyle='-', marker='o', markersize=3)
        
        plt.title('Back Trajectories')
        plt.savefig("pics/bt.pdf")      
        return


    def compute_single_traj_all(            
            self, 
            cur_time, 
            cur_location: Point, 
            delta_t=3600 * 3,
              #时间切片3h、6h
        ) -> list:
        last_path = self.__find_previous_element__(self.nc_paths,self.nc_path)
        obj = Merra2Parser(self.nc_path)
        obj2 = Merra2Parser(last_path)
        cur_time_temp = cur_time
        cur_location_temp = cur_location
        TrajsList = []
        for j in range(4):
            single_traj = [cur_location.copy() for _ in range(24)]
            cur_location = cur_location_temp
            cur_time = cur_time - 180.0*j
            Traj = [] 
            Traj.append(cur_location)
            
            for i in range(8):
                if (j==1 and i == 7) or (j==2 and i == 6) or (j==3 and i == 5):
                    obj = obj2
                    cur_time = cur_time_temp
                single_traj[i] = cur_location.copy()  # 存储当前位置的副本
                u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                        cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
                cur_wind_data = Wind(u_wind, v_wind, w_wind)
                print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
                print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

                forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
                u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                        cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
                forest_wind_data = Wind(u_wind, v_wind, w_wind)
                print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
                print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
                avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                                sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                                sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

                #avg_wind_data = (cur_wind_data+forest_wind_data)/2
                next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
                print("-----------------------------------------********************----------**************----------------------------------")
                print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
                print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
                print("-----------------------------------------******************************--------------------------------------------")

                cur_location = next_location  # 更新当前位置
                cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
                Traj.append(cur_location)
            TrajsList.append(Traj)
        return TrajsList
           
    '''
    计算24点开始
    '''
    def compute_single_traj_24(            
            self, 
            cur_time, 
            cur_location: Point, 
            delta_t=3600 * 3,
              #时间切片3h、6h
        ) -> list:
        single_traj = [cur_location.copy() for _ in range(24)]
        obj = Merra2Parser(self.nc_path)
        Traj = [] 
        Traj.append(cur_location)
        '''
        从24点开始
        '''
        for i in range(8):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            #*********************************************************************************
            # single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            # print(f"{cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            
            # u_wind,v_wind,w_wind = Merra2Parser(self.nc_path).parse_uvw2(\
            #             cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            # print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            # wind_data = Wind(u_wind, v_wind, w_wind)
            # next_location:Point = self.__compute_new_location__(cur_location,wind_data,delta_t)
            # print(f"we get next point's  is = {next_location.level} -- {next_location.latitude} -- {next_location.longitude}")
            cur_location = next_location  # 更新当前位置
            cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
    
        return
    
    '''
    计算18点开始
    '''
    def compute_single_traj_18(            
            self, 
            cur_time, 
            cur_location: Point, 
            delta_t=3600 * 3,
              #时间切片3h、6h
        ) -> list:
        single_traj = [cur_location.copy() for _ in range(24)]
        cur_time_temp = cur_time
        obj = Merra2Parser(self.nc_path)
        last_path = self.__find_previous_element__(self.nc_paths,self.nc_path)
        obj2 = Merra2Parser(last_path)
        Traj = [] 
        Traj.append(cur_location)

        for i in range(8):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            if i == 8:
                obj = obj2
                cur_time = cur_time_temp

            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            cur_location = next_location  # 更新当前位置
            cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
        return
    
    '''
    计算12点开始
    '''
    def compute_single_traj_12(            
            self, 
            cur_time, 
            cur_location: Point, 
            delta_t=3600 * 3,
              #时间切片3h、6h
        ) -> list:
    
        single_traj = [cur_location.copy() for _ in range(24)]
        cur_time_temp = cur_time
        obj = Merra2Parser(self.nc_path)
        last_path = self.__find_previous_element__(self.nc_paths,self.nc_path)
        obj2 = Merra2Parser(last_path)
        Traj = [] 
        Traj.append(cur_location)

        for i in range(8):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            if i == 7:
                obj = obj2
                cur_time = cur_time_temp

            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            cur_location = next_location  # 更新当前位置
            cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
        return
    
    '''
    计算06点开始
    '''
    def compute_single_traj_6(            
            self, 
            cur_time, 
            cur_location: Point, 
            delta_t=3600 * 3,
              #时间切片3h、6h
        ) -> list:
    
        single_traj = [cur_location.copy() for _ in range(24)]
        cur_time_temp = cur_time
        obj = Merra2Parser(self.nc_path)
        last_path = self.__find_previous_element__(self.nc_paths,self.nc_path)
        obj2 = Merra2Parser(last_path)
        Traj = [] 
        Traj.append(cur_location)

        for i in range(8):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            if i == 6:
                obj = obj2
                cur_time = cur_time_temp

            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")

            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw2(\
                                    cur_time-(60*3),forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            cur_location = next_location  # 更新当前位置
            cur_time -= 180.0  # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
        return

    def __find_previous_element__(self,ncpaths, ncpath):
        previous_element = None  # 初始化上一个元素的值为None
        for element in ncpaths:
            if element == ncpath:
                return previous_element  # 返回上一个元素的值
            previous_element = element  # 更新上一个元素的值
        return previous_element  # 如果没有找到目标值，返回None

class DrawTraj(BackTraj):
    def __init__(self, nc_path: str, start_time, start_point: Point) -> None:
        super().__init__(nc_path, start_time, start_point)
        return
    def draw_traj_grounp(self):
        return


# 假设Point和Wind类的定义如下：

cur_point = Point(39.5,-28.1,500.0)
wind_data = Wind(u_wind=10, v_wind=5, w_wind=1)
delta_t = 3600  # 1小时
start_time = 1260.0

# obj = BackTraj('cpp/merra2/MERRA2_400.inst3_3d_asm_Np.20230101.nc4',start_time,cur_point)
# obj.compute_multi_traj('cpp/merra2',start_time,cur_point)
# obj.compute_single_traj(start_time,cur_point)
# new_point = compute_new_location(cur_point, wind_data, delta_t)
# print(new_point.longitude, new_point.latitude, new_point.level)

file_list = []
folder_path = 'cpp/merra2'
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    file_list.append(item_path)
obj = BackTrajMuti('cpp/merra2/MERRA2_400.inst3_3d_asm_Np.20230101.nc4',file_list,start_time,cur_point)
obj.compute_multi_traj2(start_time,cur_point)


