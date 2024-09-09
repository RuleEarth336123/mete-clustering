# netcdf_parser.py
import netCDF4 as nc
import numpy as np
import os
import math
import numpy as np
from scipy.interpolate import interp1d
"""

"""
class Point:
    def __init__(self, latitude, longitude):
        self.longitude = longitude
        self.latitude = latitude
    
    def copy(self):
        return Point(self.longitude, self.latitude)

class Wind:
    def __init__(self, u_wind, v_wind, w_wind):
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.w_wind = w_wind

class Era5Parser:

    def __init__(self,folder_path) -> None:
        self.folder_path = folder_path
        pass

    def parse_uvw(self,time:int,cur_location:Point):
        latitude = cur_location.latitude
        longitude = cur_location.longitude
        prefixes = ['o', 'u', 'v']
        files_path = []
        # 遍历文件夹中的文件，筛选出包含特定前缀的文件
        for file_name in os.listdir(self.folder_path):
            for prefix in prefixes:
                if file_name.startswith(prefix):
                    files_path.append(os.path.join(self.folder_path, file_name))
        
        ouv_list = []

        for i in range(3):
            try:
                self.data = nc.Dataset(files_path[i])
            except Exception as e:
                    print(f"An error occurred while init parser...: {e}")
            self.latitude_sets= self.data.variables['latitude'][:]
            self.longitude_sets = self.data.variables['longitude'][:]

            mLatIndex = self.__binary_search_latitude__(self.latitude_sets, latitude)
            mLonIndex = self.__binary_search_longitude__(self.longitude_sets, longitude)
            if i == 0:
                variable = self.data.variables['w'][:]
                ouv_list.append(variable[time][mLatIndex][mLonIndex])
            else:
                variable = self.data.variables[prefixes[i]][:]
                ouv_list.append(variable[time][mLatIndex][mLonIndex])         
        
        return ouv_list[1],ouv_list[2],ouv_list[0]
    
    def parse_uvw2()->None:

        return
    
    def __binary_search_latitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.25 <= arr[mid] and arr[mid] <= target+0.25:
                return mid  # lon数据是顺序的
            elif arr[mid] < target:
                high = mid - 1
            else:
                low = mid + 1
        return -1 

    def __binary_search_longitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.4 <= arr[mid] and arr[mid] <= target+0.4:
                return mid  # lon数据是顺序的
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1  



class ComputeBackTraj():
    def __init__(self,folder_path) -> None:
        self.file_paths = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                path = os.path.join(root, dir)
                self.file_paths.append(path)
        pass

    def compute_day(self,file_path,cur_location:Point):
        Traj = []
        single_traj = [cur_location.copy() for _ in range(24)]
        obj = Era5Parser(file_path)
        Traj.append(cur_location)
        for i in range(24):
            single_traj[i] = cur_location.copy() 
            u_wind,v_wind,w_wind = obj.parse_uvw(23-i,cur_location)
            cur_wind_data = Wind(u_wind,v_wind,w_wind)
            print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} ")
            print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw(23-i,cur_location)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f}")
            print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************------next_location----**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            
            cur_location = next_location  # 更新当前位置
            Traj.append(cur_location)
            
        return Traj
    
    def compute_multi_traj(self,folder_path,cur_location:Point,delta_t=3600) -> list:

        TrajGroups = []
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                path = os.path.join(root, dir)
                TrajGroups.append(obj.compute_day(path,cur_location))
        np.save('cache/TrajGroup.npy', TrajGroups)   
        TrajGroups = np.load('cache/TrajGroup.npy',allow_pickle=True)
        
        TrajGroupsList = []
        for traj in TrajGroups:
            trajs = []
            for point in traj:
                tmp = [point.latitude,point.longitude]
                trajs.append(np.array(tmp))
            TrajGroupsList.append(trajs) 
        np.save('cache/TrajGroupsList.npy', TrajGroupsList)    
        TrajGroupsArray = np.array(TrajGroupsList)
        



        def interpolate_traj_group(TrajGroups, num_points=24):
            interpolated_traj_group = []
            for Traj in TrajGroups:
                # Extract longitude, latitude, and level from Traj
                lons = [point.longitude for point in Traj]
                lats = [point.latitude for point in Traj]

                # Calculate the ratio step for interpolation
                step = (len(lons) - 1) / (num_points - 1)

                # Create interpolation functions
                lon_interp = interp1d(range(len(lons)), lons, kind='quadratic')
                lat_interp = interp1d(range(len(lats)), lats, kind='quadratic')
    
                # Generate new points
                new_traj = []
                for i in range(num_points):
                    index = min(int(i * step), len(lons) - 1)  # Ensure not to exceed the original trajectory length
                    lon = lon_interp(index)
                    lat = lat_interp(index)
                    new_traj.append(Point(lat, lon))

                interpolated_traj_group.append(new_traj)

            return interpolated_traj_group

        interpolated_traj_group = interpolate_traj_group(TrajGroups)
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

    def compute_6h_multi_traj(self,cur_time,cur_location:Point,delta_t=3600) -> list:

        # file_list = []
        # TrajGroup = []
        # file_list = self.file_paths
        # i=0
        # for nc in file_list:
        #     if i<1:
        #         i += 1
        #         continue
        #     self.nc_path = nc
        #     Trajs:list = self.compute_single_traj_6h(nc,cur_time,cur_location,delta_t)
        #     TrajGroup += Trajs
        # np.save('cache/TrajGroups.npy', TrajGroup)   
        TrajGroup = np.load('cache/TrajGroups.npy',allow_pickle=True)


        TrajGroupsList = []
        for traj in TrajGroup:
            trajs = []
            for point in traj:
                tmp = [point.latitude,point.longitude]
                trajs.append(np.array(tmp))
            TrajGroupsList.append(trajs) 
        np.save('cache/TrajGroupsList.npy', TrajGroupsList)    
        TrajGroupsArray = np.array(TrajGroupsList)

        def interpolate_traj_group(TrajGroup, num_points=24):
            interpolated_traj_group = []
            for Traj in TrajGroup:
                # Extract longitude, latitude, and level from Traj
                lons = [point.longitude for point in Traj]
                lats = [point.latitude for point in Traj]

                # Calculate the ratio step for interpolation
                step = (len(lons) - 1) / (num_points - 1)

                # Create interpolation functions
                lon_interp = interp1d(range(len(lons)), lons, kind='quadratic')
                lat_interp = interp1d(range(len(lats)), lats, kind='quadratic')

                # Generate new points
                new_traj = []
                for i in range(num_points):
                    index = min(int(i * step), len(lons) - 1)  # Ensure not to exceed the original trajectory length
                    lon = lon_interp(index)
                    lat = lat_interp(index)
                    new_traj.append(Point(lat, lon))

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
            
            map.plot(x, y, color=colors,alpha=1,linewidth=0.6)
            #map.scatter(x, y, color='red', s=5, marker='o', edgecolors='black', alpha=1)
        
            #map.plot(x, y, color=colors[i % len(colors)], linewidth=1, linestyle='-', marker='o', markersize=3)
        
        plt.title('Back Trajectories')
        plt.savefig("pics/bt.pdf")      
        return

    def compute_single_traj_6h(self,file_path,cur_time,cur_location: Point, delta_t=3600) -> list:
        last_path = self.__find_previous_element__(file_path)
        obj = Era5Parser(file_path)
        obj2 = Era5Parser(last_path)
        cur_time_temp = cur_time
        cur_location_temp = cur_location
        TrajsList = []
        for j in range(4):
            single_traj = [cur_location.copy() for _ in range(24)]
            cur_location = cur_location_temp
            cur_time = cur_time_temp - 6*j
            Traj = [] 
            Traj.append(cur_location)
            
            for i in range(24):
                if (j==1 and i == 21) or (j==2 and i == 18) or (j==3 and i == 15):
                    obj = obj2
                    cur_time = cur_time_temp
                single_traj[i] = cur_location.copy() 
                u_wind,v_wind,w_wind = obj.parse_uvw(cur_time,cur_location)
                cur_wind_data = Wind(u_wind,v_wind,w_wind)
                print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} ")
                print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
                forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
                u_wind,v_wind,w_wind = obj.parse_uvw(cur_time-1,cur_location)            
                forest_wind_data = Wind(u_wind, v_wind, w_wind)
                print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f}")
                print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
                avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                                sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                                sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)
                next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
                print("-----------------------------------------********************------next_location----**************----------------------------------")
                print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
                print(f"next_location : {next_location.latitude:.5f} {next_location.longitude:.5f}")
                print("-----------------------------------------******************************--------------------------------------------")

                cur_location = next_location  # 更新当前位置
                cur_time -= 1  
                Traj.append(cur_location)
            TrajsList.append(Traj)
        return TrajsList    
    
    def __compute_new_location__(self,cur_point, wind_data, delta_t=3600):
        # 地球半径，单位为千米
        earth_radius = 6371.0

        next_point = cur_point.copy()  # 创建当前点的副本以进行修改

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
    def __find_previous_element__(self, ncpath):
        previous_element = None  # 初始化上一个元素的值为None
        for element in self.file_paths:
            if element == ncpath:
                return previous_element  # 返回上一个元素的值
            previous_element = element  # 更新上一个元素的值
        return previous_element  # 如果没有找到目标值，返回None
# print(Era5Parser('ERA5_data/2023/202301/01/').era5_parser(23,39.5,-28.1))
obj = ComputeBackTraj('ERA5_data/2023/202301')
cur_location = Point(39.5,-28.1)
TrajGroups = []
#obj.compute_multi_traj('ERA5_data/2023/202301',cur_location)
obj.compute_6h_multi_traj(23,cur_location)


