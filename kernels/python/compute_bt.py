from era5parser import Era5Parse
import math
import os
import numpy as np
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    def compute_multi_traj(self,folder_path,cur_time,cur_location:Point,delta_t=3600) -> list:

        file_list = []
        TrajGroup = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            file_list.append(item_path)
        for nc in file_list:
            print(f"current computing file is : {nc}")
            self.nc_path = nc
            Traj:list = self.compute_single_traj(cur_time,cur_location,delta_t)
            TrajGroup.append(Traj)
        np.save('kernels/python/cache/TrajGroup1.npy', TrajGroup)   
        TrajGroup = np.load('kernels/python/cache/TrajGroup1.npy',allow_pickle=True)

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
        
        plt.title('Back Trajectories')
        plt.savefig("res/pics/bt.pdf")      
        return

    def compute_single_traj(
            self,
            cur_time,
            cur_location:Point,
            delta_t=3600) -> list:

        single_traj = [cur_location.copy() for _ in range(24)]
        obj = Era5Parse(self.nc_path)
        Traj = [] 
        Traj.append(cur_location)
        for i in range(23):
            single_traj[i] = cur_location.copy()  # 存储当前位置的副本
            u_wind,v_wind,w_wind = obj.parse_uvw(\
                                    cur_time,cur_location.level,cur_location.latitude,cur_location.longitude)
            cur_wind_data = Wind(u_wind, v_wind, w_wind)
            # print(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            # print(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            logger.debug(f"cur_location : {cur_location.latitude:.5f} {cur_location.longitude:.5f} {cur_location.level:.5f}")
            logger.debug(f"we get cur point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            forest_location:Point = self.__compute_new_location__(cur_location,cur_wind_data)
            u_wind,v_wind,w_wind = obj.parse_uvw(\
                                    cur_time-1,forest_location.level,forest_location.latitude,forest_location.longitude)            
            forest_wind_data = Wind(u_wind, v_wind, w_wind)
            # print(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            # print(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            logger.debug(f"forest_location : {forest_location.latitude:.5f} {forest_location.longitude:.5f} {forest_location.level:.5f}")
            logger.debug(f"and get forest point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            avg_wind_data = Wind(sum(w.u_wind for w in [cur_wind_data, forest_wind_data]) / 2, \
                            sum(w.v_wind for w in [cur_wind_data, forest_wind_data]) / 2,\
                            sum(w.w_wind for w in [cur_wind_data, forest_wind_data]) / 2)

            #avg_wind_data = (cur_wind_data+forest_wind_data)/2
            next_location:Point = self.__compute_new_location__(cur_location,avg_wind_data)
            print("-----------------------------------------********************----------**************----------------------------------")
            print(f"and get avage point's wind is = {u_wind} -- {v_wind} -- {w_wind}")
            print(f"compute next_location : {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            print("-----------------------------------------******************************--------------------------------------------")
            logger.debug(f"and get avage point's wind is : {u_wind} -- {v_wind} -- {w_wind}")
            logger.debug(f"compute next_location  = {next_location.latitude:.5f} {next_location.longitude:.5f} {next_location.level:.5f}")
            
            cur_location = next_location  # 更新当前位置
            cur_time -= 1 # 减少时间，这里假设时间单位是秒
            Traj.append(cur_location)
        
        return Traj

    def __compute_new_location__(self,cur_point, wind_data, delta_t=3600):

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

class Btcompute(BackTraj):
    def __init__(self, nc_path: str, start_time, start_point: Point) -> None:
        super().__init__(nc_path, start_time, start_point)

    '''
    folder : 下载的气象文件目录
    start_time : 计算开始时间(例如23)
    start_point : 计算起始位置(例如Point(39.5,-28.1,500.0))
    hour : 间隔时间，即美多少小时计算一次轨迹
    '''
    def compute(folder,start_time,start_point:Point,hour):
        
        return








# 假设Point和Wind类的定义如下：

cur_point = Point(39.5,-28.1,500.0)
wind_data = Wind(u_wind=10, v_wind=5, w_wind=1)
delta_t = 3600  # 1小时
start_time = 23

obj = BackTraj('/mnt/d/学习资料/气象数据/era5s/2022/20221229.nc',start_time,cur_point)
obj.compute_multi_traj('/mnt/d/学习资料/气象数据/era5s/2023',start_time,cur_point)

# new_point = obj.compute_single_traj(23,cur_point)
# print(new_point.longitude, new_point.latitude, new_point.level)


