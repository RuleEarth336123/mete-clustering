import requests
import json
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
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


        
        interpolated_traj_group = TrajGroup
        interpolated_traj_group = np.array(interpolated_traj_group)

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


    def compute_single_traj(self,TrajGroup) -> list:
    
        interpolated_traj_group = TrajGroup
        interpolated_traj_group = np.array(interpolated_traj_group)

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
            lons = [point[1] for point in Traj]
            lats = [point[0] for point in Traj]

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



# 定义 URL 和端口
url = 'http://localhost:12123/compute/6h'
# 定义要发送的数据
data = {
    "file": ["/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc",\
        "/mnt/d/学习资料/气象数据/era5s/202301/20230102.nc"],
    "hour": 23,
    "latitude": 39.5,
    "longitude": -28.1,
    "level": 500.0
}

json_data = json.dumps(data)
headers = {
    'Content-Type': 'application/json'
}

obj = BackTraj()

response = requests.post(url, data=json_data, headers=headers)

import time

start = time.time()


print('Status Code:', response.status_code)
# print('Response Body:', response.text)
    
    
end = time.time()
print(f"compute use time : {end - start}" )

if response.status_code == 200:
    response_json = json.loads(response.text)
    TrajGroup = response_json['trajectories']
    
    obj.compute_single_traj(TrajGroup)

    # for point in trajectory:
    #     print(point)
else:
    print('Failed to get a valid response from the server.')






