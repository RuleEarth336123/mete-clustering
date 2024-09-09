import requests
import json
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import time
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
    def compute_multi_traj(self,TrajGroup) -> list:


        # np.save('kernels/python/cache/TrajGroup.npy', TrajGroup)   
        # TrajGroup = np.load('kernels/python/cache/TrajGroup.npy',allow_pickle=True)


        interpolated_traj_group = np.array(TrajGroup)

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
        plt.savefig("res/pics/202301.pdf")      
        return


    def compute_single_traj(self,TrajGroup) -> list:
    
        interpolated_traj_group = np.array(TrajGroup)

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


def compute1h():
    # 定义 URL 和端口
    url = 'http://localhost:12123/compute/6h'


    # 定义要发送的数据
    data = {
        "file": [
            "/mnt/d/学习资料/气象数据/era5s/202301/20230101.nc",
            "/mnt/d/学习资料/气象数据/era5s/202301/20230102.nc"
            ],
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
    return


url = 'http://localhost:12123/compute/6h'
def compute6h(folder):
    folder = "/mnt/d/学习资料/气象数据/era5s/202301"
    filelist = os.listdir(folder)
    TrajGroups = []
    obj = BackTraj()
    
    for i,file in enumerate(filelist):
        files = []
        if i == 0:
            continue
        files.append(os.path.join(folder, filelist[i-1]))
        files.append(os.path.join(folder, filelist[i]))
        
        data = {
            "file": files,
            "hour": 23,
            "latitude": 39.5,
            "longitude": -28.1,
            "level": 500.0
        }
        
        json_data = json.dumps(data, indent=4)
        
        print(json_data)
        
        headers = {
            'Content-Type': 'application/json'
        }


        response = requests.post(url, data=json_data, headers=headers)

        start = time.time()

        print('Status Code:', response.status_code)
        # print('Response Body:', response.text)
        end = time.time()
        print(f"compute use time : {end - start}" )

        if response.status_code == 200:
            response_json = json.loads(response.text)
            TrajGroup = response_json['trajectories']
            
            for item in TrajGroup:
                TrajGroups.append(item)
            
        else:
            print('Failed to get a valid response from the server.')
    
    obj.compute_multi_traj(TrajGroups)
    
    return None


compute6h('/mnt/d/学习资料/气象数据/era5s/202301')





