import requests
import json
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd

from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def bezier_smoothing(points, num=100):
    # 计算贝塞尔曲线的控制点
    n = len(points)
    control_points = [points[0]] + [(points[i] + points[i - 1]) / 2.0 for i in range(1, n - 1)] + [points[-1]]
    control_points = np.array(control_points)[:, :2]  # 只保留x和y坐标

    # 计算贝塞尔曲线上的点
    t = np.linspace(0, 1, num)
    curve_points = np.zeros((num, 2))

    for i in range(num):
        curve_point = np.zeros(2)
        for j in range(len(control_points)):
            binomial_coeff = np.math.factorial(n - 1) / (np.math.factorial(j) * np.math.factorial(n - 1 - j))
            curve_point += binomial_coeff * (t[i] ** j) * ((1 - t[i]) ** (n - 1 - j)) * control_points[j]
        curve_points[i] = curve_point

    # 将结果转换为所需的格式
    smooth_traj_segments = [curve_points[i:i+1].tolist() for i in range(len(curve_points))]

    return smooth_traj_segments


class BackTraj:
    def compute_multi_traj(self,TrajGroup) -> list:

        interpolated_traj_group = []
        
        np.save('kernels/python/cache/TrajGroup.npy', TrajGroup)   
        TrajGroup = np.load('kernels/python/cache/TrajGroup.npy',allow_pickle=True)

        for traj in TrajGroup:
            curve_points = bezier_smoothing(traj,24)
            interpolated_traj_group.append(curve_points)
            
        trajs2list = []
        for i,traj in enumerate(interpolated_traj_group):
            single = []
            for i,item in enumerate(traj):
                single.append(traj[i][0])
            trajs2list.append(single)

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
            lons = [point[0][1] for point in Traj]
            lats = [point[0][0] for point in Traj]

            concatenated_array = np.concatenate([arr.flatten() for arr in lats])
            lats = concatenated_array.tolist()

            concatenated_array = np.concatenate([arr.flatten() for arr in lons])
            lons = concatenated_array.tolist()
            
            # 将经纬度转换为地图坐标
            x, y = map(lons, lats)
            
            map.plot(x, y, color=colors, alpha=1,linewidth=0.2)
        
        plt.title('Back Trajectories 202303 smooth')
        plt.savefig("res/pics/202303.pdf")      
        return trajs2list


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

def KMeansCluster2(TrajGroups,features,num_clusters) -> None:  

    data_list = TrajGroups

    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(features)
    cluster_centers = []

    def compute_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)
    
    cluster_centers = Parallel(n_jobs=-1)(delayed(compute_cluster_center)(i, data_list, labels) for i in tqdm(range(num_clusters)))

    lat_min, lat_max= 20, 60
    lon_min, lon_max = -75, 0
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    colors = ['g', 'r', 'b', 'r', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(cluster_centers, colors)):
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            trajectory = data_list[index]
            
            lons = [point[1] for point in trajectory]
            lats = [point[0] for point in trajectory]
            
            x, y = map(lons, lats)
            map.plot(x, y, color=color, linestyle='dashed',linewidth=0.3)
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        
    
    specific_lat, specific_lon = 39.500, -28.100

    # 转换坐标
    specific_x, specific_y = map(specific_lon, specific_lat)

    # 绘制特定点
    map.scatter(specific_x, specific_y, color='red', s=5, marker='o', edgecolors='black', alpha=1)

    plt.text(specific_x, -10000, f'Longitude: {specific_lon:.2f}', fontsize=10, color='black', ha='center')
    plt.text(-10000, specific_y, f'Latitude: {specific_lat:.2f}', fontsize=10, color='black', va='center')
    plt.title('Trajectories Cluster 2023203')
    plt.show()
    plt.savefig("res/pics/cluster2023203_v1.pdf")
    return None 

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
def compute6h(folder):
    url = 'http://localhost:12123/compute/6h'
    filelist = os.listdir(folder)
    TrajGroups = []
    obj = BackTraj()
    start = time.time()
    
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

        print('Status Code:', response.status_code)
        # print('Response Body:', response.text)


        if response.status_code == 200:
            response_json = json.loads(response.text)
            TrajGroup = response_json['trajectories']
            
            for item in TrajGroup:
                TrajGroups.append(item)
            
        else:
            print('Failed to get a valid response from the server.')
    
    trajs2list = obj.compute_multi_traj(TrajGroups)
    
    end = time.time()
    print(f"compute use time : {end - start}" ) 
     
    return TrajGroups,trajs2list

def computefeatures(TrajGroups):
    url = 'http://localhost:12123/cluster/feature'

    start = time.time()
    
    data = {"trajectories": TrajGroups}

    # 将字典转换为JSON字符串
    json_data = json.dumps(data, indent=4)
    
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, data=json_data, headers=headers)

    print('Status Code:', response.status_code)

    if response.status_code == 200:
        response_json = json.loads(response.text)
        features = response_json['features']
        
    else:
        print('Failed to get a valid response from the server.')


    end = time.time()
    print(f"compute features use time : {end - start}" ) 
     
    return features






TrajGroups,trajs2list = compute6h('/mnt/d/学习资料/气象数据/era5s/202303')
# np.save('kernels/python/cache/TrajGroups.npy', TrajGroups)
# TrajGroups = np.load('kernels/python/cache/TrajGroups.npy')
features = computefeatures(TrajGroups)
KMeansCluster2(trajs2list,features,3)




