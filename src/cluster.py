from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from tqdm import tqdm
from fastdtw import fastdtw
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import euclidean,cosine
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

class TrajCluster:
    def __init__(self,folder_path) -> None:
        self.folder_path = folder_path
        pass
    def KMeansCluster(self,num_clusters:int) -> None:  
        return 
    def HierarchicalCluster(folder_path:str,num_clusters:int) -> None:  
        return
    


def conditional_decorator(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if condition():
                return func(*args, **kwargs)
            else:
                print("Condition not met, function will not execute.")
                return None
        return wrapper
    return decorator

def anima_show(data_list) -> None:
    fig, ax = plt.subplots()
    fit_lines = []
    def update(frame):
        # 清空拟合的轨迹线
        # for line in fit_lines:
        #     line.remove()
        # fit_lines.clear()
        # 画出第frame条拟合好的轨迹
        interpolated_data = data_list[frame]
        line, = ax.plot(interpolated_data[:, 0], interpolated_data[:, 1], color='blue')
        fit_lines.append(line)
        return fit_lines
    # 设置动画
    ani = FuncAnimation(fig, update, frames=len(data_list), interval=1000)
    plt.show()
    return

def remove_outliers_isolation_forest(df, contamination=0.1):
    model = IsolationForest(contamination=contamination)
    df['anomaly'] = model.fit_predict(df[['Lat', 'Lon']])
    outliers = df[df['anomaly'] == -1]
    normal = df[df['anomaly'] == 1]
    return normal, outliers

def KMeansCluster(folder_path:str,num_clusters:int) -> None:  
    file_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    data_list = []
    start_point = np.array([39.5, -28.1]) 

    def update_plot(csv_list):
        fig, ax = plt.subplots()
        num_points = 48
        for csv in csv_list:
            df = pd.read_csv(csv).dropna()
            if not df.empty:
                f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
                interpolated_data = f(np.linspace(0, 1, num_points))
                shift_vector_start = start_point - interpolated_data[0]
                interpolated_data += shift_vector_start

                for i in range(1, num_points+1):
                    ax.scatter(interpolated_data[:i, 0], interpolated_data[:i, 1], label='插值点')
                    ax.plot(interpolated_data[:i, 0], interpolated_data[:i, 1], '-o', label='曲线拟合')
                    ax.plot(start_point[0], start_point[1], 'ro', label='起点')
                    ax.set_title(f'轨迹拟合 - {csv}')
                    ax.set_xlabel('lat')
                    ax.set_ylabel('lon')
                    ax.legend()
                    plt.pause(0.01)
        plt.show()

    # Call the update_plot function with the file_list
    #update_plot(file_list)
    
    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
            interpolated_data = f(np.linspace(0, 1, num_points))
            shift_vector_start = start_point - interpolated_data[0]
            interpolated_data += shift_vector_start
            data_list.append(interpolated_data)

    #anima_show(data_list)

    np.save('data/cublic_data_list.npy', data_list)

    def dtw_distance(pair):
        i, j = pair
        dist, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
        return i, j, dist
    
    def paralcomputdistance2() -> None:
        data_array = np.array(data_list)
        np.save('data/data_array.npy', data_array)
        distances = np.zeros((len(data_list), len(data_list)))
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        results = Parallel(n_jobs=-1)(delayed(dtw_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in results:
            distances[i, j] = dist
            distances[j, i] = dist
        np.save('data/48h/dwt_distances.npy', distances)
        return None
    
    # paralcomputdistance2()

    distances = np.load('data/48h/dwt_distances.npy')
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(distances)
    cluster_centers = []

    def compute_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)
    
    cluster_centers = Parallel(n_jobs=-1)(\
        delayed(compute_cluster_center)(i, data_list, labels) for i in tqdm(range(num_clusters)))
    shared_start_point = np.array([39.5, -28.1])

    smooth_cluster_centers = []
    for cluster in cluster_centers:
        # 将轨迹的起点调整为共享起点
        cluster = cluster - cluster[0] + shared_start_point
        smooth_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[1]):
            y = cluster[:, i]
            x = np.arange(len(y))
            spl = UnivariateSpline(x, y, k=3)
            smooth_cluster[:, i] = spl(x)

        smooth_cluster_centers.append(smooth_cluster)

    lat_min, lat_max= 20, 50
    lon_min, lon_max = -60, -10
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(smooth_cluster_centers, colors)):
        # 计算聚类中心的经纬度
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        # 找到属于这个聚类中心的簇
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            # 计算簇的经纬度
            trajectory = data_list[index]
            lons, lats = trajectory[:, 1], trajectory[:, 0]
            x, y = map(lons.flatten(), lats.flatten())
            map.plot(x, y, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹
    plt.title('Clustered Trajectories')
    plt.show()
    plt.savefig("pics/DbscanClusterTraj01-03_5.pdf")
    return None 

def KMeansCluster2(folder_path:str,num_clusters:int) -> None:  
    file_list = []
    data_list = []
    start_point = np.array([39.5, -28.1]) 

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        #df, df_outliers = remove_outliers_isolation_forest(df)
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
            interpolated_data = f(np.linspace(0, 1, num_points))
            data_list.append(interpolated_data)

    np.save('data/cublic_data_list.npy', data_list)

    def dtw_distance(pair):
        i, j = pair
        dist, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
        return i, j, dist
    
    def direction_distance(pair):
        i, j = pair
        one_third = len(data_list[i]) 
        sum_cosine_sim = 0.0
        for k in range(one_third):
            vec1 = data_list[i][k]
            vec2 = data_list[j][k]
            sim = 1 - cosine(vec1, vec2)  # 计算余弦相似度
            sum_cosine_sim += sim
        early_cosine_sim = sum_cosine_sim 
        return i, j, early_cosine_sim



    #data_list = np.load('data/TrajGroupsList.npy')
    data_array = np.array(data_list)
    # np.save('data/data_array.npy', data_array)

    dtw_distances = np.zeros((len(data_list), len(data_list)))
    dir_distances = np.zeros((len(data_list), len(data_list)))   

    def paralcomputdistance() -> None:
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        dtw_results = Parallel(n_jobs=-1)(delayed(dtw_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in dtw_results:
            dtw_distances[i, j] = dist
            dtw_distances[j, i] = dist
        #dtw_distances = np.load('data/48h/dtw_distances.npy')
        dir_results = Parallel(n_jobs=-1)(delayed(direction_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in dir_results:
            dir_distances[i, j] = dist
            dir_distances[j, i] = dist
        # dtw_distances = np.load('data/48h/dtw_distances.npy')
        distances = np.concatenate([dtw_distances, dir_distances], axis=-1) # 将两种距离拼接在一起作为总的特征
        np.save('data/48h/distances.npy', distances)
        return None
    
    paralcomputdistance()

    #归一化距离到[0,1]的范围
    scaler = MinMaxScaler()
    dtw_distances = scaler.fit_transform(dtw_distances)
    dir_distances = scaler.fit_transform(dir_distances)
    np.save('data/48h/dtw_distances.npy', dtw_distances)
    np.save('data/48h/dir_distances.npy', dir_distances)


    dtw_distances = np.load('data/48h/dtw_distances.npy')
    dir_distances = np.load('data/48h/dir_distances.npy')
    # 取平均值作为最终的聚类距离
    #final_distances = dtw_distances
    final_distances = 0.7*dtw_distances-(dir_distances*0.3)
    # final_distances = (1/(1.01-dir_distances))*dtw_distances
    np.save('data/48h/final_distances.npy', final_distances)
    final_distances = scaler.fit_transform(final_distances)

    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(final_distances)
    cluster_centers = []

    def compute_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)
    
    cluster_centers = Parallel(n_jobs=-1)(delayed(compute_cluster_center)(i, data_list, labels) for i in tqdm(range(num_clusters)))
    shared_start_point = np.array([39.5, -28.1])

    smooth_cluster_centers = []
    for cluster in cluster_centers:
        # 将轨迹的起点调整为共享起点
        cluster = cluster - cluster[0] + shared_start_point
        smooth_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[1]):
            y = cluster[:, i]
            x = np.arange(len(y))
            spl = UnivariateSpline(x, y, k=3)
            smooth_cluster[:, i] = spl(x)

        smooth_cluster_centers.append(smooth_cluster)

    lat_min, lat_max= 20, 60
    lon_min, lon_max = -75, 0
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    colors = ['g', 'r', 'b', 'r', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(smooth_cluster_centers, colors)):
        # 计算聚类中心的经纬度
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        # 找到属于这个聚类中心的簇
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            # 计算簇的经纬度
            trajectory = data_list[index]
            lons, lats = trajectory[:, 1], trajectory[:, 0]
            x, y = map(lons.flatten(), lats.flatten())
            map.plot(x, y, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹
    specific_lat, specific_lon = 39.500, -28.100

    # 转换坐标
    specific_x, specific_y = map(specific_lon, specific_lat)

    # 绘制特定点
    map.scatter(specific_x, specific_y, color='red', s=5, marker='o', edgecolors='black', alpha=1)

    plt.text(specific_x, -10000, f'Longitude: {specific_lon:.2f}', fontsize=10, color='black', ha='center')
    plt.text(-10000, specific_y, f'Latitude: {specific_lat:.2f}', fontsize=10, color='black', va='center')
    plt.title('Trajectories Cluster')
    plt.show()
    plt.savefig("pics/Keans.pdf")
    return None 

def KMeansCluster3d(folder_path:str,num_clusters:int) -> None:  
    file_list = []
    data_list = []
    start_point = np.array([39.5, -28.1, 0])  # 添加Level的起始点

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon', 'Lev']], axis=0,kind='cubic')  # 添加Level
            interpolated_data = f(np.linspace(0, 1, num_points))
            data_list.append(interpolated_data)

    np.save('data/cublic_data_list.npy', data_list)

    def dtw_distance(pair):
        i, j = pair
        dist, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
        return i, j, dist

    def generate_weights(T, lambda_):
        raw_weights = np.exp(-lambda_ * np.arange(T))
        normalized_weights = raw_weights / np.sum(raw_weights)
        return normalized_weights

    def direction_distance(pair,weights):
        i, j = pair
        one_third = len(data_list[i]) 
        sum_cosine_dist = 0.0
        for k in range(one_third):
            vec1 = data_list[i][k]
            vec2 = data_list[j][k]
            dist = cosine(vec1, vec2)
            sum_cosine_dist += dist*weights[k]
        early_cosine_dist = sum_cosine_dist 
        return i, j, early_cosine_dist

    data_array = np.array(data_list)
    np.save('data/data_array.npy', data_array)
    dtw_distances = np.zeros((len(data_list), len(data_list)))
    dir_distances = np.zeros((len(data_list), len(data_list)))   

    T = 100  # 总的时间步数
    lambda_ = 0.7  # 衰减率
    weights = generate_weights(T, lambda_)

    def paralcomputdistance() -> None:
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        dtw_results = Parallel(n_jobs=-1)(delayed(dtw_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in dtw_results:
            dtw_distances[i, j] = dist
            dtw_distances[j, i] = dist
        dir_results = Parallel(n_jobs=-1)(delayed(direction_distance)(pair,weights) for pair in tqdm(pairs))
        for i, j, dist in dir_results:
            dir_distances[i, j] = dist
            dir_distances[j, i] = dist
        #dtw_distances = np.load('data/48h/dtw_distances.npy')
        distances = np.concatenate([dtw_distances, dir_distances], axis=-1) # 将两种距离拼接在一起作为总的特征
        np.save('data/48h/distances.npy', distances)
        return None
    
    #paralcomputdistance()

    # distances = np.load('data/48h/distances.npy')
    #将DTW距离和方向向量距离分开
    # dtw_distances = distances[:, :len(distances)//2]
    # dir_distances = distances[:, len(distances)//2:]

    #归一化距离到[0,1]的范围
    # scaler = MinMaxScaler()
    # dtw_distances = scaler.fit_transform(dtw_distances)
    # dir_distances = scaler.fit_transform(dir_distances)
    # np.save('data/48h/dtw_distances.npy', dtw_distances)
    # np.save('data/48h/dir_distances.npy', dir_distances)

    dtw_distances = np.load('data/48h/dtw_distances.npy')
    dir_distances = np.load('data/48h/dir_distances.npy')
    # 取平均值作为最终的聚类距离
    final_distances = 0.8 * dtw_distances +  (dir_distances*0.2)

    np.save('data/48h/final_distances.npy', final_distances)

    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(final_distances)
    cluster_centers = []

    def compute_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)
    
    cluster_centers = Parallel(n_jobs=-1)(delayed(compute_cluster_center)(i, data_list, labels) for i in tqdm(range(num_clusters)))
    shared_start_point = np.array([39.5, -28.1, 0])  # 添加Level的起始点

    smooth_cluster_centers = []
    for cluster in cluster_centers:
        # 将轨迹的起点调整为共享起点
        cluster = cluster - cluster[0] + shared_start_point
        smooth_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[1]):
            y = cluster[:, i]
            x = np.arange(len(y))
            spl = UnivariateSpline(x, y, k=3)
            smooth_cluster[:, i] = spl(x)

        smooth_cluster_centers.append(smooth_cluster)

    # 以下部分是在二维地图上绘制轨迹的代码，你可能需要找到一个适合绘制三维轨迹的方法
    lat_min, lat_max= 20, 50
    lon_min, lon_max = -60, -10
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(smooth_cluster_centers, colors)):
        # 计算聚类中心的经纬度
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        # 找到属于这个聚类中心的簇
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            # 计算簇的经纬度
            trajectory = data_list[index]
            lons, lats = trajectory[:, 1], trajectory[:, 0]
            x, y = map(lons.flatten(), lats.flatten())
            map.plot(x, y, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹
    plt.title('Clustered Trajectories')
    plt.show()
    plt.savefig("pics/kmeanscluster2d.pdf")

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(smooth_cluster_centers, colors)):
        # 计算聚类中心的经纬度
        lons, lats, levels = center[:, 1], center[:, 0], center[:, 2]
        ax.plot(lons, lats, levels, color=color, linewidth=2)
        # 找到属于这个聚类中心的簇
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            # 计算簇的经纬度
            trajectory = data_list[index]
            lons, lats, levels = trajectory[:, 1], trajectory[:, 0], trajectory[:, 2]
            ax.plot(lons, lats, levels, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Level')
    plt.title('Clustered Trajectories')
    plt.show()
    plt.savefig("pics/kmeanscluster3d.pdf")
    return None

'''
凝聚层次聚类
'''
def HierarchicalCluster(folder_path:str,num_clusters:int) -> None:  
    file_list = []
    data_list = []
    start_point = np.array([39.5, -28.1]) 
    shared_start_point = np.array([39.5, -28.1])

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
            interpolated_data = f(np.linspace(0, 1, num_points))
            data_list.append(interpolated_data)

    #anima_show(data_list)

    np.save('data/cublic_data_list.npy', data_list)

    def dtw_distance(pair):
        i, j = pair
        dist, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
        return i, j, dist
    
    def paralcomputdistance() -> None:
        data_array = np.array(data_list)
        np.save('data/data_array.npy', data_array)
        distances = np.zeros((len(data_list), len(data_list)))
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        results = Parallel(n_jobs=-1)(delayed(dtw_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in results:
            distances[i, j] = dist
            distances[j, i] = dist
        np.save('data/48h/dwt_distances.npy', distances)
        return None
    
    paralcomputdistance()

    distances = np.load('data/48h/dwt_distances.npy')

    # 替换原有的K均值聚类部分
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    agg_labels = agglomerative.fit_predict(distances)
    agg_cluster_centers = []

    def compute_agg_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)

    agg_cluster_centers = Parallel(n_jobs=-1)(\
        delayed(compute_agg_cluster_center)(i, data_list, agg_labels) for i in tqdm(range(num_clusters)))

    smooth_agg_cluster_centers = []
    for cluster in agg_cluster_centers:
        # 将轨迹的起点调整为共享起点
        cluster = cluster - cluster[0] + shared_start_point
        smooth_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[1]):
            y = cluster[:, i]
            x = np.arange(len(y))
            spl = UnivariateSpline(x, y, k=3)
            smooth_cluster[:, i] = spl(x)

        smooth_agg_cluster_centers.append(smooth_cluster)

    lat_min, lat_max= 20, 50
    lon_min, lon_max = -60, -10
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    
    for i, (center, color) in enumerate(zip(smooth_agg_cluster_centers, colors)):
        # 计算聚类中心的经纬度
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        # 找到属于这个聚类中心的簇
        cluster_indices = np.where(agg_labels == i)[0]
        for index in cluster_indices:
            # 计算簇的经纬度
            trajectory = data_list[index]
            lons, lats = trajectory[:, 1], trajectory[:, 0]
            x, y = map(lons.flatten(), lats.flatten())
            map.plot(x, y, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹
    plt.title('Clustered Trajectories')
    plt.show()
    plt.savefig("pics/HierarchicalCluster.pdf")
    return None 

def DBScanCluster(folder_path:str) -> None:
    file_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    data_list = []
    start_point = np.array([39.5, -28.1]) 

    def update_plot(csv_list):
        fig, ax = plt.subplots()
        num_points = 48
        for csv in csv_list:
            df = pd.read_csv(csv).dropna()
            if not df.empty:
                f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
                interpolated_data = f(np.linspace(0, 1, num_points))
                shift_vector_start = start_point - interpolated_data[0]
                interpolated_data += shift_vector_start

                for i in range(1, num_points+1):
                    ax.scatter(interpolated_data[:i, 0], interpolated_data[:i, 1], label='插值点')
                    ax.plot(interpolated_data[:i, 0], interpolated_data[:i, 1], '-o', label='曲线拟合')
                    ax.plot(start_point[0], start_point[1], 'ro', label='起点')
                    ax.set_title(f'轨迹拟合 - {csv}')
                    ax.set_xlabel('lat')
                    ax.set_ylabel('lon')
                    ax.legend()
                    plt.pause(0.01)
        plt.show()

    # Call the update_plot function with the file_list
    #update_plot(file_list)
    
    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
            interpolated_data = f(np.linspace(0, 1, num_points))
            shift_vector_start = start_point - interpolated_data[0]
            interpolated_data += shift_vector_start
            data_list.append(interpolated_data)

    #anima_show(data_list)

    np.save('data/cublic_data_list.npy', data_list)
        
    data_array = np.array(data_list)
    np.save('data/data_array.npy', data_array)

    distances = np.load('data/48h/dwt_distances.npy')

    def select_MinPts(dist_matrix, k):
        k_dist = []
        for i in range(dist_matrix.shape[0]):
            dist = np.sort(dist_matrix[i])
            k_dist.append(dist[k])
        return np.array(k_dist)
    
    k = 7
    k_dist = select_MinPts(distances, k)
    k_dist.sort()

    plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
    plt.ylabel('K-distance')
    plt.grid(True)
    plt.show()

    eps = k_dist[::-1][15]
    plt.scatter(15,eps,color="r")
    plt.plot([0,15],[eps,eps],linestyle="--",color = "r")
    plt.plot([15,15],[0,eps],linestyle="--",color = "r")
    plt.show()

    dbscan = DBSCAN(eps=900, min_samples=k)
    labels = dbscan.fit_predict(distances)

    def compute_cluster_center(i, data_list, labels):
        return np.mean(np.array([data_list[j] for j in range(len(data_list)) if labels[j] == i]), axis=0)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_centers = Parallel(n_jobs=-1)(delayed(compute_cluster_center)(i, data_list, labels) \
                                          for i in tqdm(range(num_clusters)) if i != -1)

    lat_min, lat_max= 20, 50
    lon_min, lon_max = -60, -10

    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    map.drawcoastlines()
    map.drawcountries()

    shared_start_point = np.array([39.5, -28.1])
    smooth_cluster_centers = []
    for cluster in cluster_centers:
        cluster = cluster - cluster[0] + shared_start_point
        smooth_cluster = np.zeros_like(cluster)
        for i in range(cluster.shape[1]):
            y = cluster[:, i]
            x = np.arange(len(y))
            p = np.polyfit(x, y, 3)
            smooth_cluster[:, i] = np.polyval(p, x)
        smooth_cluster_centers.append(smooth_cluster)

    colors = ['r', 'y', 'b','g', 'r']
    all_trajectories = []

    for i, (center, color) in enumerate(zip(smooth_cluster_centers, colors)):
        lons, lats = center[:, 1], center[:, 0]
        x, y = map(lons, lats)
        map.plot(x, y, color=color, linewidth=2)
        cluster_indices = np.where(labels == i)[0]
        for index in cluster_indices:
            trajectory = data_list[index]
            lons, lats = trajectory[:, 1], trajectory[:, 0]
            x, y = map(lons.flatten(), lats.flatten())
            map.plot(x, y, color=color, linestyle='dashed', linewidth=0.3)  # 使用虚线绘制簇的轨迹

    np.save('data\\DbscanTrajectories.npy', all_trajectories)
    plt.title('Clustered Trajectories')
    plt.savefig("pics/DbscanClusterTraj01-03_52.pdf")
    plt.show()
    return None

def DeepCluster(folder_path:str,num_clusters:int) -> None:  
    file_list = []
    data_list = []
    start_point = np.array([39.5, -28.1]) 

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)

    for csv in file_list:
        df = pd.read_csv(csv).dropna()  
        #df, df_outliers = remove_outliers_isolation_forest(df)
        num_points = 48
        if not df.empty: 
            x = np.linspace(0, 1, len(df))  # 原始数据点的参数
            f = interp1d(np.linspace(0, 1, len(df)), df[['Lat', 'Lon']], axis=0,kind='cubic')
            interpolated_data = f(np.linspace(0, 1, num_points))
            data_list.append(interpolated_data)

    np.save('data/cublic_data_list.npy', data_list)

    def dtw_distance(pair):
        i, j = pair
        dist, _ = fastdtw(data_list[i], data_list[j], dist=euclidean)
        return i, j, dist
    
    # def direction_distance(pair):
    #     i, j = pair
    #     vec1 = data_list[i][-1]
    #     vec2 = data_list[j][-1]
    #     #print(f"Vector1: {vec1}, Vector2: {vec2}")
    #     dist = cosine(vec1, vec2)
    #     return i, j, dist

    def direction_distance(pair):
        i, j = pair
        one_third = len(data_list[i]) // 3
        sum_cosine_dist = 0.0
        for k in range(one_third):
            vec1 = data_list[i][k]
            vec2 = data_list[j][k]
            dist = cosine(vec1, vec2)
            sum_cosine_dist += dist
        early_cosine_dist = sum_cosine_dist / one_third
        return i, j, early_cosine_dist


    data_array = np.array(data_list)
    np.save('data/data_array.npy', data_array)
    dtw_distances = np.zeros((len(data_list), len(data_list)))
    dir_distances = np.zeros((len(data_list), len(data_list)))   

    def paralcomputdistance() -> None:
        pairs = [(i, j) for i in range(len(data_list)) for j in range(i+1, len(data_list))]
        dtw_results = Parallel(n_jobs=-1)(delayed(dtw_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in dtw_results:
            dtw_distances[i, j] = dist
            dtw_distances[j, i] = dist
        dir_results = Parallel(n_jobs=-1)(delayed(direction_distance)(pair) for pair in tqdm(pairs))
        for i, j, dist in dir_results:
            dir_distances[i, j] = dist
            dir_distances[j, i] = dist
        distances = np.concatenate([dtw_distances, dir_distances], axis=-1) # 将两种距离拼接在一起作为总的特征
        np.save('data/48h/distances.npy', distances)
        return None
    
    #paralcomputdistance()

    distances = np.load('data/48h/distances.npy')

    #将DTW距离和方向向量距离分开
    # dtw_distances = distances[:, :len(distances)//2]
    # dir_distances = distances[:, len(distances)//2:]


    #归一化距离到[0,1]的范围
    # scaler = MinMaxScaler()
    # dtw_distances = scaler.fit_transform(dtw_distances)
    # dir_distances = scaler.fit_transform(dir_distances)
    # np.save('data/48h/dtw_distances.npy', dtw_distances)
    # np.save('data/48h/dir_distances.npy', dir_distances)

    return None 


def PltOriginCloud(folder_path:str) -> None:
    draw_traj_basemap(folder_path,'hysplitoritraj.pdf')
    return None

def draw_traj_basemap(folder_path,pdfname:str):
    file_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)
        
    data_list = []
    for csv in file_list:
        df = pd.read_csv(csv).dropna()
        coordinate = df[['Lat', 'Lon']]
        data_list.append(coordinate)
    # 创建Basemap对象
    lat_min, lat_max= 20, 60
    lon_min, lon_max = -75, 0
    map = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    # 绘制海岸线和国家边界线
    map.drawcoastlines()
    map.drawcountries()
    #map.drawmeridians(range(int(lon_min), int(lon_max), space), labels=[0, 0, 0, 1])
    #map.drawparallels(range(int(lat_min), int(lat_max), space), labels=[1, 0, 0, 0])    
    for data in tqdm(data_list):
        longitude = data['Lon'].tolist()
        latitude = data['Lat'].tolist()
        x, y = map(longitude, latitude)
        #sc = map.scatter(x, y, cmap=plt.get_cmap('YlOrRd'), s=0.5, marker='o', edgecolors='black', alpha=0.3, vmin=50, vmax=300)
        map.scatter(x[0], y[0], color='r', s=2, marker='o', edgecolors='r')
        #map.scatter(x[-1], y[-1], color='r', s=2, marker='o', edgecolors='r')
        map.plot(x, y, color='black', alpha=1,linewidth=0.2)
    # 标出特定位置并显示其坐标
    specific_lat, specific_lon = 39.500, -28.100

    # 转换坐标
    specific_x, specific_y = map(specific_lon, specific_lat)

    # 绘制特定点
    map.scatter(specific_x, specific_y, color='red', s=5, marker='o', edgecolors='black', alpha=1)

    plt.text(specific_x, -10000, f'Longitude: {specific_lon:.2f}', fontsize=10, color='black', ha='center')
    plt.text(-10000, specific_y, f'Latitude: {specific_lat:.2f}', fontsize=10, color='black', va='center')
    plt.title('Back Trajectories')
    plt.savefig("pics/"+pdfname)
    plt.close()  # 保存后关闭图形，以避免显示在屏幕上

def Main():
    #HierarchicalCluster('/home/hikaka/mete-clustering/data/48h/1-3m',3)
    #DBScanCluster('data\\48h\\1-3m')
    #KMeansCluster2('data/48h/1-3m',3)
    KMeansCluster2('data/0704/202301csv6h',4)
    # PltOriginCloud('data/0704/202301csv6h')

Main()