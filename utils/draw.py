import numpy as np
import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import webbrowser
import branca
from folium.plugins import MousePosition
import pandas as pd
import folium
import os

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
    lon_min, lon_max = -20, 70
    lat_min, lat_max = 0, 60
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
    plt.savefig("pics//"+pdfname)

def draw_traj_googlemap(traj_list:list,clusters:list):
    avg_lat = sum(data[0] for traj in traj_list for data in traj) / sum(len(traj) for traj in traj_list)
    avg_lon = sum(data[1] for traj in traj_list for data in traj) / sum(len(traj) for traj in traj_list)

    # 创建一个地图对象，中心点为所有轨迹的平均经纬度
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9)

    gaode_tiles = 'http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}'
    attribution = 'Map data &copy; <a href="http://www.openstreetmap.org">OpenStreetMap</a> contributors, ' \
                'Imagery © <a href="https://www.amap.com/">高德地图</a>'
    folium.TileLayer(tiles=gaode_tiles, attr=attribution, name='高德地图').add_to(m)
    colormap = branca.colormap.LinearColormap(
        colors=['#00FFFF', '#00FF99', '#CCFF66',  '#CCFF33',  '#FFFF00',  '#FFCC00', '#FF9900', '#FF6600',  '#FF3300', '#FF0000'],  # 蓝-红
        index=[890, 900, 910, 920,925, 930, 835, 940,845, 950],  # 颜色柱的值
        vmax=30,   # 颜色柱最大值
        caption='OVOC'   # 颜色柱备注
    )

    colormap.add_to(m)

    # 添加白色背景 39,28    42,63
    x_rect = folium.Rectangle(bounds=[[39, 63], [42, 63]], fill=True, fill_opacity=1, color='white', weight=0)
    y_rect = folium.Rectangle(bounds=[[39, 28], [39, 63]], fill=True, fill_opacity=1, color='white', weight=0)

    polygon = folium.Polygon(locations=[
        [22.1, 113.7],
        [23.62, 113.7],
        [23.62, 112.5],
        [21.8, 112.5],
        [21.8, 113.7],
        [21.8, 114.7],
        [22.1, 114.7],
    ], fill=True, fill_color='white', fill_opacity=1, color=None)

    polygon1 = folium.Polygon(locations=[
        [23.45, 114.65],
        [23.6, 114.65],
        [23.6, 115.8],
        [23.45, 115.8]
    ], fill=True, fill_color='white', fill_opacity=1, color=None)

    m.add_child(x_rect)
    m.add_child(y_rect)
    m.add_child(polygon)
    m.add_child(polygon1)
    # 以下XY反了，不要在意，即X是Y轴，Y是X轴
    # 绘制x轴
    # x_coords = [data['lon'].min(), data['lon'].max()]
    x_line = folium.PolyLine(locations=[[39, 63], [42, 63]], color='black')
    m.add_child(x_line)

    # folium.Marker([22.2, 113.55], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">22.2°N</div>')).add_to(m)
    # folium.Marker([22.4, 113.55], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">22.4°N</div>')).add_to(m)
    # folium.Marker([22.6, 113.55], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">22.6°N</div>')).add_to(m)
    # y_line = folium.PolyLine(locations=[[39, 63], [42, 63]], color='black')
    # m.add_child(y_line)
    # folium.Marker([22.07, 113.8], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">113.8°E</div>')).add_to(m)
    # folium.Marker([22.07, 114.1], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">114.1°E</div>')).add_to(m)
    # folium.Marker([22.07, 114.4], icon=folium.DivIcon(\
    #     html='<div style="font-size: 17px; display: inline; font-weight: bold;">114.4°E</div>')).add_to(m)

    # 绘制z轴 右边数显
    z_line = folium.PolyLine(locations=[[22.1, 114.6], [22.65, 114.6]], color='black')
    m.add_child(z_line)

    # 绘制c轴
    c_line = folium.PolyLine(locations=[[22.65, 113.7], [22.65, 114.6]], color='black')
    m.add_child(c_line)

    for traj in traj_list:
        folium.PolyLine(traj, color="red", weight=8, opacity=8).add_to(m) 
        m.save('map.html')
    for traj in clusters:
        folium.PolyLine(traj, color="black", weight=1, opacity=1).add_to(m)  
        m.save('map.html')
    webbrowser.open('map.html')
    return None

# draw_traj_basemap('data\\01','oritraj01.pdf')
# print('over')
# data_array1 = np.load('data\\trajectories.npy')
#data_list1 = data_array1.tolist()
data_array2 = np.load('data\\trajectories.npy')
data_list2 = data_array2.tolist()
draw_traj_googlemap(data_list2,data_list2)


