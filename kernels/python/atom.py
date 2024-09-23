import pandas as pd
def read_atom()->list : 

    atom_tracks = []

    file_path = '/mnt/d/linux/linux/ATom_nav_1613/data/ATom1_flight_tracks.csv'
    data = pd.read_csv(file_path, skiprows=13)
    selected_columns = data[['Latitude', 'Longitude', 'Altitude (m masl GPS)']].values.tolist()
    atom_tracks.append(selected_columns)

    file_path = '/mnt/d/linux/linux/ATom_nav_1613/data/ATom2_flight_tracks.csv'
    data = pd.read_csv(file_path, skiprows=13)
    selected_columns = data[['Latitude', 'Longitude', 'Altitude (m masl GPS)']].values.tolist()
    atom_tracks.append(selected_columns)

    file_path = '/mnt/d/linux/linux/ATom_nav_1613/data/ATom3_flight_tracks.csv'
    data = pd.read_csv(file_path, skiprows=13)
    selected_columns = data[['Latitude', 'Longitude', 'Altitude (m masl GPS)']].values.tolist()
    atom_tracks.append(selected_columns)

    file_path = '/mnt/d/linux/linux/ATom_nav_1613/data/ATom4_flight_tracks.csv'
    data = pd.read_csv(file_path, skiprows=13)
    selected_columns = data[['Latitude', 'Longitude', 'Altitude (m masl GPS)']].values.tolist()
    atom_tracks.append(selected_columns)


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.basemap import Basemap

    # # 假设 two_dimensional_list 是你的二维列表
    # latitudes = [row[0] for row in atom_tracks[0]]
    # longitudes = [row[1] for row in atom_tracks[0]]

    # # 创建一个新的matplotlib图
    # plt.figure(figsize=(10, 8))

    # lat_min, lat_max= min(latitudes), max(latitudes)
    # lon_min, lon_max = min(longitudes),max(longitudes) 
    # #m = Basemap(projection='merc', llcrnrlon=(lon_min), llcrnrlat=(lat_min), urcrnrlon=(lon_max), urcrnrlat=(lat_max), resolution='l')
    # m = Basemap(projection='merc', llcrnrlon=-179, llcrnrlat=-75, urcrnrlon=179, urcrnrlat=85, resolution='l')
    # m = Basemap(width=12000000,height=9000000,projection='lcc',
    #             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    # m.drawcoastlines()
    # m.drawcountries()

    # colors = ['g', 'r', 'b', 'y', 'k']
    # for i, (track, color) in enumerate(zip(atom_tracks, colors)):
    #     latitudes = [row[0] for row in track]
    #     longitudes = [row[1] for row in track]

    #     x, y = m(longitudes, latitudes)
    #     m.plot(x, y, marker='o', markersize=5, color=color, label='Track')


    # m.bluemarble()
    # plt.show()

    # plt.title('Track on Map')
    # plt.legend()
    # plt.savefig('res/pics/atom.pdf')
    # plt.show()
    
    return atom_tracks