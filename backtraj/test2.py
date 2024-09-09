import xarray as xr
import numpy as np
import os
import pandas as pd
import cdsapi
# 定义数据存储路径
directory = '/mnt/d/学习资料/气象数据/era5s/'  # 请替换为实际的文件路径

variables = {'uwnd': 'u_component_of_wind',
             'vwnd': 'v_component_of_wind',
             'sh': 'specific_humidity',
             'omega': 'vertical_velocity'}
field_era = ['u_component_of_wind','v_component_of_wind','specific_humidity','vertical_velocity']
level = [1000,950,900,850,800,750,700,\
650,600,550,500,450,400,350,300,250,200,\
150,100,70,50,30,20,10,7,5,3,2,1]
level_vector = [f'{i}' for i in level]

time_range = pd.date_range('2022-12-28', '2023-02-01', freq='D')

def create_download_path(path):
    # function to create a new directory path
    import os
    os.makedirs(path, exist_ok=True)

create_download_path(directory)

time_range = pd.date_range('2022-12-28', '2023-02-01', freq='D')

# define the geographic limits for downloading the information
lat_min = 20
lat_max = 60
lon_min = -75
lon_max = 0

delta_t = 1

time_vector = [f'{i:02}:00' for i in list(range(0, 24, delta_t))]

for i, date in enumerate(time_range):
    
    file_i = f'{date.strftime("%Y%m%d")}.nc'
    print(file_i)
    path_i = f'{directory}{date.strftime("%Y")}/'
    create_download_path(path_i)
    if not os.path.exists(f'{path_i}{file_i}'):
        print("\tDownloading...")
        c = cdsapi.Client()
                        # specify the parameters for the data download
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': field_era,
                'pressure_level': level_vector,
                'year': f'{date.strftime("%Y")}',
                'month': f'{date.strftime("%m")}',
                'day': f'{date.strftime("%d")}',
                'time': time_vector,
                'area': [lat_max, lon_min, lat_min, lon_max],
            },
            # specify the path to save the downloaded file
            f'{path_i}{file_i}')





