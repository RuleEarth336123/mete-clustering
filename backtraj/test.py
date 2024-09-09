import xarray as xr
import numpy as np

# 定义 NetCDF 文件列表
nc_files = [
    'backtraj/data/omega_500hPa_20221220.nc',
    'backtraj/data/omega_500hPa_20221221.nc',
    'backtraj/data/omega_500hPa_20221222.nc'
]

# 读取所有 NetCDF 文件到一个 xarray.Dataset 列表中
datasets = [xr.open_dataset(file) for file in nc_files]

# 确保所有文件具有相同的维度
base_ds = datasets[0]

# 获取第一个数据变量
data_var_name = list(base_ds.data_vars.keys())[0]
data_var = base_ds[data_var_name]

# 获取坐标变量
latitude = base_ds['latitude']
longitude = base_ds['longitude']
time = base_ds['time']

# 创建一个空的 DataArray 来存储合并后的数据
combined_data = xr.DataArray(
    np.empty((len(nc_files), len(time), len(latitude), len(longitude)), dtype=data_var.dtype),
    dims=('day', 'time', 'latitude', 'longitude'),
    coords={'day': range(len(nc_files)),
            'time': time,
            'latitude': latitude,
            'longitude': longitude}
)

# 将每个文件的数据放入新的 DataArray 中
for i, ds in enumerate(datasets):
    combined_data[i, ...] = ds[data_var_name].values

# 将合并后的数据转换为 DataSet
combined_ds = xr.Dataset({data_var_name: combined_data})

# 保存合并后的数据到一个新的 NetCDF 文件
combined_ds.to_netcdf('backtraj/data/combined_omega_500hPa_with_day.nc')

print('NetCDF 文件合并并添加维度完成！')
