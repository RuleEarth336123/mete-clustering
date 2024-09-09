# -*- coding: utf-8 -*-

import netCDF4 as nc

# 打开netCDF文件
file = r'E:\\MeteorologicalData\\Merra2Aer\\backtrajectory-calculator-main\\processed_bt\\BT.6h.825hPa.20220310.nc'
dataset = nc.Dataset(file)

# 获取所有变量的名称
all_vars = dataset.variables.keys()

# 遍历所有变量并打印它们的数据
for var_name in all_vars:
    print(f"变量名: {var_name}")
    print(f"数据: {dataset.variables[var_name][:]}")

# 关闭netCDF文件
dataset.close()