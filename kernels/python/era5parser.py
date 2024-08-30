# netcdf_parser.py
import netCDF4 as nc
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message1)s - %(message2)s - %(message3)s')
logger = logging.getLogger(__name__)
class Era5Parse():
    def __init__(self,file_path) -> None:
        self.data = nc.Dataset(file_path)
        self.time_sets = self.data.variables['time'][:]
        self.longitude_sets= self.data.variables['longitude'][:]
        self.latitude_sets = self.data.variables['latitude'][:]
        self.level_sets = self.data.variables['level'][:]
        self.u_sets = self.data.variables['u'][:]
        self.v_sets = self.data.variables['v'][:]
        self.w_sets = self.data.variables['w'][:]
        return
    
    def list_variable(file_path) ->list:
        data = nc.Dataset(file_path)
        variables = data.variables.keys()
        print(list(variables))
        print("Variables in the dataset:")
        for var in data.variables:
            print(var)
            print("\nVariable details:")
        for var_name, var in data.variables.items():
            print(f"Name: {var_name}")
            print(f"Dimensions: {var.dimensions}")
            print(f"Shape: {var.shape}")
            print(f"Data type: {var.datatype}")
            print(f"Attributes: {var.ncattrs()}")
            for attr in var.ncattrs():
                print(f"    {attr}: {var.getncattr(attr)}")
            print()
        return list(variables)

    def parse_uvw(self,time,lev,lat,lon):
        u_wind = self.parse_variable(time,lev,lat,lon,'u')
        v_wind = self.parse_variable(time,lev,lat,lon,'v')
        w_wind = self.parse_variable(time,lev,lat,lon,'w') #omega
        return u_wind,v_wind,w_wind

    def parse_variable(self,time,lev,lat,lon,variable) -> float:
        if variable in self.data.variables:
            variable_sets = self.data.variables[variable][:]
        else:
            print(f"Variable '{variable}' not found in data.")

        variable_sets = self.data.variables[variable][:]
        mTimeIndex,mLevIndex,mLatIndex,mLonIndex = self.__selectIndex__(time,lev,lat,lon)
        variable = variable_sets[mTimeIndex][mLevIndex][mLatIndex][mLonIndex]
        return variable
        
    def __selectIndex__(self,mTime,mLev,mLat,mLon):
        mTimeIndex = mTime
        mLevIndex = None
        min_diff = float('inf')  # 初始化为无穷大的差异值

        for index, value in enumerate(self.level_sets):
            diff = abs(value - mLev)  # 计算当前值与mLev的差异
            if diff < min_diff:
                min_diff = diff
                mLevIndex = index
                
        if mLevIndex is None:
            mLevIndex = index  # 这里index是最后一个元素的索引，即最接近mLev的值的索引     
        
        mLatIndex = self.__binary_search_latitude__(self.latitude_sets, mLat)
        mLonIndex = self.__binary_search_longitude__(self.longitude_sets, mLon)

        if(mLatIndex == -1 or mLonIndex==-1 or mLevIndex==-1 or mTimeIndex==-1 or mLevIndex==-1):
            print(f"get index error !!!  = mLatIndex:{mLatIndex} mLonIndex:{mLonIndex} mLevIndex:{mLevIndex} \
                mTimeIndex:{mTimeIndex} mLevIndex:{mLevIndex}")
            print("get index error !!!")
            return -1
        return mTimeIndex,mLevIndex,mLatIndex,mLonIndex
    
    def __binary_search_latitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.125 <= arr[mid] and arr[mid] <= target+0.125:
                return mid  # era5数据是逆序的
            elif arr[mid] < target:
                high = mid - 1
            else:
                low = mid + 1
        return -1  
    
    def __binary_search_longitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.125 <= arr[mid] and arr[mid] <= target+0.125:
                return mid  # era5数据是逆序的
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1  # 如果没有找到目标，返回-1


# obj = Era5Parse('/mnt/d/学习资料/气象数据/era5s/2022/20221228.nc')
# variable = obj.parse_variable(1078104.0,1000.0,23.75,-28.1,'w')#-0.001
# print(f"the parse variable = {variable}")

# variable = obj.parse_variable(1078104.0,1000.0,27.0,-29.5,'w')#-0.006
# print(f"the parse variable = {variable}")
