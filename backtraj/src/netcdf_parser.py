# netcdf_parser.py
import netCDF4 as nc
import numpy as np

class Era5Parse():
    def __init__(self,file_path) -> None:
        self.data = nc.Dataset(file_path)
        self.time_sets = self.data.variables['time'][:]
        self.longitude_sets= self.data.variables['longitude'][:]
        self.latitude_sets = self.data.variables['latitude'][:]
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

    def parse_uvwo(self,time,lati,long):
        u_wind = self.parse_variable(time,lati,long,'u')
        v_wind = self.parse_variable(time,lati,long,'v')
        w_wind = self.parse_variable(time,lati,long,'w') #omega
        sh = self.parse_variable(time,lati,long,'q') #sh
        return u_wind,v_wind,w_wind,sh

    def parse_variable(self,time,lati,long,variable) -> float:
        if variable in self.data.variables:
            variable_sets = self.data.variables[variable][:]
        else:
            print(f"Variable '{variable}' not found in data.")

        variable_sets = self.data.variables[variable][:]
        mTimeIndex,mLatIndex,mLonIndex = self.__selectIndex__(time,lati,long)
        variable = variable_sets[mTimeIndex][mLatIndex][mLonIndex]
        return variable
        
    def __selectIndex__(self,mTime:int,mLat,mLon):
        i=0
        for index, value in np.ndenumerate(self.time_sets):        
            if(value == mTime):
                mTimeIndex = i
                continue
            i = i+1 
        mLatIndex = self.__binary_search_latitude__(self.latitude_sets, mLat)
        mLonIndex = self.__binary_search_longitude__(self.longitude_sets, mLon)
        if(mLatIndex == -1 or mLonIndex==-1):
            print("get index error !!!")
            return -1
        return mTimeIndex,mLatIndex,mLonIndex
    
    def __binary_search_latitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.124 <= arr[mid] and arr[mid] <= target+0.124:
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
            if target-0.124 <= arr[mid] and arr[mid] <= target+0.124:
                return mid  # era5数据是逆序的
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1  # 如果没有找到目标，返回-1


# variable = Era5Parse('cpp/era5/2023/omega_500hPa_20230101.nc').parse_variable(1078200.0,23.75,-28.1,'w')
# print(f"the parse variable = {variable}")

