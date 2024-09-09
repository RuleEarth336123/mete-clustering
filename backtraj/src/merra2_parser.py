# netcdf_parser.py
import netCDF4 as nc
import numpy as np

class Merra2Parser():
    def __init__(self,file_path) -> None:
        try:
            self.data = nc.Dataset(file_path)
            self.time_sets = self.data.variables['time'][:]
            self.longitude_sets= self.data.variables['lon'][:]
            self.latitude_sets = self.data.variables['lat'][:]
            self.level_sets = self.data.variables['lev'][:]
            self.u_wind_sets = self.data.variables['U'][:]
            self.v_wind_sets = self.data.variables['V'][:]
            self.w_wind_sets = self.data.variables['OMEGA'][:]
        except Exception as e:
            print(f"An error occurred while init parser...: {e}")
        return
    
    def __del__(self) -> None:
        if self.data is not None:
            self.data.close()
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

    def parse_uvw(self,time,lev,lat,lon,variable):
        try:
            if variable == 'U':
                u_wind = self.parse_variable(time,lev,lat,lon,'U')
                return u_wind
            elif variable == 'V':    
                v_wind = self.parse_variable(time,lev,lat,lon,'V')
                return v_wind
            elif variable == 'W':
                w_wind = self.parse_variable(time,lev,lat,lon,'OMEGA') #omega
            return w_wind
        except Exception as e:
            print(f"An error occurred while parser_uvw...: {e}")

        return -999

    def parse_uvw2(self,time,lev,lat,lon):
        try:
            u_wind = self.parse_variable(time,lev,lat,lon,'U')  
            v_wind = self.parse_variable(time,lev,lat,lon,'V')
            w_wind = self.parse_variable(time,lev,lat,lon,'OMEGA') #omega

        except Exception as e:
            print(f"An error occurred while parser_uvw...: {e}")

        return u_wind,v_wind,w_wind

    def parse_variable(self,time,lev,lat,lon,variable) -> float:
        mTimeIndex,mLevelIndex,mLatIndex,mLonIndex = self.__selectIndex__(time,lev,lat,lon)
        ret = -9999
        if variable == 'U':
            ret = self.u_wind_sets[mTimeIndex][mLevelIndex][mLatIndex][mLonIndex]
        elif variable == 'V':
            ret = self.v_wind_sets[mTimeIndex][mLevelIndex][mLatIndex][mLonIndex]
        elif variable == 'OMEGA':
            ret = self.w_wind_sets[mTimeIndex][mLevelIndex][mLatIndex][mLonIndex]
        else:
            print(f"Variable '{variable}' not found in data.")
            print(f"error location Variable index is {mTimeIndex}  {mLevelIndex}  {mLatIndex} {mLonIndex}")
        
        return ret
        
    def __selectIndex__(self,mTime,mLev,mLat,mLon):
        # i=0
        # for index, value in np.ndenumerate(self.time_sets):        
        #     if(value == mTime):
        #         mTimeIndex = i
        #         break
        #     i = i+1 
        mTimeIndex = self.__search__time__(mTime)
        mLatIndex = self.__binary_search_latitude__(self.latitude_sets, mLat)
        mLonIndex = self.__binary_search_longitude__(self.longitude_sets, mLon)
        mLevIndex = self.__search_level__(mLev)
        if(mLatIndex == -1 or mLonIndex==-1):
            print("get index error !!!")
            return -1
        return mTimeIndex,mLevIndex,mLatIndex,mLonIndex
    
    def __binary_search_latitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.25 <= arr[mid] and arr[mid] <= target+0.25:
                return mid  # lon数据是顺序的
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1 
    
    def __binary_search_longitude__(self,arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if target-0.4 <= arr[mid] and arr[mid] <= target+0.4:
                return mid  # lon数据是顺序的
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1  
    
    def __search__time__(self,target):
        time_pack = [0.0,180.0,360.0,540.0,720.0,900.0,1080.0,1260.0]
        for index, value in enumerate(time_pack):
            if (value - 90) <= target < (value + 90):
                return index
            #print(f"{index}: {value}")
        return -1  
    
    def __search_level__(self,target):
        # 定义字符包列表
        character_pack = [
            1000.00, 975.000, 950.000, 925.000, 900.000, 875.000, 850.000, 825.000,
            800.000, 775.000, 750.000, 725.000, 700.000, 650.000, 600.000, 550.000,
            500.000, 450.000, 400.000, 350.000, 300.000, 250.000, 200.000, 150.000,
            100.000, 70.0000, 50.0000, 40.0000, 30.0000, 20.0000, 10.0000, 7.00000,
            5.00000, 4.00000, 3.00000, 2.00000, 1.00000, 0.700000, 0.500000, 0.400000,
            0.300000, 0.100000
        ]

        for index in range(len(character_pack) - 1):
                if character_pack[index] >= target >= character_pack[index + 1]:
                    if target >= (character_pack[index] + character_pack[index + 1]) / 2:
                        return index
                    else:
                        return index + 1
        return -1  

# variable = Merra2Parser('cpp/merra2/MERRA2_400.inst3_3d_asm_Np.20230101.nc4').parse_uvw2(0.0,551.83936,23.74983,-34.68256)
# print(f"the parse variable = {variable}")

