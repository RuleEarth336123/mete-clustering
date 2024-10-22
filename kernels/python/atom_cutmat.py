import scipy.io as sio
from datetime import datetime, timedelta
import csv
import math
import os
class AtomDatas:
    def __init__(self, times, lats, lons, \
        pres=None,CO=None,SO4=None,\
        NO=None,NO2=None,OH=None,SO2=None):
        """
        初始化AtomDatas类的实例。

        :param times: 一个列表，包含时间数据。
        :param lats: 一个列表，包含纬度数据。
        :param lons: 一个列表，包含经度数据。
        """
        self.times = times
        self.lats = lats
        self.lons = lons
        self.pres = pres
        self.CO = CO
        self.SO4 = SO4
        self.NO = NO
        self.NO2 = NO2
        self.OH = OH
        self.SO2 = SO2
        
    def __repr__(self,index):

        return f"AtomDatas(times={self.times[index]}, lats={self.lats[index]}, lons={self.lons[index]})"

class AtomMgdCut() :
    def __init__(self,mat_file_path):
        self.mat_file_path = mat_file_path
        self.lats = self.loadVariable('latitude')
        self.lons = self.loadVariable('longitude')
        for i,_ in enumerate(self.lons):
            self.lons[i] += 360.0
            
        dates = self.loadVariable('date')
        timesecs= self.loadVariable('time')
        self.times = self.__dealtime__(dates,timesecs)
        
        pass
    
    def loadVariable(self,varName:str):
        
        return sio.loadmat(mat_file_path)['data'][0][0]['mgd'][varName][0][0].ravel()
    
    def cutbySite(self,varName:list,st_index,ed_index):
        
        return varName[st_index : ed_index]
    
    def selectIndexbySite(self,lat,lon):
        for index,(latitude,longitude) in enumerate(zip(self.lats,self.lons)):
            if math.isclose(latitude, lat, abs_tol=0.1) and math.isclose(longitude, lon, abs_tol=0.1):
                return index
        print("failed to find index by sites,please check your site message.")
        return None
    
    def cutVariablebyTime(self,varName,time):
        
        return
        
    def __dealtime__(self,datas,times):
        specific_times = []
        
        for data,time in zip(datas,times):
            date_obj = datetime.strptime(str(data), '%Y%m%d')

            # 将时间转换为秒
            time_in_seconds = (time - int(time)) * 86400

            # 分别计算小时、分钟和剩余的秒
            hours = int(time_in_seconds // 3600)
            minutes = int((time_in_seconds % 3600) // 60)
            seconds = time_in_seconds % 60

            # 将秒的小数部分转换为微秒
            microseconds = int((seconds - int(seconds)) * 1e6)

            # 创建timedelta对象，包括小时、分钟、秒和微秒
            specific_time = date_obj + timedelta(hours=hours, minutes=minutes, seconds=int(seconds), microseconds=microseconds)
            specific_times.append(specific_time)        
            #print(specific_time)
        
        
        return specific_times


class SaveFile():
    def __init__(self):
        pass
    
    def toCsv(self, filename, columnName, columnData):
        # 确保列名列表和数据列表的长度匹配
        if len(columnName) != len(columnData):
            raise ValueError("列名的数量必须与数据列的数量匹配")
        
        # 使用 'w' 模式打开文件，如果文件不存在则创建
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # 创建一个csv.writer对象
            writer = csv.writer(csvfile)
            
            # 写入列名
            writer.writerow(columnName)
            
            # 获取行数，即最小的列数据长度
            row_count = min(len(data) for data in columnData)
            
            # 写入数据
            for i in range(row_count):
                # 每一行的数据是所有列的对应元素组成的列表
                row_data = [data[i] if i < len(data) else '' for data in columnData]
                writer.writerow(row_data)

        
# 指定.mat文件的路径
mat_file_path = '/mnt/d/NPFProject/ATom Data MATLAB/result/IOP01/run01@huiping/Atom_01_run01.mat'
obj = AtomMgdCut(mat_file_path)
ed = obj.selectIndexbySite(71.57,311)
st = obj.selectIndexbySite(35.72,333.2)

columnName = ['time','latitude','longitude']
columnData = [obj.times[st:ed],obj.lats[st:ed],obj.lons[st:ed]]


#O3_CL NO2_CL NO_CL SO2_CIT H2O2_CIL HCN_CIT HNO3_CIT
#CN_3nm CN_7nm CN_10nm CN_3to7nm CN_3to10nm

# CO_GMI = obj.loadVariable('CO_GMI')
# CO_GMI_CUT = obj.cutbySite(CO_GMI,st,ed)
varName = 'CN_3nm'
vars = obj.loadVariable(varName)
cuts = obj.cutbySite(vars,st,ed)
columnName.append(varName)
columnData.append(cuts)

varName = 'CN_7nm'
vars = obj.loadVariable(varName)
cuts = obj.cutbySite(vars,st,ed)
columnName.append(varName)
columnData.append(cuts)

varName = 'CN_10nm'
vars = obj.loadVariable(varName)
cuts = obj.cutbySite(vars,st,ed)
columnName.append(varName)
columnData.append(cuts)

varName = 'CN_3to7nm'
vars = obj.loadVariable(varName)
cuts = obj.cutbySite(vars,st,ed)
columnName.append(varName)
columnData.append(cuts)

varName = 'CN_3to10nm'
vars = obj.loadVariable(varName)
cuts = obj.cutbySite(vars,st,ed)
columnName.append(varName)
columnData.append(cuts)


filename = '/mnt/d/linux/Atom_IOP1_01_CN.csv'


f = SaveFile()
f.toCsv(filename,columnName,columnData)

print(1)