import json
import os
import subprocess


with open('json/configuration.json', 'r') as file:
    data = json.load(file)

parameter:dict = {}

for key, value in data['hyst_std'].items():
    parameter[key] = value
    print(f"{key}: {value}")

files = os.listdir(data['hyst_std']["gdas path"])
parameter["number of input grids"] = files.count

process = subprocess.Popen('E:/MeteorologicalData/Merra2Aer/exec/hycs_ens.exe',\
                           stdin = subprocess.PIPE, stdout = subprocess.PIPE)
starting_time = data["hyst_std"]["starting time"]
# 将四个整数转换为字符串
starting_time_str = ' '.join(map(str, starting_time)) + '\n'
process.stdin.write(starting_time_str.encode())

location_str = str(parameter["number of starting locations"]) + '\n'
process.stdin.write(location_str.encode())

location_str = str(parameter["starting location"]) + '\n'
process.stdin.write(location_str.encode())

location_str = str(parameter["total run time"]) + '\n'
process.stdin.write(location_str.encode())

location_str = str(parameter["vertical"]) + '\n'
process.stdin.write(location_str.encode())

location_str = str(parameter["top of model domain"]) + '\n'
process.stdin.write(location_str.encode())

location_str = str(parameter["number of input grids"]) + '\n'
process.stdin.write(location_str.encode())


for file in files:
    path_str = parameter["gdas path"] + '\n'
    process.stdin.write(path_str.encode())
    file_str = file + '\n'
    process.stdin.write(file_str.encode())                                            



process.stdin.flush()