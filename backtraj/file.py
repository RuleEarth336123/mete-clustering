import os
import shutil
import re
def organize_nc_files(root_folder):
    # 遍历root_folder下的所有文件和子文件夹
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            # 检查文件是否以.nc结尾
            if filename.endswith('.nc'):
                # 提取日期部分，假设日期格式为"YYYYMMDD"
                match = re.search(r'(\d{8})\.nc$', filename)
                date_str = match.group(1)
                
                date_str = match.group()
                year_month = date_str[:6]  # 提取年月部分
                day = date_str[6:8]  # 提取日部分
                
                # 创建目标文件夹路径
                target_folder = os.path.join(root_folder, year_month, day)
                
                # 如果目标文件夹不存在，则创建它
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                # 构建源文件和目标文件的完整路径
                src_file = os.path.join(foldername, filename)
                dst_file = os.path.join(target_folder, filename)
                
                # 移动文件到目标文件夹
                shutil.move(src_file, dst_file)
                print(f"Moved {filename} to {target_folder}")

# 使用示例：指定需要整理的文件夹根目录
# root_directory = 'E:/MeteorologicalData/Merra2Aer/backtrajectory-calculator-main/ERA5_data/2023'  # 请替换为实际的根目录路径
# organize_nc_files(root_directory)
