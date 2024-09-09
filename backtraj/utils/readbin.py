import os
import numpy as np

import os
import numpy as np

path = os.listdir('/home/hikaka/mete-clustering/cpp/binary')


def read_dtw_distances(folder_path):
    # 假设有 360 个轨迹，因此初始化一个 360x360 的零矩阵
    dtw_distances = np.zeros((360, 360))

    # 获取目录下所有文件名
    data_list = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in data_list if file.endswith('.bin')]

    # 遍历所有.bin文件
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 第一行是起始点信息，从第二行开始读取距离数据
            distances = [float(line.strip()) for line in lines[2:]]
            if len(distances) != 359:
                raise ValueError(f"Expected 359 distances, but got {len(distances)} in file {file_path}")
            # 因为是正方形矩阵且每个文件只包含一行数据，所以直接赋值
            dtw_distances[i,] = distances

    return dtw_distances

# 使用示例
folder_path = '/home/hikaka/mete-clustering/cpp/out'  # 替换为你的目录路径
dtw_distances = read_dtw_distances(folder_path)
print(dtw_distances)