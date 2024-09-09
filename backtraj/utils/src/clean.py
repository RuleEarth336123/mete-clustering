import numpy as np
import pandas as pd
def smooth_outliers(data, window_size=3):
    smoothed_data = data.copy()
    outliers = find_outliers(data)

    for idx in outliers:
        start_idx = max(0, idx - window_size + 1)
        end_idx = min(len(data), idx + window_size)

        smoothed_lat = np.mean([point[0] for point in data[start_idx:end_idx]])
        smoothed_lon = np.mean([point[1] for point in data[start_idx:end_idx]])

        smoothed_data[idx] = (smoothed_lat, smoothed_lon)

    return smoothed_data

def find_outliers(data, threshold=0.2):
    outliers = []
    for i in range(len(data)):
        lat = data[i][0]
        lon = data[i][1]

        if i > 0 and i < len(data) - 1:
            prev_lat = data[i - 1][0]
            next_lat = data[i + 1][0]
            prev_lon = data[i - 1][1] 
            next_lon = data[i + 1][1]

            if abs(lat - prev_lat) > threshold and abs(lat - next_lat) > threshold:
                outliers.append(i)
            if abs(lon - prev_lon) > threshold and abs(lon - next_lon) > threshold:
                outliers.append(i)

    return outliers

csv = 'data\\01'
data_array2 = np.load('data\\data_array.npy')
input_data = data_array2.tolist()

smoothed_data = smooth_outliers(input_data)

# 输出清洗后的数据
for point in smoothed_data:
    print(point)
