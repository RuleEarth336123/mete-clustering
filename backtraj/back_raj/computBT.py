import numpy as np
import pandas as pd

# 假设我们有以下函数来从数据集中获取风速分量和比湿
def get_wind_and_sh(lat, lon, time, data):
    uwnd = data[('uwnd', time)][lat, lon]
    vwnd = data[('vwnd', time)][lat, lon]
    sh = data[('sh', time)][lat, lon]
    return uwnd, vwnd, sh

# 欧拉法数值积分计算后向轨迹
def euler_integration(lat, lon, data, time_step=6, days_back=365):
    u, v, sh = get_wind_and_sh(lat, lon, pd.Timestamp('2022-01-01'), data)
    for i in range(int(24/time_step * days_back)):
        # 计算后向位移
        dx = u * time_step * 60 * 60 / 1000 # 转换m/s到度/小时，假设地球半径为1000km
        dy = v * time_step * 60 * 60 / 1000
        
        # 更新位置
        lon -= np.cos(np.radians(lat)) * dx
        lat -= dy
        
        # 准备下一次迭代
        u, v, sh = get_wind_and_sh(lat, lon, pd.Timestamp('2022-01-01') - pd.Timedelta(hours=time_step*(i+1)), data)
    
    return lat, lon

# 假设data是我们的气象数据集字典
# data = {
#     ('uwnd', pd.Timestamp('2022-01-01')): uwnd_data,
#     ('vwnd', pd.Timestamp('2022-01-01')): vwnd_data,
#     # ... 其他时间的数据 ...
# }

# 假设我们从某个特定位置开始
start_lat = 40.0  # 示例纬度
start_lon = -100.0  # 示例经度

# 计算后向轨迹
final_lat, final_lon = euler_integration(start_lat, start_lon, data)