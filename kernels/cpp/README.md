**后向轨迹计算**

*相关函数*
1.'compute_new_location'
其作用是根据风速数据和时间步长来计算一个点（在气象学中通常指空气颗粒或轨迹点）的新位置和压力水平。函数接收三个参数：

1. `Point& point`：一个`Point`结构体的引用，代表当前点的位置和压力水平。这里使用引用是为了避免复制整个结构体，提高效率。
2. `Wind& wind_data`：一个`Wind`结构体的引用，包含当前点的风速数据（东向风速`u_wind`、北向风速`v_wind`和垂直风速`w_wind`）。
3. `double delta_t`：时间步长，表示在计算新位置时要考虑的时间间隔，单位为小时。

函数内部的实现步骤如下：

- `const double degrees_to_radians = M_PI / 180.0;`：定义一个常量，用于将角度从度转换为弧度

- `double pressure_change = wind_data.w_wind * delta_t;`：计算压力水平的变化。这里假设垂直风速（`w_wind`，单位为Pa/s）每1 Pa/s 导致1 hPa的压力变化，这个假设可能需要根据实际物理模型进行调整。

- `point.level -= pressure_change;`：更新点的压力水平，减去压力变化量。

- `double dx = wind_data.u_wind * delta_t * 1000 * cos(point.latitude * degrees_to_radians);`：计算由于东向风速引起的经度变化量。这里将风速（m/s）转换为米，乘以时间步长（秒），并考虑当前纬度的余弦值，因为地球是球形的。

- `double dy = wind_data.v_wind * delta_t * 1000;`：计算由于北向风速引起的纬度变化量。

- `point.latitude -= dy / earth_radius * degrees_to_radians;`：更新纬度，减去由于北向风速引起的纬度变化量。这里将变化量除以地球半径，并转换为度数。

- `point.longitude -= dx / (earth_radius * cos(point.latitude * degrees_to_radians)) * degrees_to_radians;`：更新经度，减去由于东向风速引起的经度变化量。这里考虑了纬度变化对地球周长的影响。

- 接下来的`if`语句用于调整经度范围，确保经度值在-180到180度之间。

请注意，这个函数中的`earth_radius`变量没有在代码片段中定义，它应该在函数外部定义，并设置为地球的平均半径，大约为6371公里。此外，`Point`和`Wind`结构体需要在函数使用前定义，包含相应的成员变量。