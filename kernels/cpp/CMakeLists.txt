cmake_minimum_required(VERSION 3.22.1)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# 为 Debug 构建类型设置调试选项
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

project(main)

set(NETCDF_INCLUDE_DIRS "/usr/local/include")
set(HTTPCPP_INCLUDE_DIRS "/usr/local/include")
set(JSON11_INCLUDE_DIRS "/usr/include/x86_64-linux-gnu")

set(NETCDF_LIBRARIES "/usr/local/lib/libnetcdf-cxx4.so")
set(JSON11_LIBRARIES "/usr/lib/x86_64-linux-gnu/libjson11.a")




# 添加头文件目录
include_directories(${NETCDF_INCLUDE_DIRS} ${JSON11_INCLUDE_DIRS})

# 添加可执行文件
add_executable(main bt_compute.cpp era5_reader.cpp http_handler.cpp features.cpp main.cpp)

# 链接动态库
target_link_libraries(main ${NETCDF_LIBRARIES} ${JSON11_LIBRARIES})
