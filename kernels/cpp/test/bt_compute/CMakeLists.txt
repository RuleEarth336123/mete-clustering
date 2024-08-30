cmake_minimum_required(VERSION 3.12)
project(BT)
set(CMAKE_BUILD_TYPE Debug)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

# 使用变量设置编译标志
if(COMPILER_SUPPORTS_CXX11)
# -fsanitize=address -fno-omit-frame-pointer 与valgrind冲突, 且会导致内存增长
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fsanitize=address -fno-omit-frame-pointer")
 #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_VERSION: ${Python_INCLUDE_DIRS}")
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS} 
    )
# 生成目标文件
add_executable(bt_compute bt_compute.cpp)
# 链接库
target_link_libraries(bt_compute ${Python_LIBRARIES})