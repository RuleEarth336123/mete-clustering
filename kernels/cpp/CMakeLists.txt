cmake_minimum_required (VERSION 2.8)
set(CMAKE_C_COMPILER   /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

project (demo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(main dtw.cpp main.cpp)
#add_executable(main tmp.cpp)


