cmake_minimum_required(VERSION 2.8)

project(Net C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/src)
file(GLOB NET_SRC ${CMAKE_CURRENT_LIST_DIR}/include/*.h
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
set(NET_COMPILE_CODE ${NET_SRC})
add_executable(Net ${NET_COMPILE_CODE})
