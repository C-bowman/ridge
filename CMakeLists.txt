cmake_minimum_required(VERSION 3.21)
project(pdf_grid)

set(CMAKE_CXX_STANDARD 14)

include_directories(cpp)

add_executable(pdf_grid
        cpp/main.cpp
        cpp/main.h)
