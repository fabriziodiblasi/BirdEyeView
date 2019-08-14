cmake_minimum_required(VERSION 3.1)
project(YOUR_PROJECT_NAME)

set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=gnu11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(app video_reader.cpp)
target_link_libraries(app ${OpenCV_LIBS})

