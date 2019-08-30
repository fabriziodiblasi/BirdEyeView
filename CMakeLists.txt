cmake_minimum_required(VERSION 3.1)
project(YOUR_PROJECT_NAME)

#set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=gnu11")


find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(app src/main.cu src/utilities.cu src/cuda_remapping.cu)

target_link_libraries(app ${OpenCV_LIBS})

