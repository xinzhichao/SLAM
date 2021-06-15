# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 3.19)
# 声明一个 cmake 工程
project(SLAM)
# 设置编译模式
set(CMAKE_BUILD_TYPE "Debug")
find_package(OpenCV REQUIRED)
# 添加可执行程序
add_executable(calibration calibration/Calibration.cpp)
# 将库文件链接到可执行程序上
target_link_libraries(calibration ${OpenCV_LIBS})


#标定文件有两个，接下来的配置是标定文件２的配置文件,主要使用标定１文件
# 添加可执行程序
add_executable(calibration2 calibration/Calibration2.cpp)
# 将库文件链接到可执行程序上
target_link_libraries(calibration2 ${OpenCV_LIBS})