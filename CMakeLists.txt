cmake_minimum_required(VERSION 3.10)

project(Dynamic_Object_Detection)

set(OpenCV_DIR E:/HPA_Problem_Statement/opencv/build)

find_package (OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/app.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})