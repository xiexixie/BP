cmake_minimum_required(VERSION 3.20)
project(BPNET)

set(CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(SRC main.cpp 
${PROJECT_SOURCE_DIR}/src/net.cpp
${PROJECT_SOURCE_DIR}/src/tools.cpp
)

add_executable(bp ${SRC})