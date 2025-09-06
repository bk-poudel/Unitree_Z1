cmake_minimum_required(VERSION 3.10)
project(unitree_arm_interface)

set(CMAKE_CXX_STANDARD 14)

# Set Eigen path manually
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")

# Include Unitree SDK headers
include_directories(${PROJECT_SOURCE_DIR}/../include)

# Include Eigen headers
include_directories(${EIGEN3_INCLUDE_DIR})

# Link the directory that contains libZ1_SDK_x86_64.so
link_directories(${PROJECT_SOURCE_DIR}/../lib)

# Get pybind11 include flags
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDE_FLAGS)

add_library(unitree_arm_interface MODULE
    arm_python_interface.cpp
)

target_compile_options(unitree_arm_interface PRIVATE ${PYBIND11_INCLUDE_FLAGS})

# ✅ Link to correct architecture-specific library
target_link_libraries(unitree_arm_interface PRIVATE Z1_SDK_x86_64)

