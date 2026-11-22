####################################################################################################
#                                         PROJECT INFORMATION                                      #
####################################################################################################
cmake_minimum_required(VERSION 3.5)
project(mujoco_ros2)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()


####################################################################################################
#                                          DEPENDENCIES                                            #
####################################################################################################
find_package(ament_cmake REQUIRED)                                                                  # So ROS2 can compile?
find_package(glfw3 REQUIRED)                                                                        # Graphics Library FrameWork
find_package(rclcpp REQUIRED)                                                                       # ROS2 C++ libraries
find_package(sensor_msgs REQUIRED)                                                                  # sensor_msgs::msg::JointState
find_package(std_msgs REQUIRED)                                                                     # std_msgs::msg::Float64MultiArray
find_package(geometry_msgs REQUIRED)                                                                # geometry_msgs::msg::PoseStamped

# MuJoCo
# MuJoCo (C library). Override with:  colcon build --cmake-args -DMUJOCO_PATH=/path/to/mujoco-3.x.y
# Default: the copy committed in this repository ("Project Files/mujoco-3.13.0" at the repo root,
# or a mujoco-3.13.0 folder next to src/).
if(NOT MUJOCO_PATH)
  foreach(candidate "${CMAKE_CURRENT_SOURCE_DIR}/../mujoco-3.13.0"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../../Project Files/mujoco-3.13.0"
                    "$ENV{HOME}/.mujoco/mujoco-3.13.0")
    if(EXISTS "${candidate}/include/mujoco/mujoco.h")
      set(MUJOCO_PATH "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT EXISTS "${MUJOCO_PATH}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo not found. Pass -DMUJOCO_PATH=/path/to/mujoco-3.x.y")
endif()
message(STATUS "Using MuJoCo from ${MUJOCO_PATH}")
include_directories(include)                                                                        # Local header files
include_directories("${MUJOCO_PATH}/include/mujoco")                                                 # MuJoCo header files
include_directories("${MUJOCO_PATH}/include")                                                        # Additional MuJoCo header files
link_directories("${MUJOCO_PATH}/lib")                                                               # Location of MuJoCo libraries

####################################################################################################
#                                           EXECUTABLES                                            #
####################################################################################################
add_executable(mujoco_node src/nodes/mujoco_node.cpp
                           src/mujoco_ros.cpp)                                                      # Define new executable, list source files
    
target_link_libraries(mujoco_node mujoco glfw)                                                      # Link GLFW libraries to the executable
set_target_properties(mujoco_node PROPERTIES BUILD_RPATH "${MUJOCO_PATH}/lib"
                                             INSTALL_RPATH "${MUJOCO_PATH}/lib")              # find libmujoco.so at run time
                                     
ament_target_dependencies(mujoco_node rclcpp sensor_msgs std_msgs geometry_msgs)                                  # List ROS2 dependencies for this executable


####################################################################################################
#                                           INSTALLATION                                           #
####################################################################################################

install(TARGETS mujoco_node
        DESTINATION lib/${PROJECT_NAME})                                                            # Install so ROS2 can find it

install(DIRECTORY launch
                  model
        DESTINATION share/${PROJECT_NAME})

# Milestone 4: remember where the source is, so the launch file finds the Python nodes
# (src/nodes/*.py run straight from the source tree) no matter which folder colcon ran in.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/source_dir.txt" "${CMAKE_CURRENT_SOURCE_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/source_dir.txt"
        DESTINATION share/${PROJECT_NAME})                                                          # Install so ROS2 can find them
        
ament_package()
