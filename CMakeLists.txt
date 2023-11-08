cmake_minimum_required(VERSION 3.0.2)
project(seg_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_PREFIX_PATH "/opt/ros/noetic")
set(CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/share/catkin/cmake")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  rostime
  cv_bridge
  image_transport
  rospy
  std_msgs
  sensor_msgs
  message_generation
  message_filters
  tf
  tf_conversions
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(jsoncpp REQUIRED)
# find_package(rosbag REQUIRED)

add_message_files(
  FILES
  Masks.msg
  KeypointsWithScores.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES seg_ros
  CATKIN_DEPENDS rospy std_msgs sensor_msgs
  message_runtime
  cv_bridge
  image_transport
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ../../devel/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(save_images src/save_images.cpp)
target_link_libraries(save_images ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(truncator src/truncator.cpp)
target_link_libraries(truncator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} )

add_executable(human_model src/human_model.cpp)
target_link_libraries(human_model ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(off_line src/off_line.cpp)
target_link_libraries(off_line ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} jsoncpp_lib)

add_executable(on_line src/on_line.cpp)
target_link_libraries(on_line ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(tf_sim src/tf_sim.cpp)
target_link_libraries(tf_sim ${catkin_LIBRARIES})