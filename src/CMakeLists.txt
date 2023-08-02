# PRL ROS Node
add_executable(prl_ros_node prl_ros_node.cpp)
target_link_libraries(prl_ros_node 
    ${catkin_LIBRARIES} 
    ${Boost_LIBRARIES}
	Eigen3::Eigen
	Threads::Threads
	Core
	Models
)
target_include_directories(prl_ros_node PUBLIC
    ${catkin_INCLUDE_DIRS}
    ${TASKIT_LIBRARY_DIR}
    ${GF_LIBRARY_DIR}
)

install(TARGETS prl_ros_node DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
add_dependencies(prl_ros_node ${taskit_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})