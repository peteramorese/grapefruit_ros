cmake_minimum_required(VERSION 3.20)
project(grapefruit_ros)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_compile_options(-lstdc++fs)
find_package(catkin REQUIRED COMPONENTS
	roscpp 
	genmsg
	message_generation
	taskit
)

find_package(Boost REQUIRED system filesystem date_time thread)

catkin_package(
	CATKIN_DEPENDS 
		message_runtime
)

add_subdirectory(src)

set(BUILD_GF_EXECUTABLES OFF) # Toggle executables OFF to not pollute
add_subdirectory(grapefruit)
