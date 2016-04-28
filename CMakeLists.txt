PROJECT(BoostComputeTuto)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}")

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})

# find Boost
find_package(Boost 1.48 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/compute/include)

set(EXAMPLES
  helloworld
  devices
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
    if(APPLE)
      target_link_libraries(${EXAMPLE} "-framework OpenGL")
    elseif(UNIX)
      target_link_libraries(${EXAMPLE} GL)
    endif()
endforeach()
