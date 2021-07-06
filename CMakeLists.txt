cmake_minimum_required(VERSION 3.0)
project(LiePP VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-fconcepts-diagnostics-depth=2")

add_library(liepp INTERFACE)

target_include_directories(liepp
    INTERFACE include
    INTERFACE ${EIGEN_INCLUDE_DIRS}
    INTERFACE ${EIGEN_INCLUDE_DIR}/unsupported
)

target_link_libraries(liepp
    INTERFACE yaml-cpp
)

option(BUILD_TESTS "Build the tests" ON)
if(${BUILD_TESTS})
    enable_testing()
    message("Building the tests.")
    add_subdirectory(test)
endif()