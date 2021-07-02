cmake_minimum_required(VERSION 3.0)
project(LiePP VERSION 1.0)

set(liepp_HEADER_FILES
    include/liepp/SO3.h
    include/liepp/SE3.h
    include/liepp/SOT3.h
    include/liepp/SEn3.h
)

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