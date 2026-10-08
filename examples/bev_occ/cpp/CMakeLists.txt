cmake_minimum_required(VERSION 3.10)

project(rknn_bev_occ_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 可视化开关: 正式发布不需要可视化时可 -DENABLE_VISUALIZATION=OFF
option(ENABLE_VISUALIZATION "build with visualization output" ON)
if (ENABLE_VISUALIZATION)
    add_definitions(-DENABLE_VISUALIZATION=1)
else()
    add_definitions(-DENABLE_VISUALIZATION=0)
endif()
message(STATUS "ENABLE_VISUALIZATION=${ENABLE_VISUALIZATION}")

if (NOT (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x" OR TARGET_SOC STREQUAL "rk3576"))
    message(WARNING "bev_occ demo only supports NPU2 targets (rk3588/rk356x/rk3576), current: ${TARGET_SOC}")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

add_executable(${PROJECT_NAME}
    main.cc
    bev_occ.cc
    rknpu2/bev_occ.cc
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        log
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
