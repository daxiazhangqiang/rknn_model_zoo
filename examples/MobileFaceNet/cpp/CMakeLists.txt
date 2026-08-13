cmake_minimum_required(VERSION 3.10)

project(rknn_mobileface_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/mobileface_common.cc
)

# 注册可执行文件（带 --uid）
add_executable(rknn_mobileface_register
    main_register.cc
    ${COMMON_SRCS}
)

# 识别可执行文件（无 --uid）
add_executable(rknn_mobileface_recognize
    main.cc
    ${COMMON_SRCS}
)

foreach(tgt rknn_mobileface_register rknn_mobileface_recognize)
    target_link_libraries(${tgt}
        fileutils
        imageutils
        imagedrawing
        ${LIBRKNNRT}
        dl
    )

    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        target_link_libraries(${tgt} log)
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${tgt} Threads::Threads)
    endif()

    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNRT_INCLUDES}
    )
endforeach()

install(TARGETS rknn_mobileface_register rknn_mobileface_recognize DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test.jpg DESTINATION model)

set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/mobilefacenet.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()
