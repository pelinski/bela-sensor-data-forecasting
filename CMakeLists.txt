# Author: Rodrigo Diaz (@rodrigodzf)[https://github.com/rodrigodzf]

cmake_minimum_required(VERSION 3.18)
project(sensor-data-forecasting VERSION 0.0.0)

set(CMAKE_GENERATOR "Unix Makefiles")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

####################################

add_compile_options(-march=armv7-a -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon -Wno-psabi)
add_compile_options(-O3 -g -fPIC -ftree-vectorize -ffast-math)
add_compile_definitions(DXENOMAI_SKIN_posix)

####################################
set(BELA_ROOT "${CMAKE_SYSROOT}/root/Bela") 
set(SYS_ROOT "${CMAKE_SYSROOT}")
set(TENSORFLOW_ROOT /opt/tensorflow )
set(TFLITE_LIB_ROOT /opt/tensorflow/build )
set(FLATBUFFERS_ROOT /opt/flatbuffers-arm32 )

find_library(COBALT_LIB REQUIRED
    NAMES cobalt libcobalt
    HINTS "${CMAKE_SYSROOT}/usr/xenomai/lib"
)

find_library(NEON_LIB REQUIRED
    NAMES NE10 libNE10
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

find_library(MATHNEON_LIB REQUIRED
    NAMES mathneon libmathneon
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

set(ENABLE_TFLITE_FRONTEND ON CACHE BOOL "ENABLE_TFLITE_FRONTEND" FORCE)
set(ENABLE_ARMNN_FRONTEND ON CACHE BOOL "ENABLE_ARMNN_FRONTEND" FORCE)
set(ENABLE_RTNEURAL_FRONTEND OFF CACHE BOOL "ENABLE_RTNEURAL_FRONTEND" FORCE)

if(ENABLE_TFLITE_FRONTEND)
    add_compile_definitions(ENABLE_TFLITE_FRONTEND)
endif()

if(ENABLE_ARMNN_FRONTEND)
    add_compile_definitions(ENABLE_ARMNN_FRONTEND)
endif()

if(ENABLE_RTNEURAL_FRONTEND)
    add_compile_definitions(ENABLE_RTNEURAL_FRONTEND)
endif()

####################################

set(EXE_NAME forecasting)
file(GLOB SRC_FILES *.cpp)
add_executable(${EXE_NAME} ${SRC_FILES})
target_include_directories(${EXE_NAME} PRIVATE ${BELA_ROOT} ${BELA_ROOT}/include)
if (ENABLE_TFLITE_FRONTEND)
    ## Add TfLite dependency
    find_package(TfLiteSrc REQUIRED MODULE)
    find_package(TfLite REQUIRED MODULE)
    ## Add Flatbuffers dependency
    find_package(Flatbuffers REQUIRED MODULE)
    # include/flatbuffers/flatbuffers.h is not warning clean
    # We can't change compilation flags on header files directly, so we need to add them to an interface library first
    add_library(flatbuffer_headers INTERFACE)
    target_include_directories(flatbuffer_headers INTERFACE $<BUILD_INTERFACE:${Flatbuffers_INCLUDE_DIR}>
                                                    $<INSTALL_INTERFACE:include/flatbuffer_headers>)
    target_compile_options(flatbuffer_headers INTERFACE -Wno-sign-conversion)
    add_library(tflite_headers INTERFACE)
    target_include_directories(tflite_headers INTERFACE $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
                                                        $<INSTALL_INTERFACE:include/tflite_headers>)
    target_compile_options(tflite_headers INTERFACE -Wno-conversion
                                                    -Wno-sign-conversion
                                                    -Wno-unused-parameter
                                                    -Wno-unused-function)
endif()
target_link_libraries(${EXE_NAME}
                    PRIVATE
                    ${BELA_ROOT}/lib/libbelafull.so
                    ${COBALT_LIB}
                    ${NEON_LIB}
                    ${MATHNEON_LIB}
                    tflite_headers
                    flatbuffer_headers
                    ${TfLite_LIB}
                    ${Flatbuffers_LIB}
                    dl
                    prussdrv
                    asound
                    atomic
                    sndfile
                    pthread
)

add_custom_command(
    TARGET ${EXE_NAME}
    POST_BUILD
    COMMAND rsync --timeout=10 -P ${CMAKE_BINARY_DIR}/bin/${EXE_NAME} ../model.tflite root@$ENV{BBB_HOSTNAME}:~
)