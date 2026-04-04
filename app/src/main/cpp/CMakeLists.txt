# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.22.1)

project("game_test")

if(NOT DEFINED OPENCV_ANDROID_SDK OR OPENCV_ANDROID_SDK STREQUAL "")
    message(FATAL_ERROR
        "OPENCV_ANDROID_SDK 未设置。请下载 OpenCV Android SDK 并解压，在工程根目录 local.properties 中添加：\n"
        "opencv.dir=/你的路径/OpenCV-android-sdk\n"
        "（目录内需含 sdk/native/jni/OpenCVConfig.cmake 与 sdk/native/libs）")
endif()

# 统一路径分隔符（Windows）
file(TO_CMAKE_PATH "${OPENCV_ANDROID_SDK}" _ocv_sdk_root)

# 支持「SDK 根目录」或误指到 sdk 子目录
set(_ocv_config_dir "")
if(EXISTS "${_ocv_sdk_root}/sdk/native/jni/OpenCVConfig.cmake")
    set(_ocv_config_dir "${_ocv_sdk_root}/sdk/native/jni")
elseif(EXISTS "${_ocv_sdk_root}/native/jni/OpenCVConfig.cmake")
    set(_ocv_config_dir "${_ocv_sdk_root}/native/jni")
elseif(EXISTS "${_ocv_sdk_root}/OpenCVConfig.cmake")
    set(_ocv_config_dir "${_ocv_sdk_root}")
endif()

if(_ocv_config_dir STREQUAL "")
    message(FATAL_ERROR
        "在 OPENCV_ANDROID_SDK 下未找到 OpenCVConfig.cmake：\n  ${_ocv_sdk_root}\n"
        "请确认解压的是官方 OpenCV Android SDK，且 local.properties 里 opencv.dir 指向 "
        "OpenCV-android-sdk 根目录（内含 sdk/native/jni）。")
endif()

set(OpenCV_DIR "${_ocv_config_dir}")
list(APPEND CMAKE_PREFIX_PATH "${_ocv_config_dir}")

# 必须用 CONFIG：Android 包只提供 OpenCVConfig.cmake，没有 FindOpenCV.cmake
find_package(OpenCV REQUIRED CONFIG)

message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Creates your game shared library. The name must be the same as the
# one used for loading in your Kotlin/Java or AndroidManifest.txt files.
add_library(game_test SHARED
    main.cpp
    AndroidOut.cpp
    Renderer.cpp
    Shader.cpp
    TextureAsset.cpp
    Utility.cpp
    DnnDetector.cpp)

target_include_directories(game_test PRIVATE ${OpenCV_INCLUDE_DIRS})

# Searches for a package provided by the game activity dependency
find_package(game-activity REQUIRED CONFIG)

# Forces the linker to keep the JNI entry point for GameActivity
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u Java_com_google_androidgamesdk_GameActivity_initializeNativeCode")

# Configure libraries CMake uses to link your target library.
target_link_libraries(game_test
    game-activity::game-activity_static
    ${OpenCV_LIBS}
    EGL
    GLESv3
    jnigraphics
    android
    log)
