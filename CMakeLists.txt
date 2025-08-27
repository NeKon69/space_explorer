cmake_minimum_required(VERSION 3.20)

project(space_explorer VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
find_package(SDL3 REQUIRED CONFIG)
find_package(glad REQUIRED CONFIG)
find_package(Freetype REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
#I so unbelievably dont care, I dont wanna add each file my own, vim sucks for that
file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE include include/smartPointers)

target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>

        $<$<CONFIG:Debug>:-g>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0>

        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0>

        $<$<CONFIG:Release>:-O3>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/Ox>

        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:GNU>>:-O2 -g>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:Clang>>:-O2 -g>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:MSVC>>:/O2 /Zi>

        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:GNU>>:-Os>
        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:Clang>>:-Os>
        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:MSVC>>:/Os>
)

set(RUNTIME_OUTPUT_DIR $<TARGET_FILE_DIR:${PROJECT_NAME}>)

set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
if (EXISTS "${ASSETS_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${ASSETS_DIR}" "${RUNTIME_OUTPUT_DIR}/assets" VERBATIM)
else ()
    message(WARNING "assets directory not found: ${ASSETS_DIR}")
endif ()

set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
if (EXISTS "${SHADERS_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${SHADERS_DIR}" "${RUNTIME_OUTPUT_DIR}/shaders" VERBATIM)
else ()
    message(WARNING "shaders directory not found: ${SHADERS_DIR}")
endif ()
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 glad::glad freetype assimp::assimp CUDA::toolkit)