cmake_minimum_required(VERSION 3.20)

project(space_explorer VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SDL3 REQUIRED CONFIG)
find_package(glad REQUIRED CONFIG)


file(GLOB_RECURSE SOURCES src/*.c src/*.cpp tests/src/*.c tests/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)

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
set(SOURCE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
if(EXISTS "${SOURCE_SHADER_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${SOURCE_SHADER_DIR}" "${RUNTIME_OUTPUT_DIR}/shaders" VERBATIM)
else()
    message(WARNING "shaders directory not found: ${SOURCE_SHADER_DIR}")
endif()


target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 glad::glad)