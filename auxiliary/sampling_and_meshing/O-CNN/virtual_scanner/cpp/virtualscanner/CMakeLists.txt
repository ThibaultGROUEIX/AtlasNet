cmake_minimum_required(VERSION 3.3)

add_definitions(-DCGAL_DISABLE_ROUNDING_MATH_CHECK)

add_executable(virtualscannerexe src/main.cpp)

target_compile_features(virtualscannerexe PRIVATE cxx_auto_type)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenMP REQUIRED)

set_target_properties(virtualscannerexe PROPERTIES
      OUTPUT_NAME "virtualscanner"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_compile_options(virtualscannerexe
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -fPIC -Wno-sign-compare -Wno-uninitialized -frounding-math>
        $<$<CXX_COMPILER_ID:MSVC>: /Wall>)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

target_link_libraries(virtualscannerexe
    PRIVATE
        OpenMP::OpenMP_CXX
        Boost::filesystem
        OCNN::VirtualScanner)

