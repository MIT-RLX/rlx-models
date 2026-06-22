cmake_minimum_required(VERSION 3.20)
project(tsac-ng VERSION 0.1.0 LANGUAGES C CXX)

# ============================================================
# Architecture detection
# ============================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_ARM64 ON)
    # NEON is default on ARM64; enable auto-vec
    add_compile_options(-O3 -ffast-math -march=armv8-a+simd)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(ARCH_RISCV ON)
    # For RVV support, add -march=rv64gcv (requires RVV-capable toolchain)
    add_compile_options(-O3)
else()
    # x86-64 — enable all SIMD levels for runtime dispatch
    # AVX2 + AVX-512F/VL/BW compiled in, selected at runtime via CPUID
    set(ARCH_X86_64 ON)
    add_compile_options(-O3 -ffast-math -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw)
endif()

# ============================================================
# Options
# ============================================================
option(USE_CUDA   "Enable NVIDIA CUDA backend"   OFF)
option(USE_HIP    "Enable AMD ROCm/HIP backend"  OFF)
option(USE_VULKAN "Enable Vulkan compute backend" OFF)
option(USE_LLVM   "Enable LLVM JIT backend (experimental)" OFF)

# ============================================================
# Compile flags
# ============================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# ============================================================
# Sources — common to all backends
# ============================================================
set(TSAC_SOURCES
    src/main.c
    src/tsac_codec.c
    src/tsac_transformer.c
    src/tsac_normal_decode.c
    src/dac_model.c
    src/txc_format.c
    src/model_loader.c
    src/cpu_decoder.c
    src/range_coder.c
)

# Architecture-specific CPU sources
if(ARCH_ARM64)
    list(APPEND TSAC_SOURCES src/arch/arm/cpu_arm.c)
elseif(ARCH_RISCV)
    list(APPEND TSAC_SOURCES src/arch/riscv/cpu_riscv.c)
endif()

# Backend stubs (when GPU backend is disabled)
if(NOT USE_CUDA)
    list(APPEND TSAC_SOURCES src/cuda_stubs.c)
endif()
if(NOT USE_HIP)
    list(APPEND TSAC_SOURCES src/hip_stubs.c)
endif()
if(NOT USE_VULKAN)
    list(APPEND TSAC_SOURCES src/vulkan_stubs.c)
endif()
if(NOT USE_LLVM)
    list(APPEND TSAC_SOURCES src/llvm_stubs.c)
endif()

set(TSAC_HEADERS
    include/tsac.h
    include/libnc_arch.h
    src/tsac_codec.h
    src/txc_format.h
    src/dac_model.h
    src/model_loader.h
    src/range_coder.h
    src/cuda/nc_kernels.h
)

# ============================================================
# CUDA backend
# ============================================================
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")

    list(APPEND TSAC_SOURCES
        src/cuda/cuda_kernels.cu
        src/cuda/cuda_backend.cu
    )

    add_compile_definitions(USE_CUDA)
    message(STATUS "CUDA: ${CUDAToolkit_ROOT}")
endif()

# ============================================================
# HIP/ROCm backend
# ============================================================
if(USE_HIP)
    enable_language(HIP)
    set(HIP_PATH "/opt/rocm" CACHE PATH "HIP installation path")
    list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip")
    find_package(hip REQUIRED)

    set(HIP_SOURCE
        hip/hip_arch.hip.cpp
        hip/hip_kernels.hip.cpp
        hip/dac_decoder.hip.cpp
    )
    list(APPEND TSAC_SOURCES ${HIP_SOURCE})
    add_compile_definitions(USE_HIP)

    set_source_files_properties(${HIP_SOURCE} PROPERTIES LANGUAGE HIP)

    set(CMAKE_HIP_ARCHITECTURES "gfx1036" CACHE STRING "HIP architectures")

    message(STATUS "HIP: ${HIP_PATH}")
endif()

# ============================================================
# Vulkan compute backend
# ============================================================
if(USE_VULKAN)
    # Vulkan headers only (no lib linkage — loaded via dlopen at runtime)
    find_path(Vulkan_INCLUDE_DIRS vulkan/vulkan.h PATHS /usr/include
              /usr/local/include /usr/aarch64-linux-gnu/include NO_DEFAULT_PATH)
    if(NOT Vulkan_INCLUDE_DIRS)
        find_path(Vulkan_INCLUDE_DIRS vulkan/vulkan.h)
    endif()
    if(NOT Vulkan_INCLUDE_DIRS)
        message(WARNING "vulkan.h not found — USE_VULKAN will not include vulkan backend")
    else()
        list(APPEND TSAC_SOURCES src/vulkan/vulkan_arch.c)
        add_compile_definitions(USE_VULKAN)
    endif()

    message(STATUS "Vulkan: ${Vulkan_INCLUDE_DIRS}")
endif()

# ============================================================
# LLVM JIT backend (experimental)
# ============================================================
if(USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)
    list(APPEND TSAC_SOURCES src/llvm/llvm_backend.c)
    add_compile_definitions(USE_LLVM)
    message(STATUS "LLVM: ${LLVM_DIR}")
endif()

# ============================================================
# Targets
# ============================================================
# Shared library
add_library(tsac-ng-lib SHARED ${TSAC_SOURCES} ${TSAC_HEADERS})
target_include_directories(tsac-ng-lib
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan
)
target_link_libraries(tsac-ng-lib PRIVATE dl pthread m)

if(LIBNC_LIBRARY)
    target_link_libraries(tsac-ng-lib PRIVATE ${LIBNC_LIBRARY})
endif()
if(USE_CUDA)
    target_include_directories(tsac-ng-lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(tsac-ng-lib PRIVATE CUDA::cudart)
    set_property(TARGET tsac-ng-lib PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(USE_HIP)
    target_include_directories(tsac-ng-lib PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(tsac-ng-lib PRIVATE hip::host)
    set_property(TARGET tsac-ng-lib PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()
if(USE_VULKAN)
    target_include_directories(tsac-ng-lib PRIVATE ${Vulkan_INCLUDE_DIRS})
    # No libvulkan linkage — loaded at runtime via dlopen
endif()
if(USE_LLVM)
    target_include_directories(tsac-ng-lib PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(tsac-ng-lib PRIVATE LLVM)
endif()

set_target_properties(tsac-ng-lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME tsac-ng
)

# Executable
add_executable(tsac-ng src/main.c)
target_link_libraries(tsac-ng PRIVATE tsac-ng-lib)

# ============================================================
# Cross-compilation support
# ============================================================
# ARM64: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake ..
# RISC-V: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake ..

if(ARCH_ARM64)
    message(STATUS "Target: ARM64 NEON")
elseif(ARCH_RISCV)
    message(STATUS "Target: RISC-V (experimental)")
else()
    message(STATUS "Target: x86-64 AVX+FMA baseline")
endif()
message(STATUS "Backends: CPU (baseline) ${USE_CUDA}${USE_HIP}${USE_VULKAN}")
