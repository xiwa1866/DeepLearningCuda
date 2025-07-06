# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(ExampleCUDA LANGUAGES CXX CUDA)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Optional: tune for your GPU compute capability (e.g. 75 for Turing)
# You can list multiple architectures: 60;70;75;80
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable separable compilation if you split into multiple .cu/.cpp files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(fused_op
  fused_op.cu
)

# target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against CUDA runtime (usually automatic)
# target_link_libraries(example PRIVATE CUDA::cudart)

# (Optional) Add definitions for debug/profiling
# target_compile_definitions(example PRIVATE DEBUG)
# target_compile_options(example PRIVATE -G)  # Generate debug info for device

# Installation rules (optional)
# install(TARGETS example RUNTIME DESTINATION bin)
