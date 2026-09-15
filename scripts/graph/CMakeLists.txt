# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(groute_graph_e2e LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(kvikio CONFIG REQUIRED)

add_executable(groute_graph_e2e groute_graph_e2e.cu)
target_compile_features(groute_graph_e2e PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(groute_graph_e2e PRIVATE kvikio::kvikio CUDA::cudart)
set_target_properties(groute_graph_e2e PROPERTIES CUDA_ARCHITECTURES native)
