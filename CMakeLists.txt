cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeepLearningWithPytorchCpp)

find_package(Torch REQUIRED)

add_subdirectory(Chapter02)
add_subdirectory(Chapter03)

