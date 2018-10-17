
add_executable(chapter03 chapter03.cpp)
target_link_libraries(chapter03 "${TORCH_LIBRARIES}")
set_property(TARGET chapter03 PROPERTY CXX_STANDARD 11)
