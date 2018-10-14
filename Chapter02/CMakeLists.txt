add_executable(chapter02 chapter02.cpp)
target_link_libraries(chapter02 "${TORCH_LIBRARIES}")
set_property(TARGET chapter02 PROPERTY CXX_STANDARD 11)
