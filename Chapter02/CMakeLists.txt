
include_directories(/usr/include/python3.6)
add_executable(chapter02 chapter02.cpp)
target_link_libraries(chapter02 "${TORCH_LIBRARIES}" python3.5m)
set_property(TARGET chapter02 PROPERTY CXX_STANDARD 11)
