
add_executable(chapter03 chapter03.cpp)
target_link_libraries(chapter03 "${TORCH_LIBRARIES}")
set_property(TARGET chapter03 PROPERTY CXX_STANDARD 11)


find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_executable(dogs-and-cats dogs-and-cats.cpp)
  target_link_libraries(dogs-and-cats "${TORCH_LIBRARIES}" -lstdc++fs)
  set_property(TARGET dogs-and-cats PROPERTY CXX_STANDARD 17)
endif()
