
add_executable(chapter03 chapter03.cpp)
target_link_libraries(chapter03 "${TORCH_LIBRARIES}")
set_property(TARGET chapter03 PROPERTY CXX_STANDARD 11)


find_package(Boost)
#/home/inglada/OTB/SuperBuild/OTB/OPENCV/build
find_package(OpenCV)
if(Boost_FOUND AND OpenCV_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})

  add_executable(dogs-and-cats dogs-and-cats.cpp)
  target_include_directories(dogs-and-cats PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(dogs-and-cats PRIVATE "${TORCH_LIBRARIES}" -lstdc++fs ${OpenCV_LIBS})
  set_property(TARGET dogs-and-cats PROPERTY CXX_STANDARD 17)
endif()
