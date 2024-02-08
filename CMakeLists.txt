cmake_minimum_required(VERSION 3.14)
project(linear_solver)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source files to the project
set(SOURCES
    main.cpp
    lineq.cpp
    utils.cpp
)

# Create the executable
add_executable(linear_solver ${SOURCES})

# Optional: Add compiler flags, such as C++ version and optimization level
# For example, uncomment the line below to use C++17
# target_compile_features(linear_solver PUBLIC cxx_std_17)

# Optional: If you're using additional libraries, link them here
# For example, uncomment the line below to link against the Eigen library
# target_link_libraries(linear_solver Eigen3::Eigen)

# Optional: If you have header files in a separate include directory
# include_directories(include)

# Optional: If you have additional dependencies, find and link them here
# find_package(Boost REQUIRED)
# target_link_libraries(linear_solver ${Boost_LIBRARIES})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  tests
  tests.cpp
  lineq.cpp
  utils.cpp
)
target_link_libraries(
  tests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)
