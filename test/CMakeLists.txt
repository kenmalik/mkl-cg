include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(cg_unit_tests cg_test.cpp)
target_link_libraries(cg_unit_tests PRIVATE cg GTest::gtest_main)
target_compile_features(cg_unit_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(cg_unit_tests)
