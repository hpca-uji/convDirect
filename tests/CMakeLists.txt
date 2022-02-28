# tests/CMakeLists.txt

add_executable(convDirect_eval)
target_sources(convDirect_eval PRIVATE convDirect_eval.c array_utils.c input_utils.c)
target_link_libraries(convDirect_eval convDirect)
target_include_directories(convDirect_eval PUBLIC "${PROJECT_BINARY_DIR}/src")


add_executable(small_manual_test)
target_sources(small_manual_test PRIVATE small_manual_test.c array_utils.c input_utils.c)
target_link_libraries(small_manual_test convDirect)
target_include_directories(small_manual_test PUBLIC "${PROJECT_BINARY_DIR}/src")

add_custom_target(
        evaluation
        COMMAND cd .. && ${PROJECT_SOURCE_DIR}/tests/convDirect_eval_all.sh
        DEPENDS convDirect_eval
)
