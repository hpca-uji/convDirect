# src/convdirect_algorithms CMakeLists.txt

# Arguments to append_SOURCE_with_UKRS
# 1. Source file name.
# 2. CMake list of microkernels (list of items separated by semicolons and the whole list between quotation marks).


append_SOURCE_with_UKRS(convdirect_original.c "DEFAULT")

append_SOURCE_with_UKRS(convdirect_renamed.c "DEFAULT")

append_SOURCE_with_UKRS(convdirect_reorder.c "DEFAULT")

append_SOURCE_with_UKRS(convdirect_block.c "DEFAULT")

if (BLIS_FOUND)
    list(APPEND CD_SOURCES algorithms/packrb.c)
    list(APPEND CD_SOURCES algorithms/buffers.c)
    list(APPEND CD_SOURCES algorithms/im2row.c)

    append_SOURCE_with_UKRS(convdirect_im2row.c "DEFAULT")  # Uses bli_sgemm

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        append_SOURCE_with_UKRS(convdirect_block_blis.c "BLIS;8x12;4x12;4x16;4x20;4x4")
    else ()
        append_SOURCE_with_UKRS(convdirect_block_blis.c "BLIS")
    endif ()

    append_SOURCE_with_UKRS(convdirect_conv_gemm.c "DEFAULT")
endif ()

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    list(APPEND CD_SOURCES algorithms/ukrs/gemm_blis_neon_fp32.c)
    append_SOURCE_with_UKRS(convdirect_tzemeng.c "7x12_U4")
    append_SOURCE_with_UKRS(convdirect_block_shalom.c "7x12_NPA_U4;6x16_NPA_U4")
endif ()
