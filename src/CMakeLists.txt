# src CMakeLists.txt

set(CD_SOURCES "")

# ========================================================================
# Macro called from src subdirectories to set convDirect sources
# ------------------------------------------------------------------------
macro(append_SOURCE_with_UKRS SOURCE UKRS)
    get_filename_component(PREFIX ${SOURCE} NAME_WE)
    foreach (TF IN ITEMS NHWC;NCHW)
        string(TOLOWER ${TF} tf)
        foreach (UKR IN ITEMS ${UKRS})
            string(TOLOWER ${UKR} ukr)
            set(OBJECT_NAME ${PREFIX}_${tf}_${ukr})
            # Add object (not an actual library)
            add_library(${OBJECT_NAME} OBJECT ${SOURCE})
            target_compile_options(${OBJECT_NAME} PRIVATE -DTENSOR_FORMAT_${TF} -DMK_${UKR} -DMK_NAME=${ukr})
            list(APPEND CD_SOURCES $<TARGET_OBJECTS:${OBJECT_NAME}>)
            list(APPEND CONVDIRECT_NAMES "${OBJECT_NAME}")
            MATH(EXPR N "${N}+1")
        endforeach ()
    endforeach ()
    set(CD_SOURCES "${CD_SOURCES}" PARENT_SCOPE)
    set(N ${N} PARENT_SCOPE)
    set(CONVDIRECT_NAMES "${CONVDIRECT_NAMES}" PARENT_SCOPE)
endmacro()
# ========================================================================


# ========================================================================
# convDirect library
# ------------------------------------------------------------------------
add_subdirectory(algorithms)

if (BLIS_FOUND)
    # Sources that are not under 'algorithms/' must be added here
    list(APPEND CD_SOURCES
            convGemmNHWC/src/gemm_blis.c;
            convGemmNHWC/src/im2col_nchw.c;
            convGemmNHWC/src/gemm_blis_B3A2C0.c;
            convGemmNHWC/src/im2row_nhwc.c;
            convGemmNHWC/src/gemm_blis_B3A2C0_orig.c)
endif ()

add_library(convDirect SHARED)
target_sources(convDirect PUBLIC convdirect.c ${CD_SOURCES} gemm_reference.c)
target_link_libraries(convDirect PUBLIC OpenMP::OpenMP_C m)
if (BLIS_FOUND)
    target_link_libraries(convDirect PUBLIC ${BLIS_LIBRARY})
endif ()
target_include_directories(convDirect PUBLIC "${PROJECT_BINARY_DIR}/src")

configure_file(convdirect.h.in convdirect.h)

foreach (name IN ITEMS ${CONVDIRECT_NAMES})
    list(APPEND CONVDIRECT_N "\"${name}\"")
    list(APPEND CONVDIRECT_BS "&${name}_block_sizes")
    list(APPEND CONVDIRECT_F "&${name}")
    list(APPEND CONVDIRECT_PRE_F "&${name}_pre")
    list(APPEND CONVDIRECT_KERNEL_F "&${name}_kernel")
    list(APPEND CONVDIRECT_POST_F "&${name}_post")
    list(APPEND CONVDIRECT_EXTERN "extern convdirect_bs_t ${name}_block_sizes;")
    list(APPEND CONVDIRECT_EXTERN "extern convdirect_ft ${name};")
    list(APPEND CONVDIRECT_EXTERN "extern convdirect_pre_ft ${name}_pre;")
    list(APPEND CONVDIRECT_EXTERN "extern convdirect_kernel_ft ${name}_kernel;")
    list(APPEND CONVDIRECT_EXTERN "extern convdirect_post_ft ${name}_post;")
endforeach ()
STRING(JOIN ",\n" CONVDIRECT_N ${CONVDIRECT_N})
STRING(JOIN ",\n" CONVDIRECT_BS ${CONVDIRECT_BS})
STRING(JOIN ",\n" CONVDIRECT_F ${CONVDIRECT_F})
STRING(JOIN ",\n" CONVDIRECT_PRE_F ${CONVDIRECT_PRE_F})
STRING(JOIN ",\n" CONVDIRECT_KERNEL_F ${CONVDIRECT_KERNEL_F})
STRING(JOIN ",\n" CONVDIRECT_POST_F ${CONVDIRECT_POST_F})
STRING(JOIN ";\n" CONVDIRECT_EXTERN ${CONVDIRECT_EXTERN})
configure_file(convdirect_options.h.in convdirect_options.h)
# ========================================================================


# ========================================================================
# convDirect_info executable
# ------------------------------------------------------------------------
add_executable(convDirect_info convDirect_info.c)
target_link_libraries(convDirect_info convDirect)
# ========================================================================


# ========================================================================
# Install convDirect library nnd header, and convDirect_info executable
# ------------------------------------------------------------------------
install(TARGETS convDirect DESTINATION lib)
install(FILES "${PROJECT_BINARY_DIR}/src/convdirect.h" DESTINATION include)
install(TARGETS convDirect_info DESTINATION bin)
# ========================================================================

