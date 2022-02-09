#if defined(MK_BLIS)
    #define MR 8
    #define NR 12
#elif defined(MK_8x12)
    #define MR 8
    #define NR 12
#elif defined(MK_4x12)
    #define MR 4
    #define NR 12
#elif defined(MK_4x16)
    #define MR 4
    #define NR 16
#elif defined(MK_4x20)
    #define MR 4
    #define NR 20
#elif defined(MK_7x12_U4) || defined(MK_7x12_U2) || defined(MK_7x12) || defined(MK_7x12_NPA_U4)
    #define MR 7
    #define NR 12
#elif defined(MK_6x12_U4) || defined(MK_6x12_NPA_U4)
    #define MR 6
    #define NR 12
#elif defined(MK_4x4) || defined(MK_GEMM)
    #define MR 4
    #define NR 4
#elif defined(TVM)
    #define MR 7
    #define NR 12
#else
    #define MR 4
    #define NR 4
#endif

