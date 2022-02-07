#ifdef FP32
#define DTYPE float
#elif FP64
#define DTYPE double
#elif INT8
#define DTYPE unsigned int
#endif

#ifdef MK_BLIS
  #if defined(FP32)
    #define GEMM_KERNEL_TYPE  sgemm_ukr_ft
  #else
    #define GEMM_KERNEL_TYPE  dgemm_ukr_ft
  #endif
#endif
