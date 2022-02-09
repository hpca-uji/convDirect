#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "sutils.h"
#include "qblis.h"
#include "compute_kernel_7x12.h"
#include "compute_kernel_6x12.h"


//Micro-kernels for BLIS
void gemm_microkernel_Cresident_neon_4x4_prefetch_fp32( int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_4x12_fp32(int, int, int, float, float *, float *, float, float *, int);
void gemm_microkernel_Cresident_neon_4x16_fp32(int, int, int, float, float *, float *, float, float *, int);
void gemm_microkernel_Cresident_neon_4x20_fp32(int, int, int, float, float *, float *, float, float *, int);
void gemm_microkernel_Cresident_neon_fixed_4x20_fp32(int, int, int, float, float *, float *, float, float *, int);
void gemm_microkernel_Cresident_neon_8x12_fp32(int, int, int, float, float *, float *, float, float *, int);
void gemm_microkernel_Cresident_neon_fixed_8x12_fp32(int, int, int, float, float *, float *, float, float *, int);

//Micro-kernels for Tze-Meng
void gemm_microkernel_Cresident_neon_7x12_fp32( int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_7x12_unroll_2_fp32( int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32( int, int, int, float, float *, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32_fixed( int, int, int, float, float*, float*, float, float*, int);

//Micro-kernels for SHALOM
void gemm_microkernel_Cresident_neon_7x12_nopack_unroll_4_fp32( int, int, int, float, float *, int, float *, int, float, float *, int );
void gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32( int, int, int, float, float *, int, float *, float, float *, int );
void gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32_fixed( int, int, int, float, float*, int, float*, float, float*, int);
