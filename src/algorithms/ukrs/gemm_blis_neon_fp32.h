/* 
 * This file is part of convDirect
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "compute_kernel_7x12.h"
#include "compute_kernel_6x12.h"
#include "compute_kernel_6x16.h"
#include "compute_kernel_4x20.h"


// Micro-kernels for BLIS

void gemm_microkernel_Cresident_assembly_8x12_fixed_fp32(int kc, 
                                                         const float *Ar, const float *Br,
                                                         float *C, int ldC, 
							 void *next_Ar, void *next_Br);

void gemm_microkernel_Cresident_neon_4x4_prefetch_fp32(int, int, int, float, const float *, const float *, float,
                                                       float *, int);

void gemm_microkernel_Cresident_neon_4x12_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_4x16_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_4x20_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_4x20_fixed_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_4x20_fixed_unroll_2_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_8x12_fp32(int, int, int, float, const float *, const float *, float, float *, int, void *, void *);

void gemm_microkernel_Cresident_neon_8x12_fixed_fp32(int, int, int, float, const float *, const float *, float,
                                                     float *, int);


// Micro-kernels for Tze-Meng
void gemm_microkernel_Cresident_neon_7x12_fp32(int, int, int, float, const float *, const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_7x12_unroll_2_fp32(int, int, int, float, const float *, const float *, float,
                                                        float *, int);

void gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32(int, int, int, float, const float *, const float *, float,
                                                        float *, int);

void gemm_microkernel_Cresident_neon_7x12_fixed_unroll_4_fp32(int, int, int, float, const float *, const float *,
                                                              float, float *, int);


// Micro-kernels for SHALOM
void gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32(int, int, int, float, const float *, int,
                                                                const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_7x12_fixed_nopackA_unroll_4_fp32(int, int, int, float, const float *, int,
                                                                      const float *, float, float *, int);

void gemm_microkernel_Cresident_assembly_7x12_fixed_nopackA_unroll_4_fp32(int, int, int, float, const float *, int,
                                                                      const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_6x16_nopackA_unroll_4_fp32(int, int, int, float, const float *, int,
                                                                const float *, float, float *, int);

void gemm_microkernel_Cresident_neon_6x16_fixed_nopackA_unroll_4_fp32(int, int, int, float, const float *, int,
                                                                const float *, float, float *, int);
