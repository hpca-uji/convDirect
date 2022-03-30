/**
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

#define ALGORITHM convdirect_block_shalom

#include <stdlib.h>
#include <stdio.h>
#include "algorithms.h"
#include "arm_neon.h"
#include "../macros.h"
#include "ukrs/gemm_blis_neon_fp32.h"

#define Crow(a1, a2)  Cptr[ (a1)*(ldC)+(a2) ]
#define Arow(a1, a2)  Ar[ (a1)*(ldA)+(a2) ]
#define Atrow(a1, a2) Atmp[ (a1)*(ldAt)+(a2) ]

#ifdef MK_7x12_NPA_U4
#define MR 7
#define NR 12
#elif MK_6x16_NPA_U4
#define MR 6
#define NR 16
#endif

#define WOB 1575
#define COB 2052
#define CIB 292

// #undef TENSOR_FORMAT_NCHW /* @todo: borrar */

convdirect_bs_t BLOCK_SIZES = {MR, NR, WOB, COB, CIB};

void TRANSFORM_FILTER(int Ci, int Co,
                      int Hf, int Wf,
                      const DTYPE *F,
                      DTYPE *FB) {

    int i, j, jj, jb, j2, m, n;

    SET_Co_NR;
    SET_F_LEADING_DIMENSIONS;
    SET_FB_LEADING_DIMENSIONS;

#if TENSOR_FORMAT_NCHW
    printf("Tensor format NCHW not yet implemented in transform_filter_block_shalom!\n");
    exit(EXIT_FAILURE);
#else
    for (j = 0, j2 = 0; j < Co; j += NR, j2++) {
        jb = min(Co - j, NR);
        for (i = 0; i < Ci; i++)
            for (n = 0; n < Hf; n++)
                for (m = 0; m < Wf; m++)
                    for (jj = 0; jj < jb; jj++) {
                        FBrow_NHWC(j2, i, n, m, jj) = Frow_NHWC(j + jj, i, n, m);
                    }
    }
#endif
}


void CONVDIRECT_PRE_WITH_PARAMS {
    // DT
    *DT = (DTYPE *) D;
    // FT (FB)
    SET_Co_NR;
    SET_Co_MR;
    *FT = (DTYPE *) malloc(Hf * Wf * Co_NR * Ci * NR * sizeof(DTYPE));
    TRANSFORM_FILTER(Ci, Co, Hf, Wf, F, *FT);
    // YT
    *YT = (DTYPE *) Y;
}


void CONVDIRECT_POST_WITH_PARAMS {
    free(*FT);
}


void CONVDIRECT_KERNEL_WITH_PARAMS {

    QUICK_RETURN_IF_POSSIBLE;

    // Loops reordered as in "High Performance Zero-Memory Overhead Direct Convolution" by J. Zhang et al., 2018
    // Accommodate vectorization: j as the innermost loop
    // Ensure sufficient independent operations: k around j
    // For compatibility between output layer n and input layer n+1: n->m->i

    int h, i, j, k, l, m, n, i2, j2,
            ho, wo, ib, jb, kb, Cob_Nr = COB / NR;

    /*
    float32x4_t C00, C01, C02, 
        C10, C11, C12, 
        C20, C21, C22, 
        C30, C31, C32, 
        C40, C41, C42, 
        C50, C51, C52, 
        C60, C61, C62, 
	A0, A1, A2, A3, A4, A5, A6,
	B0, B1, B2;
    */
    int kr, baseB = 0, ldCt = NR, Amr, Bnr, ldAt = 4, ldBt = NR;
    float32x4_t 
            C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    int jr, nr, jr2, ir, mr;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    SET_LEADING_DIMENSIONS;
    SET_Co_NR;
    SET_FB_LEADING_DIMENSIONS;

    // What matrices are DT, FT, and YT?
    const DTYPE *D = DT, *FB = FT;
    DTYPE *Y = YT;

#if TENSOR_FORMAT_NCHW
    printf("Tensor format NCHW not yet implemented in convDirect_block_shalom!\n");
    exit(EXIT_FAILURE);
#else
    for (h = 0; h < t; h++) {
        for (j = 0, j2 = 0; j < Co; j += COB, j2++) {
            jb = min(Co - j, COB);
            for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
                ib = min(Ci - i, CIB);
                for (l = 0; l < ho; l++) {
                    for (k = 0; k < wo; k += WOB) {
                        kb = min(wo - k, WOB);

			// Reordered loops
                        for (jr = 0, jr2 = 0; jr < jb; jr += NR, jr2++) {
                            nr = min(jb - jr, NR);
                            for (ir = 0; ir < min(kb, Wo - k - Wf + 1); ir += MR) {

                                for (m = 0; m < Wf; m++) {
                                    mr = min(min(kb, Wo - k - m + 1) - ir, MR);
                                    if ((mr == MR) && (nr == NR) && (ib % 4 == 0)) {
                                        DTYPE *Cptr = &Yrow_NHWC(h, j + jr, l, k + ir);
                                        uint64_t uldC = ldY3;
#ifdef MK_7x12_NPA_U4
                                        #include "ukrs/load_7x12_asm.c"
#elif MK_6x16_NPA_U4
                                        #include "ukrs/load_6x16_asm.c"
#else
                                        printf("Error: Microkernel doesn't exist.\n");
                                        exit(EXIT_FAILURE);
#endif
                                        for (n = 0; n < min(Hf, Ho - l); n++) {
					    float *Ar = &Drow_NHWC(h, i, l + n, k + ir + m);
					    float *Br = &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0);

					    uint64_t ukc  = ib;
                                            uint64_t uldA = ldD3;
#ifdef MK_7x12_NPA_U4
                                            #include "ukrs/micro_7x12_asm_unroll_4.c"
#elif MK_6x16_NPA_U4
                                            #include "ukrs/micro_6x16_asm_unroll_4.c"
#endif
                                        }

#ifdef MK_7x12_NPA_U4
                                        #include "ukrs/store_7x12_asm.c"
#elif MK_6x16_NPA_U4
                                        #include "ukrs/store_6x16_asm.c"
#endif
                                    }
                                    else {
#ifdef MK_7x12_NPA_U4
                                        if ((mr == MR) && (nr == NR)) 
                                            for (n = 0; n < min(Hf, Ho - l); n++) {
                                                gemm_microkernel_Cresident_neon_7x12_fixed_nopackA_unroll_4_fp32(
                                                        mr, nr, ib,
                                                        alpha,
                                                        &Drow_NHWC(h, i, l + n, k + ir + m),
                                                        ldD3, // 4
                                                        &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                        beta,
                                                        &Yrow_NHWC(h, j + jr, l, k + ir),
                                                        ldY3);
                                            }
					else
                                            for (n = 0; n < min(Hf, Ho - l); n++) {
                                                gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32(
                                                        mr, nr, ib,
                                                        alpha,
                                                        &Drow_NHWC(h, i, l + n, k + ir + m),
                                                        ldD3, 
                                                        &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                        beta,
                                                        &Yrow_NHWC(h, j + jr, l, k + ir),
                                                        ldY3);
                                            }
#elif MK_6x16_NPA_U4
                                        if ((mr == MR) && (nr == NR)) 
                                            for (n = 0; n < min(Hf, Ho - l); n++) {
                                                gemm_microkernel_Cresident_neon_6x16_fixed_nopackA_unroll_4_fp32(
                                                            mr, nr, ib,
                                                            alpha,
                                                            &Drow_NHWC(h, i, l + n, k + ir + m),
                                                            ldD3, 
                                                            &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                            beta,
                                                            &Yrow_NHWC(h, j + jr, l, k + ir),
                                                            ldY3);
                                            }
					else
                                            for (n = 0; n < min(Hf, Ho - l); n++) {
                                                gemm_microkernel_Cresident_neon_6x16_nopackA_unroll_4_fp32(
                                                            mr, nr, ib,
                                                            alpha,
                                                            &Drow_NHWC(h, i, l + n, k + ir + m),
                                                            ldD3, 
                                                            &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                            beta,
                                                            &Yrow_NHWC(h, j + jr, l, k + ir),
                                                            ldY3);
                                            }
#endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif
}

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM

