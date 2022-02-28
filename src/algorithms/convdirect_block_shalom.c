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
#include "../macros.h"
#include "ukrs/gemm_blis_neon_fp32.h"

#ifdef MK_7x12_NPA_U4
#define MR 7
#define NR 12
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
            ho, wo, ii, jj, kk, ib, jb, kb, Cob_Nr = COB / NR;

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
                        for (n = 0; n < min(Hf, Ho - l); n++) {
                            for (m = 0; m < Wf; m++) {
                                for (jr = 0, jr2 = 0; jr < jb; jr += NR, jr2++) {
                                    nr = min(jb - jr, NR);
                                    for (ir = 0; ir < min(kb, Wo - k - m + 1); ir += MR) {
                                        mr = min(min(kb, Wo - k - m + 1) - ir, MR);
                                        /*
                                        gemm_base(mr, nr, ib,
                                                  1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,
                                                  &FBrow_NHWC(j2*Cob_nr+jr2, i, n, m, 0), ldFB4,
                                                  1.0, &Yrow_NHWC(h, j+jr, l, k+ir),     ldY3 );
                                        */
#ifdef MK_7x12_NPA_U4
                                        if ((mr == MR) && (nr == NR))
                                            gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32_fixed(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Drow_NHWC(h, i, l + n, k + ir + m),
                                                    ldD3, // 4
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir),
                                                    ldY3);
                                        else
                                            gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Drow_NHWC(h, i, l + n, k + ir + m),
                                                    ldD3, // 4
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir),
                                                    ldY3);
#else
                                        printf("Error: Microkernel doesn't exist.\n");
                                        exit(EXIT_FAILURE);
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
