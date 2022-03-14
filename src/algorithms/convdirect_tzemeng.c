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

#define ALGORITHM convdirect_tzemeng

#include <stdlib.h>
#include <stdio.h>
#include "algorithms.h"
#include "../macros.h"
#include "ukrs/gemm_blis_neon_fp32.h"

#ifdef MK_7x12_U4
#define MR 7
#define NR 12
#endif

#define WOB MR
#define COB NR
#define CIB NR

convdirect_bs_t BLOCK_SIZES = {MR, NR, WOB, COB, CIB};


#ifdef TVM
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#endif


void TRANSFORM_INPUT(int t, int Ci,
                     int Ho, int Wo,
                     int Hf, int Wf,
                     const DTYPE *D,
                     DTYPE *DT) {

    SET_D_LEADING_DIMENSIONS;
    SET_Ci_CIB;
    SET_DT_LEADING_DIMENSIONS;

    int h, i, j,
            k, l,
            m, n,
            ho, wo,
            ii, jj, kk,
            ib, jb, kb;

    int i2, x;

    if ((t == 0) || (Ci == 0) ||
        (Ho == 0) || (Wo == 0) ||
        (Hf == 0) || (Wf == 0))
        return;

#ifdef TENSOR_FORMAT_NCHW
    for (h = 0; h < t; h++)
        for (l = 0; l < Ho; l++)
            for (k = 0; k < Wo; k++)
                for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
                    ib = min(Ci - i, CIB);
                    for (ii = 0; ii < ib; ii++)
                        DT(h, i2, l, k, ii) = Drow_NCHW(h, i + ii, l, k);
                }
#else
    for (h = 0; h < t; h++)
        for (l = 0; l < Ho; l++)
            for (k = 0; k < Wo; k++)
                for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
                    ib = min(Ci - i, CIB);
                    for (ii = 0; ii < ib; ii++)
                        DT(h, i2, l, k, ii) = Drow_NHWC(h, i + ii, l, k);
                }
#endif
}


void TRANSFORM_FILTER(int Ci, int Co,
                      int Hf, int Wf,
                      const DTYPE *F,
                      DTYPE *FT) {

    int h, i, j,
            k, l,
            m, n,
            ho, wo,
            ii, jj, kk,
            ib, jb, kb;

    int i2, j2;

    SET_Co_COB;
    SET_F_LEADING_DIMENSIONS;
    SET_FT_LEADING_DIMENSIONS;

#ifdef TENSOR_FORMAT_NCHW
    for (j = 0, j2 = 0; j < Co; j += COB, j2++) {
        jb = min(Co - j, COB);
        for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
            ib = min(Ci - i, CIB);
            for (n = 0; n < Hf; n++)
                for (m = 0; m < Wf; m++)
                    for (ii = 0; ii < ib; ii++)
                        for (jj = 0; jj < jb; jj++)
                            FT(i2, j2, n, m, ii, jj) = Frow_NCHW(j + jj, i + ii, n, m);
        }
    }
#else
    for (j = 0, j2 = 0; j < Co; j += COB, j2++) {
        jb = min(Co - j, COB);
        for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
            ib = min(Ci - i, CIB);
            for (n = 0; n < Hf; n++)
                for (m = 0; m < Wf; m++)
                    for (ii = 0; ii < ib; ii++)
                        for (jj = 0; jj < jb; jj++)
                            FT(i2, j2, n, m, ii, jj) = Frow_NHWC(j + jj, i + ii, n, m);
        }
    }
#endif
}


void TRANSFORM_OUTPUT(int t, int Co,
                      int Ho, int Wo,
                      int Hf, int Wf,
                      const DTYPE *YT,
                      DTYPE *Y) {

    int h, i, j,
            k, l,
            m, n,
            ho, wo,
            ii, jj, kk,
            ib, jb, kb;

    int i2;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    SET_Y_LEADING_DIMENSIONS;
    SET_Co_COB;
    SET_YT_LEADING_DIMENSIONS;

#ifdef TENSOR_FORMAT_NCHW
    for (h = 0; h < t; h++)
        for (l = 0; l < ho; l++)
            for (k = 0; k < wo; k++)
                for (i = 0, i2 = 0; i < Co; i += COB, i2++) {
                    ib = min(Co - i, COB);
                    for (ii = 0; ii < ib; ii++)
                        Yrow_NCHW(h, i + ii, l, k) = YT(h, i2, l, k, ii);
                }
#else
    for (h = 0; h < t; h++)
        for (l = 0; l < ho; l++)
            for (k = 0; k < wo; k++)
                for (i = 0, i2 = 0; i < Co; i += COB, i2++) {
                    ib = min(Co - i, COB);
                    for (ii = 0; ii < ib; ii++)
                        Yrow_NHWC(h, i + ii, l, k) = YT(h, i2, l, k, ii);
                }
#endif
}

#ifdef TVM

void _TFMK(convDirect_block_tzemeng)(int t, int Co, int Ci,
                                     int Ho, int Wo,
                                     int Hf, int Wf,
                                     DTYPE alpha,
                                     DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
                                     DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
                                     DTYPE beta,
                                     DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
                                     tvm::runtime::PackedFunc tvm_f,
                                     DLTensor *A, DLTensor *B, DLTensor *C,
                                     int CIB, int COB, int WOB) {
    int h,
            i, j, i2, j2,
            k, l,
            m, n,
            ho, wo,
            ii, jj, kk,
            ib, jb, kb, ob;

    //int n_if  = 0;
    //int n_else = 0;
    //int o;

    // Quick return if possible
    if ((t == 0) || (Co == 0) || (Ci == 0) ||
        (Ho == 0) || (Wo == 0) ||
        (Hf == 0) || (Wf == 0))
        return;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    // Loops reordered as in "High Performance Zero-Memory Overhead Direct Convolution" by J. Zhang et al., 2018
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
                                // int mr=kb=WOB, int nr=jb=COB, int kc=ib=CIB
                                ob = min(kb, Wo - k - m + 1);
                                if ((ob == MR) && (jb == NR) && (ib == NR)) { // ib=kc=NR
                                    A->data = &DT(h, i2, l + n, k + m, 0);
                                    B->data = &FT(i2, j2, n, m, 0, 0);
                                    C->data = &YT(h, j2, l, k, 0);
                                    tvm_f(A, B, C, C);
                                } else {
                                    if ((ob == MR) && (jb == NR))
                                        gemm_microkernel_Cresident_neon_7x12_fixed_unroll_4_fp32(
                                                ob, jb, ib,
                                                1.0,
                                                &DT(h, i2, l + n, k + m, 0),
                                                &FT(i2, j2, n, m, 0, 0),
                                                1.0,
                                                &YT(h, j2, l, k, 0),
                                                ldYT4);
                                    else
                                        gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32(
                                                ob, jb, ib,
                                                1.0,
                                                &DT(h, i2, l + n, k + m, 0),
                                                &FT(i2, j2, n, m, 0, 0),
                                                1.0,
                                                &YT(h, j2, l, k, 0),
                                                ldYT4);
                                    /* gemm_microkernel_Cresident_neon_4x4_fp32( ob, jb, ib, */
                                    /* 					    1.0, &DT(h, i2, l+n, k+m, 0),  */
                                    /* 					    &FT(i2, j2, n, m, 0, 0),  */
                                    /* 					    1.0, &YT(h, j2, l, k, 0), ldYT4 ); */
                                }

                            }
                        }
                    }
                }
            }
        }
    }

}

#else //!TVM

void CONVDIRECT_PRE(CONVDIRECT_PRE_PARAMS) {

    int ho, wo;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    SET_LEADING_DIMENSIONS;
    SET_Ci_CIB;
    SET_DT_LEADING_DIMENSIONS;
    *DT = (DTYPE *) malloc(t * Ci_CIB * Ho * Wo * CIB * sizeof(DTYPE));
    TRANSFORM_INPUT(t, Ci, Ho, Wo, Hf, Wf, D, *DT);

    SET_Co_COB;
    SET_FT_LEADING_DIMENSIONS;
    *FT = (DTYPE *) malloc(Ci_CIB * Co_COB * Hf * Wf * CIB * COB * sizeof(DTYPE));
    TRANSFORM_FILTER(Ci, Co, Hf, Wf, F, *FT);

    SET_YT_LEADING_DIMENSIONS;
    *YT = (DTYPE *) malloc(t * Co_COB * ho * wo * COB * sizeof(DTYPE));
}

void CONVDIRECT_POST(CONVDIRECT_POST_PARAMS) {
    TRANSFORM_OUTPUT(t, Co, Ho, Wo, Hf, Wf, *YT, Y);
    free(*DT);
    free(*FT);
    free(*YT);
}

void CONVDIRECT_KERNEL_WITH_PARAMS {

    QUICK_RETURN_IF_POSSIBLE;

    int h, i, j,
            k, l,
            m, n,
            ho, wo,
            ii, jj, kk,
            ib, jb, kb, ob;

    int i2, j2;

    int n_if = 0;
    int n_else = 0;
    int o;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    SET_LEADING_DIMENSIONS;
    SET_Ci_CIB;
    SET_Co_COB;
    SET_DT_LEADING_DIMENSIONS;
    SET_FT_LEADING_DIMENSIONS;
    SET_YT_LEADING_DIMENSIONS;

    // What matrices are DT, FT, and YT?
    // In this case: DT, FT, and YT

    memset((void *)YT, 0,  t * Co_COB * ho * wo * COB * sizeof(DTYPE));
    /* @todo: check that memset is actually a valid alternative to the following:
    for (int in = 0; in < n; in++)
        for (int ih = 0; ih < ho; ih++)
            for (int iw = 0; iw < wo; iw++)
                for (i = 0, i2 = 0; i < k; i += COB, i2++) {
                    ib = min(k - i, COB);
                    for (ii = 0; ii < ib; ii++)
                        YT(in, i2, ih, iw, ii) = (DTYPE) 0.0;
                }
    */

    // Loops reordered as in "High Performance Zero-Memory Overhead Direct Convolution" by J. Zhang et al., 2018

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
                                ob = min(kb, Wo - k - m + 1);
#ifdef MK_7x12_U4
                                if ((ob == MR) && (jb == NR))
                                    gemm_microkernel_Cresident_neon_7x12_fixed_unroll_4_fp32(
                                            ob, jb, ib,
                                            alpha,
                                            &DT(h, i2, l + n, k + m, 0),
                                            &FT(i2, j2, n, m, 0, 0),
                                            beta,
                                            &YT(h, j2, l, k, 0),
                                            ldYT4);
                                else
                                    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32(
                                            ob, jb, ib,
                                            alpha,
                                            &DT(h, i2, l + n, k + m, 0),
                                            &FT(i2, j2, n, m, 0, 0),
                                            beta,
                                            &YT(h, j2, l, k, 0), ldYT4);
#else
                                printf("Error: Microkernel doesn't exist.\n");
                                exit(EXIT_FAILURE);
#endif
                                /*
                                gemm_base(min(kb, Wo - k - m + 1), jb, ib,
                                          1.0, &DT(h, i2, l + n, k + m, 0), ldDT4,
                                          &FT(i2, j2, n, m, 0, 0), ldFT5,
                                          1.0, &YT(h, j2, l, k, 0), ldYT4);
                                */
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif //!TVM

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM
