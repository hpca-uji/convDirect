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

#define ALGORITHM convdirect_block_blis

#include <stdlib.h>
#include <stdio.h>
#include <blis/blis.h>
#include "packrb.h"
#include "algorithms.h"
#include "convdirect_block_blis.h"
#include "buffers.h"

#ifdef ARCH__aarch64__
#include "ukrs/gemm_blis_neon_fp32.h"
#endif

convdirect_bs_t BLOCK_SIZES = {};

#ifdef MK_BLIS
cntx_t *_TFMK(cntx) = NULL;
GEMM_KERNEL_TYPE _TFMK(gemm_microkernel);
int _TFMK(MR_bs);
int _TFMK(NR_bs);
// Prepare to call micro-kernel with transposed operands
#define MK_BLIS_SET_MR_AND_NR \
    int MR = _TFMK(NR_bs);    \
    int NR = _TFMK(MR_bs);
#endif

void TRANSFORM_FILTER(int Ci, int Co,
                      int Hf, int Wf,
                      const DTYPE *F,
                      DTYPE *FB) {

#ifdef MK_BLIS
    MK_BLIS_SET_MR_AND_NR;
#endif

    int i, j, jj, jb, j2, m, n;

    SET_Co_MR;
    SET_Co_NR;
    SET_F_LEADING_DIMENSIONS;
    SET_FB_LEADING_DIMENSIONS;

#ifdef TENSOR_FORMAT_NCHW
    printf("Tensor format NCHW not yet implemented in transform_filter_block_blis!\n");
    exit(EXIT_FAILURE);
#else
    for (j = 0, j2 = 0; j < Co; j += NR, j2++) {
        jb = min(Co - j, NR);
        for (i = 0; i < Ci; i++)
            for (n = 0; n < Hf; n++)
                for (m = 0; m < Wf; m++)
                    for (jj = 0; jj < jb; jj++) {
                        FBrow_NHWC(j2, i, n, m, jj) = Frow_NHWC(j + jj, i, n, m);
                        /*
                        printf("copy %d %d %d %d --> %d %d %d %d %d, %16.10e\n",
                               j + jj, i, n, m, j2, i, n, m, jj,
                               Frow_NHWC(j + jj, i, n, m));
                        */
                    }
    }
#endif
}

void CONVDIRECT_PRE_WITH_PARAMS {
#ifdef MK_BLIS
    if (_TFMK(cntx) == NULL) {
        bli_init();
        _TFMK(cntx) = bli_gks_query_cntx();
        _TFMK(MR_bs) = (int) bli_cntx_get_blksz_def_dt(BLIS_DTYPE, BLIS_MR, _TFMK(cntx));
        _TFMK(NR_bs) = (int) bli_cntx_get_blksz_def_dt(BLIS_DTYPE, BLIS_NR, _TFMK(cntx));
        _TFMK(gemm_microkernel) = bli_cntx_get_l3_nat_ukr_dt(BLIS_DTYPE, BLIS_GEMM_UKR, _TFMK(cntx));
    }
    MK_BLIS_SET_MR_AND_NR;
#endif

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
            ho, wo, ii, jj, kk, ib, jb, kb;

/* For testing in the IDE
#undef TENSOR_FORMAT_NCHW
#define MK_BLIS
#define BLIS_ABI_VERSION 3
*/

 #ifdef MK_BLIS
    auxinfo_t aux;
    MK_BLIS_SET_MR_AND_NR;
#if BLIS_ABI_VERSION == 3
    DTYPE Cc[NR * MR];
    DTYPE zero = (DTYPE) 0.0;
#endif // BLIS_ABI_VERSION == 3
#endif // MK_BLIS

    BS_UPDATE(BLOCK_SIZES, MR, NR, WOB, COB, CIB);
    int Cob_Mr = COB / MR;
    int Cob_Nr = COB / NR;

    int jr, nr, jr2, ir, mr, in = 0;

    ho = ((Ho - Hf) / 1) + 1;
    wo = ((Wo - Wf) / 1) + 1;

    SET_LEADING_DIMENSIONS;
    SET_Co_NR;
    SET_Co_MR;
    SET_FB_LEADING_DIMENSIONS;

    // @todo: check that original size was wrong [ was ((int) ceil((WOB - 1)) / MR + 1) * MR * CIB ]
    DTYPE *Ac = get_aligned_Ac(4096, (int) ceil((double) WOB / MR) * MR * CIB * sizeof(DTYPE));

    // What matrices are DT, FT, and YT?
    const DTYPE *D = DT, *FB = FT;
    DTYPE *Y = YT;

#ifdef TENSOR_FORMAT_NCHW
    printf("Tensor format NCHW not yet implemented in convDirect_block_blis!\n");
    exit(EXIT_FAILURE);
#else
    for (h = 0; h < t; h++) {
        for (i = 0, i2 = 0; i < Ci; i += CIB, i2++) {
            ib = min(Ci - i, CIB);
            for (l = 0; l < ho; l++) {
                for (k = 0; k < wo; k += WOB) {
                    kb = min(wo - k, WOB);
                    for (n = 0; n < min(Hf, Ho - l); n++) {
                        for (m = 0; m < Wf; m++) {
                            packRB('R', 'N', kb, ib, &Drow_NHWC(h, i, l + n, k + m), ldD3, Ac, MR);
                            for (j = 0, j2 = 0; j < Co; j += COB, j2++) {
                                jb = min(Co - j, COB);
                                for (jr = 0, jr2 = 0; jr < jb; jr += NR, jr2++) {
                                    nr = min(jb - jr, NR);
                                    for (ir = 0; ir < min(kb, Wo - k - m + 1); ir += MR) {
                                        mr = min(min(kb, Wo - k - m + 1) - ir, MR);
                                        /*
                                        gemm_reference('C', 'R', 'R',
                                                       'N', 'N',
                                                       mr, nr, ib,
                                                       1.0,
                                                       &Ac[ir * ib], MR,
                                                       &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), NR,
                                                       1.0,
                                                       &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
                                        */
#ifdef MK_BLIS
                                        /* Call micro-kernel with transposed operands */
#if BLIS_ABI_VERSION == 4
                                        _TFMK(gemm_microkernel)(nr, mr, ib,
                                                                &alpha,
                                                                (DTYPE *) &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                                &Ac[ir * ib],
                                                                &beta,
                                                                &Yrow_NHWC(h, j + jr, l, k + ir), 1, ldY3,
                                                                &aux, _TFMK(cntx));
#elif BLIS_ABI_VERSION == 3
                                        if ((nr == NR) && (mr == MR))
                                            _TFMK(gemm_microkernel)(
                                                    ib,
                                                    &alpha,
                                                    (DTYPE *) &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    &Ac[ir * ib],
                                                    &beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir), 1, ldY3,
                                                    &aux, _TFMK(cntx));
                                        else {
// #ifndef ARCH__aarch64__
                                            /* BLIS alternative */
                                            _TFMK(gemm_microkernel)(
                                                    ib,
                                                    &alpha,
                                                    (DTYPE *) &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    &Ac[ir * ib],
                                                    &zero,
                                                    Cc, 1, NR,
                                                    &aux, _TFMK(cntx));
// #else
                                            /* THis alternative relies on our micro-kernel, to avoid using the one in
                                             * BLIS for the border cases */
                                            /* WARNING: NR must be used as leading dimension because he next to this
                                             * operation is shared with the no ARCH__arrch64__ gemm version */
                                            /* Our microkernel alternative should be tested again before using
                                             * it, something does not work (are always nr < 8 and mr < 12?) */
                                            /*
                                            gemm_microkernel_Cresident_neon_8x12_fp32(
                                                    nr, mr, ib,
                                                    alpha,
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    &Ac[ir * ib],
                                                    zero,
                                                    Cc, NR); // NR is ok. See above.
                                            */
// #endif
                                            /* The next operations MUST be done to achieve the correct solution.
                                             * It has a considerable impact on performance.
                                             * The overhead could be avoided by transposing the micro-tile of C
                                             * internally to the routine. However, even it dropping it, the result does
                                             * not outperform the manual micro-kernel 8x12.
                                             * The alternative is to use the BLIS micro-kernel for all cases, but the
                                             * performance is even lower */
                                            if (beta == (DTYPE) 0.0) {
                                                for (int j1 = 0; j1 < mr; j1++)
                                                    for (int i1 = 0; i1 < nr; i1++)
                                                        Yrow_NHWC(h, j + jr + i1, l, k + ir + j1) = Cc[j1 * NR + i1];
                                            } else if (beta == (DTYPE) 1.0) {
                                                for (int j1 = 0; j1 < mr; j1++)
                                                    for (int i1 = 0; i1 < nr; i1++)
                                                        Yrow_NHWC(h, j + jr + i1, l, k + ir + j1) += Cc[j1 * NR + i1];
                                            } else {
                                                for (int j1 = 0; j1 < mr; j1++)
                                                    for (int i1 = 0; i1 < nr; i1++)
                                                        Yrow_NHWC(h, j + jr + i1, l, k + ir + j1) =
                                                                beta * Yrow_NHWC(h, j + jr + i1, l, k + ir + j1) +
                                                                Cc[j1 * NR + i1];
                                            }
                                        }
#else
#pragma GCC error "Error: current BLIS ABI version not yet supported!"
#endif
#elif defined(MK_8x12)
                                        // printf("mr %d nr %d MR %d NR %d\n", mr, nr, MR, NR);
                                        if ((mr == MR) && (nr == NR))
                                            gemm_microkernel_Cresident_neon_fixed_8x12_fp32(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Ac[ir * ib],
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
                                        else
                                            gemm_microkernel_Cresident_neon_8x12_fp32(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Ac[ir * ib],
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
#elif defined(MK_4x12)
                                        gemm_microkernel_Cresident_neon_4x12_fp32(
                                                mr, nr, ib,
                                                alpha,
                                                &Ac[ir * ib],
                                                &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                beta,
                                                &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
#elif defined(MK_4x16)
                                        gemm_microkernel_Cresident_neon_4x16_fp32(
                                                mr, nr, ib,
                                                alpha,
                                                &Ac[ir * ib],
                                                &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                beta,
                                                &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
#elif defined(MK_4x20)
                                        if ((mr == MR) && (nr == NR))
                                            gemm_microkernel_Cresident_neon_fixed_4x20_fp32(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Ac[ir * ib],
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
                                        else
                                            gemm_microkernel_Cresident_neon_4x20_fp32(
                                                    mr, nr, ib,
                                                    alpha,
                                                    &Ac[ir * ib],
                                                    &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                    beta,
                                                    &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
#elif defined(MK_4x4)
                                        gemm_microkernel_Cresident_neon_4x4_prefetch_fp32(
                                                mr, nr, ib,
                                                alpha,
                                                &Ac[ir * ib],
                                                &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0),
                                                beta,
                                                &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);

#else
                                        printf("Error: Microkernel doesn't exist.\n");
                                        exit(-1);
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
