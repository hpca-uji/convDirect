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

#define ALGORITHM convdirect_im2row

#include <blis/blis.h>

#include "im2row.h"
#include "algorithms.h"
#include "buffers.h"

convdirect_bs_t BLOCK_SIZES = {0, 0, 0, 0, 0};

void CONVDIRECT_PRE_NOP;

void CONVDIRECT_POST_NOP;

void CONVDIRECT_KERNEL_WITH_PARAMS {

    QUICK_RETURN_IF_POSSIBLE;

    // Copy the dimensions parameters to the names used in convGemm
    int n = t, k = Co, c = Ci, h = Ho, w = Wo, r = Hf, s = Wf;

    char transa = 'N';
    char transb = 'N';

    int ho = ((h - r) / 1) + 1;
    int wo = ((w - s) / 1) + 1;

    int mm = k;
    int nn = ho * wo * n;
    int kk = r * s * c;
    int lda = k;
    int ldb = r * s * c;
    int ldc = k;

    DTYPE *DEXT = get_DEXT(kk * nn * sizeof(DTYPE));

    // What matrices are DT, FT, and YT?
    const DTYPE *D = DT, *F = FT;
    DTYPE *Y = YT;

#if TENSOR_FORMAT_NCHW
    printf("Error: convdirect_im2row does not yet support the NCHW tensor format!\n");
    exit(1);
#else
    im2row(DEXT, c * r * s, D, n, h, w, c, ho, wo, r, s, vpadding, hpadding, vstride, hstride, vdilation, hdilation);

    bli_sgemm(transa == 'T' ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              transb == 'T' ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE,
              mm, nn, kk,
              &alpha,
              F, 1, lda,
              DEXT, 1, ldb,
              &beta,
              Y, 1, ldc);
#endif
}

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM
