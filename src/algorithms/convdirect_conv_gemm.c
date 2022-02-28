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

#define ALGORITHM convdirect_conv_gemm

#include <stdlib.h>
#include <stdio.h>
#include <blis/blis.h>
#include "packrb.h"
#include "buffers.h"
#include "algorithms.h"

#undef min

#include "../convGemmNHWC/src/gemm_blis.h"
#include "../convGemmNHWC/src/im2row_nhwc.h"

convdirect_bs_t BLOCK_SIZES = {0, 0, 0, 0, 0};

void CONVDIRECT_PRE_NOP;

void CONVDIRECT_POST_NOP;

void CONVDIRECT_KERNEL_WITH_PARAMS {

    QUICK_RETURN_IF_POSSIBLE;

    // Copy the dimensions parameters to the names used in convGemm
    int n = t, k = Co, c = Ci, h = Ho, w = Wo, r = Hf, s = Wf;

    cntx_t *cntx;
    bli_init();
    cntx = bli_gks_query_cntx();

    int vpadding = 0;
    int hpadding = 0;
    int vdilation = 1;
    int hdilation = 1;
    int vstride = 1;
    int hstride = 1;

    int ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
    int wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;

    conv_p conv_params = {n, h, w, c,
                          k, r, s,
                          vstride, hstride, vpadding, hpadding,
                          vdilation, hdilation, ho, wo,
                          NULL, NULL, NULL, NULL, NULL, false};

    // Get Gemm BLIS blocks sizes, Ac_pack, and Bc_pack
    int _m = k;
    int _n = ho * wo * n;
    int _k = r * s * c;
    int MR_bs, NR_bs, MC_bs, NC_bs, KC_bs;
    gemm_blis_blocks_sizes(_m, _n, _k, &MR_bs, &NR_bs, &MC_bs, &NC_bs, &KC_bs);
    BS_UPDATE(BLOCK_SIZES, MR_bs, NR_bs, 0, 0, 0);
    DTYPE *Ac_pack = get_aligned_Ac_pack(4096, MC_bs * KC_bs * sizeof(BLIS_DTYPE));
    DTYPE *Bc_pack = get_aligned_Bc_pack(4096, KC_bs * NC_bs * sizeof(BLIS_DTYPE));

    // What matrices are DT, FT, and YT?
    const DTYPE *D = DT, *F = FT;
    DTYPE *Y = YT;

#ifdef TENSOR_FORMAT_NCHW
    printf("Tensor format NCHW not yet implemented in convDirect_block_blis!\n");
    exit(EXIT_FAILURE);
#else
    gemm_blis_B3A2C0_orig('C', 'C', 'C',
                          'N', 'N',
                          _m, _n, _k,
                          alpha, F, k,
                          D, r * s * c,
                          beta, Y, k,
                          Ac_pack, pack_RB,
                          Bc_pack, pack_CB_nhwc, NULL, cntx, &conv_params);
#endif
}

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM
