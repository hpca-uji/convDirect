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

#define ALGORITHM convdirect_reorder

#include "algorithms.h"

convdirect_bs_t BLOCK_SIZES = {0, 0, 0, 0, 0};

void CONVDIRECT_PRE_NOP;

void CONVDIRECT_POST_NOP;

void CONVDIRECT_KERNEL_WITH_PARAMS {

    // Loops reordered as in "High Performance Zero-Memory Overhead Direct Convolution" by J. Zhang et al., 2018
    // Accommodate vectorization: j as the innermost loop
    // Ensure sufficient independent operations: k around j
    // For compatibility between output layer n and input layer n+1: n->m->i

    QUICK_RETURN_IF_POSSIBLE;

    int h, i, j, k, l, m, n, x_x, x_y, ho, wo;

    ho = (Ho + 2 * vpadding - vdilation * (Hf - 1) - 1) / vstride + 1;
    wo = (Wo + 2 * hpadding - hdilation * (Wf - 1) - 1) / hstride + 1;

    SET_LEADING_DIMENSIONS;

    // What matrices are DT, FT, and YT?
    const DTYPE *D = DT, *F = FT;
    DTYPE *Y = YT;

#if TENSOR_FORMAT_NCHW
    for (h = 0; h < t; h++)
        for (l = 0; l < ho; l++)
            for (n = 0; n < Hf; n++) {
                x_x = vstride * l + vdilation * n - vpadding;
                if (0 <= x_x && x_x < Ho)
                    for (m = 0; m < Wf; m++)
                        for (i = 0; i < Ci; i++)
                            for (k = 0; k < wo; k++) {
                                x_y = hstride * k + hdilation * m - hpadding;
                                if (0 <= x_y && x_y < Wo)
                                    for (j = 0; j < Co; j++)
                                        Yrow_NCHW(h, j, l, k) += Drow_NCHW(h, i, x_x, x_y) * Frow_NCHW(j, i, n, m);
                            }
            }
#else
    for (h = 0; h < t; h++)
        for (l = 0; l < ho; l++)
            for (n = 0; n < Hf; n++) {
                x_x = vstride * l + vdilation * n - vpadding;
                if (0 <= x_x && x_x < Ho)
                    for (m = 0; m < Wf; m++)
                        for (i = 0; i < Ci; i++)
                            for (k = 0; k < wo; k++) {
                                x_y = hstride * k + hdilation * m - hpadding;
                                if (0 <= x_y && x_y < Wo)
                                    for (j = 0; j < Co; j++)
                                        Yrow_NHWC(h, j, l, k) += Drow_NHWC(h, i, x_x, x_y) * Frow_NHWC(j, i, n, m);
                            }
            }
#endif
}

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM