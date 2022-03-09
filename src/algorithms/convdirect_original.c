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

#define ALGORITHM convdirect_original

#include "algorithms.h"

convdirect_bs_t BLOCK_SIZES = {0, 0, 0, 0, 0};

void CONVDIRECT_PRE_NOP;

void CONVDIRECT_POST_NOP;

void CONVDIRECT_KERNEL(int n, int k, int c,
                       int h, int w,
                       int r, int s,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation,
                       DTYPE alpha,
                       const DTYPE *D,
                       const DTYPE *F,
                       DTYPE beta,
                       DTYPE *Yg) {

    // Copy the dimensions parameters to the standard dimensions names (as the QUICK_RETURN_IF_POSSIBLE and
    // SET_LEADING_DIMENSIONS macros require these)
    int t = n, Co = k, Ci = c, Ho = h, Wo = w, Hf = r, Wf = s;

    QUICK_RETURN_IF_POSSIBLE;

    int in, ik, ic, ih, iw, ir, is, x_x, x_y, ho, wo;

    ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
    wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;

    SET_LEADING_DIMENSIONS;

#if TENSOR_FORMAT_NCHW
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ic = 0; ic < c; ic++)
                for (ih = 0; ih < ho; ih++)
                    for (iw = 0; iw < wo; iw++)
                        for (ir = 0; ir < r; ir++) {
                            x_x = vstride * ih + vdilation * ir - vpadding;
                            if (0 <= x_x && x_x < h)
                                for (is = 0; is < s; is++) {
                                    x_y = hstride * iw + hdilation * is - hpadding;
                                    if (0 <= x_y && x_y < w)
                                        Ygrow_NCHW(in, ik, ih, iw) +=
                                                Drow_NCHW(in, ic, x_x, x_y) * Frow_NCHW(ik, ic, ir, is);
                                }
                        }
#else
    for (in = 0; in < n; in++)
        for (ik = 0; ik < k; ik++)
            for (ic = 0; ic < c; ic++)
                for (ih = 0; ih < ho; ih++)
                    for (iw = 0; iw < wo; iw++)
                        for (ir = 0; ir < r; ir++) {
                            x_x = vstride * ih + vdilation * ir - vpadding;
                            if (0 <= x_x && x_x < h)
                                for (is = 0; is < s; is++) {
                                    x_y = hstride * iw + hdilation * is - hpadding;
                                    if (0 <= x_y && x_y < w) {
                                        // printf("FB %d %d %d %d %16.10e\n", ik, ic, ir, is, Frow_NHWC(ik,ic,ir,is));
                                        Ygrow_NHWC(in, ik, ih, iw) +=
                                                Drow_NHWC(in, ic, x_x, x_y) * Frow_NHWC(ik, ic, ir, is);
                                    }
                                }
                        }
#endif
}

void CONVDIRECT_PRE_KERNEL_POST;

#undef ALGORITHM
