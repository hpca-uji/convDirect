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

#include "packrb.h"

/**
 * BLIS pack for A-->Ac
*/
void packRB(char orderA, char transA, int mc, int nc, const DTYPE *A,
        int start_y, int ky, int dim_w,
        int hpadding, int hstride, int hdilation,
        int ld3, DTYPE *Ac, int RR) {

    if (((transA == 'N') && (orderA == 'C')) ||
        ((transA == 'T') && (orderA == 'R'))) {
      printf("Packing not yet implemented\n");
      exit(-1);
    } else {
        start_y = hstride * start_y + hdilation * ky - hpadding;
        #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            for (int j = 0; j < nc; j++) {
                int ii = 0;
                int y = start_y + hstride * i;
                for (; ii < rr && y < 0    ; ii++, y += hstride, k++) Ac[k] = 0.0;
                for (; ii < rr && y < dim_w; ii++, y += hstride, k++) Ac[k] = A[y * ld3 + j];
                for (; ii < rr             ; ii++,               k++) Ac[k] = 0.0;
                k += (RR - rr);
            }
        }
    }
}
