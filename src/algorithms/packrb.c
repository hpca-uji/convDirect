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
void packRB(char orderA, char transA, int mc, int nc, const DTYPE *A, int ldA, DTYPE *Ac, int RR) {

    int i, j, ii, k, rr;

    if (((transA == 'N') && (orderA == 'C')) ||
        ((transA == 'T') && (orderA == 'R')))
        //#pragma omp parallel for private(j, ii, rr, k)
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Ac[k] = Acol(i + ii, j);
                    k++;
                }
                k += (RR - rr);
            }
        }
    else
        //#pragma omp parallel for private(j, ii, rr, k)
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Ac[k] = Acol(j, i + ii);
                    k++;
                }
                k += (RR - rr);
            }
        }
}

