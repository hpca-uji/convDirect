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

#include <stdio.h>
#include <stdlib.h>
#include "arrays.h"
#include "gemm_reference.h"


void gemm_reference(char orderA, char orderB, char orderC,
                    char transA, char transB,
                    int m, int n, int k,
                    DTYPE alpha,
                    const DTYPE *A, int ldA,
                    const DTYPE *B, int ldB,
                    DTYPE beta,
                    DTYPE *C, int ldC) {

    int ic, jc, pc, i, j, p;
    DTYPE zero = (DTYPE) 0.0, one = (DTYPE) 1.0, tmp;

    // Quick return if possible
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

    if ((transA == 'N') && (transB == 'N')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Bcol(p, j);
                } else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Brow(p, j);
                } else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Bcol(p, j);
                } else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Brow(p, j);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                } else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    } else if ((transA == 'N') && (transB == 'T')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Bcol(j, p);
                } else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Brow(j, p);
                } else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Bcol(j, p);
                } else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Brow(j, p);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                } else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    } else if ((transA == 'T') && (transB == 'N')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Bcol(p, j);
                } else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Brow(p, j);
                } else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Bcol(p, j);
                } else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Brow(p, j);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                } else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    } else if ((transA == 'T') && (transB == 'T')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Bcol(j, p);
                } else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Brow(j, p);
                } else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Bcol(j, p);
                } else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Brow(j, p);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                } else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    } else {
        printf("Error: Invalid options for transA, transB: %c %c\n", transA, transB);
        exit(EXIT_FAILURE);
    }
}


