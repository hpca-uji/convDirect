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

#ifndef CONVDIRECT_GEMM_REFERENCE_H
#define CONVDIRECT_GEMM_REFERENCE_H

#include "dtypes.h"

#define min(a, b)     ( (a) > (b) ? (b) : (a) )

void gemm_reference(char orderA, char orderB, char orderC,
                    char transA, char transB,
                    int m, int n, int k,
                    DTYPE alpha,
                    const DTYPE *A, int ldA,
                    const DTYPE *B, int ldB,
                    DTYPE beta,
                    DTYPE *C, int ldC);

#endif // CONVDIRECT_GEMM_REFERENCE_H
