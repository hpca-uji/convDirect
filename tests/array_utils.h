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

#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include "../src/dtypes.h"

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

// @todo: set ATOL and RTOL for INT8 and FP64, as the error measure is now performed as in numpy's allclose

#if defined(INT8)
#define ATOL 0.5
//-----------------
#elif defined(FP16)
#define ATOL 1.0e-3
//-----------------
#elif defined(FP32)
#define ATOL 1.e-8   // 1.0e-5
#define RTOL 1.e-5
//-----------------
#elif defined(FP64)
#define ATOL 1.0e-14
#endif

void fill_tensor4D_rand(int, int, int, int, DTYPE *, int, int, int);

void print_tensor4D(char *, int, int, int, int, DTYPE *, int, int, int);

bool all_close_tensor4D(int m1, int m2, int m3, int m4, const DTYPE *A, const DTYPE *B, double *error,
                        int ldT1, int ldT2, int ldT3);

void print_tensor5D(char *, int, int, int, int, int, DTYPE *, int, int, int, int);

void print_tensor6D(char *, int, int, int, int, int, int, DTYPE *, int, int, int, int, int);

void print_matrix(char *, char, int, int, DTYPE *, int);

double dclock();

#endif // ARRAY_UTILS_H
