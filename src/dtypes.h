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

#ifdef BLIS_FOUND

#include <blis/blis.h>

#endif //BLIS_FOUND


#if defined(INT8)
#define DTYPE unsigned int
//-----------------
#elif defined(FP32)
#define DTYPE float
//-----------------
#elif defined(FP64)
#define DTYPE double
#endif


#ifdef BLIS_FOUND
#ifdef FP32
#define GEMM_KERNEL_TYPE sgemm_ukr_ft
#define BLIS_DTYPE BLIS_FLOAT
//------------------------------------
#elif defined(FP64)
#define GEMM_KERNEL_TYPE dgemm_ukr_ft
#define BLIS_DTYPE BLIS_DOUBLE
//------------------------------------
#else // FP32 and FP64 not defined
#pragma GCC error "ERROR: BLIS and INT8 are not yet supported!"
//------------------------------------
#endif

#endif // BLIS_FOUND
