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

#ifndef CONVDIRECT_BUFFERS_H
#define CONVDIRECT_BUFFERS_H

#include "../dtypes.h"

#define EXTERN_BUFFER_DECLARATION(x) \
    extern DTYPE *_##x;              \
    extern size_t _##x##_size;       \
    DTYPE *get_##x(size_t size);

#define EXTERN_ALIGNED_BUFFER_DECLARATION(x) \
    extern DTYPE *_##x;                      \
    extern size_t _##x##_size;               \
    DTYPE *get_aligned_##x (size_t alignment, size_t size);

// convdirect_conv_gemm aligned buffers
EXTERN_ALIGNED_BUFFER_DECLARATION(Ac_pack);
EXTERN_ALIGNED_BUFFER_DECLARATION(Bc_pack);

// convdirect_blis aligned buffer
EXTERN_ALIGNED_BUFFER_DECLARATION(Ac);

// convdirect_im2ros buffer
EXTERN_BUFFER_DECLARATION(DEXT);

#endif //CONVDIRECT_BUFFERS_H
