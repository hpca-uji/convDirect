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

#include "buffers.h"

#define BUFFER_DECLARATION(x) DTYPE *_##x = NULL; size_t _##x##_size = 0;

#define GET_BUFFER(x) \
    DTYPE *get_##x(size_t size) { \
        if (_##x##_size < size) { \
            _##x##_size = size;   \
            free(_##x);           \
            _##x = malloc(size);  \
        }                         \
        return _##x;              \
    }

#define GET_ALIGNED_BUFFER(x) \
    DTYPE *get_aligned_##x(size_t alignment, size_t size) { \
        if (_##x##_size < size) {                           \
            _##x##_size = size;                             \
            free(_##x);                                     \
            _##x = aligned_alloc(alignment, size);          \
        }                                                   \
        return _##x;                                        \
    }

// convdirect_conv_gemm aligned buffers

BUFFER_DECLARATION(Ac_pack);

GET_ALIGNED_BUFFER(Ac_pack);

BUFFER_DECLARATION(Bc_pack);

GET_ALIGNED_BUFFER(Bc_pack);

// convdirect_conv_blis aligned buffers

BUFFER_DECLARATION(Ac);

GET_ALIGNED_BUFFER(Ac);

// convdirect_im2row DEXT

BUFFER_DECLARATION(DEXT);

GET_BUFFER(DEXT);
