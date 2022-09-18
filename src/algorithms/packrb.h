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

#include "../dtypes.h"
#include "../arrays.h"
#include "../macros.h"
#include <arm_neon.h>

#ifndef CONVDIRECT_PACKRB_H
#define CONVDIRECT_PACKRB_H


void packRB_neon( char orderA, char transA, int mc, int nc, DTYPE *A, int ldA, DTYPE *Ac, int RR );

void packRB(char orderA, char transA, int mc, int nc, const DTYPE *A,
        int start_y, int ky, int dim_w,
        int hpadding, int hstride, int hdilation,
        int ld3, DTYPE *Ac, int RR);
#endif //CONVDIRECT_PACKRB_H
