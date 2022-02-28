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
#include <string.h>
#include "array_utils.h"
#include "colors.h"
#include "input_utils.h"

#include "convdirect.h"

/*
#include "dtypes.h"
#include "gemm_reference.h"
#include "arrays.h"
*/


#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

int main(int argc, char *argv[]) {

    int n = 1, c = 1, h = 4, w = 2, k = 1, r = 2, s = 2;

    DTYPE D[] = {1, 2,
                 4, 8,
                 4, 8,
                 16, 32
    };

    DTYPE F[] = {1, 1,
                 1, 1};

    DTYPE Y[] = {0, 0, 0};

    DTYPE Yg[] = {0, 0, 0};

    print_tensor4D("D", n, h, w, c, D, 0, 0, 0);
    printf("\n");

    print_tensor4D("F", k, c, r, s, F, 0, 0, 0);
    printf("\n");

    convdirect_get_algorithm("convdirect_original_nhwc_default")(
            n, k, c,
            h, w,
            r, s,
            (DTYPE) 1.0,
            D,
            F,
            (DTYPE) 1.0,
            Yg);

    int ho = ((h - r) / 1) + 1;
    int wo = ((w - s) / 1) + 1;

    print_tensor4D("Yg", k, ho, wo, c, Yg, 0, 0, 0);
    printf("\n");

    convdirect_get_algorithm("convdirect_block_blis_nhwc_blis")(
            n, k, c,
            h, w,
            r, s,
            (DTYPE) 1.0,
            D,
            F,
            (DTYPE) 1.0,
            Y);

    print_tensor4D("Y", k, ho, wo, c, Yg, 0, 0, 0);
    printf("\n");


    return EXIT_SUCCESS;
}
