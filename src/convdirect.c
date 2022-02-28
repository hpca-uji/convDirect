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
#include <string.h>
#include <stdlib.h>
#include "convdirect_options.h"

// In case we ever need a constructor
/*
static void con() __attribute__((constructor));

void con() {
    printf("I'm a constructor\n");
}
*/

// Obtain tensor format from name
tensor_format_t convdirect_get_tensor_format(char* name) {
    if (strstr(name, "nchw") != NULL) {
        return nchw;
    } else if (strstr(name, "nhwc") != NULL) {
        return nhwc;
    } else {
        printf("Error: The algorithm name '%s' must have either 'nhwc' or 'nchw' in it.\n"
               "Please, execute convDirect_info to see the available algorithms.\n", name);
        exit(EXIT_FAILURE);
    }
}

// Search for algorithm name in convdirect_nchw_n[] or convdirect_nhwc_n[] and return its index
int convdirect_get_option(char* name) {
    for (int i = 0; i < convdirect_options; i++) {
        if (strcmp(name, convdirect_n[i]) == 0)
            return i;
    }
    printf("Error: The algorithm '%s' is not available.\n"
           "Please, execute convDirect_info to see the available algorithms.\n", name);
    exit(EXIT_FAILURE);
}

// Obtain algorithm by name
convdirect_ft* convdirect_get_algorithm(char* name) {
    int n = convdirect_get_option(name);
    return convdirect_f[n];
}

// Obtain block sizes by name
convdirect_bs_t* convdirect_get_block_sizes(char* name) {
    int n = convdirect_get_option(name);
    return convdirect_bss[n];
}

// Obtain algorithm parts by name
void convdirect_get_algorithm_parts(char* name, convdirect_algorithm_parts_t *convdirect_algorithm_parts) {
    int n = convdirect_get_option(name);
    convdirect_algorithm_parts->pre = convdirect_pre_f[n];
    convdirect_algorithm_parts->kernel = convdirect_kernel_f[n];
    convdirect_algorithm_parts->post = convdirect_post_f[n];
}

