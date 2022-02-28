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
#include "convdirect.h"

int main(int argc, char *argv[]) {
    printf(".--------------------------------.\n");
    printf("| convDirect library information |\n");
    printf(".--------------------------------.\n");
    printf("\n");
#ifdef BLIS_FOUND
    printf("Compiled against BLIS ABI version %d.\n", BLIS_ABI_VERSION);
#else
    printf("The BLIS library was not available when compiled. Not using it.\n");
#endif
    printf("\n");
    char *format = " + %s\n";
    printf("The next NCHW algorithms are available:\n");
    for (int n = 0; n < convdirect_options; n++) {
        if (strstr(convdirect_n[n], "nchw") != NULL)
            printf(format, convdirect_n[n]);
    }
    printf("\n");
    printf("The next NHWC algorithms are available:\n");
    for (int n = 0; n < convdirect_options; n++) {
        if (strstr(convdirect_n[n], "nhwc") != NULL)
            printf(format, convdirect_n[n]);
    }
    printf("\n");
    printf("See how to access the different algorithms and their information in 'convdirect.h'.\n");
}

