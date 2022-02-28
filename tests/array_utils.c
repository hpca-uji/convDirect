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
#include <time.h>

#include "array_utils.h"

#define Trow4D(a1, a2, a3, a4)          T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4) ]
#define Trow5D(a1, a2, a3, a4, a5)      T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4)*(ldT4) + (a5) ]
#define Trow6D(a1, a2, a3, a4, a5, a6)  T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4)*(ldT4) + (a5)*(ldT5) + (a6) ]

#define ROW4D(X, a1, a2, a3, a4)        (X)[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4) ]

#define Mcol(a1, a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1, a2)  M[ (a1)*(ldM)+(a2) ]

#if defined(INT8)
#define RANDOM_NUMBER ((DTYPE) rand()) % 10 + 1
#else
#define RANDOM_NUMBER ((DTYPE) rand()) / (DTYPE) RAND_MAX
#endif

#if defined(INT8)
#define FORMAT "%d"
// ---------------------
#elif defined(FP16)
#define FORMAT "%8.2e"
// ---------------------
#elif defined(FP32)
#define FORMAT "%14.8e"
// ---------------------
#elif defined(FP64)
#define FORMAT "%22.16e"
// ---------------------
#endif


/**
 * Fills a 4D tensor with random entries
 *
 * @param m1 First dimension of the tensor
 * @param m2 Second dimension of the tensor
 * @param m3 Third dimension of the tensor
 * @param m4 Fourth dimension of the tensor
 * @param T Tensor to be filled with random entries
 * @param ldT1 Leading dimension for the first dimension of the tensor
 * @param ldT2 Leading dimension for the second dimension of the tensor
 * @param ldT3 Leading dimension for the third dimension of the tensor
 */
void fill_tensor4D_rand(int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3) {

    if (ldT3 == 0 && ldT2 == 0 && ldT1 == 0) {
        ldT3 = m4;
        ldT2 = m3 * ldT3;
        ldT1 = m2 * ldT2;
    }

    int i1, i2, i3, i4;

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++) {
                    Trow4D(i1, i2, i3, i4) = RANDOM_NUMBER;
                }
}


/**
 * Prints a 4D tensor to the standard output
 *
 * @param name Label that will be used to identify the tensor
 * @param m1 First dimension of the tensor
 * @param m2 Second dimension of the tensor
 * @param m3 Third dimension of the tensor
 * @param m4 Fourth dimension of the tensor
 * @param T Tensor to be filled with random entries
 * @param ldT1 Leading dimension for the first dimension of the tensor
 * @param ldT2 Leading dimension for the second dimension of the tensor
 * @param ldT3 Leading dimension for the third dimension of the tensor
 */
void print_tensor4D(char *name, int m1, int m2, int m3, int m4, DTYPE *T, int ldT1, int ldT2, int ldT3) {

    if (ldT3 == 0 && ldT2 == 0 && ldT1 == 0) {
        ldT3 = m4;
        ldT2 = m3 * ldT3;
        ldT1 = m2 * ldT2;
    }

    int i1, i2, i3, i4;

    /*
    int i;
    printf("Matrix Memory Disposition: ");
    for (i = 0; i < m1 * m2 * m3 * m4; i++)
        printf("%d,", T[i]);
    printf("\n");
    */

    char *format = "%s[%d,%d,%d,%d] = "FORMAT";\n";

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++)
                    printf(format, name, i1, i2, i3, i4, ((DTYPE) Trow4D(i1, i2, i3, i4)));
}

/**
 * Compares two tensor4D element-wise (based on numpy allclose()).
 *
 * If the following equation is element-wise True, then returns True:
 *   absolute(a - b) <= (atol + rtol * absolute(b))
 *
 * @param m1 First dimension of the tensors
 * @param m2 Second dimension of the tensors
 * @param m3 Third dimension of the tensors
 * @param m4 Fourth dimension of the tensors
 * @param A The tensor to be tested
 * @param B The tensor that is considered as the correct one
 * @param error
 * @param ldT1 Leading dimension for the first dimension of the tensor
 * @param ldT2 Leading dimension for the second dimension of the tensor
 * @param ldT3 Leading dimension for the third dimension of the tensor
 * @return true if both tensors have all their elements close enough
 */
bool all_close_tensor4D(int m1, int m2, int m3, int m4, const DTYPE *A, const DTYPE *B, double *error,
                        int ldT1, int ldT2, int ldT3) {

    if (ldT3 == 0 && ldT2 == 0 && ldT1 == 0) {
        ldT3 = m4;
        ldT2 = m3 * ldT3;
        ldT1 = m2 * ldT2;
    }

    int i1, i2, i3, i4;
    double a, b;
    double tmp;

    *error = 0.0;

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++) {
                    a = ROW4D(A, i1, i2, i3, i4);
                    b = ROW4D(B, i1, i2, i3, i4);
                    tmp = dabs(a - b);
                    if (tmp / b > *error) *error = tmp / b;
                    if (tmp > (ATOL + RTOL * dabs(b))) {
                        return false;
                    }
                }
    return true;
}

/**
 * Prints a 5D tensor to the standard output
 *
 * @param name Label that will be used to identify the tensor
 * @param m1 First dimension of the tensor
 * @param m2 Second dimension of the tensor
 * @param m3 Third dimension of the tensor
 * @param m4 Fourth dimension of the tensor
 * @param m5 Fifth dimension of the tensor
 * @param T Tensor to be filled with random entries
 * @param ldT1 Leading dimension for the first dimension of the tensor
 * @param ldT2 Leading dimension for the second dimension of the tensor
 * @param ldT3 Leading dimension for the third dimension of the tensor
 * @param ldT4 Leading dimension for the fourth dimension of the tensor
 */
void print_tensor5D(char *name, int m1, int m2, int m3, int m4, int m5, DTYPE *T,
                    int ldT1, int ldT2, int ldT3, int ldT4) {

    if (ldT4 == 0 && ldT3 == 0 && ldT2 == 0 && ldT1 == 0) {
        ldT4 = m5;
        ldT3 = m4 * ldT4;
        ldT2 = m3 * ldT3;
        ldT1 = m2 * ldT2;
    }

    int i1, i2, i3, i4, i5;

    /*
    int i;
    printf("Matrix Memory Disposition %s[ ", name);
    for (i = 0; i < m1 * m2 * m3 * m4 * m5; i++) printf("%d,", T[i]);
    printf("]\n");
    printf("m1=%d x m2=%d x m3=%d x m4=%d x m5=%d:\n", m1, m2, m3, m4, m5);
    */

    char *format = "%s[%d,%d,%d,%d,%d] = "FORMAT";\\n";

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++)
                    for (i5 = 0; i5 < m5; i5++)
                        printf(format, name, i1, i2, i3, i4, i5, ((double) Trow5D(i1, i2, i3, i4, i5)));
}


/**
 * Prints a 6D tensor to the standard output
 *
 * @param name Label that will be used to identify the tensor
 * @param m1 First dimension of the tensor
 * @param m2 Second dimension of the tensor
 * @param m3 Third dimension of the tensor
 * @param m4 Fourth dimension of the tensor
 * @param m5 Fifth dimension of the tensor
 * @param m6 Sixth dimension of the tensor
 * @param T Tensor to be filled with random entries
 * @param ldT1 Leading dimension for the first dimension of the tensor
 * @param ldT2 Leading dimension for the second dimension of the tensor
 * @param ldT3 Leading dimension for the third dimension of the tensor
 * @param ldT4 Leading dimension for the fourth dimension of the tensor
 * @param ldT5 Leading dimension for the fifth dimension of the tensor
 */
void print_tensor6D(char *name, int m1, int m2, int m3, int m4, int m5, int m6, DTYPE *T,
                    int ldT1, int ldT2, int ldT3, int ldT4, int ldT5) {

    if (ldT5 == 0 && ldT4 == 0 && ldT3 == 0 && ldT2 == 0 && ldT1 == 0) {
        ldT5 = m6;
        ldT4 = m5 * ldT5;
        ldT3 = m4 * ldT4;
        ldT2 = m3 * ldT3;
        ldT1 = m2 * ldT2;
    }

    int i1, i2, i3, i4, i5, i6;

    /*
    int i;
    printf("Matrix Memory Disposition: ");
    for (i = 0; i < m1 * m2 * m3 * m4 * m5 * m6; i++)
        printf("%d,", T[i]);
    printf("\n");
    */

    char *format = "%s[%d,%d,%d,%d,%d,%d] = "FORMAT";\n";

    for (i1 = 0; i1 < m1; i1++)
        for (i2 = 0; i2 < m2; i2++)
            for (i3 = 0; i3 < m3; i3++)
                for (i4 = 0; i4 < m4; i4++)
                    for (i5 = 0; i5 < m5; i5++)
                        for (i6 = 0; i6 < m6; i6++)
                            printf(format, name, i1, i2, i3, i4, i5, i6, ((int) Trow6D(i1, i2, i3, i4, i5, i6)));
}


/**
 * Prints a matrix to the standard output
 *
 * @param name Label that will be used to identify the tensor
 * @param orderM Order of the elements in the matrix: 'C' or 'F'
 * @param m Row dimension
 * @param n Column dimension
 * @param M Matrix
 * @param ldM Leading dimension
 */
void print_matrix(char *name, char orderM, int m, int n, DTYPE *M, int ldM) {

    int i, j;

    char *format = "%s[%d,%d] = "FORMAT";\n";

    if (orderM == 'C')
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++)
                printf(format, name, i, j, ((DTYPE) Mcol(i, j)));
    else
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++)
                printf(format, name, i, j, ((DTYPE) Mrow(i, j)));
}


/**
 * Provides the number of seconds and nanoseconds from a point in the past as a double
 *
 * @return The number of seconds and nanoseconds from a point in the past
 */
double dclock() {
    /*
     * From man gettimeofday:
     *
     *  The  time returned by gettimeofday() is affected by discontinuous jumps in the system time (e.g., if the system
     *  administrator manually changes the system time).  If you need a monotonically increasing clock,
     *  see clock_gettime(2).
     *
     */
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (double) tp.tv_sec + (double) tp.tv_nsec * 1.0e-9;
}
