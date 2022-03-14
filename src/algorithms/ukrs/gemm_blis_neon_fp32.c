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

#include "gemm_blis_neon_fp32.h"


#define Arow(a1, a2)  Ar[ (a1)*(ldA)+(a2) ]
#define Brow(a1, a2)  Br[ (a1)*(ldB)+(a2) ]
#define Crow(a1, a2)  C[ (a1)*(ldC)+(a2) ]
#define Ctrow(a1, a2) Ctmp[ (a1)*(ldCt)+(a2) ]
#define Atrow(a1, a2) Atmp[ (a1)*(ldAt)+(a2) ]
#define Btrow(a1, a2) Btmp[ (a1)*(ldBt)+(a2) ]

#define Ctref(a1, a2) Ctmp[ (a2)*(ldCt)+(a1) ]
#define Atref(a1, a2) Atmp[ (a2)*(ldAt)+(a1) ]


#define SET_MR_NR(_MR, _NR) \
    const int MR = _MR;     \
    const int NR = _NR;


void gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                        const float *Ar,
                                                        const float *Br,
                                                        float beta,
                                                        float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = 4;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);

    Bptr = &Br[0];
    Bnr = NR;
    for (i = 0; i < MR; i++)
        for (j = 0; j < ldAt; j++)
            Atrow(i, j) = 0.0;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc - 3; k += 4) {

            // printf("Inside iteration k %d %d %d\n", k, k+1, k+2);
            // Load columns/rows of A/B for current iteration
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
                Atrow(i, 2) = Arow(i, k + 2);
                Atrow(i, 3) = Arow(i, k + 3);
            }

            COMPUTE_KERNEL_7x12_UNROLL_4  // Code for single iteration 7x12 in file compute_kernel_7x12.h

            baseB = baseB + 4 * Bnr;
        }

        if (k == kc - 1) {
            // printf("Outside iteration k %d\n", k);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++)
                Atrow(i, 0) = Arow(i, k);

            COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        } else if (k == kc - 2) {
            // printf("Outside iteration k %d %d\n", k, k+1);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
            }

            COMPUTE_KERNEL_7x12_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        } else if (k == kc - 3) {
            // printf("Outside iteration k %d %d\n", k, k+1);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
                Atrow(i, 2) = Arow(i, k + 2);
            }

            COMPUTE_KERNEL_7x12_UNROLL_3  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        }

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
            C40 = -C40;
            C41 = -C41;
            C42 = -C42;
            C50 = -C50;
            C51 = -C51;
            C52 = -C52;
            C60 = -C60;
            C61 = -C61;
            C62 = -C62;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
            C40 = alpha * C40;
            C41 = alpha * C41;
            C42 = alpha * C42;
            C50 = alpha * C50;
            C51 = alpha * C51;
            C52 = alpha * C52;
            C60 = alpha * C60;
            C61 = alpha * C61;
            C62 = alpha * C62;
        }
    }

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40);
        vst1q_f32(&Ctrow(4, 4), C41);
        vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50);
        vst1q_f32(&Ctrow(5, 4), C51);
        vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60);
        vst1q_f32(&Ctrow(6, 4), C61);
        vst1q_f32(&Ctrow(6, 8), C62);

        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            A0 = vld1q_f32(&Crow(0, 0));
            A1 = vld1q_f32(&Crow(0, 4));
            A2 = vld1q_f32(&Crow(0, 8));
            C00 = beta * A0 + C00;
            C01 = beta * A1 + C01;
            C02 = beta * A2 + C02;

            A0 = vld1q_f32(&Crow(1, 0));
            A1 = vld1q_f32(&Crow(1, 4));
            A2 = vld1q_f32(&Crow(1, 8));
            C10 = beta * A0 + C10;
            C11 = beta * A1 + C11;
            C12 = beta * A2 + C12;

            A0 = vld1q_f32(&Crow(2, 0));
            A1 = vld1q_f32(&Crow(2, 4));
            A2 = vld1q_f32(&Crow(2, 8));
            C20 = beta * A0 + C20;
            C21 = beta * A1 + C21;
            C22 = beta * A2 + C22;

            A0 = vld1q_f32(&Crow(3, 0));
            A1 = vld1q_f32(&Crow(3, 4));
            A2 = vld1q_f32(&Crow(3, 8));
            C30 = beta * A0 + C30;
            C31 = beta * A1 + C31;
            C32 = beta * A2 + C32;

            A0 = vld1q_f32(&Crow(4, 0));
            A1 = vld1q_f32(&Crow(4, 4));
            A2 = vld1q_f32(&Crow(4, 8));
            C40 = beta * A0 + C40;
            C41 = beta * A1 + C41;
            C42 = beta * A2 + C42;

            A0 = vld1q_f32(&Crow(5, 0));
            A1 = vld1q_f32(&Crow(5, 4));
            A2 = vld1q_f32(&Crow(5, 8));
            C50 = beta * A0 + C50;
            C51 = beta * A1 + C51;
            C52 = beta * A2 + C52;

            A0 = vld1q_f32(&Crow(6, 0));
            A1 = vld1q_f32(&Crow(6, 4));
            A2 = vld1q_f32(&Crow(6, 8));
            C60 = beta * A0 + C60;
            C61 = beta * A1 + C61;
            C62 = beta * A2 + C62;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
    } else {
        printf("Error: Incorrect use of 7x12 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_7x12_fixed_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                              const float *Ar,
                                                              const float *Br,
                                                              float beta,
                                                              float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = 4;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    // C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0);
    // C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
    // C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0);
    // C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0);
    // C40 = vmovq_n_f32(0); C41 = vmovq_n_f32(0); C42 = vmovq_n_f32(0);
    // C50 = vmovq_n_f32(0); C51 = vmovq_n_f32(0); C52 = vmovq_n_f32(0);
    // C60 = vmovq_n_f32(0); C61 = vmovq_n_f32(0); C62 = vmovq_n_f32(0);
    C00 = vld1q_f32(&Crow(0, 0));
    C01 = vld1q_f32(&Crow(0, 4));
    C02 = vld1q_f32(&Crow(0, 8));
    C10 = vld1q_f32(&Crow(1, 0));
    C11 = vld1q_f32(&Crow(1, 4));
    C12 = vld1q_f32(&Crow(1, 8));
    C20 = vld1q_f32(&Crow(2, 0));
    C21 = vld1q_f32(&Crow(2, 4));
    C22 = vld1q_f32(&Crow(2, 8));
    C30 = vld1q_f32(&Crow(3, 0));
    C31 = vld1q_f32(&Crow(3, 4));
    C32 = vld1q_f32(&Crow(3, 8));
    C40 = vld1q_f32(&Crow(4, 0));
    C41 = vld1q_f32(&Crow(4, 4));
    C42 = vld1q_f32(&Crow(4, 8));
    C50 = vld1q_f32(&Crow(5, 0));
    C51 = vld1q_f32(&Crow(5, 4));
    C52 = vld1q_f32(&Crow(5, 8));
    C60 = vld1q_f32(&Crow(6, 0));
    C61 = vld1q_f32(&Crow(6, 4));
    C62 = vld1q_f32(&Crow(6, 8));

    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    for (k = 0; k < kc - 3; k += 4) {

        // Load columns/rows of A/B for current iteration
        A0 = vld1q_f32(&Arow(0, k)); \
      A1 = vld1q_f32(&Arow(1, k)); \
      A2 = vld1q_f32(&Arow(2, k)); \
      A3 = vld1q_f32(&Arow(3, k)); \
      A4 = vld1q_f32(&Arow(4, k)); \
      A5 = vld1q_f32(&Arow(5, k)); \
      A6 = vld1q_f32(&Arow(6, k)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB + 4]); \
      B2 = vld1q_f32(&Bptr[baseB + 8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 0);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 0);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 0);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 0);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 0);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 0);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB + 12]); \
      B1 = vld1q_f32(&Bptr[baseB + 16]); \
      B2 = vld1q_f32(&Bptr[baseB + 20]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 1);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 1);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 1);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 1);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 1);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 1);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 1);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB + 24]); \
      B1 = vld1q_f32(&Bptr[baseB + 28]); \
      B2 = vld1q_f32(&Bptr[baseB + 32]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 2);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 2);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 2);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 2);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 2);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 2);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 2);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
\
      B0 = vld1q_f32(&Bptr[baseB + 36]); \
      B1 = vld1q_f32(&Bptr[baseB + 40]); \
      B2 = vld1q_f32(&Bptr[baseB + 44]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 3);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 3);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 3); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 3);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 3);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 3); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 3);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 3);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 3); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 3);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 3);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 3); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 3);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 3);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 3); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 3);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 3);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 3); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 3);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 3);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 3); \

        baseB = baseB + 4 * Bnr;
    }

    if (k == kc - 1) {
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++)
            Atrow(i, 0) = Arow(i, k);

        COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 2) {
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
        }

        COMPUTE_KERNEL_7x12_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 3) {
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
            Atrow(i, 2) = Arow(i, k + 2);
        }

        COMPUTE_KERNEL_7x12_UNROLL_3  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    }

    vst1q_f32(&Crow(0, 0), C00);
    vst1q_f32(&Crow(0, 4), C01);
    vst1q_f32(&Crow(0, 8), C02);
    vst1q_f32(&Crow(1, 0), C10);
    vst1q_f32(&Crow(1, 4), C11);
    vst1q_f32(&Crow(1, 8), C12);
    vst1q_f32(&Crow(2, 0), C20);
    vst1q_f32(&Crow(2, 4), C21);
    vst1q_f32(&Crow(2, 8), C22);
    vst1q_f32(&Crow(3, 0), C30);
    vst1q_f32(&Crow(3, 4), C31);
    vst1q_f32(&Crow(3, 8), C32);
    vst1q_f32(&Crow(4, 0), C40);
    vst1q_f32(&Crow(4, 4), C41);
    vst1q_f32(&Crow(4, 8), C42);
    vst1q_f32(&Crow(5, 0), C50);
    vst1q_f32(&Crow(5, 4), C51);
    vst1q_f32(&Crow(5, 8), C52);
    vst1q_f32(&Crow(6, 0), C60);
    vst1q_f32(&Crow(6, 4), C61);
    vst1q_f32(&Crow(6, 8), C62);
}

void gemm_microkernel_Cresident_neon_4x4_prefetch_fp32(int mr, int nr, int kc, float alpha,
                                                       const float *Ar,
                                                       const float *Br,
                                                       float beta,
                                                       float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x4, with kc<=4
*/
    SET_MR_NR(4, 4);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C0, C1, C2, C3,
            A0, A1, A2, A3, A0n, B0, B0n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

    if (kc == 0)
        return;

    C0 = vmovq_n_f32(0);
    C1 = vmovq_n_f32(0);
    C2 = vmovq_n_f32(0);
    C3 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;

    Bptr = &Br[0];
    Bnr = NR;

    A0 = vld1q_f32(&Aptr[0]);
    B0 = vld1q_f32(&Bptr[0]);

    // Iterate from 1 to kc-1 (last iteration outside loop)
    if (alpha != zero) {
        for (k = 0; k < kc - 1; k++) {

            // Prefect colum/row of A/B for next iteration
            baseA = baseA + Amr;
            baseB = baseB + Bnr;

            A0n = vld1q_f32(&Aptr[baseA]);
            B0n = vld1q_f32(&Bptr[baseB]);

            C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
            C1 = vfmaq_laneq_f32(C1, B0, A0, 1);
            C2 = vfmaq_laneq_f32(C2, B0, A0, 2);
            C3 = vfmaq_laneq_f32(C3, B0, A0, 3);

            A0 = A0n;
            B0 = B0n;
        }

        // Last iteration
        C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
        C1 = vfmaq_laneq_f32(C1, B0, A0, 1);
        C2 = vfmaq_laneq_f32(C2, B0, A0, 2);
        C3 = vfmaq_laneq_f32(C3, B0, A0, 3);

        if (alpha == -one) {
            C0 = -C0;
            C1 = -C1;
            C2 = -C2;
            C3 = -C3;
        } else if (alpha != one) {
            C0 = alpha * C0;
            C1 = alpha * C1;
            C2 = alpha * C2;
            C3 = alpha * C3;
        }
    }

    if ((mr < MR) || (nr < NR)) {
        vst1q_f32(&Ctrow(0, 0), C0);
        vst1q_f32(&Ctrow(0, 1), C1);
        vst1q_f32(&Ctrow(0, 2), C2);
        vst1q_f32(&Ctrow(0, 3), C3);
        //printf("Ci(0,0) = %16.10g\n", Crow(0,0));
        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
        //printf("Cf(0,0) = %16.10g\n", Crow(0,0));
    } else if ((mr == MR) && (nr == NR)) {
        printf("here 1\n");
        if (beta != zero) {
            A0 = vld1q_f32(&Crow(0, 0));
            A1 = vld1q_f32(&Crow(1, 0));
            A2 = vld1q_f32(&Crow(2, 0));
            A3 = vld1q_f32(&Crow(3, 0));

            C0 = beta * A0 + C0;
            C1 = beta * A1 + C1;
            C2 = beta * A2 + C2;
            C3 = beta * A3 + C3;
        }

        vst1q_f32(&Crow(0, 0), C0);
        vst1q_f32(&Crow(1, 0), C1);
        vst1q_f32(&Crow(2, 0), C2);
        vst1q_f32(&Crow(3, 0), C3);
    } else {
        printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

/*
void gemm_microkernel_Cresident_neon_4x4_nopackA_fp32(int mr, int nr, int kc, float alpha,
                                                      const float *Ar,
                                                      const float *Br,
                                                      float beta,
                                                      float *C, int ldC ){

  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Specific: only for MRxNR = 4x4, with kc<=4
    SET_MR_NR(4, 4);

  int         i, j, k, baseB = 0, ldCt = MR, Amr, Bnr, ldA = NR;
  float32x4_t C0, C1, C2, C3,
	      A0, A1, A2, A3, B0;
  float       zero = 0.0, one = 1.0, *Bptr, Ctmp[MR*NR], Atmp[MR];

  if ( kc==0 )
    return;

  C0 = vmovq_n_f32(0);
  C1 = vmovq_n_f32(0);
  C2 = vmovq_n_f32(0);
  C3 = vmovq_n_f32(0);

  Bptr = &Br[0];
  Bnr  = NR;
  for ( i=mr; i<MR; i++ )
    Atmp[i] = 0.0;

  // Iterate from 1 to kc
  if ( alpha!=zero ) {
    for ( k=0; k<kc; k++ ) {

      // Load column/row of A/B for current iteration
      for ( i=0; i<mr; i++ )
        Atmp[i] = Arow(i,k);
      A0 = vld1q_f32(&Atmp[0]);
      B0 = vld1q_f32(&Bptr[baseB]);

      C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
      C1 = vfmaq_laneq_f32(C1, B0, A0, 1);
      C2 = vfmaq_laneq_f32(C2, B0, A0, 2);
      C3 = vfmaq_laneq_f32(C3, B0, A0, 3);

      baseB = baseB+Bnr;
    }

    if ( alpha==-one ) {
      C0 = -C0; C1 = -C1; C2 = -C2; C3 = -C3;
    }
    else if ( alpha!=one ) {
      C0 = alpha*C0; C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3;
    }
  }

  if ( (mr<MR)||(nr<NR) ) {
    vst1q_f32(&Ctref(0,0), C0);
    vst1q_f32(&Ctref(0,1), C1);
    vst1q_f32(&Ctref(0,2), C2);
    vst1q_f32(&Ctref(0,3), C3);
    if ( beta!=zero ) {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = beta*Crow(i,j) + Ctrow(i,j);
    }
    else {
      for ( j=0; j<nr; j++ )
        for ( i=0; i<mr; i++ )
          Crow(i,j) = Ctrow(i,j);
    }
  }
  else if ( (mr==MR)&&(nr==NR) ) {
    if ( beta!=zero ) {
      A0 = vld1q_f32(&Crow(0,0));
      A1 = vld1q_f32(&Crow(1,0));
      A2 = vld1q_f32(&Crow(2,0));
      A3 = vld1q_f32(&Crow(3,0));

      C0 = beta*A0 + C0;
      C1 = beta*A1 + C1;
      C2 = beta*A2 + C2;
      C3 = beta*A3 + C3;
    }

    vst1q_f32(&Crow(0,0), C0);
    vst1q_f32(&Crow(1,0), C1);
    vst1q_f32(&Crow(2,0), C2);
    vst1q_f32(&Crow(3,0), C3);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
    exit(-1);
  }
  }*/

void gemm_microkernel_Cresident_neon_7x12_fp32(int mr, int nr, int kc, float alpha,
                                               const float *Ar,
                                               const float *Br,
                                               float beta,
                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = 4;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);

    Bptr = &Br[0];
    Bnr = NR;
    for (i = 0; i < MR; i++)
        for (j = 0; j < ldAt; j++)
            Atrow(i, j) = 0.0;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc; k++) {

            // Load column/row of A/B for current iteration
            for (i = 0; i < mr; i++)
                Atrow(i, 0) = Arow(i, k);

            COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h

            baseB = baseB + Bnr;
        }

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
            C40 = -C40;
            C41 = -C41;
            C42 = -C42;
            C50 = -C50;
            C51 = -C51;
            C52 = -C52;
            C60 = -C60;
            C61 = -C61;
            C62 = -C62;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
            C40 = alpha * C40;
            C41 = alpha * C41;
            C42 = alpha * C42;
            C50 = alpha * C50;
            C51 = alpha * C51;
            C52 = alpha * C52;
            C60 = alpha * C60;
            C61 = alpha * C61;
            C62 = alpha * C62;
        }
    }

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40);
        vst1q_f32(&Ctrow(4, 4), C41);
        vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50);
        vst1q_f32(&Ctrow(5, 4), C51);
        vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60);
        vst1q_f32(&Ctrow(6, 4), C61);
        vst1q_f32(&Ctrow(6, 8), C62);

        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            A0 = vld1q_f32(&Crow(0, 0));
            A1 = vld1q_f32(&Crow(0, 4));
            A2 = vld1q_f32(&Crow(0, 8));
            C00 = beta * A0 + C00;
            C01 = beta * A1 + C01;
            C02 = beta * A2 + C02;

            A0 = vld1q_f32(&Crow(1, 0));
            A1 = vld1q_f32(&Crow(1, 4));
            A2 = vld1q_f32(&Crow(1, 8));
            C10 = beta * A0 + C10;
            C11 = beta * A1 + C11;
            C12 = beta * A2 + C12;

            A0 = vld1q_f32(&Crow(2, 0));
            A1 = vld1q_f32(&Crow(2, 4));
            A2 = vld1q_f32(&Crow(2, 8));
            C20 = beta * A0 + C20;
            C21 = beta * A1 + C21;
            C22 = beta * A2 + C22;

            A0 = vld1q_f32(&Crow(3, 0));
            A1 = vld1q_f32(&Crow(3, 4));
            A2 = vld1q_f32(&Crow(3, 8));
            C30 = beta * A0 + C30;
            C31 = beta * A1 + C31;
            C32 = beta * A2 + C32;

            A0 = vld1q_f32(&Crow(4, 0));
            A1 = vld1q_f32(&Crow(4, 4));
            A2 = vld1q_f32(&Crow(4, 8));
            C40 = beta * A0 + C40;
            C41 = beta * A1 + C41;
            C42 = beta * A2 + C42;

            A0 = vld1q_f32(&Crow(5, 0));
            A1 = vld1q_f32(&Crow(5, 4));
            A2 = vld1q_f32(&Crow(5, 8));
            C50 = beta * A0 + C50;
            C51 = beta * A1 + C51;
            C52 = beta * A2 + C52;

            A0 = vld1q_f32(&Crow(6, 0));
            A1 = vld1q_f32(&Crow(6, 4));
            A2 = vld1q_f32(&Crow(6, 8));
            C60 = beta * A0 + C60;
            C61 = beta * A1 + C61;
            C62 = beta * A2 + C62;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
    } else {
        printf("Error: Incorrect use of 7x12 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_7x12_unroll_2_fp32(int mr, int nr, int kc, float alpha,
                                                        const float *Ar,
                                                        const float *Br,
                                                        float beta,
                                                        float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = 4;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);

    Bptr = &Br[0];
    Bnr = NR;
    for (i = 0; i < MR; i++)
        for (j = 0; j < ldAt; j++)
            Atrow(i, j) = 0.0;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc - 1; k += 2) {

            // Load columns/rows of A/B for current iteration
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
            }

            COMPUTE_KERNEL_7x12_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h

            baseB = baseB + 2 * Bnr;
        }

        if (k == kc - 1) {

            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++)
                Atrow(i, 0) = Arow(i, k);

            COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        }

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
            C40 = -C40;
            C41 = -C41;
            C42 = -C42;
            C50 = -C50;
            C51 = -C51;
            C52 = -C52;
            C60 = -C60;
            C61 = -C61;
            C62 = -C62;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
            C40 = alpha * C40;
            C41 = alpha * C41;
            C42 = alpha * C42;
            C50 = alpha * C50;
            C51 = alpha * C51;
            C52 = alpha * C52;
            C60 = alpha * C60;
            C61 = alpha * C61;
            C62 = alpha * C62;
        }
    }

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40);
        vst1q_f32(&Ctrow(4, 4), C41);
        vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50);
        vst1q_f32(&Ctrow(5, 4), C51);
        vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60);
        vst1q_f32(&Ctrow(6, 4), C61);
        vst1q_f32(&Ctrow(6, 8), C62);

        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            A0 = vld1q_f32(&Crow(0, 0));
            A1 = vld1q_f32(&Crow(0, 4));
            A2 = vld1q_f32(&Crow(0, 8));
            C00 = beta * A0 + C00;
            C01 = beta * A1 + C01;
            C02 = beta * A2 + C02;

            A0 = vld1q_f32(&Crow(1, 0));
            A1 = vld1q_f32(&Crow(1, 4));
            A2 = vld1q_f32(&Crow(1, 8));
            C10 = beta * A0 + C10;
            C11 = beta * A1 + C11;
            C12 = beta * A2 + C12;

            A0 = vld1q_f32(&Crow(2, 0));
            A1 = vld1q_f32(&Crow(2, 4));
            A2 = vld1q_f32(&Crow(2, 8));
            C20 = beta * A0 + C20;
            C21 = beta * A1 + C21;
            C22 = beta * A2 + C22;

            A0 = vld1q_f32(&Crow(3, 0));
            A1 = vld1q_f32(&Crow(3, 4));
            A2 = vld1q_f32(&Crow(3, 8));
            C30 = beta * A0 + C30;
            C31 = beta * A1 + C31;
            C32 = beta * A2 + C32;

            A0 = vld1q_f32(&Crow(4, 0));
            A1 = vld1q_f32(&Crow(4, 4));
            A2 = vld1q_f32(&Crow(4, 8));
            C40 = beta * A0 + C40;
            C41 = beta * A1 + C41;
            C42 = beta * A2 + C42;

            A0 = vld1q_f32(&Crow(5, 0));
            A1 = vld1q_f32(&Crow(5, 4));
            A2 = vld1q_f32(&Crow(5, 8));
            C50 = beta * A0 + C50;
            C51 = beta * A1 + C51;
            C52 = beta * A2 + C52;

            A0 = vld1q_f32(&Crow(6, 0));
            A1 = vld1q_f32(&Crow(6, 4));
            A2 = vld1q_f32(&Crow(6, 8));
            C60 = beta * A0 + C60;
            C61 = beta * A1 + C61;
            C62 = beta * A2 + C62;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
    } else {
        printf("Error: Incorrect use of 7x12 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}


void gemm_microkernel_Cresident_neon_7x12_nopack_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                               const float *Ar, int ldA,
                                                               const float *Br, int ldB,
                                                               float beta,
                                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldAt = 4, ldBt = NR;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);
    /*
      C00  = vld1q_f32(&Crow(0,0)); C01  = vld1q_f32(&Crow(0,4)); C02  = vld1q_f32(&Crow(0,8));
      C10  = vld1q_f32(&Crow(1,0)); C11  = vld1q_f32(&Crow(1,4)); C12  = vld1q_f32(&Crow(1,8));
      C20  = vld1q_f32(&Crow(2,0)); C21  = vld1q_f32(&Crow(2,4)); C22  = vld1q_f32(&Crow(2,8));
      C30  = vld1q_f32(&Crow(3,0)); C31  = vld1q_f32(&Crow(3,4)); C32  = vld1q_f32(&Crow(3,8));
      C40  = vld1q_f32(&Crow(4,0)); C41  = vld1q_f32(&Crow(4,4)); C42  = vld1q_f32(&Crow(4,8));
      C50  = vld1q_f32(&Crow(5,0)); C51  = vld1q_f32(&Crow(5,4)); C52  = vld1q_f32(&Crow(5,8));
      C60  = vld1q_f32(&Crow(6,0)); C61  = vld1q_f32(&Crow(6,4)); C62  = vld1q_f32(&Crow(6,8));
  */

    Bptr = &Br[0];
    Bnr = NR;
    for (i = 0; i < MR; i++)
        for (j = 0; j < ldAt; j++)
            Atrow(i, j) = 0.0;

    // Iterate from 1 to kc
    //if ( alpha!=zero ) {
    for (k = 0; k < kc - 3; k += 4) {

        // Load columns/rows of A/B for current iteration
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
            Atrow(i, 2) = Arow(i, k + 2);
            Atrow(i, 3) = Arow(i, k + 3);
        }

        COMPUTE_KERNEL_7x12_NOPACK_UNROLL_4  // Code for single iteration 7x12 in file compute_kernel_7x12.h

        baseB = baseB + 4 * ldB;
    }

    if (k == kc - 1) {
        // printf("Outside iteration k %d\n", k);
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++)
            Atrow(i, 0) = Arow(i, k);

        COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 2) {
        // printf("Outside iteration k %d %d\n", k, k+1);
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
        }

        COMPUTE_KERNEL_7x12_NOPACK_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 3) {
        // printf("Outside iteration k %d %d\n", k, k+1);
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
            Atrow(i, 2) = Arow(i, k + 2);
        }

        COMPUTE_KERNEL_7x12_NOPACK_UNROLL_3  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    }

    /*
    if ( alpha==-one ) {
      C00 = -C00; C01 = -C01; C02 = -C02;
      C10 = -C10; C11 = -C11; C12 = -C12;
      C20 = -C20; C21 = -C21; C22 = -C22;
      C30 = -C30; C31 = -C31; C32 = -C32;
      C40 = -C40; C41 = -C41; C42 = -C42;
      C50 = -C50; C51 = -C51; C52 = -C52;
      C60 = -C60; C61 = -C61; C62 = -C62;
    }
    else if ( alpha!=one ) {
      C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02;
      C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12;
      C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22;
      C30 = alpha*C30; C31 = alpha*C31; C32 = alpha*C32;
      C40 = alpha*C40; C41 = alpha*C41; C42 = alpha*C42;
      C50 = alpha*C50; C51 = alpha*C51; C52 = alpha*C52;
      C60 = alpha*C60; C61 = alpha*C61; C62 = alpha*C62;
    }
    */
    //}

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40);
        vst1q_f32(&Ctrow(4, 4), C41);
        vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50);
        vst1q_f32(&Ctrow(5, 4), C51);
        vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60);
        vst1q_f32(&Ctrow(6, 4), C61);
        vst1q_f32(&Ctrow(6, 8), C62);

        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        //if ( beta!=zero ) {
        A0 = vld1q_f32(&Crow(0, 0));
        A1 = vld1q_f32(&Crow(0, 4));
        A2 = vld1q_f32(&Crow(0, 8));
        C00 = A0 + C00;
        C01 = A1 + C01;
        C02 = A2 + C02;

        A0 = vld1q_f32(&Crow(1, 0));
        A1 = vld1q_f32(&Crow(1, 4));
        A2 = vld1q_f32(&Crow(1, 8));
        C10 = A0 + C10;
        C11 = A1 + C11;
        C12 = A2 + C12;

        A0 = vld1q_f32(&Crow(2, 0));
        A1 = vld1q_f32(&Crow(2, 4));
        A2 = vld1q_f32(&Crow(2, 8));
        C20 = A0 + C20;
        C21 = A1 + C21;
        C22 = A2 + C22;

        A0 = vld1q_f32(&Crow(3, 0));
        A1 = vld1q_f32(&Crow(3, 4));
        A2 = vld1q_f32(&Crow(3, 8));
        C30 = A0 + C30;
        C31 = A1 + C31;
        C32 = A2 + C32;

        A0 = vld1q_f32(&Crow(4, 0));
        A1 = vld1q_f32(&Crow(4, 4));
        A2 = vld1q_f32(&Crow(4, 8));
        C40 = A0 + C40;
        C41 = A1 + C41;
        C42 = A2 + C42;

        A0 = vld1q_f32(&Crow(5, 0));
        A1 = vld1q_f32(&Crow(5, 4));
        A2 = vld1q_f32(&Crow(5, 8));
        C50 = A0 + C50;
        C51 = A1 + C51;
        C52 = A2 + C52;

        A0 = vld1q_f32(&Crow(6, 0));
        A1 = vld1q_f32(&Crow(6, 4));
        A2 = vld1q_f32(&Crow(6, 8));
        C60 = A0 + C60;
        C61 = A1 + C61;
        C62 = A2 + C62;
        //}

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
    } else {
        printf("Error: Incorrect use of 7x12 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                                const float *Ar, int ldA,
                                                                const float *Br,
                                                                float beta,
                                                                float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldAt = 4, ldBt = NR;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);

    Bptr = &Br[0];
    Bnr = NR;
    for (i = 0; i < MR; i++)
        for (j = 0; j < ldAt; j++)
            Atrow(i, j) = 0.0;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc - 3; k += 4) {

            // Load columns/rows of A/B for current iteration
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
                Atrow(i, 2) = Arow(i, k + 2);
                Atrow(i, 3) = Arow(i, k + 3);
            }

            COMPUTE_KERNEL_7x12_UNROLL_4  // Code for single iteration 7x12 in file compute_kernel_7x12.h

            baseB = baseB + 4 * Bnr;
        }

        if (k == kc - 1) {
            // printf("Outside iteration k %d\n", k);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++)
                Atrow(i, 0) = Arow(i, k);

            COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        } else if (k == kc - 2) {
            // printf("Outside iteration k %d %d\n", k, k+1);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
            }

            COMPUTE_KERNEL_7x12_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        } else if (k == kc - 3) {
            // printf("Outside iteration k %d %d\n", k, k+1);
            for (i = 0; i < MR; i++)
                for (j = 0; j < ldAt; j++)
                    Atrow(i, j) = 0.0;
            for (i = 0; i < mr; i++) {
                Atrow(i, 0) = Arow(i, k);
                Atrow(i, 1) = Arow(i, k + 1);
                Atrow(i, 2) = Arow(i, k + 2);
            }

            COMPUTE_KERNEL_7x12_UNROLL_3  // Code for single iteration 7x12 in file compute_kernel_7x12.h
        }

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
            C40 = -C40;
            C41 = -C41;
            C42 = -C42;
            C50 = -C50;
            C51 = -C51;
            C52 = -C52;
            C60 = -C60;
            C61 = -C61;
            C62 = -C62;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
            C40 = alpha * C40;
            C41 = alpha * C41;
            C42 = alpha * C42;
            C50 = alpha * C50;
            C51 = alpha * C51;
            C52 = alpha * C52;
            C60 = alpha * C60;
            C61 = alpha * C61;
            C62 = alpha * C62;
        }
    }

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40);
        vst1q_f32(&Ctrow(4, 4), C41);
        vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50);
        vst1q_f32(&Ctrow(5, 4), C51);
        vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60);
        vst1q_f32(&Ctrow(6, 4), C61);
        vst1q_f32(&Ctrow(6, 8), C62);

        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            A0 = vld1q_f32(&Crow(0, 0));
            A1 = vld1q_f32(&Crow(0, 4));
            A2 = vld1q_f32(&Crow(0, 8));
            C00 = beta * A0 + C00;
            C01 = beta * A1 + C01;
            C02 = beta * A2 + C02;

            A0 = vld1q_f32(&Crow(1, 0));
            A1 = vld1q_f32(&Crow(1, 4));
            A2 = vld1q_f32(&Crow(1, 8));
            C10 = beta * A0 + C10;
            C11 = beta * A1 + C11;
            C12 = beta * A2 + C12;

            A0 = vld1q_f32(&Crow(2, 0));
            A1 = vld1q_f32(&Crow(2, 4));
            A2 = vld1q_f32(&Crow(2, 8));
            C20 = beta * A0 + C20;
            C21 = beta * A1 + C21;
            C22 = beta * A2 + C22;

            A0 = vld1q_f32(&Crow(3, 0));
            A1 = vld1q_f32(&Crow(3, 4));
            A2 = vld1q_f32(&Crow(3, 8));
            C30 = beta * A0 + C30;
            C31 = beta * A1 + C31;
            C32 = beta * A2 + C32;

            A0 = vld1q_f32(&Crow(4, 0));
            A1 = vld1q_f32(&Crow(4, 4));
            A2 = vld1q_f32(&Crow(4, 8));
            C40 = beta * A0 + C40;
            C41 = beta * A1 + C41;
            C42 = beta * A2 + C42;

            A0 = vld1q_f32(&Crow(5, 0));
            A1 = vld1q_f32(&Crow(5, 4));
            A2 = vld1q_f32(&Crow(5, 8));
            C50 = beta * A0 + C50;
            C51 = beta * A1 + C51;
            C52 = beta * A2 + C52;

            A0 = vld1q_f32(&Crow(6, 0));
            A1 = vld1q_f32(&Crow(6, 4));
            A2 = vld1q_f32(&Crow(6, 8));
            C60 = beta * A0 + C60;
            C61 = beta * A1 + C61;
            C62 = beta * A2 + C62;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
    } else {
        printf("Error: Incorrect use of 7x12 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_7x12_fixed_nopackA_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                                      const float *Ar, int ldA,
                                                                      const float *Br,
                                                                      float beta,
                                                                      float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 7x12, with kc<=12
*/
    SET_MR_NR(7, 12);

    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldAt = 4, ldBt = NR;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, *Bptr, Ctmp[MR * NR], Atmp[MR * 4];

    if (kc == 0)
        return;

    // C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0);
    // C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
    // C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0);
    // C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0);
    // C40 = vmovq_n_f32(0); C41 = vmovq_n_f32(0); C42 = vmovq_n_f32(0);
    // C50 = vmovq_n_f32(0); C51 = vmovq_n_f32(0); C52 = vmovq_n_f32(0);
    // C60 = vmovq_n_f32(0); C61 = vmovq_n_f32(0); C62 = vmovq_n_f32(0);
    C00 = vld1q_f32(&Crow(0, 0));
    C01 = vld1q_f32(&Crow(0, 4));
    C02 = vld1q_f32(&Crow(0, 8));
    C10 = vld1q_f32(&Crow(1, 0));
    C11 = vld1q_f32(&Crow(1, 4));
    C12 = vld1q_f32(&Crow(1, 8));
    C20 = vld1q_f32(&Crow(2, 0));
    C21 = vld1q_f32(&Crow(2, 4));
    C22 = vld1q_f32(&Crow(2, 8));
    C30 = vld1q_f32(&Crow(3, 0));
    C31 = vld1q_f32(&Crow(3, 4));
    C32 = vld1q_f32(&Crow(3, 8));
    C40 = vld1q_f32(&Crow(4, 0));
    C41 = vld1q_f32(&Crow(4, 4));
    C42 = vld1q_f32(&Crow(4, 8));
    C50 = vld1q_f32(&Crow(5, 0));
    C51 = vld1q_f32(&Crow(5, 4));
    C52 = vld1q_f32(&Crow(5, 8));
    C60 = vld1q_f32(&Crow(6, 0));
    C61 = vld1q_f32(&Crow(6, 4));
    C62 = vld1q_f32(&Crow(6, 8));

    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    for (k = 0; k < kc - 3; k += 4) {

        A0 = vld1q_f32(&Arow(0, k)); \
      A1 = vld1q_f32(&Arow(1, k)); \
      A2 = vld1q_f32(&Arow(2, k)); \
      A3 = vld1q_f32(&Arow(3, k)); \
      A4 = vld1q_f32(&Arow(4, k)); \
      A5 = vld1q_f32(&Arow(5, k)); \
      A6 = vld1q_f32(&Arow(6, k)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB + 4]); \
      B2 = vld1q_f32(&Bptr[baseB + 8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 0);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 0);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 0);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 0);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 0);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 0);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB + 12]); \
      B1 = vld1q_f32(&Bptr[baseB + 16]); \
      B2 = vld1q_f32(&Bptr[baseB + 20]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 1);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 1);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 1);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 1);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 1);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 1);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 1);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB + 24]); \
      B1 = vld1q_f32(&Bptr[baseB + 28]); \
      B2 = vld1q_f32(&Bptr[baseB + 32]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 2);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 2);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 2);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 2);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 2);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 2);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 2);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
\
      B0 = vld1q_f32(&Bptr[baseB + 36]); \
      B1 = vld1q_f32(&Bptr[baseB + 40]); \
      B2 = vld1q_f32(&Bptr[baseB + 44]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 3);
        C01 = vfmaq_laneq_f32(C01, B1, A0, 3);
        C02 = vfmaq_laneq_f32(C02, B2, A0, 3); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 3);
        C11 = vfmaq_laneq_f32(C11, B1, A1, 3);
        C12 = vfmaq_laneq_f32(C12, B2, A1, 3); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 3);
        C21 = vfmaq_laneq_f32(C21, B1, A2, 3);
        C22 = vfmaq_laneq_f32(C22, B2, A2, 3); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 3);
        C31 = vfmaq_laneq_f32(C31, B1, A3, 3);
        C32 = vfmaq_laneq_f32(C32, B2, A3, 3); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 3);
        C41 = vfmaq_laneq_f32(C41, B1, A4, 3);
        C42 = vfmaq_laneq_f32(C42, B2, A4, 3); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 3);
        C51 = vfmaq_laneq_f32(C51, B1, A5, 3);
        C52 = vfmaq_laneq_f32(C52, B2, A5, 3); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 3);
        C61 = vfmaq_laneq_f32(C61, B1, A6, 3);
        C62 = vfmaq_laneq_f32(C62, B2, A6, 3); \

        baseB = baseB + 4 * Bnr;
    }

    if (k == kc - 1) {
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++)
            Atrow(i, 0) = Arow(i, k);

        COMPUTE_KERNEL_7x12  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 2) {
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
        }

        COMPUTE_KERNEL_7x12_UNROLL_2  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    } else if (k == kc - 3) {
        // printf("Outside iteration k %d %d\n", k, k+1);
        for (i = 0; i < MR; i++)
            for (j = 0; j < ldAt; j++)
                Atrow(i, j) = 0.0;
        for (i = 0; i < mr; i++) {
            Atrow(i, 0) = Arow(i, k);
            Atrow(i, 1) = Arow(i, k + 1);
            Atrow(i, 2) = Arow(i, k + 2);
        }

        COMPUTE_KERNEL_7x12_UNROLL_3  // Code for single iteration 7x12 in file compute_kernel_7x12.h
    }

    /*
      A0  = vld1q_f32(&Crow(0,0)); A1  = vld1q_f32(&Crow(0,4)); A2  = vld1q_f32(&Crow(0,8));
      C00 = A0 + C00;         C01 = A1 + C01;         C02 = A2 + C02;

      A0  = vld1q_f32(&Crow(1,0)); A1  = vld1q_f32(&Crow(1,4)); A2  = vld1q_f32(&Crow(1,8));
      C10 = A0 + C10;         C11 = A1 + C11;         C12 = A2 + C12;

      A0  = vld1q_f32(&Crow(2,0)); A1  = vld1q_f32(&Crow(2,4)); A2  = vld1q_f32(&Crow(2,8));
      C20 = A0 + C20;         C21 = A1 + C21;         C22 = A2 + C22;

      A0  = vld1q_f32(&Crow(3,0)); A1  = vld1q_f32(&Crow(3,4)); A2  = vld1q_f32(&Crow(3,8));
      C30 = A0 + C30;         C31 = A1 + C31;         C32 = A2 + C32;

      A0  = vld1q_f32(&Crow(4,0)); A1  = vld1q_f32(&Crow(4,4)); A2  = vld1q_f32(&Crow(4,8));
      C40 = A0 + C40;         C41 = A1 + C41;         C42 = A2 + C42;

      A0  = vld1q_f32(&Crow(5,0)); A1  = vld1q_f32(&Crow(5,4)); A2  = vld1q_f32(&Crow(5,8));
      C50 = A0 + C50;         C51 = A1 + C51;         C52 = A2 + C52;

      A0  = vld1q_f32(&Crow(6,0)); A1  = vld1q_f32(&Crow(6,4)); A2  = vld1q_f32(&Crow(6,8));
      C60 = A0 + C60;         C61 = A1 + C61;         C62 = A2 + C62;
    */

    vst1q_f32(&Crow(0, 0), C00);
    vst1q_f32(&Crow(0, 4), C01);
    vst1q_f32(&Crow(0, 8), C02);
    vst1q_f32(&Crow(1, 0), C10);
    vst1q_f32(&Crow(1, 4), C11);
    vst1q_f32(&Crow(1, 8), C12);
    vst1q_f32(&Crow(2, 0), C20);
    vst1q_f32(&Crow(2, 4), C21);
    vst1q_f32(&Crow(2, 8), C22);
    vst1q_f32(&Crow(3, 0), C30);
    vst1q_f32(&Crow(3, 4), C31);
    vst1q_f32(&Crow(3, 8), C32);
    vst1q_f32(&Crow(4, 0), C40);
    vst1q_f32(&Crow(4, 4), C41);
    vst1q_f32(&Crow(4, 8), C42);
    vst1q_f32(&Crow(5, 0), C50);
    vst1q_f32(&Crow(5, 4), C51);
    vst1q_f32(&Crow(5, 8), C52);
    vst1q_f32(&Crow(6, 0), C60);
    vst1q_f32(&Crow(6, 4), C61);
    vst1q_f32(&Crow(6, 8), C62);
}

void gemm_microkernel_Cresident_neon_8x12_fp32(int mr, int nr, int kc, float alpha,
                                               const float *Ar,
                                               const float *Br,
                                               float beta,
                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
*/
    SET_MR_NR(8, 12);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            C70, C71, C72,
            A0, A1,
            B0, B1, B2,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C00n A0
#define C01n A1
#define C02n B0n
#define C10n B1n
#define C11n B2n
#define C12n B0
#define C0n A0
#define C1n A1
#define C2n B0
#define C3n B1
#define C4n B2
#define C5n B0n
#define C6n B1n
#define C7n B2n

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0); C41 = vmovq_n_f32(0); C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0); C51 = vmovq_n_f32(0); C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0); C61 = vmovq_n_f32(0); C62 = vmovq_n_f32(0);
    C70 = vmovq_n_f32(0); C71 = vmovq_n_f32(0); C72 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    if ( alpha!=zero ) {
        if ((mr <= 4) && (nr <= 4))
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if ((mr <= 4) && (nr <= 8))
            for (k = 0; k < kc; k++) {
    
                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]);
    
                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3);
    
                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if ((mr <= 4) && (nr <= 12))
            for (k = 0; k < kc; k++) {
    
                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]);
    
                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3);
    
                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if ((mr <= 8) && (nr <= 4))
            for (k = 0; k < kc; k++) {
    
                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                A1 = vld1q_f32(&Aptr[baseA + 4]);
                B0 = vld1q_f32(&Bptr[baseB + 0]);
    
                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3);
    
                C40 = vfmaq_laneq_f32(C40, B0, A1, 0);
                C50 = vfmaq_laneq_f32(C50, B0, A1, 1);
                C60 = vfmaq_laneq_f32(C60, B0, A1, 2);
                C70 = vfmaq_laneq_f32(C70, B0, A1, 3);
    
                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if ((mr <= 8) && (nr <= 8))
            for (k = 0; k < kc; k++) {
    
                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                A1 = vld1q_f32(&Aptr[baseA + 4]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]);
    
                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3);
    
                C40 = vfmaq_laneq_f32(C40, B0, A1, 0); C41 = vfmaq_laneq_f32(C41, B1, A1, 0);
                C50 = vfmaq_laneq_f32(C50, B0, A1, 1); C51 = vfmaq_laneq_f32(C51, B1, A1, 1);
                C60 = vfmaq_laneq_f32(C60, B0, A1, 2); C61 = vfmaq_laneq_f32(C61, B1, A1, 2);
                C70 = vfmaq_laneq_f32(C70, B0, A1, 3); C71 = vfmaq_laneq_f32(C71, B1, A1, 3);
    
                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else
            for (k = 0; k < kc; k++) {
    
                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                A1 = vld1q_f32(&Aptr[baseA + 4]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]);
    
                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3);
    
                C40 = vfmaq_laneq_f32(C40, B0, A1, 0); C41 = vfmaq_laneq_f32(C41, B1, A1, 0); C42 = vfmaq_laneq_f32(C42, B2, A1, 0);
                C50 = vfmaq_laneq_f32(C50, B0, A1, 1); C51 = vfmaq_laneq_f32(C51, B1, A1, 1); C52 = vfmaq_laneq_f32(C52, B2, A1, 1);
                C60 = vfmaq_laneq_f32(C60, B0, A1, 2); C61 = vfmaq_laneq_f32(C61, B1, A1, 2); C62 = vfmaq_laneq_f32(C62, B2, A1, 2);
                C70 = vfmaq_laneq_f32(C70, B0, A1, 3); C71 = vfmaq_laneq_f32(C71, B1, A1, 3); C72 = vfmaq_laneq_f32(C72, B2, A1, 3);
    
                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }

        if ( alpha==-one ) {
            C00 = -C00; C01 = -C01; C02 = -C02;
            C10 = -C10; C11 = -C11; C12 = -C12;
            C20 = -C20; C21 = -C21; C22 = -C22;
            C30 = -C30; C31 = -C31; C32 = -C32;
            C40 = -C40; C41 = -C41; C42 = -C42;
            C50 = -C50; C51 = -C51; C52 = -C52;
            C60 = -C60; C61 = -C61; C62 = -C62;
            C70 = -C70; C71 = -C71; C72 = -C72;
        }
        else if ( alpha!=one ) {
            C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02;
            C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12;
            C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22;
            C30 = alpha*C30; C31 = alpha*C31; C32 = alpha*C32;
            C40 = alpha*C40; C41 = alpha*C41; C42 = alpha*C42;
            C50 = alpha*C50; C51 = alpha*C51; C52 = alpha*C52;
            C60 = alpha*C60; C61 = alpha*C61; C62 = alpha*C62;
            C70 = alpha*C70; C71 = alpha*C71; C72 = alpha*C72;
        }
    }

    if ((mr < MR) || (nr < NR)) {
        vst1q_f32(&Ctrow(0, 0), C00); vst1q_f32(&Ctrow(0, 4), C01); vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10); vst1q_f32(&Ctrow(1, 4), C11); vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20); vst1q_f32(&Ctrow(2, 4), C21); vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30); vst1q_f32(&Ctrow(3, 4), C31); vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(4, 0), C40); vst1q_f32(&Ctrow(4, 4), C41); vst1q_f32(&Ctrow(4, 8), C42);
        vst1q_f32(&Ctrow(5, 0), C50); vst1q_f32(&Ctrow(5, 4), C51); vst1q_f32(&Ctrow(5, 8), C52);
        vst1q_f32(&Ctrow(6, 0), C60); vst1q_f32(&Ctrow(6, 4), C61); vst1q_f32(&Ctrow(6, 8), C62);
        vst1q_f32(&Ctrow(7, 0), C70); vst1q_f32(&Ctrow(7, 4), C71); vst1q_f32(&Ctrow(7, 8), C72);
        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else {
        printf("Error: Incorrect use of non-fixed micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_8x12_fixed_fp32(int mr, int nr, int kc, float alpha,
                                                     const float *Ar,
                                                     const float *Br,
                                                     float beta,
                                                     float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 8x12
*/
    SET_MR_NR(8, 12);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            C40, C41, C42,
            C50, C51, C52,
            C60, C61, C62,
            C70, C71, C72,
            A0, A1,
            B0, B1, B2,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C00n A0
#define C01n A1
#define C02n B0n
#define C10n B1n
#define C11n B2n
#define C12n B0
#define C0n A0
#define C1n A1
#define C2n B0
#define C3n B1
#define C4n B2
#define C5n B0n
#define C6n B1n
#define C7n B2n

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0); C41 = vmovq_n_f32(0); C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0); C51 = vmovq_n_f32(0); C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0); C61 = vmovq_n_f32(0); C62 = vmovq_n_f32(0);
    C70 = vmovq_n_f32(0); C71 = vmovq_n_f32(0); C72 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    if ( alpha!=zero ) {
        for (k = 0; k < kc; k++) {

            // Load colums/rows of A/B for current iteration
            A0 = vld1q_f32(&Aptr[baseA + 0]);
            A1 = vld1q_f32(&Aptr[baseA + 4]);
            B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]);

            /* Compute */
            C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
            C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
            C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
            C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3);

            C40 = vfmaq_laneq_f32(C40, B0, A1, 0); C41 = vfmaq_laneq_f32(C41, B1, A1, 0); C42 = vfmaq_laneq_f32(C42, B2, A1, 0);
            C50 = vfmaq_laneq_f32(C50, B0, A1, 1); C51 = vfmaq_laneq_f32(C51, B1, A1, 1); C52 = vfmaq_laneq_f32(C52, B2, A1, 1);
            C60 = vfmaq_laneq_f32(C60, B0, A1, 2); C61 = vfmaq_laneq_f32(C61, B1, A1, 2); C62 = vfmaq_laneq_f32(C62, B2, A1, 2);
            C70 = vfmaq_laneq_f32(C70, B0, A1, 3); C71 = vfmaq_laneq_f32(C71, B1, A1, 3); C72 = vfmaq_laneq_f32(C72, B2, A1, 3);

            baseA = baseA + Amr;
            baseB = baseB + Bnr;
        }

        if ( alpha==-one ) {
            C00 = -C00; C01 = -C01; C02 = -C02;
            C10 = -C10; C11 = -C11; C12 = -C12;
            C20 = -C20; C21 = -C21; C22 = -C22;
            C30 = -C30; C31 = -C31; C32 = -C32;
            C40 = -C40; C41 = -C41; C42 = -C42;
            C50 = -C50; C51 = -C51; C52 = -C52;
            C60 = -C60; C61 = -C61; C62 = -C62;
            C70 = -C70; C71 = -C71; C72 = -C72;
        }
        else if ( alpha!=one ) {
            C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02;
            C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12;
            C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22;
            C30 = alpha*C30; C31 = alpha*C31; C32 = alpha*C32;
            C40 = alpha*C40; C41 = alpha*C41; C42 = alpha*C42;
            C50 = alpha*C50; C51 = alpha*C51; C52 = alpha*C52;
            C60 = alpha*C60; C61 = alpha*C61; C62 = alpha*C62;
            C70 = alpha*C70; C71 = alpha*C71; C72 = alpha*C72;
        }
    }

    if ((mr == MR) && (nr == NR)) {
        C0n = vld1q_f32(&Crow(0, 0));
        C1n = vld1q_f32(&Crow(1, 0));
        C2n = vld1q_f32(&Crow(2, 0));
        C3n = vld1q_f32(&Crow(3, 0));
        C4n = vld1q_f32(&Crow(4, 0));
        C5n = vld1q_f32(&Crow(5, 0));
        C6n = vld1q_f32(&Crow(6, 0));
        C7n = vld1q_f32(&Crow(7, 0));

        C00 = C0n + C00;
        vst1q_f32(&Crow(0, 0), C00);
        C10 = C1n + C10;
        vst1q_f32(&Crow(1, 0), C10);
        C20 = C2n + C20;
        vst1q_f32(&Crow(2, 0), C20);
        C30 = C3n + C30;
        vst1q_f32(&Crow(3, 0), C30);
        C40 = C4n + C40;
        vst1q_f32(&Crow(4, 0), C40);
        C50 = C5n + C50;
        vst1q_f32(&Crow(5, 0), C50);
        C60 = C6n + C60;
        vst1q_f32(&Crow(6, 0), C60);
        C70 = C7n + C70;
        vst1q_f32(&Crow(7, 0), C70);

        C0n = vld1q_f32(&Crow(0, 4));
        C1n = vld1q_f32(&Crow(1, 4));
        C2n = vld1q_f32(&Crow(2, 4));
        C3n = vld1q_f32(&Crow(3, 4));
        C4n = vld1q_f32(&Crow(4, 4));
        C5n = vld1q_f32(&Crow(5, 4));
        C6n = vld1q_f32(&Crow(6, 4));
        C7n = vld1q_f32(&Crow(7, 4));

        C01 = C0n + C01;
        vst1q_f32(&Crow(0, 4), C01);
        C11 = C1n + C11;
        vst1q_f32(&Crow(1, 4), C11);
        C21 = C2n + C21;
        vst1q_f32(&Crow(2, 4), C21);
        C31 = C3n + C31;
        vst1q_f32(&Crow(3, 4), C31);
        C41 = C4n + C41;
        vst1q_f32(&Crow(4, 4), C41);
        C51 = C5n + C51;
        vst1q_f32(&Crow(5, 4), C51);
        C61 = C6n + C61;
        vst1q_f32(&Crow(6, 4), C61);
        C71 = C7n + C71;
        vst1q_f32(&Crow(7, 4), C71);

        C0n = vld1q_f32(&Crow(0, 8));
        C1n = vld1q_f32(&Crow(1, 8));
        C2n = vld1q_f32(&Crow(2, 8));
        C3n = vld1q_f32(&Crow(3, 8));
        C4n = vld1q_f32(&Crow(4, 8));
        C5n = vld1q_f32(&Crow(5, 8));
        C6n = vld1q_f32(&Crow(6, 8));
        C7n = vld1q_f32(&Crow(7, 8));

        C02 = C0n + C02;
        vst1q_f32(&Crow(0, 8), C02);
        C12 = C1n + C12;
        vst1q_f32(&Crow(1, 8), C12);
        C22 = C2n + C22;
        vst1q_f32(&Crow(2, 8), C22);
        C32 = C3n + C32;
        vst1q_f32(&Crow(3, 8), C32);
        C42 = C4n + C42;
        vst1q_f32(&Crow(4, 8), C42);
        C52 = C5n + C52;
        vst1q_f32(&Crow(5, 8), C52);
        C62 = C6n + C62;
        vst1q_f32(&Crow(6, 8), C62);
        C72 = C7n + C72;
        vst1q_f32(&Crow(7, 8), C72);

	/*
        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(4, 0), C40);
        vst1q_f32(&Crow(4, 4), C41);
        vst1q_f32(&Crow(4, 8), C42);
        vst1q_f32(&Crow(5, 0), C50);
        vst1q_f32(&Crow(5, 4), C51);
        vst1q_f32(&Crow(5, 8), C52);
        vst1q_f32(&Crow(6, 0), C60);
        vst1q_f32(&Crow(6, 4), C61);
        vst1q_f32(&Crow(6, 8), C62);
        vst1q_f32(&Crow(7, 0), C70);
        vst1q_f32(&Crow(7, 4), C71);
        vst1q_f32(&Crow(7, 8), C72);
	*/
    } else {
        printf("Error: Incorrect use of fixed micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_4x12_fp32(int mr, int nr, int kc, float alpha,
                                               const float *Ar,
                                               const float *Br,
                                               float beta,
                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x12
*/
    SET_MR_NR(4, 12);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02,
            C10, C11, C12,
            C20, C21, C22,
            C30, C31, C32,
            A0,
            B0, B1, B2,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C0p A0
#define C1p B0
#define C2p B1
#define C3p B2

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc; k++) {

            // Load colums/rows of A/B for current iteration
            A0 = vld1q_f32(&Aptr[baseA + 0]);
            B0 = vld1q_f32(&Bptr[baseB + 0]);
            B1 = vld1q_f32(&Bptr[baseB + 4]);
            B2 = vld1q_f32(&Bptr[baseB + 8]);

            /*** */
            C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
            C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
            C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
            C10 = vfmaq_laneq_f32(C10, B0, A0, 1);
            C11 = vfmaq_laneq_f32(C11, B1, A0, 1);
            C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
            C20 = vfmaq_laneq_f32(C20, B0, A0, 2);
            C21 = vfmaq_laneq_f32(C21, B1, A0, 2);
            C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
            C30 = vfmaq_laneq_f32(C30, B0, A0, 3);
            C31 = vfmaq_laneq_f32(C31, B1, A0, 3);
            C32 = vfmaq_laneq_f32(C32, B2, A0, 3);

            baseA = baseA + Amr;
            baseB = baseB + Bnr;
        }

        // Last iteration

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
        }
    }

    if ((mr < MR) || (nr < NR)) {
        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            C0p = vld1q_f32(&Crow(0, 0));
            C1p = vld1q_f32(&Crow(1, 0));
            C2p = vld1q_f32(&Crow(2, 0));
            C3p = vld1q_f32(&Crow(3, 0));

            C00 = beta * C0p + C00;
            C10 = beta * C1p + C10;
            C20 = beta * C2p + C20;
            C30 = beta * C3p + C30;

            C0p = vld1q_f32(&Crow(0, 4));
            C1p = vld1q_f32(&Crow(1, 4));
            C2p = vld1q_f32(&Crow(2, 4));
            C3p = vld1q_f32(&Crow(3, 4));

            C01 = beta * C0p + C01;
            C11 = beta * C1p + C11;
            C21 = beta * C2p + C21;
            C31 = beta * C3p + C31;

            C0p = vld1q_f32(&Crow(0, 8));
            C1p = vld1q_f32(&Crow(1, 8));
            C2p = vld1q_f32(&Crow(2, 8));
            C3p = vld1q_f32(&Crow(3, 8));

            C02 = beta * C0p + C02;
            C12 = beta * C1p + C12;
            C22 = beta * C2p + C22;
            C32 = beta * C3p + C32;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
    } else {
        printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_4x16_fp32(int mr, int nr, int kc, float alpha,
                                               const float *Ar,
                                               const float *Br,
                                               float beta,
                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x16
*/
    SET_MR_NR(4, 16);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02, C03,
            C10, C11, C12, C13,
            C20, C21, C22, C23,
            C30, C31, C32, C33,
            A0,
            B0, B1, B2, B3,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C0p A0
#define C1p B0
#define C2p B1
#define C3p B2

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C03 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C13 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C23 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C33 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    if (alpha != zero) {
        for (k = 0; k < kc; k++) {

            // Load colums/rows of A/B for current iteration
            A0 = vld1q_f32(&Aptr[baseA + 0]);
            B0 = vld1q_f32(&Bptr[baseB + 0]);
            B1 = vld1q_f32(&Bptr[baseB + 4]);
            B2 = vld1q_f32(&Bptr[baseB + 8]);
            B3 = vld1q_f32(&Bptr[baseB + 12]);

            /*** */
            C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
            C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
            C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
            C03 = vfmaq_laneq_f32(C03, B3, A0, 0);
            C10 = vfmaq_laneq_f32(C10, B0, A0, 1);
            C11 = vfmaq_laneq_f32(C11, B1, A0, 1);
            C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
            C13 = vfmaq_laneq_f32(C13, B3, A0, 1);
            C20 = vfmaq_laneq_f32(C20, B0, A0, 2);
            C21 = vfmaq_laneq_f32(C21, B1, A0, 2);
            C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
            C23 = vfmaq_laneq_f32(C23, B3, A0, 2);
            C30 = vfmaq_laneq_f32(C30, B0, A0, 3);
            C31 = vfmaq_laneq_f32(C31, B1, A0, 3);
            C32 = vfmaq_laneq_f32(C32, B2, A0, 3);
            C33 = vfmaq_laneq_f32(C33, B3, A0, 3);

            baseA = baseA + Amr;
            baseB = baseB + Bnr;
        }

        // Last iteration

        if (alpha == -one) {
            C00 = -C00;
            C01 = -C01;
            C02 = -C02;
            C03 = -C03;
            C10 = -C10;
            C11 = -C11;
            C12 = -C12;
            C13 = -C13;
            C20 = -C20;
            C21 = -C21;
            C22 = -C22;
            C23 = -C23;
            C30 = -C30;
            C31 = -C31;
            C32 = -C32;
            C33 = -C33;
        } else if (alpha != one) {
            C00 = alpha * C00;
            C01 = alpha * C01;
            C02 = alpha * C02;
            C03 = alpha * C03;
            C10 = alpha * C10;
            C11 = alpha * C11;
            C12 = alpha * C12;
            C13 = alpha * C13;
            C20 = alpha * C20;
            C21 = alpha * C21;
            C22 = alpha * C22;
            C23 = alpha * C23;
            C30 = alpha * C30;
            C31 = alpha * C31;
            C32 = alpha * C32;
            C33 = alpha * C33;
        }
    }

    if ((mr < MR) || (nr < NR)) {

        vst1q_f32(&Ctrow(0, 0), C00);
        vst1q_f32(&Ctrow(0, 4), C01);
        vst1q_f32(&Ctrow(0, 8), C02);
        vst1q_f32(&Ctrow(0, 12), C03);
        vst1q_f32(&Ctrow(1, 0), C10);
        vst1q_f32(&Ctrow(1, 4), C11);
        vst1q_f32(&Ctrow(1, 8), C12);
        vst1q_f32(&Ctrow(1, 12), C13);
        vst1q_f32(&Ctrow(2, 0), C20);
        vst1q_f32(&Ctrow(2, 4), C21);
        vst1q_f32(&Ctrow(2, 8), C22);
        vst1q_f32(&Ctrow(2, 12), C23);
        vst1q_f32(&Ctrow(3, 0), C30);
        vst1q_f32(&Ctrow(3, 4), C31);
        vst1q_f32(&Ctrow(3, 8), C32);
        vst1q_f32(&Ctrow(3, 12), C33);
        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            C0p = vld1q_f32(&Crow(0, 0));
            C1p = vld1q_f32(&Crow(1, 0));
            C2p = vld1q_f32(&Crow(2, 0));
            C3p = vld1q_f32(&Crow(3, 0));

            C00 = beta * C0p + C00;
            C10 = beta * C1p + C10;
            C20 = beta * C2p + C20;
            C30 = beta * C3p + C30;

            C0p = vld1q_f32(&Crow(0, 4));
            C1p = vld1q_f32(&Crow(1, 4));
            C2p = vld1q_f32(&Crow(2, 4));
            C3p = vld1q_f32(&Crow(3, 4));

            C01 = beta * C0p + C01;
            C11 = beta * C1p + C11;
            C21 = beta * C2p + C21;
            C31 = beta * C3p + C31;

            C0p = vld1q_f32(&Crow(0, 8));
            C1p = vld1q_f32(&Crow(1, 8));
            C2p = vld1q_f32(&Crow(2, 8));
            C3p = vld1q_f32(&Crow(3, 8));

            C02 = beta * C0p + C02;
            C12 = beta * C1p + C12;
            C22 = beta * C2p + C22;
            C32 = beta * C3p + C32;

            C0p = vld1q_f32(&Crow(0, 12));
            C1p = vld1q_f32(&Crow(1, 12));
            C2p = vld1q_f32(&Crow(2, 12));
            C3p = vld1q_f32(&Crow(3, 12));

            C03 = beta * C0p + C03;
            C13 = beta * C1p + C13;
            C23 = beta * C2p + C23;
            C33 = beta * C3p + C33;
        }

        vst1q_f32(&Crow(0, 0), C00);
        vst1q_f32(&Crow(0, 4), C01);
        vst1q_f32(&Crow(0, 8), C02);
        vst1q_f32(&Crow(0, 12), C03);
        vst1q_f32(&Crow(1, 0), C10);
        vst1q_f32(&Crow(1, 4), C11);
        vst1q_f32(&Crow(1, 8), C12);
        vst1q_f32(&Crow(1, 12), C13);
        vst1q_f32(&Crow(2, 0), C20);
        vst1q_f32(&Crow(2, 4), C21);
        vst1q_f32(&Crow(2, 8), C22);
        vst1q_f32(&Crow(2, 12), C23);
        vst1q_f32(&Crow(3, 0), C30);
        vst1q_f32(&Crow(3, 4), C31);
        vst1q_f32(&Crow(3, 8), C32);
        vst1q_f32(&Crow(3, 12), C33);
    } else {
        printf("Error: Incorrect use of 4x16 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_4x20_fp32(int mr, int nr, int kc, float alpha,
                                               const float *Ar,
                                               const float *Br,
                                               float beta,
                                               float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
*/
    SET_MR_NR(4, 20);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02, C03, C04,
            C10, C11, C12, C13, C14,
            C20, C21, C22, C23, C24,
            C30, C31, C32, C33, C34,
            A0,
            B0, B1, B2, B3, B4,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C0r A0
#define C1r B0
#define C2r B1
#define C3r B2

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0); C03 = vmovq_n_f32(0); C04 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0); C13 = vmovq_n_f32(0); C14 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0); C23 = vmovq_n_f32(0); C24 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0); C33 = vmovq_n_f32(0); C34 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    if (alpha != zero) {
        if (nr <= 4)
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if (nr <= 8)
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if (nr <= 12)
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else if (nr <= 16)
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]); B3 = vld1q_f32(&Bptr[baseB + 12]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); C03 = vfmaq_laneq_f32(C03, B3, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1); C13 = vfmaq_laneq_f32(C13, B3, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2); C23 = vfmaq_laneq_f32(C23, B3, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3); C33 = vfmaq_laneq_f32(C33, B3, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }
        else
            for (k = 0; k < kc; k++) {

                // Load colums/rows of A/B for current iteration
                A0 = vld1q_f32(&Aptr[baseA + 0]);
                B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]); B3 = vld1q_f32(&Bptr[baseB + 12]); B4 = vld1q_f32(&Bptr[baseB + 16]);

                /* Compute */
                C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); C03 = vfmaq_laneq_f32(C03, B3, A0, 0); C04 = vfmaq_laneq_f32(C04, B4, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1); C13 = vfmaq_laneq_f32(C13, B3, A0, 1); C14 = vfmaq_laneq_f32(C14, B4, A0, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2); C23 = vfmaq_laneq_f32(C23, B3, A0, 2); C24 = vfmaq_laneq_f32(C24, B4, A0, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3); C33 = vfmaq_laneq_f32(C33, B3, A0, 3); C34 = vfmaq_laneq_f32(C34, B4, A0, 3);

                baseA = baseA + Amr;
                baseB = baseB + Bnr;
            }

        if ( alpha==-one ) {
          C00 = -C00; C01 = -C01; C02 = -C02; C03 = -C03; C04 = -C04;
          C10 = -C10; C11 = -C11; C12 = -C12; C13 = -C13; C14 = -C14;
          C20 = -C20; C21 = -C21; C22 = -C22; C23 = -C23; C24 = -C24;
          C30 = -C30; C31 = -C31; C32 = -C32; C33 = -C33; C34 = -C34;
        }
        else if ( alpha!=one ) {
          C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03; C04 = alpha*C04;
          C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12; C13 = alpha*C13; C14 = alpha*C14;
          C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22; C23 = alpha*C23; C24 = alpha*C24;
          C30 = alpha*C30; C31 = alpha*C31; C32 = alpha*C32; C33 = alpha*C33; C34 = alpha*C34;
        }
    }

    if ((mr < MR) || (nr < NR)) {
        vst1q_f32(&Ctrow(0, 0), C00); vst1q_f32(&Ctrow(0, 4), C01); vst1q_f32(&Ctrow(0, 8), C02); vst1q_f32(&Ctrow(0, 12), C03); vst1q_f32(&Ctrow(0, 16), C04);
        vst1q_f32(&Ctrow(1, 0), C10); vst1q_f32(&Ctrow(1, 4), C11); vst1q_f32(&Ctrow(1, 8), C12); vst1q_f32(&Ctrow(1, 12), C13); vst1q_f32(&Ctrow(1, 16), C14);
        vst1q_f32(&Ctrow(2, 0), C20); vst1q_f32(&Ctrow(2, 4), C21); vst1q_f32(&Ctrow(2, 8), C22); vst1q_f32(&Ctrow(2, 12), C23); vst1q_f32(&Ctrow(2, 16), C24);
        vst1q_f32(&Ctrow(3, 0), C30); vst1q_f32(&Ctrow(3, 4), C31); vst1q_f32(&Ctrow(3, 8), C32); vst1q_f32(&Ctrow(3, 12), C33); vst1q_f32(&Ctrow(3, 16), C34);
        if (beta != zero) {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = beta * Crow(i, j) + Ctrow(i, j);
        } else {
            for (j = 0; j < nr; j++)
                for (i = 0; i < mr; i++)
                    Crow(i, j) = Ctrow(i, j);
        }
    } else {
        printf("Error: Incorrect use of non-fixed micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}

void gemm_microkernel_Cresident_neon_4x20_fixed_fp32(int mr, int nr, int kc, float alpha,
                                                     const float *Ar,
                                                     const float *Br,
                                                     float beta,
                                                     float *C, int ldC) {
/*
  BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  Specific: only for MRxNR = 4x20
*/
    SET_MR_NR(4, 20);

    int i, j, k, baseB = 0, baseA = 0, ldCt = NR, Amr, Bnr;
    float32x4_t C00, C01, C02, C03, C04,
            C10, C11, C12, C13, C14,
            C20, C21, C22, C23, C24,
            C30, C31, C32, C33, C34,
            A0,
            B0, B1, B2, B3, B4,
            B0n, B1n, B2n;
    float zero = 0.0, one = 1.0, *Aptr, *Bptr, Ctmp[MR * NR];

#define C0r A0
#define C1r B0
#define C2r B1
#define C3r B2

    if (kc == 0)
        return;

    C00 = vmovq_n_f32(0); C01 = vmovq_n_f32(0); C02 = vmovq_n_f32(0); C03 = vmovq_n_f32(0); C04 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0); C11 = vmovq_n_f32(0); C12 = vmovq_n_f32(0); C13 = vmovq_n_f32(0); C14 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0); C21 = vmovq_n_f32(0); C22 = vmovq_n_f32(0); C23 = vmovq_n_f32(0); C24 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0); C31 = vmovq_n_f32(0); C32 = vmovq_n_f32(0); C33 = vmovq_n_f32(0); C34 = vmovq_n_f32(0);

    Aptr = &Ar[0];
    Amr = MR;
    Bptr = &Br[0];
    Bnr = NR;

    // Iterate from 1 to kc
    // printf("mr %d nr %d kc %d alpha %16.10e beta %16.10e\n", MR, NR, kc, alpha, beta);
    if (alpha != zero) {
        for (k = 0; k < kc; k++) {

            // Load colums/rows of A/B for current iteration
            A0 = vld1q_f32(&Aptr[baseA + 0]);
            B0 = vld1q_f32(&Bptr[baseB + 0]); B1 = vld1q_f32(&Bptr[baseB + 4]); B2 = vld1q_f32(&Bptr[baseB + 8]); B3 = vld1q_f32(&Bptr[baseB + 12]); B4 = vld1q_f32(&Bptr[baseB + 16]);

            /* Compute */
            C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); C03 = vfmaq_laneq_f32(C03, B3, A0, 0); C04 = vfmaq_laneq_f32(C04, B4, A0, 0);
            C10 = vfmaq_laneq_f32(C10, B0, A0, 1); C11 = vfmaq_laneq_f32(C11, B1, A0, 1); C12 = vfmaq_laneq_f32(C12, B2, A0, 1); C13 = vfmaq_laneq_f32(C13, B3, A0, 1); C14 = vfmaq_laneq_f32(C14, B4, A0, 1);
            C20 = vfmaq_laneq_f32(C20, B0, A0, 2); C21 = vfmaq_laneq_f32(C21, B1, A0, 2); C22 = vfmaq_laneq_f32(C22, B2, A0, 2); C23 = vfmaq_laneq_f32(C23, B3, A0, 2); C24 = vfmaq_laneq_f32(C24, B4, A0, 2);
            C30 = vfmaq_laneq_f32(C30, B0, A0, 3); C31 = vfmaq_laneq_f32(C31, B1, A0, 3); C32 = vfmaq_laneq_f32(C32, B2, A0, 3); C33 = vfmaq_laneq_f32(C33, B3, A0, 3); C34 = vfmaq_laneq_f32(C34, B4, A0, 3);

            baseA = baseA + Amr;
            baseB = baseB + Bnr;
        }

        if ( alpha==-one ) {
            C00 = -C00; C01 = -C01; C02 = -C02; C03 = -C03; C04 = -C04;
            C10 = -C10; C11 = -C11; C12 = -C12; C13 = -C13; C14 = -C14;
            C20 = -C20; C21 = -C21; C22 = -C22; C23 = -C23; C24 = -C24;
            C30 = -C30; C31 = -C31; C32 = -C32; C33 = -C33; C34 = -C34;
        }
        else if ( alpha!=one ) {
            C00 = alpha*C00; C01 = alpha*C01; C02 = alpha*C02; C03 = alpha*C03; C04 = alpha*C04;
            C10 = alpha*C10; C11 = alpha*C11; C12 = alpha*C12; C13 = alpha*C13; C14 = alpha*C14;
            C20 = alpha*C20; C21 = alpha*C21; C22 = alpha*C22; C23 = alpha*C23; C24 = alpha*C24;
            C30 = alpha*C30; C31 = alpha*C31; C32 = alpha*C32; C33 = alpha*C33; C34 = alpha*C34;
        }
    }

    if ((mr == MR) && (nr == NR)) {
        if (beta != zero) {
            C0r = vld1q_f32(&Crow(0, 0));
            C1r = vld1q_f32(&Crow(1, 0));
            C2r = vld1q_f32(&Crow(2, 0));
            C3r = vld1q_f32(&Crow(3, 0));

            C00 = beta * C0r + C00;
            C10 = beta * C1r + C10;
            C20 = beta * C2r + C20;
            C30 = beta * C3r + C30;

            C0r = vld1q_f32(&Crow(0, 4));
            C1r = vld1q_f32(&Crow(1, 4));
            C2r = vld1q_f32(&Crow(2, 4));
            C3r = vld1q_f32(&Crow(3, 4));

            C01 = beta * C0r + C01;
            C11 = beta * C1r + C11;
            C21 = beta * C2r + C21;
            C31 = beta * C3r + C31;

            C0r = vld1q_f32(&Crow(0, 8));
            C1r = vld1q_f32(&Crow(1, 8));
            C2r = vld1q_f32(&Crow(2, 8));
            C3r = vld1q_f32(&Crow(3, 8));

            C02 = beta * C0r + C02;
            C12 = beta * C1r + C12;
            C22 = beta * C2r + C22;
            C32 = beta * C3r + C32;

            C0r = vld1q_f32(&Crow(0, 12));
            C1r = vld1q_f32(&Crow(1, 12));
            C2r = vld1q_f32(&Crow(2, 12));
            C3r = vld1q_f32(&Crow(3, 12));

            C03 = beta * C0r + C03;
            C13 = beta * C1r + C13;
            C23 = beta * C2r + C23;
            C33 = beta * C3r + C33;

            C0r = vld1q_f32(&Crow(0, 16));
            C1r = vld1q_f32(&Crow(1, 16));
            C2r = vld1q_f32(&Crow(2, 16));
            C3r = vld1q_f32(&Crow(3, 16));

            C04 = beta * C0r + C04;
            C14 = beta * C1r + C14;
            C24 = beta * C2r + C24;
            C34 = beta * C3r + C34;
        }

        vst1q_f32(&Crow(0, 0), C00); vst1q_f32(&Crow(0, 4), C01); vst1q_f32(&Crow(0, 8), C02); vst1q_f32(&Crow(0, 12), C03); vst1q_f32(&Crow(0, 16), C04);
        vst1q_f32(&Crow(1, 0), C10); vst1q_f32(&Crow(1, 4), C11); vst1q_f32(&Crow(1, 8), C12); vst1q_f32(&Crow(1, 12), C13); vst1q_f32(&Crow(1, 16), C14);
        vst1q_f32(&Crow(2, 0), C20); vst1q_f32(&Crow(2, 4), C21); vst1q_f32(&Crow(2, 8), C22); vst1q_f32(&Crow(2, 12), C23); vst1q_f32(&Crow(2, 16), C24);
        vst1q_f32(&Crow(3, 0), C30); vst1q_f32(&Crow(3, 4), C31); vst1q_f32(&Crow(3, 8), C32); vst1q_f32(&Crow(3, 12), C33); vst1q_f32(&Crow(3, 16), C34);
    } else {
        printf("Error: Incorrect use of 4x20 micro-kernel with %d x %d block\n", mr, nr);
        exit(-1);
    }
}
