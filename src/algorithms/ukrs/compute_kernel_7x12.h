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

#define COMPUTE_KERNEL_7x12 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
}    

#define COMPUTE_KERNEL_7x12_UNROLL_2 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+12]); \
      B1 = vld1q_f32(&Bptr[baseB+16]); \
      B2 = vld1q_f32(&Bptr[baseB+20]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
}    

#define COMPUTE_KERNEL_7x12_UNROLL_3 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+12]); \
      B1 = vld1q_f32(&Bptr[baseB+16]); \
      B2 = vld1q_f32(&Bptr[baseB+20]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB+24]); \
      B1 = vld1q_f32(&Bptr[baseB+28]); \
      B2 = vld1q_f32(&Bptr[baseB+32]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2); C01 = vfmaq_laneq_f32(C01, B1, A0, 2); C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2); C11 = vfmaq_laneq_f32(C11, B1, A1, 2); C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2); C21 = vfmaq_laneq_f32(C21, B1, A2, 2); C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2); C31 = vfmaq_laneq_f32(C31, B1, A3, 2); C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2); C41 = vfmaq_laneq_f32(C41, B1, A4, 2); C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2); C51 = vfmaq_laneq_f32(C51, B1, A5, 2); C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2); C61 = vfmaq_laneq_f32(C61, B1, A6, 2); C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
}    

#define COMPUTE_KERNEL_7x12_UNROLL_4 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+12]); \
      B1 = vld1q_f32(&Bptr[baseB+16]); \
      B2 = vld1q_f32(&Bptr[baseB+20]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB+24]); \
      B1 = vld1q_f32(&Bptr[baseB+28]); \
      B2 = vld1q_f32(&Bptr[baseB+32]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2); C01 = vfmaq_laneq_f32(C01, B1, A0, 2); C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2); C11 = vfmaq_laneq_f32(C11, B1, A1, 2); C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2); C21 = vfmaq_laneq_f32(C21, B1, A2, 2); C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2); C31 = vfmaq_laneq_f32(C31, B1, A3, 2); C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2); C41 = vfmaq_laneq_f32(C41, B1, A4, 2); C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2); C51 = vfmaq_laneq_f32(C51, B1, A5, 2); C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2); C61 = vfmaq_laneq_f32(C61, B1, A6, 2); C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
\
      B0 = vld1q_f32(&Bptr[baseB+36]); \
      B1 = vld1q_f32(&Bptr[baseB+40]); \
      B2 = vld1q_f32(&Bptr[baseB+44]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 3); C01 = vfmaq_laneq_f32(C01, B1, A0, 3); C02 = vfmaq_laneq_f32(C02, B2, A0, 3); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 3); C11 = vfmaq_laneq_f32(C11, B1, A1, 3); C12 = vfmaq_laneq_f32(C12, B2, A1, 3); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 3); C21 = vfmaq_laneq_f32(C21, B1, A2, 3); C22 = vfmaq_laneq_f32(C22, B2, A2, 3); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 3); C31 = vfmaq_laneq_f32(C31, B1, A3, 3); C32 = vfmaq_laneq_f32(C32, B2, A3, 3); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 3); C41 = vfmaq_laneq_f32(C41, B1, A4, 3); C42 = vfmaq_laneq_f32(C42, B2, A4, 3); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 3); C51 = vfmaq_laneq_f32(C51, B1, A5, 3); C52 = vfmaq_laneq_f32(C52, B2, A5, 3); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 3); C61 = vfmaq_laneq_f32(C61, B1, A6, 3); C62 = vfmaq_laneq_f32(C62, B2, A6, 3); \
}

#define COMPUTE_KERNEL_7x12_NOPACK_UNROLL_2 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
}    

#define COMPUTE_KERNEL_7x12_NOPACK_UNROLL_3 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB+2*ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+2*ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+2*ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2); C01 = vfmaq_laneq_f32(C01, B1, A0, 2); C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2); C11 = vfmaq_laneq_f32(C11, B1, A1, 2); C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2); C21 = vfmaq_laneq_f32(C21, B1, A2, 2); C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2); C31 = vfmaq_laneq_f32(C31, B1, A3, 2); C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2); C41 = vfmaq_laneq_f32(C41, B1, A4, 2); C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2); C51 = vfmaq_laneq_f32(C51, B1, A5, 2); C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2); C61 = vfmaq_laneq_f32(C61, B1, A6, 2); C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
}    

#define COMPUTE_KERNEL_7x12_NOPACK_UNROLL_4 \
{ \
      A0 = vld1q_f32(&Atrow(0,0)); \
      A1 = vld1q_f32(&Atrow(1,0)); \
      A2 = vld1q_f32(&Atrow(2,0)); \
      A3 = vld1q_f32(&Atrow(3,0)); \
      A4 = vld1q_f32(&Atrow(4,0)); \
      A5 = vld1q_f32(&Atrow(5,0)); \
      A6 = vld1q_f32(&Atrow(6,0)); \
\
      B0 = vld1q_f32(&Bptr[baseB]);   \
      B1 = vld1q_f32(&Bptr[baseB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 0); C01 = vfmaq_laneq_f32(C01, B1, A0, 0); C02 = vfmaq_laneq_f32(C02, B2, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 0); C11 = vfmaq_laneq_f32(C11, B1, A1, 0); C12 = vfmaq_laneq_f32(C12, B2, A1, 0); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 0); C21 = vfmaq_laneq_f32(C21, B1, A2, 0); C22 = vfmaq_laneq_f32(C22, B2, A2, 0); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 0); C31 = vfmaq_laneq_f32(C31, B1, A3, 0); C32 = vfmaq_laneq_f32(C32, B2, A3, 0); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 0); C41 = vfmaq_laneq_f32(C41, B1, A4, 0); C42 = vfmaq_laneq_f32(C42, B2, A4, 0); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 0); C51 = vfmaq_laneq_f32(C51, B1, A5, 0); C52 = vfmaq_laneq_f32(C52, B2, A5, 0); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 0); C61 = vfmaq_laneq_f32(C61, B1, A6, 0); C62 = vfmaq_laneq_f32(C62, B2, A6, 0); \
\
      B0 = vld1q_f32(&Bptr[baseB+ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 1); C01 = vfmaq_laneq_f32(C01, B1, A0, 1); C02 = vfmaq_laneq_f32(C02, B2, A0, 1); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 1); C11 = vfmaq_laneq_f32(C11, B1, A1, 1); C12 = vfmaq_laneq_f32(C12, B2, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 1); C21 = vfmaq_laneq_f32(C21, B1, A2, 1); C22 = vfmaq_laneq_f32(C22, B2, A2, 1); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 1); C31 = vfmaq_laneq_f32(C31, B1, A3, 1); C32 = vfmaq_laneq_f32(C32, B2, A3, 1); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 1); C41 = vfmaq_laneq_f32(C41, B1, A4, 1); C42 = vfmaq_laneq_f32(C42, B2, A4, 1); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 1); C51 = vfmaq_laneq_f32(C51, B1, A5, 1); C52 = vfmaq_laneq_f32(C52, B2, A5, 1); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 1); C61 = vfmaq_laneq_f32(C61, B1, A6, 1); C62 = vfmaq_laneq_f32(C62, B2, A6, 1); \
\
      B0 = vld1q_f32(&Bptr[baseB+2*ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+2*ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+2*ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 2); C01 = vfmaq_laneq_f32(C01, B1, A0, 2); C02 = vfmaq_laneq_f32(C02, B2, A0, 2); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 2); C11 = vfmaq_laneq_f32(C11, B1, A1, 2); C12 = vfmaq_laneq_f32(C12, B2, A1, 2); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 2); C21 = vfmaq_laneq_f32(C21, B1, A2, 2); C22 = vfmaq_laneq_f32(C22, B2, A2, 2); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 2); C31 = vfmaq_laneq_f32(C31, B1, A3, 2); C32 = vfmaq_laneq_f32(C32, B2, A3, 2); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 2); C41 = vfmaq_laneq_f32(C41, B1, A4, 2); C42 = vfmaq_laneq_f32(C42, B2, A4, 2); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 2); C51 = vfmaq_laneq_f32(C51, B1, A5, 2); C52 = vfmaq_laneq_f32(C52, B2, A5, 2); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 2); C61 = vfmaq_laneq_f32(C61, B1, A6, 2); C62 = vfmaq_laneq_f32(C62, B2, A6, 2); \
\
      B0 = vld1q_f32(&Bptr[baseB+3*ldB]); \
      B1 = vld1q_f32(&Bptr[baseB+3*ldB+4]); \
      B2 = vld1q_f32(&Bptr[baseB+3*ldB+8]); \
\
      C00 = vfmaq_laneq_f32(C00, B0, A0, 3); C01 = vfmaq_laneq_f32(C01, B1, A0, 3); C02 = vfmaq_laneq_f32(C02, B2, A0, 3); \
      C10 = vfmaq_laneq_f32(C10, B0, A1, 3); C11 = vfmaq_laneq_f32(C11, B1, A1, 3); C12 = vfmaq_laneq_f32(C12, B2, A1, 3); \
      C20 = vfmaq_laneq_f32(C20, B0, A2, 3); C21 = vfmaq_laneq_f32(C21, B1, A2, 3); C22 = vfmaq_laneq_f32(C22, B2, A2, 3); \
      C30 = vfmaq_laneq_f32(C30, B0, A3, 3); C31 = vfmaq_laneq_f32(C31, B1, A3, 3); C32 = vfmaq_laneq_f32(C32, B2, A3, 3); \
      C40 = vfmaq_laneq_f32(C40, B0, A4, 3); C41 = vfmaq_laneq_f32(C41, B1, A4, 3); C42 = vfmaq_laneq_f32(C42, B2, A4, 3); \
      C50 = vfmaq_laneq_f32(C50, B0, A5, 3); C51 = vfmaq_laneq_f32(C51, B1, A5, 3); C52 = vfmaq_laneq_f32(C52, B2, A5, 3); \
      C60 = vfmaq_laneq_f32(C60, B0, A6, 3); C61 = vfmaq_laneq_f32(C61, B1, A6, 3); C62 = vfmaq_laneq_f32(C62, B2, A6, 3); \
}

