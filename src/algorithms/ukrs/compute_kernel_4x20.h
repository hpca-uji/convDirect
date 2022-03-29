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

#define COMPUTE_KERNEL_4x20 \
{ \
      A0 = vld1q_f32(&Aptr[baseA + 0]); \
      B00 = vld1q_f32(&Bptr[baseB + 0]);  B01 = vld1q_f32(&Bptr[baseB + 4]);  B02 = vld1q_f32(&Bptr[baseB + 8]);  B03 = vld1q_f32(&Bptr[baseB + 12]); B04 = vld1q_f32(&Bptr[baseB + 16]); \
 \
      C00 = vfmaq_laneq_f32(C00, B00, A0, 0); C01 = vfmaq_laneq_f32(C01, B01, A0, 0); C02 = vfmaq_laneq_f32(C02, B02, A0, 0); C03 = vfmaq_laneq_f32(C03, B03, A0, 0); C04 = vfmaq_laneq_f32(C04, B04, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B00, A0, 1); C11 = vfmaq_laneq_f32(C11, B01, A0, 1); C12 = vfmaq_laneq_f32(C12, B02, A0, 1); C13 = vfmaq_laneq_f32(C13, B03, A0, 1); C14 = vfmaq_laneq_f32(C14, B04, A0, 1); \
      C20 = vfmaq_laneq_f32(C20, B00, A0, 2); C21 = vfmaq_laneq_f32(C21, B01, A0, 2); C22 = vfmaq_laneq_f32(C22, B02, A0, 2); C23 = vfmaq_laneq_f32(C23, B03, A0, 2); C24 = vfmaq_laneq_f32(C24, B04, A0, 2); \
      C30 = vfmaq_laneq_f32(C30, B00, A0, 3); C31 = vfmaq_laneq_f32(C31, B01, A0, 3); C32 = vfmaq_laneq_f32(C32, B02, A0, 3); C33 = vfmaq_laneq_f32(C33, B03, A0, 3); C34 = vfmaq_laneq_f32(C34, B04, A0, 3); \
}    

#define COMPUTE_KERNEL_4x20_UNROLL_2 \
{ \
      A0 = vld1q_f32(&Aptr[baseA + 0]); \
      A1 = vld1q_f32(&Aptr[baseA + 4]); \
      B00 = vld1q_f32(&Bptr[baseB + 0]);  B01 = vld1q_f32(&Bptr[baseB + 4]);  B02 = vld1q_f32(&Bptr[baseB + 8]);  B03 = vld1q_f32(&Bptr[baseB + 12]); B04 = vld1q_f32(&Bptr[baseB + 16]); \
      B10 = vld1q_f32(&Bptr[baseB + 20]); B11 = vld1q_f32(&Bptr[baseB + 24]); B12 = vld1q_f32(&Bptr[baseB + 28]); B13 = vld1q_f32(&Bptr[baseB + 32]); B14 = vld1q_f32(&Bptr[baseB + 36]); \
 \
      C00 = vfmaq_laneq_f32(C00, B00, A0, 0); C01 = vfmaq_laneq_f32(C01, B01, A0, 0); C02 = vfmaq_laneq_f32(C02, B02, A0, 0); C03 = vfmaq_laneq_f32(C03, B03, A0, 0); C04 = vfmaq_laneq_f32(C04, B04, A0, 0); \
      C10 = vfmaq_laneq_f32(C10, B00, A0, 1); C11 = vfmaq_laneq_f32(C11, B01, A0, 1); C12 = vfmaq_laneq_f32(C12, B02, A0, 1); C13 = vfmaq_laneq_f32(C13, B03, A0, 1); C14 = vfmaq_laneq_f32(C14, B04, A0, 1); \
      C20 = vfmaq_laneq_f32(C20, B00, A0, 2); C21 = vfmaq_laneq_f32(C21, B01, A0, 2); C22 = vfmaq_laneq_f32(C22, B02, A0, 2); C23 = vfmaq_laneq_f32(C23, B03, A0, 2); C24 = vfmaq_laneq_f32(C24, B04, A0, 2); \
      C30 = vfmaq_laneq_f32(C30, B00, A0, 3); C31 = vfmaq_laneq_f32(C31, B01, A0, 3); C32 = vfmaq_laneq_f32(C32, B02, A0, 3); C33 = vfmaq_laneq_f32(C33, B03, A0, 3); C34 = vfmaq_laneq_f32(C34, B04, A0, 3); \
 \
      C00 = vfmaq_laneq_f32(C00, B10, A1, 0); C01 = vfmaq_laneq_f32(C01, B11, A1, 0); C02 = vfmaq_laneq_f32(C02, B12, A1, 0); C03 = vfmaq_laneq_f32(C03, B13, A1, 0); C04 = vfmaq_laneq_f32(C04, B14, A1, 0); \
      C10 = vfmaq_laneq_f32(C10, B10, A1, 1); C11 = vfmaq_laneq_f32(C11, B11, A1, 1); C12 = vfmaq_laneq_f32(C12, B12, A1, 1); C13 = vfmaq_laneq_f32(C13, B13, A1, 1); C14 = vfmaq_laneq_f32(C14, B14, A1, 1); \
      C20 = vfmaq_laneq_f32(C20, B10, A1, 2); C21 = vfmaq_laneq_f32(C21, B11, A1, 2); C22 = vfmaq_laneq_f32(C22, B12, A1, 2); C23 = vfmaq_laneq_f32(C23, B13, A1, 2); C24 = vfmaq_laneq_f32(C24, B14, A1, 2); \
      C30 = vfmaq_laneq_f32(C30, B10, A1, 3); C31 = vfmaq_laneq_f32(C31, B11, A1, 3); C32 = vfmaq_laneq_f32(C32, B12, A1, 3); C33 = vfmaq_laneq_f32(C33, B13, A1, 3); C34 = vfmaq_laneq_f32(C34, B14, A1, 3); \
}    
