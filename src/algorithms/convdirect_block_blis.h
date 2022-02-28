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

#ifndef CONVDIRECT_CONVDIRECT_BLOCK_BLIS_H
#define CONVDIRECT_CONVDIRECT_BLOCK_BLIS_H

#if defined(MK_BLIS)
#undef MR  // (provided by BLIS)
#undef NR  // (provided by BLIS)
#define WOB 888
#define COB 640
#define CIB 512
// WOB = 1792;
// COB = 1008;
// CIB = 256;
// WOB = 120;  // MC
// COB = 3072; // NC
// CIB = 640;  // KC
// BLIS parameters
// COB = 120;  // MC
// WOB = 3072; // NC
// CIB = 640;  // KC
//--------------------
#elif defined(MK_8x12)
#define MR 8
#define NR 12
#define WOB 1792
#define COB 3072
#define CIB 640
// WOB = 896;
// COB = 636;
// CIB = 512;
// BLIS parameters
// WOB = 120;  // MC
// COB = 3072; // NC
// CIB = 640;  // KC
//--------------------
#elif defined(MK_4x4)
#define MR 4
#define NR 4
#define WOB 896
#define COB 1024
#define CIB 512
//--------------------
#elif defined(MK_4x8)
#define MR 4
#define NR 8
#define WOB 896
#define COB 1024
#define CIB 512
//--------------------
#elif defined(MK_4x12)
#define MR 4
#define NR 12
#define WOB 896
#define COB 1008
#define CIB 512
//--------------------
#elif defined(MK_4x16)
#define MR 4
#define NR 16
#define WOB 896
#define COB 1024
#define CIB 512
//--------------------
#elif defined(MK_4x20)
#define MR 4
#define NR 20
#define WOB 896
#define COB 3060
#define CIB 512
//--------------------
#endif

#endif //CONVDIRECT_CONVDIRECT_BLOCK_BLIS_H
