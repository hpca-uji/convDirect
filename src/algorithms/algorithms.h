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

#include "blocks_sizes.h"
#include "../dtypes.h"
#include "../arrays.h"


// Tensor format suffix from TENSOR_FORMAT_X definition
#if TENSOR_FORMAT_NHWC
#define TF _nhwc
#elif TENSOR_FORMAT_NCHW
#define TF _nchw
#else
#pragma GCC error "Tensor format has not been defined!"
#endif


// Join two strings/macros and stringify each macro
#define _JOIN2_INNER_INNER(X, Y) X ## Y
#define _JOIN2_INNER(X, Y) _JOIN2_INNER_INNER(X, Y)


// Microkernel suffix from MK_NAME definition
#define MK _JOIN2_INNER(_, MK_NAME)


// _TFMK macro adds to the name of a function the tensor format and microkernels suffixes
#if defined(TF) && defined(MK)
#define _TFMK_INNER_INNER(F, TF, MK) F##TF##MK
#define _TFMK_INNER(F, TF, MK) _TFMK_INNER_INNER(F, TF, MK)
#define _TFMK(F) _TFMK_INNER(F, TF, MK)
// ---------------------------------------------------------
#define CONVDIRECT _TFMK(ALGORITHM)
#define CONVDIRECT_PRE _JOIN2_INNER(CONVDIRECT, _pre)
#define CONVDIRECT_KERNEL _JOIN2_INNER(CONVDIRECT, _kernel)
#define CONVDIRECT_POST _JOIN2_INNER(CONVDIRECT, _post)
#define BLOCK_SIZES _JOIN2_INNER(CONVDIRECT, _block_sizes)
#define TRANSFORM_INPUT _JOIN2_INNER(CONVDIRECT, _transform_input)
#define TRANSFORM_FILTER _JOIN2_INNER(CONVDIRECT, _transform_filter)
#define TRANSFORM_OUTPUT _JOIN2_INNER(CONVDIRECT, _transform_output)
// ---------------------------------------------------------
#define CONVDIRECT_PARAMS         \
    int t, int Co, int Ci,        \
    int Ho, int Wo,               \
    int Hf, int Wf,               \
    int vpadding, int hpadding,   \
    int vstride, int hstride,     \
    int vdilation, int hdilation, \
    DTYPE alpha,                  \
    const DTYPE *D,               \
    const DTYPE *F,               \
    DTYPE beta,                   \
    DTYPE *Y

#define CONVDIRECT_PRE_PARAMS \
    int t, int Co, int Ci,    \
    int Ho, int Wo,           \
    int Hf, int Wf,           \
    const DTYPE *D,           \
    const DTYPE *F,           \
    const DTYPE *Y,           \
    DTYPE **DT,               \
    DTYPE **FT,               \
    DTYPE **YT

#define CONVDIRECT_PRE_WITH_PARAMS \
    CONVDIRECT_PRE(CONVDIRECT_PRE_PARAMS)

#define CONVDIRECT_PRE_NOP       \
    CONVDIRECT_PRE_WITH_PARAMS { \
        *DT = (DTYPE *) D;       \
        *FT = (DTYPE *) F;       \
        *YT = (DTYPE *) Y;       \
    }

#define CONVDIRECT_KERNEL_PARAMS  \
    int t, int Co, int Ci,        \
    int Ho, int Wo,               \
    int Hf, int Wf,               \
    int vpadding, int hpadding,   \
    int vstride, int hstride,     \
    int vdilation, int hdilation, \
    DTYPE alpha,                  \
    const DTYPE *DT,              \
    const DTYPE *FT,              \
    DTYPE beta,                   \
    DTYPE *YT

#define CONVDIRECT_KERNEL_WITH_PARAMS \
    CONVDIRECT_KERNEL(CONVDIRECT_KERNEL_PARAMS)

#define CONVDIRECT_POST_PARAMS \
    int t, int Co, int Ci,     \
    int Ho, int Wo,            \
    int Hf, int Wf,            \
    DTYPE **DT,                \
    DTYPE **FT,                \
    DTYPE **YT,                \
    DTYPE *Y

#define CONVDIRECT_POST_WITH_PARAMS \
    CONVDIRECT_POST(CONVDIRECT_POST_PARAMS)

#define CONVDIRECT_POST_NOP \
    CONVDIRECT_POST_WITH_PARAMS {}

#define CONVDIRECT_PRE_KERNEL_POST              \
    CONVDIRECT(CONVDIRECT_PARAMS) {             \
        DTYPE *DT, *FT, *YT;                    \
                                                \
        CONVDIRECT_PRE(t, Co, Ci,               \
                       Ho, Wo, Hf, Wf,          \
                       D, F, Y,                 \
                       &DT, &FT, &YT);          \
                                                \
        CONVDIRECT_KERNEL(t, Co, Ci,            \
                          Ho, Wo, Hf, Wf,       \
                          vpadding, hpadding,   \
                          vstride, hstride,     \
                          vdilation, hdilation, \
                          alpha,                \
                          DT,                   \
                          FT,                   \
                          beta,                 \
                          YT);                  \
                                                \
        CONVDIRECT_POST(t, Co, Ci,              \
                        Ho, Wo, Hf, Wf,         \
                        &DT, &FT, &YT,          \
                        Y);                     \
    };

#endif // defined(TF) && defined(MK)


#define QUICK_RETURN_IF_POSSIBLE              \
    if ((t == 0) || (Co == 0) || (Ci == 0) || \
        (Ho == 0) || (Wo == 0) ||             \
        (Hf == 0) || (Wf == 0))               \
        return;


#ifdef TENSOR_FORMAT_NCHW
#define SET_D_LEADING_DIMENSIONS \
        int ldD3 = Wo;           \
        int ldD2 = Ho * ldD3;    \
        int ldD1 = Ci * ldD2;

#define SET_F_LEADING_DIMENSIONS \
        int ldF3 = Wf;           \
        int ldF2 = Hf * ldF3;    \
        int ldF1 = Ci * ldF2;

#define SET_Y_LEADING_DIMENSIONS \
        int ldY3 = wo;           \
        int ldY2 = ho * ldY3;    \
        int ldY1 = Co * ldY2;
//----------------------------------
#else //NHWC
#define SET_D_LEADING_DIMENSIONS \
        int ldD3 = Ci;           \
        int ldD2 = Wo * ldD3;    \
        int ldD1 = Ho * ldD2;

#define SET_F_LEADING_DIMENSIONS \
        int ldF3 = Co;           \
        int ldF2 = Wf * ldF3;    \
        int ldF1 = Hf * ldF2;

#define SET_Y_LEADING_DIMENSIONS \
        int ldY3 = Co;           \
        int ldY2 = wo * ldY3;    \
        int ldY1 = ho * ldY2;

//----------------------------------
#endif

#define SET_LEADING_DIMENSIONS    \
        SET_D_LEADING_DIMENSIONS; \
        SET_F_LEADING_DIMENSIONS; \
        SET_Y_LEADING_DIMENSIONS;

// DT, FT, YT, and FB leading dimensions are the same for NCHW and NHWC tensor formats
#define SET_Ci_CIB \
    int Ci_CIB = (int) ceil((double) Ci / CIB);
//----------------------------------------------
#define SET_Co_COB \
    int Co_COB = (int) ceil((double) Co / COB);
//----------------------------------------------
#define SET_DT_LEADING_DIMENSIONS               \
    int ldDT4 = CIB;                            \
    int ldDT3 = Wo * ldDT4;                     \
    int ldDT2 = Ho * ldDT3;                     \
    int ldDT1 = Ci_CIB * ldDT2;
//----------------------------------------------
#define SET_FT_LEADING_DIMENSIONS               \
    int ldFT5 = COB;                            \
    int ldFT4 = CIB * ldFT5;                    \
    int ldFT3 = Wf * ldFT4;                     \
    int ldFT2 = Hf * ldFT3;                     \
    int ldFT1 = Co_COB * ldFT2;
//----------------------------------------------
#define SET_YT_LEADING_DIMENSIONS               \
    int ldYT4 = COB;                            \
    int ldYT3 = wo * ldYT4;                     \
    int ldYT2 = ho * ldYT3;                     \
    int ldYT1 = Co_COB * ldYT2;
//----------------------------------------------
#define SET_FB_LEADING_DIMENSIONS               \
    int ldFB4 = NR;                             \
    int ldFB3 = Ci * ldFB4;                     \
    int ldFB2 = Co_NR * ldFB3;                  \
    int ldFB1 = Wf * ldFB2;
//----------------------------------------------
#define SET_Co_MR \
    int Co_MR = (int) ceil((double) Co / MR);
//----------------------------------------------
#define SET_Co_NR \
    int Co_NR = (int) ceil((double) Co / NR);
//----------------------------------------------

/*
 * To test the defined macros, execute (from this directory):
 *   echo '#include "./macros.h"' | gcc -E -dM -DTENSOR_FORMAT_NHWC -DMK_4x4 -P - | grep -v __ | sort
 *
 * To test the _TFMK(F) macro, uncomment the next function and execute (from this directory):
 *   echo '#include "./macros.h"' | gcc -E  -DTENSOR_FORMAT_NHWC -DMK_4x4 -P -
 */

// void _TFMK(function_name)() { printf("Hey!\n"); }
