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

#ifndef CONVDIRECT_ALGORITHM_BLOCK_SIZES_H
#define CONVDIRECT_ALGORITHM_BLOCK_SIZES_H

typedef struct {
    int mr;
    int nr;
    int wob;
    int cob;
    int cib;
} convdirect_bs_t;

#define BS_UPDATE(bs, MR_bs, NR_bs, WOB_bs, COB_bs, CIB_bs) \
    bs.mr = MR_bs;                                          \
    bs.nr = NR_bs;                                          \
    bs.wob = WOB_bs;                                        \
    bs.cob = COB_bs;                                        \
    bs.cib = CIB_bs;

#endif //CONVDIRECT_ALGORITHM_BLOCK_SIZES_H
