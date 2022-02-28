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


#ifndef CONVDIRECT_IM2ROW_H
#define CONVDIRECT_IM2ROW_H

void im2row(float *rows, int ld, float *in,
            int batch, int height, int width, int channel, int oheight, int owidth,
            int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
            hstride, int vdilation, int hdilation);

#endif //CONVDIRECT_IM2ROW_H