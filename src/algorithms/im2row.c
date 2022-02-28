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

#include "im2row.h"

void im2row(float *rows, int ld, float *in,
            int batch, int height, int width, int channel, int oheight, int owidth,
            int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
            hstride, int vdilation, int hdilation) {

    int b, x, y, row, kx, ix, ky, iy, c, col;

    for (b = 0; b < batch; b++)
        for (x = 0; x < oheight; x++)
            for (y = 0; y < owidth; y++) {
                row = b * oheight * owidth + x * owidth + y;
                for (kx = 0; kx < kheight; kx++) {
                    ix = vstride * x + vdilation * kx - vpadding;
                    if (0 <= ix && ix < height)
                        for (ky = 0; ky < kwidth; ky++) {
                            iy = hstride * y + hdilation * ky -
                                 hpadding;
                            if (0 <= iy && iy < width)
                                for (c = 0; c < channel; c++) {
                                    col = c * kheight * kwidth + kx * kwidth + ky;
                                    rows[row * channel * kheight * kwidth + col] = in[
                                            ((b * height + ix) * width + iy) * channel + c];
                                }
                        }
                }
            }
}
