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

#include "packrb.h"

/**
 * BLIS pack for A-->Ac
*/

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]

void packRB_neon( char orderA, char transA, int mc, int nc, DTYPE *A, int ldA, DTYPE *Ac, int RR ) {
  
    int    i, j, ii, ii_tmp, k, rr, iters, iters_left;
    float32x4_t A00, A01, A10, A11, A20, A21, A30, A31;

    if (((transA == 'N') && (orderA == 'C')) ||
        ((transA == 'T') && (orderA == 'R'))) {
      printf("Packing not yet implemented\n");
      exit(-1);
    } else {

      for ( i=0; i < mc; i += RR ) { 
        k = i * nc;
        rr = min( mc - i, RR );
         
        for ( i=0; i<mc; i+=RR ) {
          k = i*nc;
          rr = min( mc-i, RR );
          
	  if (rr == RR) { //rr == 8

	    for ( j=0; j < nc - 4; j += 4 ) {

	      A00[0] = Acol(j + 0, i);
	      A10[0] = Acol(j + 1, i);
	      A20[0] = Acol(j + 2, i);
	      A30[0] = Acol(j + 3, i);
	      
	      A00[1] = Acol(j + 0, i + 1);
	      A10[1] = Acol(j + 1, i + 1);
	      A20[1] = Acol(j + 2, i + 1);
	      A30[1] = Acol(j + 3, i + 1);
	      
	      A00[2] = Acol(j + 0, i + 2);
	      A10[2] = Acol(j + 1, i + 2);
	      A20[2] = Acol(j + 2, i + 2);
	      A30[2] = Acol(j + 3, i + 2);
              
	      A00[3] = Acol(j + 0, i + 3);
	      A10[3] = Acol(j + 1, i + 3);
	      A20[3] = Acol(j + 2, i + 3);
	      A30[3] = Acol(j + 3, i + 3);

	      A01[0] = Acol(j + 0, i + 4);
	      A11[0] = Acol(j + 1, i + 4);
	      A21[0] = Acol(j + 2, i + 4);
	      A31[0] = Acol(j + 3, i + 4);
	      
	      A01[1] = Acol(j + 0, i + 5);
	      A11[1] = Acol(j + 1, i + 5);
	      A21[1] = Acol(j + 2, i + 5);
	      A31[1] = Acol(j + 3, i + 5);
	      
	      A01[2] = Acol(j + 0, i + 6);
	      A11[2] = Acol(j + 1, i + 6);
	      A21[2] = Acol(j + 2, i + 6);
	      A31[2] = Acol(j + 3, i + 6);
              
	      A01[3] = Acol(j + 0, i + 7);
	      A11[3] = Acol(j + 1, i + 7);
	      A21[3] = Acol(j + 2, i + 7);
	      A31[3] = Acol(j + 3, i + 7);

              vst1q_f32(&Ac[k], A00); k += 4;
              vst1q_f32(&Ac[k], A01); k += 4;
              k += (RR - rr);
	      
              vst1q_f32(&Ac[k], A10); k += 4;
              vst1q_f32(&Ac[k], A11); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A20); k += 4;
              vst1q_f32(&Ac[k], A21); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A30); k += 4;
              vst1q_f32(&Ac[k], A31); k += 4;
              k += (RR - rr);

	    }

	    for(; j < nc; j++) {
              A00[0] = Acol(j, i+0);
              A00[1] = Acol(j, i+1);
              A00[2] = Acol(j, i+2);
              A00[3] = Acol(j, i+3);

              A01[0] = Acol(j, i+4);
              A01[1] = Acol(j, i+5);
              A01[2] = Acol(j, i+6);
              A01[3] = Acol(j, i+7);

              vst1q_f32(&Ac[k], A00); k += 4;
              vst1q_f32(&Ac[k], A01); k += 4;

              k += (RR - rr);
	    }
	  } else if (rr == 4) {
	  
	    for ( j=0; j < nc - 4; j += 4 ) {

	      A00[0] = Acol(j + 0, i);
	      A10[0] = Acol(j + 1, i);
	      A20[0] = Acol(j + 2, i);
	      A30[0] = Acol(j + 3, i);
	      
	      A00[1] = Acol(j + 0, i + 1);
	      A10[1] = Acol(j + 1, i + 1);
	      A20[1] = Acol(j + 2, i + 1);
	      A30[1] = Acol(j + 3, i + 1);
	      
	      A00[2] = Acol(j + 0, i + 2);
	      A10[2] = Acol(j + 1, i + 2);
	      A20[2] = Acol(j + 2, i + 2);
	      A30[2] = Acol(j + 3, i + 2);
              
	      A00[3] = Acol(j + 0, i + 3);
	      A10[3] = Acol(j + 1, i + 3);
	      A20[3] = Acol(j + 2, i + 3);
	      A30[3] = Acol(j + 3, i + 3);

              vst1q_f32(&Ac[k], A00); k += 4;
              k += (RR - rr);
	      
              vst1q_f32(&Ac[k], A10); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A20); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A30); k += 4;
              k += (RR - rr);

	    }

	    for(; j < nc; j++) {
              A00[0] = Acol(j, i + 0);
              A00[1] = Acol(j, i + 1);
              A00[2] = Acol(j, i + 2);
              A00[3] = Acol(j, i + 3);
              
              vst1q_f32(&Ac[k], A00); k += 4; 
              k += (RR - rr);

	    }
	  
	  } else { // rr != 8 && rr != 4
	  
	    for ( j=0; j<nc; j++ ) {
              for ( ii=0; ii<rr; ii++ ) {
                Ac[k] = Acol(j,i+ii);
                k++;
              }
              k += (RR-rr);
            }
	  
	  }
        }

      }
    }

}

void packRB(char orderA, char transA, int mc, int nc, const DTYPE *A,
        int start_y, int ky, int dim_w,
        int hpadding, int hstride, int hdilation,
        int ld3, DTYPE *Ac, int RR) {

    if (((transA == 'N') && (orderA == 'C')) ||
        ((transA == 'T') && (orderA == 'R'))) {
      printf("Packing not yet implemented\n");
      exit(-1);
    } else {
        start_y = hstride * start_y + hdilation * ky - hpadding;
        #pragma omp parallel for
        for (int i = 0; i < mc; i += RR) {
            int k = i * nc;
            int rr = min(mc - i, RR);
            for (int j = 0; j < nc; j++) {
                int ii = 0;
                int y = start_y + hstride * i;
                for (; ii < rr && y < 0    ; ii++, y += hstride, k++) Ac[k] = 0.0;
                for (; ii < rr && y < dim_w; ii++, y += hstride, k++) Ac[k] = A[y * ld3 + j];
                for (; ii < rr             ; ii++,               k++) Ac[k] = 0.0;
                k += (RR - rr);
            }
        }
    }
}
