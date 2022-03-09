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
#include <string.h>
#include "array_utils.h"
#include "colors.h"
#include "input_utils.h"

#include "convdirect.h"

/*
#include "dtypes.h"
#include "gemm_reference.h"
#include "arrays.h"
*/


#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

int main(int argc, char *argv[]) {

    char *variant;
    DTYPE *D, *F, *Y, *Yg;

    double t1, t2, t3, t4, time_pre, time_kernel, time_post, tmin, error, nrm, tmp, flops, GFLOPS;
    int m, t,
            nmin, nmax, nstep,
            kmin, kmax, kstep,
            cmin, cmax, cstep,
            hmin, hmax, hstep,
            wmin, wmax, wstep,
            rmin, rmax, rstep,
            smin, smax, sstep,
            prmax, psmax, ret,
            n, k, c,
            h, w,
            r, s,
            pr, ps,
            in, ir, is, ic, ik, ih, iw,
            visual, nreps,
            ho, wo, homax, womax;

    tensor_format_t tensor_format;
    bool all_close = true;

    // int ib, i, i2, ii, Ci_CIB, Co_COB, Co_NR, Co_MR;
    // char *filename;
    // FILE *fd;
    int cnn_test_num, cnn_i;
    int CIB, COB, WOB;
    size_t test_n = 0;

    /* @todo: recover this but reading from the bloc_sizes structure

#if defined(BLOCKED_SHALOM) || defined(BLOCKED_TZEMENG) || defined(BLOCKED_BLIS)
#if MK_BLIS
    if (WOB % NR != 0) {
      printf("Error: WOB must be multiple of NR. Now WOB=%d and NR=%d\n", WOB, NR);
      exit(-1);
    } else if (COB % MR != 0) {
      printf("Error: COB must be multiple of MR. Now COB=%d and MR=%d\n", COB, MR);
#else
    if (WOB % MR != 0) {
      printf("Error: WOB must be multiple of MR. Now WOB=%d and MR=%d\n", WOB, MR);
      exit(-1);
    } else if (COB % NR != 0) {
      printf("Error: COB must be multiple of NR. Now COB=%d and NR=%d\n", COB, NR);
#endif
      exit(-1);
    }
#endif
     */

    evalConfig_t *ec = get_evalConfig(argv);
    /* TODO this should in the configuration function */
    const int vpadding  = 0;
    const int hpadding  = 0;
    const int vstride   = 1;
    const int hstride   = 1;
    const int vdilation = 1;
    const int hdilation = 1;

    m = 2;
    t = 6;

    /* @todo: The next snippet should go to the main code
// #ifdef TVM
    //  ******* PREPARING TVM *******
    //LOG(INFO) << "Verify load function from system lib";
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
    std::string fname="microkernel_mult";
    
    // Get the function from the module.
    tvm::runtime::PackedFunc f = mod.GetFunction(fname);
    ICHECK(f != nullptr);
    
    DLTensor* A;
    DLTensor* B;
    DLTensor* C;
    
    int ndim = 2;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int tvm_m=MR;
    int tvm_n=NR;
    int tvm_k=NR;
    int64_t shapeA[2] = {tvm_m, tvm_k};
    int64_t shapeB[2] = {tvm_k, tvm_n};
    int64_t shapeC[2] = {tvm_m, tvm_n};
    
    TVMArrayAlloc(shapeA, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &A);
    TVMArrayAlloc(shapeB, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &B);
    TVMArrayAlloc(shapeC, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &C);
    
    for (int i = 0; i < tvm_m*tvm_k; ++i)
      static_cast<float*>(A->data)[i] = 0;

    for (int i = 0; i < tvm_n*tvm_k; ++i)
      static_cast<float*>(B->data)[i] = 0;

    for (int i = 0; i < tvm_m*tvm_n; ++i)
      static_cast<float*>(C->data)[i] = 0;
    
    void *A_aux = A->data;
    void *B_aux = B->data;
    void *C_aux = C->data;
    // ******* PREPARING TVM END ******* //
#endif
    */

    tmin = ec->tmin;
    tensor_format = ec->tensor_format;
    // convdirect_ft *convdirect_algorithm = convdirect_get_algorithm(ec->algorithm_name);
    convdirect_algorithm_parts_t convdirect_algorithm_parts;
    convdirect_get_algorithm_parts(ec->algorithm_name, &convdirect_algorithm_parts);
    convdirect_bs_t *convdirect_bs = convdirect_get_block_sizes(ec->algorithm_name);

    if (!ec->test) {
        if (ec->type == CNN_TYPE)
            fprintf(ec->fd_out, "l;");
        fprintf(ec->fd_out, "Variant;CIB;COB;WOB;n;k;c;h;w;kh;kw;Time;GFLOPS;PreTime;PostTime;Error\n");
    } else {
        printf("\n%s In test mode, output files will not be generated!%s\n\n", COLOR_BOLDYELLOW, COLOR_RESET);
    }

    printf(" ==============================================================================================================================\n");
    printf(" |%s                     D R I V E R    F O R    D I R E C T    C O N V O L U T I O N    E V A L U A T I O N       %s             |\n",
           COLOR_BOLDYELLOW, COLOR_RESET);
    printf(" ==============================================================================================================================\n");
    //printf(" |                                          [%s*%s]MR:%2d                  [%s*%s]NR:%2d                                                |\n",
    //	   COLOR_BOLDYELLOW, COLOR_RESET, MR,
    //	   COLOR_BOLDYELLOW, COLOR_RESET, NR);
    //printf(" ==============================================================================================================================\n");
    printf(" |%s   CIB   COB   WOB     n     k     c     h     w    kh    kw"
           "    Time     GFLOPS   PreTime   PostTime   Error    All close?%s  |\n",
           COLOR_RESET, COLOR_RESET);
    printf(" ==============================================================================================================================\n");

    for (cnn_i = 0; cnn_i < ec->cnn_layers; cnn_i++) {
        nmin = ec->cnn[cnn_i].nmin;
        nmax = ec->cnn[cnn_i].nmax;
        nstep = ec->cnn[cnn_i].nstep;

        kmin = ec->cnn[cnn_i].kmin;
        kmax = ec->cnn[cnn_i].kmax;
        kstep = ec->cnn[cnn_i].kstep;

        cmin = ec->cnn[cnn_i].cmin;
        cmax = ec->cnn[cnn_i].cmax;
        cstep = ec->cnn[cnn_i].cstep;

        hmin = ec->cnn[cnn_i].hmin;
        hmax = ec->cnn[cnn_i].hmax;
        hstep = ec->cnn[cnn_i].hstep;

        wmin = ec->cnn[cnn_i].wmin;
        wmax = ec->cnn[cnn_i].wmax;
        wstep = ec->cnn[cnn_i].wstep;

        rmin = ec->cnn[cnn_i].rmin;
        rmax = ec->cnn[cnn_i].rmax;
        rstep = ec->cnn[cnn_i].rstep;

        smin = ec->cnn[cnn_i].smin;
        smax = ec->cnn[cnn_i].smax;
        sstep = ec->cnn[cnn_i].sstep;

        homax = (hmax + 2 * vpadding - vdilation * (rmin - 1) - 1) / vstride + 1;
        womax = (wmax + 2 * hpadding - hdilation * (smin - 1) - 1) / hstride + 1;

        D = (DTYPE *) malloc(nmax * cmax * hmax * wmax * sizeof(DTYPE));
        F = (DTYPE *) malloc(kmax * cmax * rmax * smax * sizeof(DTYPE));
        Y = (DTYPE *) malloc(nmax * kmax * homax * womax * sizeof(DTYPE));

        if (ec->test)
            Yg = (DTYPE *) malloc(nmax * kmax * homax * womax * sizeof(DTYPE));

        DTYPE *DT, *FT, *YT;

        for (n = nmin; n <= nmax; n += nstep) {
            for (k = kmin; k <= kmax; k += kstep) {
                for (c = cmin; c <= cmax; c += cstep) {
                    for (h = hmin; h <= hmax; h += hstep) {
                        for (w = wmin; w <= wmax; w += wstep) {
                            for (r = rmin; r <= rmax; r += rstep) {

                                s = r;
                                ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
                                wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;

                                if (tensor_format == nchw) { // nchw
                                    fill_tensor4D_rand(n, c, h, w, D, 0, 0, 0);
                                    fill_tensor4D_rand(k, c, r, s, F, 0, 0, 0);
                                } else { // NHWC
                                    fill_tensor4D_rand(n, h, w, c, D, 0, 0, 0);
                                    fill_tensor4D_rand(c, r, s, k, F, 0, 0, 0);
                                }

                                // Fill the result tensors with zeros
                                memset((void *) Y, 0, n * k * ho * wo * sizeof(DTYPE));
                                if (ec->test)
                                    memset((void *) Yg, 0, n * k * ho * wo * sizeof(DTYPE));
                                /* @todo: check that memset is actually a valid alternative to the following: */
                                /*
                                // (as all the tensor is filled, the tensor format does not affect the result)
                                for (in = 0; in < n; in++)
                                    for (ik = 0; ik < k; ik++)
                                        for (ih = 0; ih < ho; ih++)
                                            for (iw = 0; iw < wo; iw++) {
                                                Y[in * k * ho * wo + ik * ho * wo + ih * wo + wo] = (DTYPE) 0.0;
                                                if (ec->test)
                                                    Yg[in * k * ho * wo + ik * ho * wo + ih * wo +
                                                       wo] = (DTYPE) 0.0;
                                            }
                                */

                                if (ec->debug) {
                                    if (tensor_format == nchw) {
                                        print_tensor4D("D", n, c, h, w, D, 0, 0, 0);
                                        print_tensor4D("F", k, c, r, s, F, 0, 0, 0);
                                    } else {
                                        print_tensor4D("D", n, h, w, c, D, 0, 0, 0);
                                        print_tensor4D("F", c, r, s, k, F, 0, 0, 0);
                                    }
                                }


                                time_pre = 0.0;
                                time_kernel = 0.0;
                                time_post = 0.0;
                                nreps = 0;
                                while (time_kernel <= tmin) {

                                    t1 = dclock();
                                    convdirect_algorithm_parts.pre(
                                            n, k, c,
                                            h, w,
                                            r, s,
                                            D, F, Y,
                                            &DT, &FT, &YT);

                                    t2 = dclock();
                                    convdirect_algorithm_parts.kernel(
                                            n, k, c,
                                            h, w,
                                            r, s,
                                            vpadding, hpadding,
                                            vstride, hstride,
                                            vdilation, hdilation,
                                            (DTYPE) 1.0,
                                            DT,
                                            FT,
                                            (DTYPE) 1.0,
                                            YT
                                    );

                                    t3 = dclock();
                                    convdirect_algorithm_parts.post(
                                            n, k, c,
                                            h, w,
                                            r, s,
                                            &DT, &FT, &YT,
                                            Y);

                                    t4 = dclock();

                                    nreps++;

                                    time_pre += t2 - t1;
                                    time_kernel += t3 - t2;
                                    time_post += t4 - t3;
                                }

                                if (nreps == 0) continue;

                                time_pre = time_pre / nreps;
                                time_kernel = time_kernel / nreps;
                                time_post = time_post / nreps;

                                // Test result
                                if (ec->test) {
                                    char test_nhwc[] = "convdirect_original_nhwc_default";
                                    char test_nchw[] = "convdirect_original_nchw_default";
                                    convdirect_get_algorithm((tensor_format == nhwc) ? test_nhwc : test_nchw)(
                                            n, k, c,
                                            h, w,
                                            r, s,
                                            vpadding, hpadding,
                                            vstride, hstride,
                                            vdilation, hdilation,
                                            (DTYPE) 1.0,
                                            D,
                                            F,
                                            (DTYPE) 1.0,
                                            Yg);

                                    // print_tensor4D( "Yg", n, ho, wo, k, Yg, ldY1, ldY2, ldY3 );

                                    all_close = all_close_tensor4D(n, k, ho, wo, Y, Yg, &error, 0, 0, 0);

                                } else {
                                    error = 0.0;
                                }

                                flops = 2.0 * n * k * c * ho * wo * r * s;
                                GFLOPS = flops / (1.0e+9 * time_kernel);

                                if (ec->debug) {
                                    if (tensor_format == nhwc) {
                                        print_tensor4D("Y_test", n, ho, wo, k, Y, 0, 0, 0);
                                        print_tensor4D("Y_correct", n, ho, wo, k, Yg, 0, 0, 0);
                                    } else {
                                        print_tensor4D("Y_test", n, k, ho, wo, Y, 0, 0, 0);
                                        print_tensor4D("Y_correct", n, k, ho, wo, Yg, 0, 0, 0);
                                    }
                                }

                                // Get CIB, COB, WOB
                                CIB = convdirect_bs->cib;
                                COB = convdirect_bs->cob;
                                WOB = convdirect_bs->wob;

                                test_n++;

                                if (test_n % 2)
                                    printf(COLOR_CYAN);

                                printf("  %6d%6d%6d%6d%6d%6d%6d%6d%6d%6d "
                                       "%9.2e %9.2e %9.2e %9.2e %9.2e",
                                       CIB, COB, WOB, n, k, c, h, w, r, s,
                                       time_kernel, GFLOPS, time_pre, time_post, error);

                                if (test_n % 2)
                                    printf(COLOR_RESET);

                                if (ec->test)
                                    if (all_close)
                                        printf("      %sOK%s", COLOR_GREEN, COLOR_RESET);
                                    else {
                                        printf("   %sERROR%s", COLOR_RED, COLOR_RESET);
                                        // exit(EXIT_FAILURE);
                                    }
                                else
                                    printf("   %sDisabled%s", COLOR_BOLDYELLOW, COLOR_RESET);

                                if (!ec->test) {
                                    if (ec->type == CNN_TYPE)
                                        fprintf(ec->fd_out, "%d;", ec->cnn[cnn_i].layer);

                                    fprintf(ec->fd_out,
                                            "%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e;%.2e;%.2e\n",
                                            (tensor_format == nchw) ? "NCHW" : "NHWC", CIB,
                                            COB, WOB, n, k, c, h, w, r, s,
                                            time_kernel, GFLOPS, time_pre, time_post, error);
                                }

                                printf("\n");

                            }
                        }
                    }
                }
            }
        }

        /* Free data */
        free(D);
        free(F);
        free(Y);

        if (ec->test) {
            free(Yg);
        }
    }

#ifdef TVM
    A->data = A_aux;
    B->data = B_aux;
    C->data = C_aux;
    TVMArrayFree(A);
    TVMArrayFree(B);
    TVMArrayFree(C);
#endif

    if (ec->fd_out != NULL) fclose(ec->fd_out);
    free_evalConfig(ec);

    printf(" ==============================================================================================================================\n");

    return EXIT_SUCCESS;
}
