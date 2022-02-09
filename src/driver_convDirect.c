/* 
   Direct convolution 

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti" contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include <sys/time.h>

#include "omp.h"
#include "dtypes.h"
#include "formats.h"
#include "arrays.h"
#include "sutils.h"
#include "convDirect.h"
#include "colors.h"
#include "qblis.h"
#include "inutils.h"

#if defined(IM2COL) || defined(MK_BLIS) || defined(CONVGEMM)
  #include "blis/blis.h"
#endif

#ifdef CONVGEMM
  #undef min
  #include "../convGemmNHWC/src/gemm_blis.h"
  #include "../convGemmNHWC/src/im2row_nhwc.h"
#endif

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

int main(int argc, char *argv[])
{
  // The definition of these matrices are not necessary as the vectorized
  // versions implicitly contain them in the corresponding codes
  // This is only necessary for the generic version based on gemm operations
  // These parameteres for the vectorized variants can be NULL  

  char* variant;
  DTYPE *D, *F, *Y, *Yg, *DT, *FT, *YT, *FB, *DEXT, *Ac;
  
  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  int    m, t,
         nmin,  nmax,  nstep,
         kmin,  kmax,  kstep,
         cmin,  cmax,  cstep,
         hmin,  hmax,  hstep,
         wmin,  wmax,  wstep,
         rmin,  rmax,  rstep,
         smin,  smax,  sstep,
         prmax, psmax, ret,
         tformat, tformatmin, tformatmax,
         n, k, c,
         h, w,
         r, s,
         pr, ps,
         in, ir, is, ic, ik, ih, iw,
         ldD1,  ldD2,  ldD3,
         ldDT1, ldDT2, ldDT3, ldDT4,
         ldF1,  ldF2,  ldF3,
         ldFB1, ldFB2, ldFB3, ldFB4,
    ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
         ldY1,  ldY2,  ldY3,
         ldYT1, ldYT2, ldYT3, ldYT4,
         visual, nreps, 
    ho, wo, homax, womax;
  
  int ib, i, i2, ii, Ci_Cib, Co_Cob, Co_Nr, Co_Mr;
  char *filename;
  FILE *fd;
  int cnn_test_num, cnn_i;
  int CIB, COB, WOB;
  size_t test_n = 0;
  //VARIABLES FOR GEMM IM2COL
  int mm, nn, kk;
  DTYPE alphap;
  DTYPE betap;
  int lda, ldb, ldc;
  
  #ifdef BLOCKED_TZEMENG
    WOB = MR;
    COB = NR;
    CIB = NR;
  #elif BLOCKED_SHALOM
    WOB = 1575;
    COB = 2052;
    CIB = 292;
  #elif BLOCKED_BLIS
    #ifdef MK_8x12
    // MK_8x12
      WOB = 1792;
      COB = 3072;
      CIB = 640;
      // WOB = 896;
      // COB = 636;
      // CIB = 512;
      // BLIS parameters
      // WOB = 120;  // MC
      // COB = 3072; // NC
      // CIB = 640;  // KC
    #elif MK_4x4
    // MK_4x4
      WOB = 896;
      COB = 1024;
      CIB = 512;
    #elif MK_4x8
    // MK_4x8
      WOB = 896;
      COB = 1024;
      CIB = 512;
    #elif MK_4x12
    // MK_4x12
      WOB = 896;
      COB = 1008;
      CIB = 512;
    #elif MK_4x16
    // MK_4x16
      WOB = 896;
      COB = 1024;
      CIB = 512;
    #elif MK_4x20
    // MK_4x20
      WOB = 896;
      COB = 3060;
      CIB = 512;
    #elif MK_BLIS
    // MK_BLIS
      // WOB = 1792;
      // COB = 1008;
      // CIB = 256;
      // WOB = 120;  // MC
      // COB = 3072; // NC
      // CIB = 640;  // KC
      WOB = 888;
      COB = 640;
      CIB = 512;
      // BLIS parameters
      // COB = 120;  // MC
      // WOB = 3072; // NC
      // CIB = 640;  // KC
    #else
      printf("ERROR: WOB, CIB, COB not defined\n");
      exit(-1);
    #endif
  #else
      WOB = MR;
      COB = NR;
      CIB = NR;
  #endif

  #if defined(BLOCKED_SHALOM) || defined(BLOCKED_TZEMENG) || defined(BLOCKED_BLIS)
    #if MK_BLIS
    if (WOB % NR != 0) {
      printf("ERROR: WOB must be multiple of NR. Now WOB=%d and NR=%d\n", WOB, NR);
      exit(-1);
    } else if (COB % MR != 0) {
      printf("ERROR: COB must be multiple of MR. Now COB=%d and MR=%d\n", COB, MR);
    #else
    if (WOB % MR != 0) {
      printf("ERROR: WOB must be multiple of MR. Now WOB=%d and MR=%d\n", WOB, MR);
      exit(-1);
    } else if (COB % NR != 0) {
      printf("ERROR: COB must be multiple of NR. Now COB=%d and NR=%d\n", COB, NR);
    #endif
      exit(-1);
    }
  #endif
    
  testConfig_t* testConf=new_CNN_Test_Config(argv);
  
  m = 2; t = 6;
  
  tformatmin = NCHW;
  tformatmax = NCHW + NHWC;
  
  if (testConf->format == NCHW)
    tformatmax = NHWC;
  else if (testConf->format == NHWC)
    tformatmin = NHWC;
  
  #if defined(INT8)
    errorthd = 0.5;
  #elif defined(FP16)
    errorthd = 1.0e-3;
  #elif defined(FP32)
    errorthd = 1.0e-5;
  #elif defined(FP64)
    errorthd = 1.0e-14;
  #endif

  #if defined(MK_BLIS) || defined(CONVGEMM)
    //** MICRO-KERNEL BLIS **//
    #if defined(FP32)
      #define BTYPE BLIS_FLOAT
    #else
      #define BTYPE BLIS_DOUBLE
    #endif
    auxinfo_t aux;
    cntx_t * cntx;
    bli_init();
    cntx = bli_gks_query_cntx();
    sgemm_ukr_ft gemm_kernel = bli_cntx_get_l3_nat_ukr_dt(BLIS_FLOAT, /*BLIS_GEMM_UKR*/0, cntx);

    int mr = bli_cntx_get_blksz_def_dt(BTYPE, BLIS_MR, cntx);
    int nr = bli_cntx_get_blksz_def_dt(BTYPE, BLIS_NR, cntx);
    
    #ifdef CONVGEMM
      int KC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_KC, cntx);
      /*
      if (CIB >= KC) {
        printf("ERROR: CIB=%d must be less than PACKKC=%d", CIB, KC);
        exit(-1);
      }
      */
    
      int MC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_MC, cntx);
      int NC = bli_cntx_get_blksz_max_dt(BTYPE, BLIS_NC, cntx);
      DTYPE *Ac_pack = aligned_alloc(4096, MC * KC * sizeof(BTYPE));
      DTYPE *Bc_pack = aligned_alloc(4096, KC * NC * sizeof(BTYPE));
    #endif

  #endif
    
  #ifdef TVM
    //******* PREPARING TVM *******//
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
    //******* PREPARING TVM END *******//
  #endif

    if (testConf->type == CNN_TYPE)
      fprintf(testConf->fd_csv, "l;Variant;CIB;COB;WOB;n;k;c;h;w;kh;kw;Time;GFLOPS;Error\n");    
    else
      fprintf(testConf->fd_csv, "Variant;CIB;COB;WOB;n;k;c;h;w;kh;kw;Time;GFLOPS;Error\n");    

    printf(" ==============================================================================================================================\n");
    printf(" |%s                     D R I V E R    F O R    D I R E C T    C O N V O L U T I O N    E V A L U A T I O N       %s             |\n",
	   COLOR_BOLDYELLOW, COLOR_RESET);
    printf(" ==============================================================================================================================\n");
    printf(" |                                          [%s*%s]MR:%2d                  [%s*%s]NR:%2d                                                |\n",
	   COLOR_BOLDYELLOW, COLOR_RESET, MR,
	   COLOR_BOLDYELLOW, COLOR_RESET, NR);
    printf(" ==============================================================================================================================\n");
    printf(" |  %s Variant     CIB     COB     WOB     n     k     c    h     w      kh    kw    Time     GFLOPS     Error     Evaluation%s   |\n",
	   COLOR_RESET, COLOR_RESET);
    printf(" ==============================================================================================================================\n");
    
    tmin = testConf->tmin;
        
    for (cnn_i = 0; cnn_i < testConf->cnn_num; cnn_i++)
    {
      
        nmin  = testConf->cnn[cnn_i].nmin;
	nmax  = testConf->cnn[cnn_i].nmax;
	nstep = testConf->cnn[cnn_i].nstep;
	
	kmin  = testConf->cnn[cnn_i].kmin;
	kmax  = testConf->cnn[cnn_i].kmax;
	kstep = testConf->cnn[cnn_i].kstep;
	
	cmin  = testConf->cnn[cnn_i].cmin;
	cmax  = testConf->cnn[cnn_i].cmax;
	cstep = testConf->cnn[cnn_i].cstep;
	
	hmin  = testConf->cnn[cnn_i].hmin;
	hmax  = testConf->cnn[cnn_i].hmax;
	hstep = testConf->cnn[cnn_i].hstep;
	
	wmin  = testConf->cnn[cnn_i].wmin;
	wmax  = testConf->cnn[cnn_i].wmax;
	wstep = testConf->cnn[cnn_i].wstep;
	
	rmin  = testConf->cnn[cnn_i].rmin;
	rmax  = testConf->cnn[cnn_i].rmax;
	rstep = testConf->cnn[cnn_i].rstep;
	
	smin  = testConf->cnn[cnn_i].smin;
	smax  = testConf->cnn[cnn_i].smax;
	sstep = testConf->cnn[cnn_i].sstep;

	//WARNING: ONLY FOR GEMM TEST; TODO: FIX THIS WITH PADDING!!
	if ((rmax == 3) && (smax == 3)) {
	  hmin += 2;
	  hmax += 2;
	  wmin += 2;
	  wmax += 2;
	}
	    
      homax = floor(((double) hmax - rmin) / 1) + 1;
      womax = floor(((double) wmax - smin) / 1) + 1;
     
      D = (DTYPE *) malloc( nmax*cmax*hmax*wmax*sizeof(DTYPE));
      DEXT = (DTYPE *) malloc( (hmax*wmax*nmax)*(rmax*smax*cmax)*sizeof(DTYPE));
      
      F = (DTYPE *) malloc( kmax*cmax*rmax*smax*sizeof(DTYPE));   
      Y = (DTYPE *) malloc( nmax*kmax*homax*womax*sizeof(DTYPE));
      
      DT = (DTYPE *) malloc( nmax*ceil(((float) cmax)/CIB)*CIB*hmax*wmax*sizeof(DTYPE));   
      FT = (DTYPE *) malloc( ceil(((float) kmax)/COB)*COB*ceil(((float) cmax)/CIB)*CIB*rmax*smax*sizeof(DTYPE));   
      YT = (DTYPE *) malloc( nmax*ceil(((float) kmax)/COB)*COB*homax*womax*sizeof(DTYPE));   
      
      //FB = (DTYPE *) malloc( ceil(((float) kmax)/COB)*COB*cmax*rmax*smax*sizeof(DTYPE));
#ifdef MK_BLIS
      FB = (DTYPE *) malloc( ceil(((float) kmax)/MR)*MR*cmax*rmax*smax*sizeof(DTYPE));
#else
      FB = (DTYPE *) malloc( ceil(((float) kmax)/NR)*NR*cmax*rmax*smax*sizeof(DTYPE));
#endif
      
      if ( testConf->test=='T' )
	Yg = (DTYPE *) malloc( nmax*kmax*homax*womax*sizeof(DTYPE) );   
      
      for ( n=nmin; n<=nmax; n+=nstep ){
      for ( k=kmin; k<=kmax; k+=kstep ){
      for ( c=cmin; c<=cmax; c+=cstep ){
      for ( h=hmin; h<=hmax; h+=hstep ){
      for ( w=wmin; w<=wmax; w+=wstep ){
      for ( r=rmin; r<=rmax; r+=rstep ){
      //for ( s=smin; s<=smax; s+=sstep ){
      for ( tformat=tformatmin; tformat<tformatmax; tformat+=1 ){
	//for (COB = 192; COB <= 256; COB += 36){
	//for (CIB = 48; CIB <= 384; CIB += 12){
	//for (WOB = 512; WOB <= 2048; WOB += 128){
	s = r;
	ho     = floor(((double) h - r) / 1) + 1;
	wo     = floor(((double) w - s) / 1) + 1;
	Ci_Cib = (int)ceil(((float) c)/CIB);
	Co_Cob = (int)ceil(((float) k)/COB);
	Co_Nr  = (int)ceil(((float) k)/NR);
	Co_Mr  = (int)ceil(((float) k)/MR);
#ifdef MK_BLIS
        // Prepare to call micro-kernel with transposed operands
        Ac = (DTYPE *) aligned_alloc( 4096, ((int) ceil((WOB-1))/NR+1)*NR*CIB*sizeof(DTYPE));
#else
        Ac = (DTYPE *) aligned_alloc( 4096, ((int) ceil((WOB-1))/MR+1)*MR*CIB*sizeof(DTYPE));
#endif
	// printf("size of Ac %d\n", ((int) ceil((WOB-1))/MR+1)*MR*CIB);
	// printf("size of Ac %d %d %d\n", WOB, (int) ceil((WOB-1.0)/MR+1.0), ((int) ceil((WOB-1.0)/MR+1.0))*MR*CIB);
	
	if ( tformat == NCHW ) { // NCHW
	  ldD3 = w;
	  ldD2 = h*ldD3;
	  ldD1 = c*ldD2;
	  
	  ldF3 = s;
	  ldF2 = r*ldF3;
	  ldF1 = c*ldF2;
	  
	  ldY3 = wo;
	  ldY2 = ho*ldY3;
	  ldY1 = k*ldY2;
	  
	  //==// CONVOLUTION-FRIENDLY LAYOUT //==//
	  //NHWC MACRO DT[]
	  ldDT4 = CIB;
	  ldDT3 = w      * ldDT4;
	  ldDT2 = h      * ldDT3;      
	  ldDT1 = Ci_Cib * ldDT2;
	  //NHWC MACRO YT[]
	  ldYT4 = COB;
	  ldYT3 = wo     * ldYT4;
	  ldYT2 = ho     * ldYT3;      
	  ldYT1 = Co_Cob * ldYT2;
	  //NHWC MACRO FT[]
	  ldFT5 = COB;
	  ldFT4 = CIB    * ldFT5;
	  ldFT3 = s      * ldFT4;
	  ldFT2 = r      * ldFT3;
	  ldFT1 = Co_Cob * ldFT2;
	  //==//===========================//==//

	  //NHWC MACRO FB[] 
          #ifdef MK_BLIS
	    ldFB4 = MR;
	    ldFB3 = c*ldFB4;
	    ldFB2 = Co_Mr*ldFB3;
	    ldFB1 = s*ldFB2;
          #else
	    ldFB4 = NR;
	    ldFB3 = c*ldFB4;
	    ldFB2 = Co_Nr*ldFB3;
	    ldFB1 = s*ldFB2;
          #endif

	  generate_tensor4D( n, c, h, w, D, ldD1, ldD2, ldD3 );
	  generate_tensor4D( k, c, r, s, F, ldF1, ldF2, ldF3 );
	} 
	else { // NHWC
	  //NHWC MACRO D[] 
	  ldD3 = c;
	  ldD2 = w * ldD3;
	  ldD1 = h * ldD2;
	  //NHWC MACRO F[] 
	  ldF3 = k;
	  ldF2 = s*ldF3;
	  ldF1 = r*ldF2;
	  //NHWC MACRO Y[] 
	  ldY3 = k;
	  ldY2 = wo*ldY3;
	  ldY1 = ho*ldY2;
	  
	  //==// CONVOLUTION-FRIENDLY LAYOUT //==//
	  //NHWC MACRO DT[]
	  ldDT4 = CIB;
	  ldDT3 = w      * ldDT4;
	  ldDT2 = h      * ldDT3;      
	  ldDT1 = Ci_Cib * ldDT2;
	  //NHWC MACRO YT[]
	  ldYT4 = COB;
	  ldYT3 = wo     * ldYT4;
	  ldYT2 = ho     * ldYT3;      
	  ldYT1 = Co_Cob * ldYT2;
	  //NHWC MACRO FT[]
	  ldFT5 = COB;
	  ldFT4 = CIB    * ldFT5;
	  ldFT3 = s      * ldFT4;
	  ldFT2 = r      * ldFT3;
	  ldFT1 = Co_Cob * ldFT2;
	  //==//===========================//==//
	  
	  //NHWC MACRO FB[] 
#ifdef MK_BLIS
	  ldFB4 = MR;
	  ldFB3 = c*ldFB4;
	  ldFB2 = Co_Mr*ldFB3;
	  ldFB1 = s*ldFB2;
#else
	  ldFB4 = NR;
	  ldFB3 = c*ldFB4;
	  ldFB2 = Co_Nr*ldFB3;
	  ldFB1 = s*ldFB2;
#endif
	  
	  generate_tensor4D( n, h, w, c, D, ldD1, ldD2, ldD3 );
	  generate_tensor4D( c, r, s, k, F, ldF1, ldF2, ldF3 );
	  
	}
	
	// Set result to zeros
	for ( in=0; in<n; in++ )
	for ( ik=0; ik<k; ik++ )
	for ( ih=0; ih<ho; ih++ )
	for ( iw=0; iw<wo; iw++ ) {
	  if (tformat == NHWC)
	    Yrow_NHWC(in,ik,ih,iw) = 0.0;
	  else
	    Yrow_NCHW(in,ik,ih,iw) = 0.0;
	  
	  if ( testConf->test=='T' )
	    if (tformat == NHWC)
	      Ygrow_NHWC(in,ik,ih,iw) = 0.0;
	    else
	      Ygrow_NCHW(in,ik,ih,iw) = 0.0;
	}

    
        #if BLOCKED_TZEMENG
	for ( in=0; in<n; in++)
        for ( ih=0; ih<ho; ih++)
    	for ( iw=0; iw<wo; iw++)
    	for ( i=0,i2=0; i<k; i+=COB,i2++) {
	  ib = min(k-i, COB);
	  for ( ii=0; ii<ib; ii++)
	    YT(in, i2, ih, iw, ii) = 0.0;
	  }
        #endif

	if ( testConf->debug=='T' ) {
          if ( tformat == NCHW ) {
            print_tensor4D( "D", n, c, h, w, D, ldD1, ldD2, ldD3 );
            print_tensor4D( "F", k, c, r, s, F, ldF1, ldF2, ldF3 );
          } else {
            print_tensor4D( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
            print_tensor4D( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
          }
        }

	// Convolution
        #if BLOCKED_BLIS
	  transform_filter_block_blis(c, k,
				      r, s,
				      F,  ldF1,  ldF2,  ldF3,
				      FB, ldFB1, ldFB2, ldFB3, ldFB4, 
				      tformat);
        #elif BLOCKED_SHALOM
	  transform_filter_block_shalom(c, k,
				        r, s,
				        F,  ldF1,  ldF2,  ldF3,
				        FB, ldFB1, ldFB2, ldFB3, ldFB4, 
				        tformat);
        #elif BLOCKED_TZEMENG	
	  transform_input_tzemeng(n, c,
				  h, w, 
				  r, s,
				  D,  ldD1,  ldD2,  ldD3,
				  DT, ldDT1, ldDT2, ldDT3, ldDT4,
				  tformat, CIB);
	
	  transform_filter_tzemeng(c, k,
				   r, s,
				   F,  ldF1,  ldF2,  ldF3,
				   FT, ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
				   tformat, CIB, COB);
        #endif      
	  
	time  = 0.0; 
	t1    = dclock();
	nreps = 0;
	while ( time <= tmin ) {
	  // Convolution
          #ifdef IM2COL
	    im2row(DEXT, c * r * s, D, n, h, w, c, ho, wo, r,
	           s, 0, 0, 1, 1, 1, 1);

	    mm = k;
	    nn = ho * wo * n;
	    kk = r * s * c;
	    alphap = 1.0;
	    betap  = 0.0;
	    lda = k;
	    ldb = r * s * c;
	    ldc = k;
	  
	    sgemm_( "N", "N", &mm, &nn, &kk, 
	            &alphap,
	             F, &lda,
	             DEXT, &ldb,
	             &betap,
	             Y, &ldc );

         #elif CONVGEMM

	    int vpadding  = 0;
	    int hpadding  = 0;
	    int vdilation = 1;
	    int hdilation = 1;
	    int vstride   = 1;
	    int hstride   = 1;

	    int ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
	    int wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;

	    conv_p conv_params = { n, h, w, c, k, r, s,
	    vstride, hstride, vpadding, hpadding,
	    vdilation, hdilation, ho, wo,
	    NULL, NULL, NULL, NULL, NULL, false };

            gemm_blis_B3A2C0_orig('C', 'C', 'C',
				  'N', 'N',
				  k, ho * wo * n, r * s * c,
				  1.0, F, k,
				  D, r * s * c,
				  0.0, Y, k,
				  Ac_pack, pack_RB, 
				  Bc_pack, pack_CB_nhwc, NULL, cntx, &conv_params);
	
         #elif RENAMED
	    convDirect_renamed(n, k, c,
			       h, w,
			       r, s, 
			       D, ldD1, ldD2, ldD3, 
			       F, ldF1, ldF2, ldF3, 
			       Y, ldY1, ldY2, ldY3,
			       tformat);
          #elif REORDER
          convDirect_reorder(n, k, c,
			     h, w,
			     r, s, 
			     D, ldD1, ldD2, ldD3, 
			     F, ldF1, ldF2, ldF3, 
			     Y, ldY1, ldY2, ldY3,
			     tformat);
          #elif BLOCKED
          convDirect_block(n, k, c,
			   h, w,
			   r, s, 
			   D, ldD1, ldD2, ldD3, 
			   F, ldF1, ldF2, ldF3, 
			   Y, ldY1, ldY2, ldY3,
			   tformat, CIB, COB, WOB);
          #elif BLOCKED_BLIS
	    #ifdef MK_BLIS
              convDirect_block_blis(n, k, c,
	  		            h, w,
			            r, s, 
			            D,  ldD1,  ldD2,  ldD3, 
			            FB, ldFB1, ldFB2, ldFB3, ldFB4,
			            Y,  ldY1,  ldY2,  ldY3,
                                    Ac, 
			            tformat, CIB, COB, WOB,
	                            cntx, &aux, gemm_kernel, mr, nr);
	    #else
	      convDirect_block_blis(n, k, c,
	  		            h, w,
			            r, s, 
			            D,  ldD1,  ldD2,  ldD3, 
			            FB, ldFB1, ldFB2, ldFB3, ldFB4,
			            Y,  ldY1,  ldY2,  ldY3,
                                    Ac, 
			            tformat, CIB, COB, WOB);
	    #endif
          #elif BLOCKED_SHALOM
          convDirect_block_shalom(n, k, c,
			        h, w,
			        r, s, 
			        D,  ldD1,  ldD2,  ldD3, 
			        FB, ldFB1, ldFB2, ldFB3, ldFB4,
			        Y,  ldY1,  ldY2,  ldY3,
			        tformat, CIB, COB, WOB);
          #elif BLOCKED_TZEMENG	
            #ifdef TVM
	      convDirect_block_tzemeng(n, k, c,
				       h, w,
				       r, s,
				       DT, ldDT1, ldDT2, ldDT3, ldDT4,
				       FT, ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
				       YT, ldYT1, ldYT2, ldYT3, ldYT4,
				       f, A, B, C,
				       tformat, CIB, COB, WOB);
            #else
	      convDirect_block_tzemeng(n, k, c,
	  			       h, w,
	  			       r, s,
				       DT, ldDT1, ldDT2, ldDT3, ldDT4,
				       FT, ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
				       YT, ldYT1, ldYT2, ldYT3, ldYT4,
				       tformat, CIB, COB, WOB);
            #endif
          #endif      
	  nreps++;
	  
	  t2   = dclock();
	  time = ( t2 > t1 ? t2 - t1 : 0.0 );
	}
	time = time/nreps;
	if ( nreps == 0 ) continue; 
	
        #if BLOCKED_TZEMENG	
	  transform_output_tzemeng(n, k,
				   h, w,
				   r, s,
				   Y,  ldY1,  ldY2,  ldY3,
				   YT, ldYT1, ldYT2, ldYT3, ldYT4,
				   tformat, COB);
        #endif   
	  
	// Test result
	if ( testConf->test=='T' ) {
	  convDirect_original(n, k, c, h, w, r, s, 
			      D,  ldD1, ldD2, ldD3, 
			      F,  ldF1, ldF2, ldF3, 
			      Yg, ldY1, ldY2, ldY3,
			      tformat);
	  //print_tensor4D( "Yg", n, ho, wo, k, Yg, ldY1, ldY2, ldY3 );
	  
	  error = 0.0;
	  nrm   = 0.0;
	  for ( in=0; in<n; in++ )
	  for ( ik=0; ik<k; ik++ )
	  for ( ih=0; ih<ho; ih++ )
	  for ( iw=0; iw<wo; iw++ ) {
	    if (tformat == NHWC) {
	      tmp = (double) Ygrow_NHWC(in,ik,ih,iw);
	      nrm += tmp*tmp;
	      tmp = (double) dabs(Yrow_NHWC(in,ik,ih,iw)-Ygrow_NHWC(in,ik,ih,iw));
	      //printf("Y=%14.8e vs Yg=%14.8e\n", Yrow_NCHW(in,ik,ih,iw), Ygrow_NCHW(in,ik,ih,iw));
	      error += tmp*tmp;
	    } else {
	      tmp = (double) Ygrow_NCHW(in,ik,ih,iw);
	      nrm += tmp*tmp;
	      tmp = (double) dabs(Yrow_NCHW(in,ik,ih,iw)-Ygrow_NCHW(in,ik,ih,iw));	  
	      //printf("Y=%14.8e vs Yg=%14.8e\n", Yrow_NCHW(in,ik,ih,iw), Ygrow_NCHW(in,ik,ih,iw));
	      error += tmp*tmp;
	    }
	  }
	  if ( nrm!=0.0 )
	    error = sqrt(error) / sqrt(nrm);
	  else
	    error = sqrt(error);
	}
	else
	  error = -1.0;
        
	flops   = 2.0 * n * k * c * h * w * r * s;
	GFLOPS  = flops / (1.0e+9 * time );
	
	if ( testConf->debug=='T' ) {
          if ( tformat == NCHW ) {
            print_tensor4D( "Ytest", n, k, h, w, Y, ldY1, ldY2, ldY3 );
            print_tensor4D( "Ycorrect", n, k, h, w, Yg, ldY1, ldY2, ldY3 );
          } else {
            print_tensor4D( "Ytest", n, h, w, k, Y, ldY1, ldY2, ldY3 );
            print_tensor4D( "Ycorrect", n, h, w, k, Yg, ldY1, ldY2, ldY3 );
          }
        }
		
	if ((test_n++ % 2) == 0)
	  printf("%s  %8s %9d %6d %7d %6d %5d %5d %5d %5d %5d %5d %10.2e %9.2e %9.2e%s",
		 COLOR_CYAN, (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error,  COLOR_RESET);
	else
	  printf("  %8s %9d %6d %7d %6d %5d %5d %5d %5d %5d %5d %10.2e %9.2e %9.2e",
		 (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB,  n, k, c, h, w, r, s, time, GFLOPS, error);
	
	
	if ( testConf->test=='T' )
	  if ( error < errorthd )
	    printf("     %sOK%s", COLOR_GREEN, COLOR_RESET);
	  else {
	    printf("     %sERROR%s\n", COLOR_RED, COLOR_RESET);
	    exit(-1);
	  }
	else
	  printf("     %sDisabled%s", COLOR_BOLDYELLOW, COLOR_RESET);
	
    if (testConf->type == CNN_TYPE)
      fprintf(testConf->fd_csv,"%d;%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e\n",testConf->cnn[cnn_i].layer, (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error);
    else
      fprintf(testConf->fd_csv,"%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e\n",(tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error);
		
	printf("\n");

      } } }
      free(Ac); 
      } } } } //} } // Variation of CIB, WOB, COB

      /* Free data */
      free(Y);
      free(D);
      free(F);
      free(YT);
      free(FB);
      free(DEXT);
      
      if ( testConf->test=='T' )
	free(Yg);

    }
    
    #ifdef TVM
      A->data = A_aux;
      B->data = B_aux;
      C->data = C_aux;	
      TVMArrayFree(A);
      TVMArrayFree(B);
      TVMArrayFree(C);
    #endif

    fclose(testConf->fd_csv);
    free_CNN_Test_Config(testConf);
    
    printf(" ==============================================================================================================================\n");

  return 0;
}
