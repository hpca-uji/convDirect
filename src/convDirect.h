#ifdef TVM
    #include <dlpack/dlpack.h>
    #include <tvm/runtime/module.h>
    #include <tvm/runtime/packed_func.h>
    #include <tvm/runtime/registry.h>
#endif

#ifdef MK_BLIS
    #include <blis/blis.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "qblis.h"
#include "dtypes.h"
#include "formats.h"
#include "arrays.h"
#include "dtypes.h"
#include "gemm_blis_neon_fp32.h"

#define min(a,b)     ( (a) > (b) ? (b) : (a) )

void gemm_reference( char, char, char,
                     char, char,
                     int, int, int,
	             DTYPE, DTYPE *, int,
	             DTYPE *, int,
	             DTYPE, DTYPE *, int );

/* void gemm_base( int, int, int, */
/* 		DTYPE, DTYPE *, int, */
/* 		DTYPE *, int, */
/* 		DTYPE, DTYPE *, int ); */

/* void gemm_base_col( int, int, int, */
/* 		    DTYPE, DTYPE *, int, */
/* 		    DTYPE *, int, */
/* 		    DTYPE, DTYPE *, int ); */

void im2row(float *, int, float *,
	    int, int, int, int, int, int, int, int, int,
	    int, int, int, int, int);

void convDirect_original( int, int, int,
                          int, int,
                          int, int,
                          DTYPE *, int, int, int,
                          DTYPE *, int, int, int,
                          DTYPE *, int, int, int,
			  int);

void convDirect_renamed( int, int, int, 
                         int, int, 
                         int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int,
			 int);

void convDirect_reorder( int, int, int, 
                         int, int, 
                         int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int,
			 int);


void convDirect_block( int, int, int, 
                       int, int, 
                       int, int,
                       DTYPE *, int, int, int, 
                       DTYPE *, int, int, int, 
                       DTYPE *, int, int, int,
		       int, int, int, int);


void transform_input_tzemeng( int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int,
			      DTYPE *, int, int, int, int,
			      int, int);

void transform_output_tzemeng( int, int, 
			       int, int, 
			       int, int,
			       DTYPE *, int, int, int, 
			       DTYPE *, int, int, int, int,
			       int, int);


void transform_filter_tzemeng( int, int, 
			       int, int, 
			       DTYPE *, int, int, int,
			       DTYPE *, int, int, int, int, int,
			       int, int, int);


#ifdef TVM
    void convDirect_block_tzemeng( int, int, int, 
				     int, int, 
				     int, int,
				     DTYPE *, int, int, int, int,
				     DTYPE *, int, int, int, int, int,
				     DTYPE *, int, int, int, int,
				     tvm::runtime::PackedFunc,
				     DLTensor*, DLTensor*, DLTensor*, 
				     int, int, int, int);
#else
    void convDirect_block_tzemeng( int, int, int, 
				   int, int, 
				   int, int,
				   DTYPE *, int, int, int, int,
				   DTYPE *, int, int, int, int, int,
				   DTYPE *, int, int, int, int,
				   int, int, int, int);
#endif

void transform_filter_block_shalom( int, int, 
				    int, int, 
				    DTYPE *, int, int, int,
				    DTYPE *, int, int, int, int,
				    int);

void convDirect_block_shalom( int, int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *, int, int, int, int,
			      DTYPE *, int, int, int, 
			      int, int, int, int);


void transform_filter_block_blis( int, int, 
				  int, int, 
				  DTYPE *, int, int, int,
				  DTYPE *, int, int, int, int,
				  int);

#ifdef MK_BLIS
  void convDirect_block_blis( int, int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *, int, int, int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *,
			      int, int, int, int,
			      cntx_t *, auxinfo_t *, sgemm_ukr_ft,
			      int, int);
#else
  void convDirect_block_blis( int, int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *, int, int, int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *,
			      int, int, int, int);
#endif

void packRB( char, char, int, int, DTYPE *, int, DTYPE *, int);
