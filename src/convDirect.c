#include "convDirect.h"

/*
void gemm_base( int m, int n, int k, 
                DTYPE alpha, DTYPE *A, int ldA, 
                             DTYPE *B, int ldB, 
                DTYPE beta,  DTYPE *C, int ldC ){

  //Baseline micro-kernel 
  //Replace with specialized micro-kernel where C-->m x n is resident in registers

  int    i, j, p;
  DTYPE  tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Arow(i,p) * Brow(p,j);
      Crow(i,j) = alpha*tmp + beta*Crow(i,j);
    }
}

void gemm_base_col( int m, int n, int k, 
		    DTYPE alpha, DTYPE *A, int ldA, 
		    DTYPE *B, int ldB, 
		    DTYPE beta,  DTYPE *C, int ldC ) {

  //Baseline micro-kernel 
  //Replace with specialized micro-kernel where C-->m x n is resident in registers

  int    i, j, p;
  DTYPE  tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Bcol(p,j);
      Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
    }
}
*/

void im2row(float *rows, int ld, float *in,
	    int batch, int height, int width, int channel, int oheight, int owidth,
	    int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
	    hstride, int vdilation, int hdilation)
{

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
		  rows[row * channel * kheight * kwidth + col] = in[((b * height + ix) * width + iy) * channel + c];
		}
	    }
	}
      }
}

void convDirect_original( int n, int k, int c, 
                          int h, int w, 
                          int r, int s, 
                          DTYPE *D, int ldD1, int ldD2, int ldD3,
	                  DTYPE *F, int ldF1, int ldF2, int ldF3,
                          DTYPE *Yg, int ldY1, int ldY2, int ldY3,
			  int tformat)
{ 
  int in, ik, ic, ih, iw, ir, is, x_x, x_y, ho, wo;

  // Quick return if possible
  if ( (n==0)||(k==0)||(c==0)||
       (h==0)||(w==0)||
       (r==0)||(s==0))
    return;

  ho = floor(((double) h - r) / 1) + 1;
  wo = floor(((double) w - s) / 1) + 1;

  if (tformat == NHWC) {
    for ( in=0;  in<n;   in++ ) 
    for ( ik=0;  ik<k;   ik++ ) 
    for ( ic=0;  ic<c;   ic++ ) 
    for ( ih=0;  ih<ho;  ih++ ) 
    for ( iw=0;  iw<wo;  iw++ ) 
    for ( ir=0;  ir<r;   ir++ ) {
      x_x = ih + ir;
      if (0 <= x_x && x_x < h) 
	for ( is=0; is<s; is++ ) {
	  x_y = iw + is;
	  if (0 <= x_y && x_y < w) {
            //printf("FB %d %d %d %d %16.10e\n", ik, ic, ir, is, Frow_NHWC(ik,ic,ir,is));
	    Ygrow_NHWC(in,ik,ih,iw) += Drow_NHWC(in,ic,x_x,x_y) * Frow_NHWC(ik,ic,ir,is);        
          }
	}
    }
  } else {
    for ( in=0;  in<n;   in++ ) 
    for ( ik=0;  ik<k;   ik++ ) 
    for ( ic=0;  ic<c;   ic++ ) 
    for ( ih=0;  ih<ho;  ih++ ) 
    for ( iw=0;  iw<wo;  iw++ ) 
    for ( ir=0;  ir<r;   ir++ ) {
      x_x = ih + ir;
      if (0 <= x_x && x_x < h) 
	for ( is=0; is<s; is++ ) {
	  x_y = iw + is;
	  if (0 <= x_y && x_y < w)
	    Ygrow_NCHW(in,ik,ih,iw) += Drow_NCHW(in,ic,x_x,x_y) * Frow_NCHW(ik,ic,ir,is);        
	}
    }
  }

}  


void convDirect_renamed( int t, int Co, int Ci, 
                         int Ho, int Wo, 
                         int Hf, int Wf, 
                         DTYPE *D, int ldD1, int ldD2, int ldD3,
	                 DTYPE *F, int ldF1, int ldF2, int ldF3,
                         DTYPE *Y, int ldY1, int ldY2, int ldY3,
			 int tformat)
{ 
  int h, i, j, k, l, m, n, x_x, x_y, ho, wo;

  // Quick return if possible
  if ( (t==0)||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0))
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

  if (tformat == NHWC) {
    for ( h=0;  h<t;   h++ ) 
    for ( i=0;  i<Ci;   i++ ) 
    for ( j=0;  j<Co;   j++ ) 
    for ( k=0;  k<wo;   k++ ) 
    for ( l=0;  l<ho;   l++ ) 
    for ( m=0;  m<Wf;   m++ ) {
      x_y = k + m;
      if (0 <= x_y && x_y < Wo) {
	for ( n=0;  n<Hf;   n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    Yrow_NHWC(h,j,l,k) += Drow_NHWC(h,i,x_x,x_y) * Frow_NHWC(j,i,n,m);
	}
      }
    }
  } else {
    for ( h=0;  h<t;   h++ ) 
    for ( i=0;  i<Ci;   i++ ) 
    for ( j=0;  j<Co;   j++ ) 
    for ( k=0;  k<wo;   k++ ) 
    for ( l=0;  l<ho;   l++ ) 
    for ( m=0;  m<Wf;   m++ ) {
      x_y = k + m;
      if (0 <= x_y && x_y < Wo) {
	for ( n=0;  n<Hf;   n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    Yrow_NCHW(h,j,l,k) += Drow_NCHW(h,i,x_x,x_y) * Frow_NCHW(j,i,n,m);
	}
      }
    }

  }
}



void convDirect_reorder( int t, int Co, int Ci, 
                         int Ho, int Wo, 
                         int Hf, int Wf, 
                         DTYPE *D, int ldD1, int ldD2, int ldD3,
	                 DTYPE *F, int ldF1, int ldF2, int ldF3,
                         DTYPE *Y, int ldY1, int ldY2, int ldY3,
			 int tformat)
{ 
  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  // Accommodate vectorization: j as the innermost loop
  // Ensure sufficient independent operations: k around j
  // For compatibility between output layer n and input layer n+1: n->m->i

  int h, i, j, k, l, m, n, x_x, x_y, ho, wo;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

  if (tformat == NHWC) {
    for ( h=0;  h<t; h++ )
      for ( l=0;  l<ho; l++ )
	for ( n=0;  n<Hf; n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    for ( m=0;  m<Wf; m++ )
	      for ( i=0;  i<Ci; i++ )
		for ( k=0;  k<wo; k++ ) {
		  x_y = k + m;
		  if (0 <= x_y && x_y < Wo)
		    for ( j=0;  j<Co;   j++ )
		      Yrow_NHWC(h,j,l,k) += Drow_NHWC(h,i,x_x,x_y) * Frow_NHWC(j,i,n,m);
		}
	}
  } else {
    for ( h=0;  h<t; h++ )
      for ( l=0;  l<ho; l++ )
	for ( n=0;  n<Hf; n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    for ( m=0;  m<Wf; m++ )
	      for ( i=0;  i<Ci; i++ )
		for ( k=0;  k<wo; k++ ) {
		  x_y = k + m;
		  if (0 <= x_y && x_y < Wo)
		    for ( j=0;  j<Co;   j++ )
		      Yrow_NCHW(h,j,l,k) += Drow_NCHW(h,i,x_x,x_y) * Frow_NCHW(j,i,n,m);
		}
	}
  }
  
}

void convDirect_block( int t,     int Co,   int Ci, 
                       int Ho,    int Wo, 
                       int Hf,    int Wf,
		       DTYPE *D,  int ldD1, int ldD2, int ldD3,
	               DTYPE *F,  int ldF1, int ldF2, int ldF3,
                       DTYPE *Y, int ldY1, int ldY2, int ldY3,
		       int tformat, int CIB, int COB, int WOB)
{ 
  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  // Accommodate vectorization: j as the innermost loop
  // Ensure sufficient independent operations: k around j
  // For compatibility between output layer n and input layer n+1: n->m->i

  int h, i, j, k, l, m, n, x_x, x_y, 
      ho, wo, ii, jj, kk, ib, jb, kb;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++ ) 
      for ( j=0; j<Co; j+=COB ) {
	jb = min(Co-j, COB);
	for ( i=0; i<Ci; i+=CIB ) {
	  ib = min(Ci-i, CIB);
	  for ( l=0; l<ho; l++ ) 
	    for ( k=0; k<wo; k+=WOB ) {
	      kb = min(wo-k, WOB);
	      for ( n=0; n<Hf; n++ ) {
		x_x = l + n;
		if (0 <= x_x && x_x < Ho)
		  for ( m=0; m<Wf; m++ )
		    for ( ii=0; ii<ib; ii++ )
		      for ( kk=0; kk<kb; kk++ ) {
			x_y = k + kk + m;
			if (0 <= x_y && x_y < Wo)
			  for ( jj=0; jj<jb; jj++ )
			    Yrow_NHWC(h,j+jj,l,k+kk) += Drow_NHWC(h,i+ii,l+n,k+kk+m) * Frow_NHWC(j+jj,i+ii,n,m);
		      }
	      }
	    }
	}
      }
  } else {
    for ( h=0; h<t; h++ ) 
      for ( j=0; j<Co; j+=COB ) {
	jb = min(Co-j, COB);
	for ( i=0; i<Ci; i+=CIB ) {
	  ib = min(Ci-i, CIB);
	  for ( l=0; l<ho; l++ ) 
	    for ( k=0; k<wo; k+=WOB ) {
	      kb = min(wo-k, WOB);
	      for ( n=0; n<Hf; n++ ) {
		x_x = l + n;
		if (0 <= x_x && x_x < Ho)
		  for ( m=0; m<Wf; m++ )
		    for ( ii=0; ii<ib; ii++ )
		      for ( kk=0; kk<kb; kk++ ) {
			x_y = k + kk + m;
			if (0 <= x_y && x_y < Wo)
			  for ( jj=0; jj<jb; jj++ )
			    Yrow_NCHW(h,j+jj,l,k+kk) += Drow_NCHW(h,i+ii,l+n,k+kk+m) * Frow_NCHW(j+jj,i+ii,n,m);
		      }
	      }
	    }
	}
      }

  }
} 


void transform_input_tzemeng( int t, int Ci,
			      int Ho, int Wo,
			      int Hf, int Wf,
			      DTYPE *D, int ldD1, int ldD2, int ldD3,
			      DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			      int tformat, int CIB) {
  int     h,
          i, j,
          k, l,
          m, n,
          ho, wo,
          ii, jj, kk,
          ib, jb, kb;

  int i2, x;

  if ( (t==0) ||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++)
      for ( l=0; l<Ho; l++)
	for ( k=0; k<Wo; k++)
	  for ( i=0,i2=0; i<Ci; i+=CIB,i2++) {
	    ib = min(Ci-i, CIB);
	    for ( ii=0; ii<ib; ii++)
	      DT(h, i2, l, k, ii) = Drow_NHWC(h, i+ii, l, k);	  
	      }
  } else {
    for ( h=0; h<t; h++)
      for ( l=0; l<Ho; l++)
	for ( k=0; k<Wo; k++)
	  for ( i=0,i2=0; i<Ci; i+=CIB,i2++) {
	    ib = min(Ci-i, CIB);
	    for ( ii=0; ii<ib; ii++)
	      DT(h, i2, l, k, ii) = Drow_NCHW(h, i+ii, l, k);	  
	      }
    
  }

}

void transform_output_tzemeng( int t, int Co,
			       int Ho, int Wo,
			       int Hf, int Wf,
			       DTYPE *Y, int ldY1, int ldY2, int ldY3,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       int tformat, int COB) {
  int h,
      i, i2, j,
      k, l,
      m, n,
      ho, wo,
      ii, jj, kk,
      ib, jb, kb;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++)
      for ( l=0; l<ho; l++)
	for ( k=0; k<wo; k++)
	  for ( i=0,i2=0; i<Co; i+=COB,i2++) {
	    ib = min(Co-i, COB);
	    for ( ii=0; ii<ib; ii++)
	      Yrow_NHWC(h, i+ii, l, k) = YT(h, i2, l, k, ii);	  
	  }
  } else {
    for ( h=0; h<t; h++)
      for ( l=0; l<ho; l++)
	for ( k=0; k<wo; k++)
	  for ( i=0,i2=0; i<Co; i+=COB,i2++) {
	    ib = min(Co-i, COB);
	    for ( ii=0; ii<ib; ii++)
	      Yrow_NCHW(h, i+ii, l, k) = YT(h, i2, l, k, ii);	  
	  }
  }
}



void transform_filter_tzemeng( int Ci, int Co,
			       int Hf, int Wf,
			       DTYPE *F, int ldF1, int ldF2, int ldF3,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       int tformat, int CIB, int COB) {
  int     h,
          i, j,
          k, l,
          m, n,
          ho, wo,
          ii, jj, kk,
          ib, jb, kb;
  int i2, j2;

  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
	ib = min(Ci-i, CIB);       
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( ii=0; ii<ib; ii++ )
	      for ( jj=0; jj<jb; jj++ )
		FT(i2, j2, n, m, ii, jj ) = Frow_NHWC(j+jj, i+ii, n,  m);
      }
    }
  } else {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
	ib = min(Ci-i, CIB);       
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( ii=0; ii<ib; ii++ )
	      for ( jj=0; jj<jb; jj++ )
		FT(i2, j2, n, m, ii, jj ) = Frow_NCHW(j+jj, i+ii, n,  m);
      }
    }
  }
  
}

#ifdef TVM

void convDirect_block_tzemeng( int t, int Co, int Ci,
			       int Ho, int Wo,
			       int Hf, int Wf,
			       DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       tvm::runtime::PackedFunc tvm_f,
			       DLTensor *A, DLTensor *B, DLTensor *C, 
			       int tformat, int CIB, int COB, int WOB) {
  int  h,
    i, j, i2, j2,
    k, l,
    m, n,
    ho, wo,
    ii, jj, kk,
    ib, jb, kb, ob;
  
  //int n_if  = 0;
  //int n_else = 0;
  //int o;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;
  
  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;
  
  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  for ( h=0; h<t; h++ ) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
        ib = min(Ci-i, CIB);
        for ( l=0; l<ho; l++ ) {
          for ( k=0; k<wo; k+=WOB ) {
            kb = min(wo-k, WOB);
            for ( n=0; n<min(Hf,Ho-l); n++ ) {
              for ( m=0; m<Wf; m++ ) {		
		// int mr=kb=WOB, int nr=jb=COB, int kc=ib=CIB
		ob = min(kb,Wo-k-m+1);
		if ((ob == MR) && (jb == NR) && (ib == NR)) { //ib=kc=NR
		  A->data = &DT(h, i2, l+n, k+m, 0);
		  B->data = &FT(i2, j2, n, m, 0, 0);
		  C->data = &YT(h, j2, l, k, 0);
		  tvm_f(A, B, C, C);
		} else {
		  if ((ob == MR) && (jb == NR))
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32_fixed( ob, jb, ib,
									      1.0, &DT(h, i2, l+n, k+m, 0), 
									      &FT(i2, j2, n, m, 0, 0), 
									      1.0, &YT(h, j2, l, k, 0), ldYT4 );
		  else
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );
		  /* gemm_microkernel_Cresident_neon_4x4_fp32( ob, jb, ib, */
		  /* 					    1.0, &DT(h, i2, l+n, k+m, 0),  */
		  /* 					    &FT(i2, j2, n, m, 0, 0),  */
		  /* 					    1.0, &YT(h, j2, l, k, 0), ldYT4 ); */
		}
	      
	      } } } } } } }
  
}

#else

void convDirect_block_tzemeng( int t, int Co, int Ci,
			       int Ho, int Wo,
			       int Hf, int Wf,
			       DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       int tformat, int CIB, int COB, int WOB) {
  int  h,
    i, j, i2, j2,
    k, l,
    m, n,
    ho, wo,
    ii, jj, kk,
    ib, jb, kb, ob;

  int n_if  = 0;
  int n_else = 0;
  int o;
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  for ( h=0; h<t; h++ ) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
        ib = min(Ci-i, CIB);
        for ( l=0; l<ho; l++ ) {
          for ( k=0; k<wo; k+=WOB ) {
            kb = min(wo-k, WOB);
            for ( n=0; n<min(Hf,Ho-l); n++ ) {
              for ( m=0; m<Wf; m++ ) {
		 ob = min(kb,Wo-k-m+1);
                 #ifdef MK_7x12_U4
		  if ((ob == MR) && (jb == NR))
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32_fixed( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );
                  else
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );
		#else
		    printf("ERROR: Microkernel doesn't exist.\n");
		    exit(-1);
                #endif
		    
		    //gemm_base( min(kb,Wo-k-m+1), jb, ib,
		    //         1.0, &DT(h, i2, l+n, k+m, 0), ldDT4,
		    //	       &FT(i2, j2, n, m, 0, 0), ldFT5,
		    //	       1.0, &YT(h, j2, l, k, 0),     ldYT4 );

	      } } } } } } }
  
}

#endif


void transform_filter_block_shalom( int Ci, int Co,
				    int Hf, int Wf,
				    DTYPE *F,  int ldF1,  int ldF2,  int ldF3,
				    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
				    int tformat) {
  int  i, j, jj, jb, j2, m, n;  
  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( j=0,j2=0; j<Co; j+=NR,j2++ ) {
      jb = min(Co-j, NR);
      for ( i=0; i<Ci; i++ )
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( jj=0; jj<jb; jj++ ) {
              FBrow_NHWC(j2, i, n, m, jj) = Frow_NHWC(j+jj, i, n, m);
	    }
    }
  } else {
    printf("Case not yet implemented!\n");
    exit(-1);
  }
}



void convDirect_block_shalom( int t,     int Co,   int Ci, 
			      int Ho,    int Wo, 
			      int Hf,    int Wf,
			      DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
			      DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
			      DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
			      int tformat, int CIB, int COB, int WOB)
{ 
  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  // Accommodate vectorization: j as the innermost loop
  // Ensure sufficient independent operations: k around j
  // For compatibility between output layer n and input layer n+1: n->m->i

  int h, i, j, k, l, m, n, i2, j2,
      ho, wo, ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR;

  int jr, nr, jr2, ir, mr;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

   
  if (tformat == NHWC) { 
     for ( h=0; h<t; h++ ) 
       for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
         jb = min(Co-j, COB); 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=0; l<ho; l++ ) 
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ )
                 for ( m=0; m<Wf; m++ ) 
		  for ( jr=0, jr2=0; jr < jb; jr += NR, jr2++) {
		    nr = min(jb-jr, NR);
		    for ( ir=0; ir < min(kb, Wo-k-m+1); ir += MR) {
		      mr = min(min(kb, Wo-k-m+1)-ir, MR);
                      /*
			gemm_base( mr, nr, ib,
  	  	                 1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,
  	  	                      &FBrow_NHWC(j2*Cob_nr+jr2, i, n, m, 0), ldFB4,
  	  	                 1.0, &Yrow_NHWC(h, j+jr, l, k+ir),     ldY3 );
                      */
                      #if MK_7x12_NPA_U4
		       if ((mr == MR) && (nr == NR))
			 gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32_fixed( mr, nr, ib, 
		                                                                          1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
											  &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
											  1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
		      else
			gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32( mr, nr, ib, 
										    1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
										    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
										    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                     #else
		       printf("ERROR: Microkernel doesn't exist.\n");
		       exit(-1);
                     #endif

		    }
                  }
             } 
         } 
       } 
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   } 
}

void transform_filter_block_blis( int Ci, int Co,
				  int Hf, int Wf,
				  DTYPE *F,  int ldF1,  int ldF2,  int ldF3,
				  DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
				  int tformat) {
  int  i, j, jj, jb, j2, m, n;  
  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
#ifdef MK_BLIS
    /* Prepare to call micro-kernel with transposed operands */
    for ( j=0,j2=0; j<Co; j+=MR,j2++ ) {
      jb = min(Co-j, MR);
#else
    for ( j=0,j2=0; j<Co; j+=NR,j2++ ) {
      jb = min(Co-j, NR);
#endif
      for ( i=0; i<Ci; i++ )
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( jj=0; jj<jb; jj++ ) {
              FBrow_NHWC(j2, i, n, m, jj) = Frow_NHWC(j+jj, i, n, m);
              //printf("copy %d %d %d %d --> %d %d %d %d %d, %16.10e\n", j+jj, i, n, m, j2, i, n, m, jj, Frow_NHWC(j+jj, i, n, m));
	    }
    }
  } else {
    printf("Case not yet implemented!\n");
    exit(-1);
  }
}


#ifdef MK_BLIS
void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac,
		            int tformat, int CIB, int COB, int WOB,
			    cntx_t * cntx, auxinfo_t * aux, sgemm_ukr_ft gemm_kernel,
			    int b_mr, int b_nr)
#else
void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac,
		            int tformat, int CIB, int COB, int WOB)
#endif
{ 
  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  // Accommodate vectorization: j as the innermost loop
  // Ensure sufficient independent operations: k around j
  // For compatibility between output layer n and input layer n+1: n->m->i

  int blis_mr, blis_nr;
  
  #ifdef MK_BLIS
    // Prepare to call micro-kernel with transposed operands
    blis_mr = b_nr;
    blis_nr = b_mr;
  #else
    blis_mr = MR;
    blis_nr = NR;
  #endif
  
  int h, i, j, k, l, m, n, i2, j2,
      ho, wo, ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR, Cob_Mr = COB/MR;
      //DTYPE Cc[MR*NR], blis_beta = 0.0;

  int jr, nr, jr2, ir, mr, in = 0;

  float alpha = 1.0;
  float beta  = 1.0;
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  ho = floor(((float) Ho - Hf) / 1) + 1;
  wo = floor(((float) Wo - Wf) / 1) + 1;

   
  if (tformat == NHWC) { 
     for ( h=0; h<t; h++ ) 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=0; l<ho; l++ ) 
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ )
                 for ( m=0; m<Wf; m++ ) {
                   packRB( 'R', 'N', kb, ib, &Drow_NHWC(h, i, l+n, k+m), ldD3, Ac, blis_mr);
		   for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
		     jb = min(Co-j, COB); 
		     for ( jr=0, jr2=0; jr < jb; jr += blis_nr, jr2++) {
		       nr = min(jb-jr, blis_nr);
		       for ( ir=0; ir < min(kb, Wo-k-m+1); ir += blis_mr) {
			 mr = min(min(kb, Wo-k-m+1)-ir, blis_mr);
                        /*
			gemm_reference( 'C', 'R', 'R', 
                              'N', 'N',
                              mr, nr, ib,
  	  	              1.0, &Ac[ir*ib], MR, 
  	  	                   &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0), NR,
  	  	              1.0, &Yrow_NHWC(h, j+jr, l, k+ir),     ldY3 );
                        */
                       #ifdef MK_BLIS
                         /* Call micro-kernel with transposed operands */
                         //printf("mr %d nr %d blis_mr %d blis_nr %d\n", mr, nr, blis_mr, blis_nr);
			   gemm_kernel(nr, mr, ib, 
                                       &alpha, &FBrow_NHWC(j2*Cob_Mr+jr2, i, n, m, 0), 
                                               &Ac[ir*ib], 
			 	       &beta,  &Yrow_NHWC(h, j+jr, l, k+ir), 1, ldY3, aux, cntx);
                         /* THis alternative relies in our micro-kernel, to avoid using the one in BLIS for border cases 
                         if ((nr==MR)&&(mr==NR))
			   gemm_kernel(nr, mr, ib, 
                                       &alpha, &FBrow_NHWC(j2*Cob_Mr+jr2, i, n, m, 0), 
                                               &Ac[ir*ib], 
			 	       &beta,  &Yrow_NHWC(h, j+jr, l, k+ir), 1, ldY3, aux, cntx);
                         else {
		           gemm_microkernel_Cresident_neon_8x12_fp32( nr, mr, ib, 
                                                                      1.0, &FBrow_NHWC(j2*Cob_Mr+jr2, i, n, m, 0), 
                                                                           &Ac[ir*ib], 
			 	                                      0.0, &Cc, mr);
                         */
                         /* This MUST be done for correct solution 
                            It has a considerable impact on performance 
                            The overhead could be avoided by transposing the micro-tile of C internally to the routine. 
                            Howver, even it droping it, the result does not outperform the manual micro-kernel 8x12.*/
                         /* The alternative is to use the BLIS micro-kernel for all cases, but the performance is even lower */
                         /*
                           for (int i1=0; i1<nr; i1++)
                             for (int j1=0; j1<mr; j1++)
			 	Yrow_NHWC(h, j+jr+i1, l, k+ir+j1) += Cc[i1*mr+j1];
                         }
                         */
                         /* ORIGINAL
			 gemm_kernel(mr, nr, ib, &alpha, &Ac[ir*ib], &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
			 	     &beta, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3, 1, aux, cntx);
                         */
		       #elif MK_8x12
                         //printf("mr %d nr %d blis_mr %d blis_nr %d\n", mr, nr, blis_mr, blis_nr);
                         if ((mr==MR)&&(nr==NR))
		             gemm_microkernel_Cresident_neon_fixed_8x12_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                        &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                        1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                         else
		             gemm_microkernel_Cresident_neon_8x12_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                        &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                        1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                       #elif MK_4x12
		         gemm_microkernel_Cresident_neon_4x12_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                       #elif MK_4x16
		         gemm_microkernel_Cresident_neon_4x16_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                       #elif MK_4x20
                         if ((mr==MR)&&(nr==NR))
		         gemm_microkernel_Cresident_neon_fixed_4x20_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                         else
		         gemm_microkernel_Cresident_neon_4x20_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                       #elif MK_4x4
		         gemm_microkernel_Cresident_neon_4x4_prefetch_fp32( mr, nr, ib, 1.0, &Ac[ir*ib], 
                                                                           &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),
                                                                           1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                       #else
		         printf("ERROR: Microkernel doesn't exist.\n");
		         exit(-1);
                       #endif
                       //}
		    }
                  }
                }
             } 
         } 
       } 
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   }

  //printf("In MK_BLIS = %s\n", in == 0 ? "FALSE" : "TRUE");
  
}

void packRB( char orderA, char transA, int mc, int nc, DTYPE *A, int ldA, DTYPE *Ac, int RR ){
/*
  BLIS pack for A-->Ac
*/
  int    i, j, ii, k, rr;

  if ( ((transA=='N')&&( orderA=='C'))||
       ((transA=='T')&&( orderA=='R')) )
    //#pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Ac[k] = Acol(i+ii,j);
          k++;
        }
        k += (RR-rr);
      }
    }
  else
    //#pragma omp parallel for private(j, ii, rr, k)
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
           Ac[k] = Acol(j,i+ii);
          k++;
        }
        k += (RR-rr);
      }
    }
}

void gemm_reference( char orderA, char orderB, char orderC, 
                     char transA, char transB, 
                     int m, int n, int k, 
                     DTYPE alpha, DTYPE *A, int ldA, 
	                          DTYPE *B, int ldB, 
                     DTYPE beta,  DTYPE *C, int ldC ){
   int    ic, jc, pc, i, j, p;
   DTYPE  zero = 0.0, one = 1.0, tmp;

   // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  if ( (transA=='N')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='N')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0.0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else {
    printf("Error: Invalid options for transA, transB: %c %c\n", transA, transB);
    exit(-1);
  }
}


