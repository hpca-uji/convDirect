//*NHWC
#define  Drow_NHWC(a1,a2,a3,a4)  D[(a1)*(ldD1) + (a3)*(ldD2) + (a4)*(ldD3) + (a2)]
#define  Yrow_NHWC(a1,a2,a3,a4)  Y[(a1)*(ldY1) + (a3)*(ldY2) + (a4)*(ldY3) + (a2)]
#define Ygrow_NHWC(a1,a2,a3,a4) Yg[(a1)*(ldY1) + (a3)*(ldY2) + (a4)*(ldY3) + (a2)]
#define  Frow_NHWC(a1,a2,a3,a4)  F[(a2)*(ldF1) + (a3)*(ldF2) + (a4)*(ldF3) + (a1)]

//*NCHW
#define  Drow_NCHW(a1,a2,a3,a4)  D[(a1)*(ldD1) + (a2)*(ldD2) + (a3)*(ldD3) + (a4)]
#define  Frow_NCHW(a1,a2,a3,a4)  F[(a1)*(ldF1) + (a2)*(ldF2) + (a3)*(ldF3) + (a4)]
#define  Yrow_NCHW(a1,a2,a3,a4)  Y[(a1)*(ldY1) + (a2)*(ldY2) + (a3)*(ldY3) + (a4)]
#define Ygrow_NCHW(a1,a2,a3,a4) Yg[(a1)*(ldY1) + (a2)*(ldY2) + (a3)*(ldY3) + (a4)]

// DATA LAYAOUT TRANSFORM
#define DT(a1,a2,a3,a4,a5)    DT[(a1)*(ldDT1) + (a2)*(ldDT2) + (a3)*(ldDT3) + (a4)*(ldDT4) + (a5)]
#define YT(a1,a2,a3,a4,a5)    YT[(a1)*(ldYT1) + (a2)*(ldYT2) + (a3)*(ldYT3) + (a4)*(ldYT4) + (a5)]
#define FT(a1,a2,a3,a4,a5,a6) FT[(a1)*(ldFT1) + (a2)*(ldFT2) + (a3)*(ldFT3) + (a4)*(ldFT4) + (a5)*ldFT5 + (a6)]

// DATA LAYAOUT TRANSFORM (BLOCK)
#define  FBrow_NHWC(a1,a2,a3,a4,a5)  FB[(a3)*(ldFB1) + (a4)*(ldFB2) + (a1)*(ldFB3) + (a2)*(ldFB4) + (a5)]

// GEMM
#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
