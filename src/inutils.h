#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "colors.h"
#include "formats.h"

#define CNN_MAX_TEST 128
#define CNN_TYPE     0
#define BATCH_TYPE   1


typedef struct cnn {
  int nmin;
  int nmax; 
  int nstep;

  int kmin; 
  int kmax; 
  int kstep;

  int cmin;
  int cmax;
  int cstep;

  int hmin;
  int hmax;
  int hstep;

  int wmin;
  int wmax;
  int wstep;

  int rmin;
  int rmax;
  int rstep;

  int smin;
  int smax;
  int sstep;

  int layer;
} cnn_t;

typedef struct testConfig {
  cnn_t cnn[CNN_MAX_TEST];
  unsigned int cnn_num;
  double tmin;
  char test;
  unsigned char type;
  char debug;
  FILE *fd_csv;  
  unsigned char format;
} testConfig_t;
  

void set_CNN(int, int, char *, int, cnn_t *);
testConfig_t * new_CNN_Test_Config(char **);
void free_CNN_Test_Config(testConfig_t *);
