#include "inutils.h"

void set_CNN(int col, int cnn_num, char *tmp, int type, cnn_t *cnn) {
  if (type == CNN_TYPE)  
    switch(col) {
    case 0:
      cnn[cnn_num].layer  = atoi(tmp);
      break;
    case 1: //Kn
      cnn[cnn_num].kmin  = atoi(tmp);
      cnn[cnn_num].kmax  = atoi(tmp);
      cnn[cnn_num].kstep = 1;
      break;
    case 2: //Wo
      break;
    case 3: //Ho
      break;
    case 4: //t
      cnn[cnn_num].nmin  = atoi(tmp);
      cnn[cnn_num].nmax  = atoi(tmp);
      cnn[cnn_num].nstep = 1;
      break;
    case 5: //Kh
      cnn[cnn_num].rmin  = atoi(tmp);
      cnn[cnn_num].rmax  = atoi(tmp);
      cnn[cnn_num].rstep = 1;
      break;
    case 6: //Kw
      cnn[cnn_num].smin  = atoi(tmp);
      cnn[cnn_num].smax  = atoi(tmp);
      cnn[cnn_num].sstep = 1;
      break;
    case 7: //Ci
      cnn[cnn_num].cmin  = atoi(tmp);
      cnn[cnn_num].cmax  = atoi(tmp);
      cnn[cnn_num].cstep = 1;
      break; 
    case 8: //Wi
      cnn[cnn_num].wmin  = atoi(tmp);
      cnn[cnn_num].wmax  = atoi(tmp);
      cnn[cnn_num].wstep = 1;
      break;
    case 9: //Hi
      cnn[cnn_num].hmin = atoi(tmp);
      cnn[cnn_num].hmax = atoi(tmp);
      cnn[cnn_num].hstep = 1;
      break;
    }
  else
    switch(col) {
    case 0:
      cnn[cnn_num].nmin  = atoi(tmp);
      break;
    case 1:
      cnn[cnn_num].nmax  = atoi(tmp);
      break;
    case 2:
      cnn[cnn_num].nstep = atoi(tmp);
      break;
    case 3: 
      cnn[cnn_num].kmin  = atoi(tmp);
      break;
    case 4: 
      cnn[cnn_num].kmax  = atoi(tmp);
      break;
    case 5: 
      cnn[cnn_num].kstep = atoi(tmp);
      break;
    case 6: 
      cnn[cnn_num].cmin  = atoi(tmp);
      break;
    case 7: 
      cnn[cnn_num].cmax  = atoi(tmp);
      break; 
    case 8: 
      cnn[cnn_num].cstep = atoi(tmp);
      break;
    case 9: 
      cnn[cnn_num].hmin = atoi(tmp);
      break;
    case 10: 
      cnn[cnn_num].hmax = atoi(tmp);
      break;
    case 11: 
      cnn[cnn_num].hstep = atoi(tmp);
      break;
    case 12: 
      cnn[cnn_num].wmin = atoi(tmp);
      break;
    case 13: 
      cnn[cnn_num].wmax = atoi(tmp);
      break;
    case 14: 
      cnn[cnn_num].wstep = atoi(tmp);
      break;
    case 15: 
      cnn[cnn_num].rmin = atoi(tmp);
      break;
    case 16: 
      cnn[cnn_num].rmax = atoi(tmp);
      break;
    case 17: 
      cnn[cnn_num].rstep = atoi(tmp);
      break;
    case 18: 
      cnn[cnn_num].smin = atoi(tmp);
      break;
    case 19: 
      cnn[cnn_num].smax = atoi(tmp);
      break;
    case 20: 
      cnn[cnn_num].sstep = atoi(tmp);
      break;
    }
}


testConfig_t* new_CNN_Test_Config(char * argv[]) {
  FILE *fd_conf = fopen(argv[2], "r"); //open config file
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  const char delimiter[] = "\t";
  char *tmp;
  int col;
  unsigned char type;  
  testConfig_t *new_testConfig = (testConfig_t *)malloc(sizeof(testConfig_t));
  int cnn_num;
  char format_str[12];
  char algorithm[32];
  char kernel[32];

  new_testConfig->tmin   = atof(argv[3]);
  new_testConfig->test   = argv[4][0];
  new_testConfig->debug  = argv[5][0];
  new_testConfig->fd_csv = fopen(argv[6], "w");

  if (strcmp(argv[7], "BOTH") == 0) {
    new_testConfig->format = NHWC + NCHW;
    sprintf(format_str, "%s", "NHWC|NCHW");
  } else if (strcmp(argv[7], "NCHW") == 0) {
    new_testConfig->format = NCHW;
    sprintf(format_str, "%s", "NCHW");
  } else if (strcmp(argv[7], "NHWC") == 0) {
    new_testConfig->format = NHWC;
    sprintf(format_str, "%s", "NHWC");
  } else {
    printf("ERROR: Matrix Format unrecognized.\n");
    exit(-1);
  }
  
  #ifdef IM2COL
    sprintf(algorithm, "%s", "IM2COL");
  #elif CONVGEMM
    sprintf(algorithm, "%s", "CONVGEMM");
  #elif RENAMED
    sprintf(algorithm, "%s", "RENAMED");
  #elif REORDER
    sprintf(algorithm, "%s", "REORDER");
  #elif BLOCKED
    sprintf(algorithm, "%s", "BLOCKED");
  #elif BLOCKED_TZEMENG
    sprintf(algorithm, "%s", "BLOCKED_TZEMENG");
  #elif BLOCKED_SHALOM
    sprintf(algorithm, "%s", "BLOCKED_SHALOM");
  #elif BLOCKED_BLIS
    sprintf(algorithm, "%s", "BLOCKED_BLIS");
  #else
    sprintf(algorithm, "%s", "UNKNOWN");
  #endif

  #ifdef GEMM
    sprintf(kernel, "%s", "GEMM");
  #elif  MK_4x4
    sprintf(kernel, "%s", "MK_4X4");
  #elif MK_4x12
    sprintf(kernel, "%s", "MK_4X12");
  #elif MK_4x16
    sprintf(kernel, "%s", "MK_4X16");
  #elif MK_4x20
    sprintf(kernel, "%s", "MK_4X20");
  #elif MK_8x12
    sprintf(kernel, "%s", "MK_8X12");
  #elif MK_BLIS
    sprintf(kernel, "%s", "MK_BLIS");
  #elif MK_7x12_U4
    sprintf(kernel, "%s", "MK_7X12_U4");
  #elif TVM
    sprintf(kernel, "%s", "TVM");
  #elif MK_7x12_NPA_U4
    sprintf(kernel, "%s", "MK_7X12_NPA_U4");
  #else
    sprintf(kernel, "%s", "-");
  #endif
    
  printf("\n =====================================================================\n");
  printf(" |%s               T E S T     C O N F I G U R A T I O N               %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" =====================================================================\n");
  printf(" |  [%s*%s] Matrix Format Selected |  %-35s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, format_str);
  printf(" |  [%s*%s] Test Verification      |  %-35s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, argv[4][0] == 'T' ? "ON" : "OFF");
  printf(" |  [%s*%s] Mode Debug             |  %-35s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, argv[5][0] == 'T' ? "ON" : "OFF");
  printf(" |  [%s*%s] Configuration Selected |  %-35s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, argv[2]);
  printf(" |  [%s*%s] File Results Selected  |  %-35s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, argv[6]);
  printf(" =====================================================================\n");
  printf(" |  [%s*%s] Algorithm Selected     |  %s%-35s%s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, COLOR_BOLDMAGENTA, algorithm, COLOR_RESET);
  printf(" |  [%s*%s] Micro-Kernel Selected  |  %s%-35s%s|\n",  COLOR_BOLDYELLOW, COLOR_RESET, COLOR_BOLDMAGENTA, kernel, COLOR_RESET);
  printf(" =====================================================================\n\n");

   
  if ((new_testConfig->debug == 'T') && (new_testConfig->test != 'T')) {
    printf("WARNING: Mode debug enable. Test mode automatically enabled.\n");
    new_testConfig->test = 'T';
    new_testConfig->tmin = 0.0;
  }
  
  if ((new_testConfig->tmin > 0.0) && (new_testConfig->test == 'T')) {
    printf("ERROR: If 'Test check' enabled, 'Tmin' must be 0. Fix it and run the Test Driver again.\n");
    exit(-1);
  }
  
  cnn_num=0;  

  if ((argv[1][0] == 'c') && (argv[1][1] == 'n') && (argv[1][2] == 'n'))
    type = CNN_TYPE;
  else
    type = BATCH_TYPE;
  
  while ((read = getline(&line, &len, fd_conf)) != -1)
    if (line[0] != '#') {      
      col = 0;
      tmp = strtok(line, delimiter);
      if (tmp == NULL)
	break;
      set_CNN(col, cnn_num, tmp, type, new_testConfig->cnn);
      col++;
      for (;;) {
	tmp = strtok(NULL, delimiter);
	if (tmp == NULL)
	  break;
	set_CNN(col, cnn_num, tmp, type, new_testConfig->cnn);
	col++;
      }

      cnn_num++;
    }

  fclose(fd_conf); 

  new_testConfig->cnn_num = cnn_num;
  new_testConfig->type = type;
  
  return new_testConfig;
}

void free_CNN_Test_Config(testConfig_t *testConfig) {
  free(testConfig);
}

//printf("nmin=%d; nmax=%d; nstep=%d; kmin=%d; kmax=%d; kstep=%d; cmin=%d; cmax=%d; cstep=%d; hmin=%d; hmax=%d; hstep=%d, wmin=%d; wmax=%d; wstep=%d; rmin=%d; rmax=%d; rstep=%d; smin=%d; smax=%d; sstep=%d\n",
//       cnn[*cnn_num].nmin, cnn[*cnn_num].nmax, cnn[*cnn_num].nstep,
//       cnn[*cnn_num].kmin, cnn[*cnn_num].kmax, cnn[*cnn_num].kstep,
//       cnn[*cnn_num].cmin, cnn[*cnn_num].cmax, cnn[*cnn_num].cstep,
//       cnn[*cnn_num].hmin, cnn[*cnn_num].hmax, cnn[*cnn_num].hstep,
//      cnn[*cnn_num].wmin, cnn[*cnn_num].wmax, cnn[*cnn_num].wstep,
//       cnn[*cnn_num].rmin, cnn[*cnn_num].rmax, cnn[*cnn_num].rstep,
//       cnn[*cnn_num].smin, cnn[*cnn_num].smax, cnn[*cnn_num].sstep);
