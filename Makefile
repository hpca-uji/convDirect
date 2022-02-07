#*--- CONVOLUTION ALGORITHM ---*#
#*    ALGORITHM OPTIONS:       
#*      [*] -DIM2COL
#*      [*] -DCONVGEMM
#*      [*] -DRENAMED
#*      [*] -DREORDER
#*      [*] -DBLOCKED
#*      [*] -DBLOCKED_TZEMENG
#*      [*] -DBLOCKED_SHALOM
#*      [*] -DBLOCKED_BLIS
#*-----------------------------*#
ALGORITHM = -DCONVGEMM
#------------------------------*#


#*--- MATRIX ELEMENTS DTYPE ---*#
#*    DTYPE OPTIONS:           
#*      [*] -DINT8
#*      [*] -DFP16
#*      [*] -DFP32
#*      [*] -DFP64
#*-----------------------------*#
DTYPE     = -DFP32
#------------------------------*#


#*-- MICRO-KERNEL CALL TYPE ---*#
#*    DTYPE OPTIONS:
#*      [*] -DGEMM
#-------------------------------
#*      MICRO-KERNEL FOR BLIS
#*      [*] -DMK_4x4
#*      [*] -DMK_4x12
#*      [*] -DMK_4x16
#*      [*] -DMK_4x20
#*      [*] -DMK_8x12
#*      [*] -DMK_BLIS
#-------------------------------
#*      MICRO-KERNEL FOR TZE-MENG
#*      [*] -DMK_7x12_U4
#*      [*] -DTVM
#-------------------------------
#*      MICRO-KERNEL FOR SHALOM
#*      [*] -DMK_7x12_NPA_U4
#*-----------------------------*#
MKERNEL    =
#------------------------------*#


#*----- COMPILER OPTIONS ------*#
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
# Linux
	CC       = gcc
	CLINKER  = gcc
	OPTFLAGS = -march=armv8-a -fPIC -O3 #-std=c++14 -Wall
else
# MacOs
	CC       = gcc -arch arm64
	CLINKER  = gcc -arch arm64
	OPTFLAGS = # -march=armv8-a -std=c++14 -fPIC -O3 #-Wall
endif
#*-----------------------------*#


#*------- LIBRARIES -----------*#
LIBS_PATH      = 
INCLUDES_PATH  = 
LIBS           = -lm -ldl -pthread
#-----------------------------*#

vpath %.c ./src
vpath %.h ./src

BIN        = driver_convDirect.x
OBJDIR     = build
_OBJ       = driver_convDirect.o convDirect.o sutils.o inutils.o gemm_blis_neon_fp32.o 

#CHANGE WITH YOUR BLIS INSTALLATION PATH
BLIS_ROOT = /home/nano/software/blis_install/
#CHANGE WITH YOUR TVM INSTALLATION PATH	
TVM_ROOT  = /home/nano/software/apache-tvm-src-v0.8.0.rc0/


ifeq ($(ALGORITHM), -DIM2COL)
	LIBS_PATH     := $(LIBS_PATH) $(BLIS_ROOT)/lib/libblis.a
	INCLUDES_PATH := $(INCLUDES_PATH) -I$(BLIS_ROOT)/include/blis/
	MKERNEL        =
else ifeq ($(ALGORITHM), -DCONVGEMM)
	_OBJ          := $(_OBJ) gemm_blis_B3A2C0_orig.o im2row_nhwc.o gemm_blis.o
	LIBS_PATH     := $(LIBS_PATH) $(BLIS_ROOT)/lib/libblis.a
	INCLUDES_PATH := $(INCLUDES_PATH) -I$(BLIS_ROOT)/include/blis/
	MKERNEL        =
endif

ifeq ($(MKERNEL), -DMK_BLIS)
	LIBS_PATH     := $(LIBS_PATH) $(BLIS_ROOT)/lib/libblis.a
	INCLUDES_PATH := $(COMMON_INC_PATH) -I$(BLIS_ROOT)/include/blis/
else ifeq ($(MKERNEL), -DTVM)
	#TVM ONLY AVAILABLE WITH DL_BLOCKED IMPLEMENTATION
	ALGORITHM      = -DBLOCKED_TZEMENG
	#TVM ONLY AVAILABLE WITH FLOAT 32 DTYPE
	DTYPE          = -DFP32       
	DMLC_CORE      = ${TVM_ROOT}/3rdparty/dmlc-core
	LIBS_PATH     := $(LIBS_PATH) -L${TVM_ROOT}/build
	INCLUDES_PATH := $(INCLUDES_PATH) -I${TVM_ROOT}/include\
				-I${DMLC_CORE}/include\
				-I${TVM_ROOT}/3rdparty/dlpack/include\
				-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
	LIBS          := $(LIBS) -ltvm_runtime
	_OBJ          := $(_OBJ) microkernel_gen.o
	CC             = g++
	CLINKER        = g++
endif

OBJ         = $(patsubst %, $(OBJDIR)/%, $(_OBJ))
RUN_OPT     = $(ALGORITHM) $(DTYPE) $(MKERNEL)

default: $(OBJDIR)/$(BIN)

$(OBJDIR)/%.o: %.c 
	@mkdir -p $(OBJDIR)
	$(CC) $(OPTFLAGS) $(RUN_OPT) -c -o  $@ $< $(INCLUDES_PATH)

$(OBJDIR)/$(BIN): $(OBJ)
	$(CLINKER) $(OPTFLAGS) $(INCLUDES_PATH) -o $@ $^ $(LIBS) $(LIBS_PATH)

$(OBJDIR)/microkernel_gen.o:
	python3 ./tvm/microkernel_generator.py

#RULES FOR CONGEMM 
$(OBJDIR)/gemm_blis_B3A2C0_orig.o:
	$(CLINKER) $(OPTFLAGS) -c -o $@ convGemmNHWC/gemm_blis_B3A2C0_orig.c $(INCLUDES_PATH)

$(OBJDIR)/im2row_nhwc.o:
	$(CLINKER) $(OPTFLAGS) -c -o $@ convGemmNHWC/im2row_nhwc.c $(INCLUDES_PATH)

$(OBJDIR)/gemm_blis.o:
	$(CLINKER) $(OPTFLAGS) -c -o $@ convGemmNHWC/gemm_blis.c $(INCLUDES_PATH)

clean:
	rm $(OBJDIR)/*
