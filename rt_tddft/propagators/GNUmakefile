OBJ = exp_diag.o exp_pseries.o prop_euler.o prop_magnus_exp_bch.o prop_magnus_exp_bch_scaled.o prop_magnus_ord2_interpol.o prop_magnus_ord2_step.o prop_rk4.o rttddft_gpu.o rttddft_gpu_basis.o

OBJ_OPTIMIZE = 

LIBRARY = libnwdft.a

USES_BLAS =

# Hardcoding CUDA support for debugging
# Adding to both LIB_DEFINES and FOPTIMIZE to ensure it's picked up
MY_CUDA_FLAGS = -DDEBUG_PRINT -DRTTDDFT_CUDA -I/usr/local/cuda-12.4/include/
LIB_DEFINES = $(MY_CUDA_FLAGS)
FOPTIMIZE   += $(MY_CUDA_FLAGS)

##
## KAL: hack to allow symlinked directory
##
LIB_INCLUDES = -I../../include -I../../../ddscf

HEADERS = 

##
## KAL: hack to allow symlinked directory
##
include ../../../config/makefile.h
include ../../../config/makelib.h

# Explicit rule to ensure make knows how to build the cuda object
.SUFFIXES: .cu

rttddft_gpu.o: rttddft_gpu.cu
	/usr/local/cuda-12.4/bin/nvcc -c -arch=sm_80 -I/usr/local/cuda-12.4/include/ $< -o $@