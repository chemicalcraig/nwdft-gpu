#
#   $Id$
#

    OBJ_OPTIMIZE = dft_gpu_basis.o dft_gpu_fock.o

#   Only compile .cu if CUDA is enabled
ifdef RTTDDFT_CUDA
    OBJ_OPTIMIZE += nwdft_gpu.o
endif

    LIBRARY = libnwdft.a

    # Standard NWChem Includes
    LIB_INCLUDES = -I../include -I../../include -I../../ddscf

ifdef RTTDDFT_CUDA
    # CUDA flags and includes
    MY_CUDA_FLAGS = -arch=sm_60 -O3
    
    # Add CUDA include path if provided
    ifdef CUDA_INCLUDE
        MY_CUDA_FLAGS += -I$(CUDA_INCLUDE)
    endif
endif

include ../../config/makefile.h
include ../../config/makelib.h

# Explicit rule for CUDA compilation
%.o: %.cu
	nvcc $(MY_CUDA_FLAGS) -c $< -o $@
