COMPUTER_NAME := $(shell uname -n)

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.2

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)


# operating system
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
TARGET_SIZE := 64
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -std=c++14
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
endif

NVCCFLAGS  += --expt-relaxed-constexpr -w
NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
NVCC_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES   := -I include -I cupla/include -I cupla/alpaka/include
LIBRARIES  :=
CUDA_FLAGS := -x cu

CUPLA_CPUTBB_ACC := -DFOR_TBB -I /usr/include/tbb -L /usr/lib/x86_64-linux-gnu -ltbb 
ifeq ($(COMPUTER_NAME), patatrack02.cern.ch)
TBB_BASE := /cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/tbb/2019_U8
$(info >>> TBB_BASE=$(TBB_BASE))
CUPLA_CPUTBB_ACC := -DFOR_TBB -I $(TBB_BASE)/include -L $(TBB_BASE)/lib -ltbb
endif

CUPLA_FLAGS := -DUSE_CUPLA
CUPLA_CUDA_ACC := -DFOR_CUDA -DALPAKA_ACC_GPU_CUDA_ENABLED=1

################################################################################

# Gencode arguments
SMS ?= 60 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

################################################################################

# Target rules
all: build


# TARGET EXPLANATION
#
# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.
#
# mainCuplaCPUTBB: this will build the native C++ implementation of CLUE and
#       its corresponding TBB one  built using CUPLA. Which one to use must be
#       selected at runtime via a flag (misleadingly enough called useGPU in
#       the help text).
#
# mainCuplaCUDA: this will build the native C++ implementation of CLUE and its
#       corresponding CUDA one built using CUPLA. Which one to use must be
#       selected at runtime via a flag.

build: main mainCuplaCPUTBB mainCuplaCUDA

# Native C++ implementation bundled with native CUDA
CLUEAlgo.cuda.o:src/CLUEAlgoGPU.cu include/CLUEAlgoGPU.h
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

CLUEAlgo.o:src/CLUEAlgo.cc include/CLUEAlgo.h
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:src/main.cc include/CLUEAlgo.h include/CLUEAlgoGPU.h
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(CUDA_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main: main.o CLUEAlgo.cuda.o CLUEAlgo.o
	$(EXEC) $(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIBRARIES)

# Native C++ implementation bundled TBB via CUPLA
CLUEAlgo.tbb.o:src/CLUEAlgo.cc include/CLUEAlgo.h
	$(EXEC) $(HOST_COMPILER) $(INCLUDES) -g $(CUPLA_CPUTBB_ACC) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCPUTBB.o:src/main.cc include/CLUEAlgoCupla.h include/GPUVecArrayCupla.h
	$(EXEC) $(HOST_COMPILER) $(INCLUDES) -g $(CUPLA_CPUTBB_ACC) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCPUTBB: mainCuplaCPUTBB.o CLUEAlgo.tbb.o
	$(EXEC) $(HOST_COMPILER) -o $@ $^ $(LIBRARIES) $(CUPLA_CPUTBB_ACC)


# Native C++ implementation bundled with CUDA via CUPLA
CLUEAlgo.gpu.o:src/CLUEAlgo.cc include/CLUEAlgo.h
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCUDA.o:src/main.cc include/CLUEAlgoCupla.h include/GPUVecArrayCupla.h
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(CUDA_FLAGS) $(CUPLA_CUDA_ACC) $(GENCODE_FLAGS) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCUDA:mainCuplaCUDA.o CLUEAlgo.gpu.o
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCCFLAGS) $(CUPLA_CUDA_ACC) $(GENCODE_FLAGS) $(CUPLA_FLAGS) -o $@ $^

run: build
	$(EXEC) main

clean:
	rm -f main mainCuplaCPUTBB mainCuplaCUDA *.o

clobber: clean
