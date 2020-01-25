
COMPUTER_NAME := $(shell uname -n)

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)


# operating system
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
TARGET_SIZE := 64
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -std=c++14 -O2 -g 
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS := --expt-relaxed-constexpr -w
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


# Common includes and paths for CUDA
INCLUDES   := -I../../common/inc -I include -I cupla/include -I cupla/alpaka/include
LIBRARIES  :=
CUDA_FLAGS := -x cu

CUPLA_CPUTBB_ACC := -DFOR_TBB -I /usr/include/tbb -L /usr/lib/x86_64-linux-gnu -ltbb
ifeq ($(COMPUTER_NAME), patatrack02.cern.ch)
TBB_BASE := /home/cmssw/slc7_amd64_gcc820/external/tbb/2019_U8
$(info >>> TBB_BASE=$(TBB_BASE))
CUPLA_CPUTBB_ACC := -DFOR_TBB -I $(TBB_BASE)/include -L $(TBB_BASE)/lib -ltbb
endif


CUPLA_FLAGS := -DUSE_CUPLA
CUPLA_CUDA_ACC := -DFOR_CUDA

################################################################################

# Gencode arguments
SMS ?= 60 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

################################################################################

# Target rules
all: build

build: main mainCuplaCPUTBB 
#mainCuplaCUDA mainCuplaCPUSerial 

CLUEAlgo.o:src/CLUEAlgo.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

CLUEAlgoGPU.o:src/CLUEAlgoGPU.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:src/main.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mainCuplaCUDA.o:src/main.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(CUDA_FLAGS) $(CUPLA_CUDA_ACC) $(GENCODE_FLAGS) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCPUSerial.o:src/main.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(CUDA_FLAGS) $(CUPLA_CPUSERIAL_ACC) $(GENCODE_FLAGS) $(CUPLA_FLAGS) -o $@ -c $<

mainCuplaCPUTBB.o:src/main.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(CUDA_FLAGS) $(CUPLA_CPUTBB_ACC) $(GENCODE_FLAGS) $(CUPLA_FLAGS) -o $@ -c $<

main: main.o CLUEAlgoGPU.o CLUEAlgo.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

mainCuplaCPUTBB: mainCuplaCPUTBB.o include/CLUEAlgoCupla.h CLUEAlgo.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ mainCuplaCPUTBB.o CLUEAlgo.o $(LIBRARIES) $(CUPLA_CPUTBB_ACC)

#mainCuplaCUDA: mainCuplaCUDA.o include/CLUEAlgoCupla.h CLUEAlgo.o
#	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ mainCuplaCUDA.o CLUEAlgo.o $(LIBRARIES)

#mainCuplaCPUSerial: mainCuplaCPUSerial.o include/CLUEAlgoCupla.h CLUEAlgo.o
#	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ mainCuplaCPUSerial.o CLUEAlgo.o $(LIBRARIES) $(CUPLA_CPUSERIAL_ACC)


run: build
	$(EXEC) main

clean:
	rm -f main main.o mainCupla mainCupla.o mainCuplaCUDA mainCuplaCUDA.o mainCuplaCPUSerial mainCuplaCPUSerial.o mainCuplaCPUTBB mainCuplaCPUTBB.o CLUEAlgo.o CLUEAlgoGPU.o

clobber: clean
