include $(abs_top_srcdir)/Makefrag

tests = \
	cifar_quant \
	#cifar_quant0_og \
	#cifar_quant0_rt \
	#cifar_quant1_og \
	#cifar_quant1_rt \
	#cifar_quant2_og \
	#cifar_quant2_rt \
	#cifar_quant3_og \
	#cifar_quant3_rt \
	#cifar_quant4_og \
	#cifar_quant4_rt \
	#identity \
	#identity_negative \
	#double \
	#double_keep_weights \
	#double_keep_weights_then_change \
	#random_matmuls \
	#large_matmul \
	#large_matmul_without_cpu \
	#very_large_matmul \

tests_baremetal = $(tests:=-baremetal)
ifdef BAREMETAL_ONLY
	tests_linux =
else
	tests_linux = $(tests:=-linux)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
    -L$(abs_top_srcdir)/softfloat/build/BareMetal-RISCV-GCC/ \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

LFLAGS := $(LFLAGS) \
    -lsoftfloat \

all: $(tests_baremetal) $(tests_linux)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

junk += $(tests_baremetal) $(tests_linux)

