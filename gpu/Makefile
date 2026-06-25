NVCC      ?= nvcc
ARCH      ?= sm_120
NVCCFLAGS ?= -O3 -arch=$(ARCH)

HEADERS = field.cuh ec.cuh sha256.cuh ripemd160.cuh hash160.cuh search.cuh \
          u256.h host_hash.h

cyclone_gpu: cyclone_gpu.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ cyclone_gpu.cu

.PHONY: selftest bench clean
selftest: cyclone_gpu
	./cyclone_gpu --selftest

bench: cyclone_gpu
	./cyclone_gpu --bench 20 --grid 512,256 --slices 16

clean:
	rm -f cyclone_gpu
