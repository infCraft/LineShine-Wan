CXX ?= clang++

ARCH_FLAGS := -march=armv9-a+sme+bf16+sve
COMMON_FLAGS := -O3 -std=c++20 $(ARCH_FLAGS) -mcpu=hip11 -g -pthread --rtlib=compiler-rt
PIC_FLAGS := $(COMMON_FLAGS) -fPIC
LDFLAGS_SO := -shared -pthread -L. -lnuma --rtlib=compiler-rt -Wl,-rpath,$(WAN21_BISHENG_ROOT)/lib

BUILD_DIR := build
LIB_NAME := libsme_gemm.so

DISPATCH_SRC := bgemm.cpp
HEADER := bgemm.h

DEMO1_SRC := backend_demo1.cpp
DEMO2_SRC := backend_demo2.cpp
OUTER_SRC := backend_outer.cpp
VECTOR_SRC := backend_vector.cpp

DISPATCH_OBJ := $(BUILD_DIR)/bgemm_dispatch.o
DEMO1_OBJ := $(BUILD_DIR)/bgemm_demo1_impl.o
DEMO2_OBJ := $(BUILD_DIR)/bgemm_demo2_impl.o
OUTER_OBJ := $(BUILD_DIR)/bgemm_outer_impl.o
VECTOR_OBJ := $(BUILD_DIR)/bgemm_vector_impl.o

POOL_OBJ := $(BUILD_DIR)/wan21_gemm_thread_pool.o
BACKEND_OBJS := $(DEMO1_OBJ) $(DEMO2_OBJ) $(OUTER_OBJ) $(VECTOR_OBJ) $(POOL_OBJ)

.PHONY: all lib clean

all: lib

lib: $(LIB_NAME)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DISPATCH_OBJ): $(DISPATCH_SRC) $(HEADER) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -c $< -o $@

$(DEMO1_OBJ): $(DEMO1_SRC) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=last_gemm_demo1_impl -c $< -o $@

$(DEMO2_OBJ): $(DEMO2_SRC) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=last_gemm_demo2_impl -c $< -o $@

$(OUTER_OBJ): $(OUTER_SRC) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=last_gemm_outer_impl -c $< -o $@

$(VECTOR_OBJ): $(VECTOR_SRC) | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -Dxhblas_gemm=last_gemm_vector_impl -c $< -o $@

$(POOL_OBJ): wan21_gemm_thread_pool.cpp wan21_gemm_thread_pool.h | $(BUILD_DIR)
	$(CXX) $(PIC_FLAGS) -c $< -o $@

$(LIB_NAME): $(DISPATCH_OBJ) $(BACKEND_OBJS)
	$(CXX) $(COMMON_FLAGS) $(DISPATCH_OBJ) $(BACKEND_OBJS) $(LDFLAGS_SO) -o $@

clean:
	rm -f $(LIB_NAME)
	rm -rf $(BUILD_DIR)
