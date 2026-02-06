NVCC ?= nvcc
ARCH ?= -arch=sm_70  # change to meet the requirements of the GPU you have
CXXFLAGS ?= -O3 -std=c++17
INCLUDES := -I./src
LIBS := -lcublas

SRC := $(wildcard src/*.cu)
BIN_DIR := bin
TARGET := $(BIN_DIR)/dmlp

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) $(ARCH) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean

