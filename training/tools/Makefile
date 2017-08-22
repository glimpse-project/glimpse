all: image-pre-processor pfm-debug exr-to-pfm pfm-to-exr

#OPT ?= -g3 -O3 -march=native
OPT ?= -g3 -O0 -DDEBUG

CC ?= gcc
CFLAGS ?= $(OPT)
CFLAGS += -Wall
CXX ?= g++
CXXFLAGS ?= $(OPT)
CXXFLAGS += -std=c++14 -Wall

LDFLAGS=

image-pre-processor: image-pre-processor.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng` -lpthread

pfm-debug: pfm-debug.c
	$(CC) -o $(@) $(^) $(CFLAGS) $(LDFLAGS)

exr-to-pfm: exr-to-pfm.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

pfm-to-exr: pfm-to-exr.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR`

clean:
	rm -f image-pre-processor pfm-debug exr-to-pfm pfm-to-exr
