all: image-pre-processor pfm-debug exr-to-pfm

#OPT=-g3 -O3 -march=native
OPT=-g3 -O0

CC=gcc
CFLAGS=$(OPT)

CXX=g++
CXXFLAGS=$(OPT)

LDFLAGS=


pfm-debug: pfm-debug.c
	$(CC) -o $(@) $(^) $(CFLAGS) $(LDFLAGS)

image-pre-processor: image-pre-processor.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

exr-to-pfm: exr-to-pfm.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

clean:
	rm -f image-pre-processor pfm-debug exr-to-pfm
