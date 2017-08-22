all: image-pre-processor pfm-debug exr-to-pfm pfm-to-exr

OPT=-g3 -O3 -march=native
#OPT=-g3 -O0 -DDEBUG

CC=gcc
CFLAGS=$(OPT) -Wall

CXX=g++ -std=c++14 -Wall
CXXFLAGS=$(OPT)

LDFLAGS=


pfm-debug: pfm-debug.c
	$(CC) -o $(@) $(^) $(CFLAGS) $(LDFLAGS)

image-pre-processor: image-pre-processor.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng` -lpthread

exr-to-pfm: exr-to-pfm.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

pfm-to-exr: pfm-to-exr.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR`

clean:
	rm -f image-pre-processor pfm-debug exr-to-pfm
