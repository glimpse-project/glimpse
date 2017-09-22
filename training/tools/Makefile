all: pack-training-data image-pre-processor pfm-debug exr-to-pfm pfm-to-exr

#OPT ?= -g3 -O3 -march=native
OPT ?= -g3 -O0 -DDEBUG

CC ?= gcc
CFLAGS ?= $(OPT)
CFLAGS += -Wall
CXX ?= g++
CXXFLAGS ?= $(OPT)
CXXFLAGS += -std=c++14 -Wall

LDFLAGS=

pack-training-data: pack-training-data.cc llist.cc xalloc.cc pack.c
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng` -lsnappy -lpthread

image-pre-processor: image-pre-processor.cc parson.c
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng` -lpthread

pfm-debug: pfm-debug.c
	$(CC) -o $(@) $(^) $(CFLAGS) $(LDFLAGS)

exr-to-pfm: exr-to-pfm.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng`

pfm-to-exr: pfm-to-exr.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f pack-training-data image-pre-processor pfm-debug exr-to-pfm pfm-to-exr
