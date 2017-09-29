ifndef OPT
ifeq ($(RELEASE), 1)
OPT = -g3 -O3 -march=native -mtune=native
else
OPT = -g3 -O0 -DDEBUG
endif
endif

CC ?= gcc
CFLAGS ?= $(OPT)
CXX ?= g++
CXXFLAGS ?= $(OPT)

LDFLAGS ?=

_CFLAGS = $(CFLAGS) -Wall
_CXXFLAGS = $(CXXFLAGS) -std=c++14 -Wall

TOOLS = image-pre-processor \
	train_rdt \
	train_joint_params \
	depth2labels \
	exr-to-pfm \
	pfm-to-exr \
	pfm-debug \
	pack-training-data

all: $(TOOLS)

image-pre-processor: image-pre-processor.cc parson.c half.hpp
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng` -lpthread

train_rdt: train_rdt.cc xalloc.cc llist.cc utils.h train_utils.cc
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

train_joint_params: train_joint_params.cc infer.cc xalloc.cc llist.cc utils.h train_utils.cc loader.cc
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

depth2labels: depth2labels.cc infer.cc xalloc.cc utils.h loader.cc loader.h
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

exr-to-pfm: exr-to-pfm.cc half.hpp
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng`

pfm-to-exr: pfm-to-exr.cc half.hpp
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS)

pfm-debug: pfm-debug.c
	$(CC) -o $(@) $(^) $(_CFLAGS) $(LDFLAGS)

pack-training-data: pack-training-data.cc llist.cc xalloc.cc pack.c half.hpp
	$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --cflags --libs libpng` -lsnappy -lpthread

clean:
	-rm -f $(TOOLS)
