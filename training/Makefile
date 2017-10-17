E=$(if $(V),,@)

ifeq ($(RELEASE), 1)
OPT ?= -g3 -O3 -march=native -mtune=native
else
OPT ?= -g3 -O0 -DDEBUG
endif

CC ?= gcc
CFLAGS ?= $(OPT)
CXX ?= g++
CXXFLAGS ?= $(OPT)

LDFLAGS ?=

_CFLAGS = $(CFLAGS) -Wall -std=c11
_CXXFLAGS = $(CXXFLAGS) -std=c++14 -Wall


INCLUDES = `pkg-config --cflags OpenEXR libpng`

TOOLS = image-pre-processor \
	train_rdt \
	train_joint_params \
	depth2labels \
	exr-to-pfm \
	pfm-to-exr \
	rdt-to-json \
	pfm-debug \
	pack-training-data

LIBS = libglimpse.so

all: $(TOOLS) $(LIBS)

image-pre-processor: image-pre-processor.o tinyexr.o parson.o image_utils.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --libs libpng` -lpthread

train_rdt: train_rdt.o image_utils.o train_utils.o tinyexr.o parson.o llist.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(LDFLAGS) `pkg-config --libs libpng` -lpthread

train_joint_params: train_joint_params.o infer.o image_utils.o train_utils.o loader.o tinyexr.o parson.o llist.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(LDFLAGS) `pkg-config --libs libpng` -lpthread

depth2labels: depth2labels.o infer.o loader.o xalloc.o llist.o tinyexr.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --libs OpenEXR libpng`

exr-to-pfm: exr-to-pfm.o tinyexr.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --libs libpng`

pfm-to-exr: pfm-to-exr.o tinyexr.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS)

rdt-to-json: rdt-to-json.o loader.o parson.o llist.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS)

pfm-debug: pfm-debug.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CC) -o $(@) $(^) $(_CFLAGS) $(LDFLAGS)

pack-training-data: pack-training-data.o pack.o tinyexr.o llist.o xalloc.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --libs libpng` -lsnappy -lpthread -lc

libglimpse.so: xalloc.o tinyexr.o llist.o loader.o infer.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -shared -o $(@) $(^) $(_CXXFLAGS) $(LDFLAGS) `pkg-config --libs libpng`


%.o: %.c Makefile | dirs
	$(E)echo "(CC)      $(@)"
	$(E)$(CC) $(filter %.c,$(^)) -o $@ -MMD -MF .deps/$(@).rules $(_CFLAGS) $(INCLUDES) -c -fPIC -std=gnu99

%.o: %.cc Makefile | dirs
	$(E)echo "(CXX)     $(@)"
	$(E)$(CXX) $(filter %.cc,$(^)) -o $@ -MMD -MF .deps/$(@).rules $(_CXXFLAGS) $(INCLUDES) -c -fPIC

-include .deps/*.rules

dirs:
	$(E)mkdir -p .deps

clean:
	-rm -f $(TOOLS)
	-rm -f *.o
	-rm -fr .deps

.PHONY: all clean dirs
