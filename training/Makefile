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

TOOLS = image-pre-processor \
	train_rdt \
	train_joint_params \
	depth2labels \
	exr-to-pfm \
	pfm-to-exr \
	rdt-to-json \
	json-to-rdt \
	jip-to-json \
	pfm-debug \
	pack-training-data \
	glimpse_viewer

LIBS = libglimpse.so

IMGUI_OBJS=\
	imgui/imgui.o \
	imgui/imgui_demo.o \
	imgui/imgui_draw.o \
	imgui/imgui_impl_glfw_gles3.o

THIRD_PARTY_DEPS ?= ../third_party_deps
THIRD_PARTY_INCLUDE ?= $(THIRD_PARTY_DEPS)/include
THIRD_PARTY_LIBS ?= $(THIRD_PARTY_DEPS)/lib

_CFLAGS = $(CFLAGS) -Wall -std=gnu99 -fPIC
_CXXFLAGS = $(CXXFLAGS) -std=c++14 -Wall -fPIC
_LDFLAGS = $(LDFLAGS) -L$(THIRD_PARTY_LIBS)

_CPPFLAGS = \
	-I. \
	-Iandroid_stub/ \
	-Iimgui \
	-DGLM_ENABLE_EXPERIMENTAL=1 \
	-I$(THIRD_PARTY_INCLUDE) \
	`pkg-config --cflags libpng libfreenect`


all: $(TOOLS) $(LIBS)

image-pre-processor: image-pre-processor.o tinyexr.o parson.o image_utils.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) `pkg-config --libs libpng` -lpthread

train_rdt: train_rdt.o image_utils.o train_utils.o tinyexr.o parson.o llist.o xalloc.o loader.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(_LDFLAGS) `pkg-config --libs libpng` -lpthread

train_joint_params: train_joint_params.o infer.o parson.o image_utils.o train_utils.o loader.o tinyexr.o parson.o llist.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) -lrt $(_LDFLAGS) `pkg-config --libs libpng` -lpthread

depth2labels: depth2labels.o infer.o parson.o loader.o xalloc.o llist.o tinyexr.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) `pkg-config --libs libpng`

exr-to-pfm: exr-to-pfm.o tinyexr.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) `pkg-config --libs libpng`

pfm-to-exr: pfm-to-exr.o tinyexr.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS)

rdt-to-json: rdt-to-json.o loader.o parson.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS)

json-to-rdt: json-to-rdt.o loader.o parson.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS)

jip-to-json: jip-to-json.o loader.o parson.o llist.o xalloc.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS)

pfm-debug: pfm-debug.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CC) -o $(@) $(^) $(_CFLAGS) $(_LDFLAGS)

pack-training-data: pack-training-data.o pack.o tinyexr.o llist.o xalloc.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) `pkg-config --libs libpng` -lsnappy -lpthread -lc

libglimpse.so: xalloc.o tinyexr.o llist.o loader.o infer.o parson.o image_utils.o
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -shared -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) `pkg-config --libs libpng`

glimpse_viewer: glimpse_viewer.o glimpse_context.o infer.o parson.o loader.o parson.o xalloc.o image_utils.o tinyexr.o $(IMGUI_OBJS)
	$(E)echo "(LD)      $(@)"
	$(E)$(CXX) -o $(@) $(^) $(_CXXFLAGS) $(_LDFLAGS) -lpthread -lpcl_common -lpcl_features -lpcl_filters -lpcl_io_ply -lpcl_io -lpcl_kdtree -lpcl_keypoints -lpcl_ml -lpcl_octree -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_stereo -lpcl_surface -lpcl_tracking -ldlib -lopencv_core -lopencv_objdetect -lflann_cpp -lpng -lboost_date_time -lboost_iostreams -lboost_filesystem -lboost_wserialization -lboost_serialization -lboost_thread -lboost_system -llapack -lcblas -lusb-1.0 `pkg-config --libs libfreenect libpng glfw3 epoxy`

%.o: %.c Makefile | dirs
	$(E)echo "(CC)      $(@)"
	$(E)$(CC) $(filter %.c,$(^)) -o $@ -MMD -MF .deps/$(@).rules $(_CFLAGS) $(_CPPFLAGS) -c

%.o: %.cc Makefile | dirs
	$(E)echo "(CXX)     $(@)"
	$(E)$(CXX) $(filter %.cc,$(^)) -o $@ -MMD -MF .deps/$(@).rules $(_CXXFLAGS) $(_CPPFLAGS) -c

%.o: %.cpp Makefile | dirs
	$(E)echo "(CXX)     $(@)"
	$(E)$(CXX) $(filter %.cpp,$(^)) -o $@ -MMD -MF .deps/$(@).rules $(_CXXFLAGS) $(_CPPFLAGS) -c

-include .deps/*.rules

dirs:
	$(E)mkdir -p .deps
	$(E)mkdir -p .deps/imgui

clean:
	-rm -f $(TOOLS)
	-rm -f *.o
	-rm -fr .deps

.PHONY: all clean dirs
