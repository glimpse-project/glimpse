all: train_rdt train_joint_params depth2labels

CFLAGS ?= -O0 -g -Wall
LDFLAGS ?= 
CXX ?= g++

train_rdt: train_rdt.cc xalloc.cc llist.cc utils.h train_utils.cc
	$(CXX) -o $(@) $(^) $(CFLAGS) -std=c++11 -lrt $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

train_joint_params: train_joint_params.cc infer.cc xalloc.cc llist.cc utils.h train_utils.cc loader.cc
	$(CXX) -o $(@) $(^) $(CFLAGS) -std=c++11 -lrt $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

depth2labels: depth2labels.cc infer.cc xalloc.cc utils.h loader.cc
	$(CXX) -o $(@) $(^) $(CFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

clean:
	rm -f train_rdt train_joint_params depth2labels
