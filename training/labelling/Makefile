all: train infer

CFLAGS ?= -O0 -g -Wall
LDFLAGS ?= 
CXX ?= g++

train: train.cc xalloc.cc llist.cc utils.cc
	$(CXX) -o $(@) $(^) $(CFLAGS) -std=c++11 -lrt $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

infer: infer.cc utils.cc xalloc.cc
	$(CXX) -o $(@) $(^) $(CFLAGS) $(LDFLAGS) `pkg-config --cflags --libs OpenEXR libpng`

clean:
	rm -f train infer
