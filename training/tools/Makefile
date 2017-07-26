all: image-pre-processor

CXX=g++
CXXFLAGS=`pkg-config --cflags OpenEXR libpng`
LDFLAGS=`pkg-config --libs OpenEXR libpng`


image-pre-processor: image-pre-processor.cc
	$(CXX) -o $(@) $(^) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f image-pre-processor
