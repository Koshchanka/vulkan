CFLAGS = -std=c++17 -O2
CRFLAGS = -DNDEBUG
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkan: main.cpp
	g++ $(CFLAGS) -o vulkan main.cpp $(LDFLAGS)

rvulkan: main.cpp
	g++ $(CFLAGS) $(CRFLAGS) -o rvulkan main.cpp $(LDFLAGS)

vert.spv: vert.glsl
	glslc vert.glsl -o vert.spv -O

flag.spv: flag.glsl
	glslc flag.glsl -o flag.spv -O

shaders: vert.spv frag.spv

.PHONY: rtest test clean

test: vulkan shaders
	./vulkan

rtest: rvulkan shaders
	./rvulkan

clean:
	rm -f ./vulkan ./rvulkan ./vert.spv ./frag.spv
