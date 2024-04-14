CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkan: main.cpp
	g++ $(CFLAGS) -o vulkan main.cpp $(LDFLAGS)

.PHONY: test clean

test: vulkan
	./vulkan

clean:
	rm -f ./vulkan
