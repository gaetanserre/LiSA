C++ = g++
CFLAGS = -lGLEW -lglfw -lGL
libs = ../src/libs

all : ray_tracing

ray_tracing : main.o opengl_utils.o engine.o
	$(C++) $^ -o $@ $(CFLAGS)

main.o : ../src/main.cpp
	$(C++) $^ -c $(CFLAGS)

opengl_utils.o : $(libs)/opengl_utils.cpp
	$(C++) $^ -c $(CFLAGS)

engine.o : $(libs)/engine.cpp
	$(C++) $^ -c $(CFLAGS)

clean :
	rm -rf *.o
	rm ray_tracing

run :
	./ray_tracing 500