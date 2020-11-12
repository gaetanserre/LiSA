C++ = g++
CFLAGS = -lGLEW -lglfw -lGL
libs = ../src/libs

all : ray_tracing

ray_tracing : main.o opengl_utils.o engine.o scene_builder.o parse_obj.o
	$(C++) $^ -o $@ $(CFLAGS)

main.o : ../src/main.cpp
	$(C++) $^ -c

opengl_utils.o : $(libs)/opengl_utils.cpp
	$(C++) $^ -c

engine.o : $(libs)/engine.cpp
	$(C++) $^ -c

scene_builder.o : $(libs)/scene_builder.cpp
	$(C++) $^ -c

parse_obj.o : $(libs)/parse_obj.cpp
	$(C++) $^ -c


clean :
	rm -rf *.o
	rm ray_tracing

run :
	./ray_tracing ../scene.rto 500