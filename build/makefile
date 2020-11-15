C++ = g++
CFLAGS = -lGLEW -lglfw -lGL
libs = ../src/libs

all : LiSA

LiSA : main.o opengl_utils.o engine.o scene_builder.o parse_obj.o export_image.o
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

export_image.o : $(libs)/export_image.cpp
	$(C++) $^ -c

clean :
	rm -rf *.o
	rm LiSA
