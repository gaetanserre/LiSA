CUDA = nvcc
libs = ../src/libs

all : LiSA

LiSA : main.o engine.o scene_builder.o parse_obj.o export_image.o cudaEngine.o 3Dstructs.o render_thread.o
	$(CUDA) $^ -o $@

main.o : ../src/main.cpp
	$(CUDA) -dc $^
	
engine.o : $(libs)/engine.cpp
	$(CUDA) -dc $^

scene_builder.o : $(libs)/scene_builder.cpp
	$(CUDA) -dc $^

parse_obj.o : $(libs)/parse_obj.cpp
	$(CUDA) -dc $^

export_image.o : $(libs)/export_image.cpp
	$(CUDA) -dc $^

cudaEngine.o : $(libs)/cuda/cudaEngine.cu
	$(CUDA) -dc $^

3Dstructs.o : $(libs)/cuda/3Dstructs.cu
	$(CUDA) -dc $^

render_thread.o : $(libs)/cuda/render_thread.cu
	$(CUDA) -dc $^

clean :
	rm -rf *.o
	rm LiSA
