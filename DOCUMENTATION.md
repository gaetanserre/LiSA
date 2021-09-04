## Documentation

### Detailed operations
1. The main function creates an instance of `Engine` by passing the scene file to it.
2. Then, the engine creates an instance of `SceneBuilder` which parse the scene file in order to get
every geometrical data (spheres, triangles, normals, material, camera, ...) and stores these data in attributes.
3. The `Engine` instance creates an instance of `cudaEngine` and asks for the `SceneBuilder` instance to pass all its data
to the `cudaEngine` instance. Then, this instance allocates memory on the stack of the GPU and copy the previous data inside it.
4. To finish, the `Engine` instance asks for the `cudaEngine` to run. It will create a `CudaPool` instance which will, 
for each pixel of the final image, create a thread which will compute the color of it, using all
the geometrical data previously stored inside the GPU memory.
5. Then, every pointers are freed (I hope 😅) and the final image is saved.

+ `Engine` → libs/engine.cpp headers/engine.hpp
+ `SceneBuilder` → libs/scene_builder.cpp headers/scene_builder.hpp
+ `cudaEngine` → libs/cuda/cudaEngine.cu headers/cuda/cudaEngine.hpp
+ `CudaPool` → libs/cuda/render_thread.cu headers/cuda/render_thread.hpp

### Path tracing algorithm
There are a lot of resources online to learn how path tracing works but sometimes they are to complicated for a beginner or to simple for someone who already implemented a little path tracing renderer and wants to add some specifics features.
To implement my program, I mostly used [Scratchpixel](https://www.scratchapixel.com/) guides and the excellent tutorial on [Tyro](https://wwwtyro.net/2018/02/25/caffeine.html).