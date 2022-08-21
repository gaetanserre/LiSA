# LiSA

LiSA is a path tracing render engine developped in C++ using NVidia Optix.

LiSA runs in multiple CUDA cores and uses the Monte-Carlo rendering technique associated with a simple Lambertian BSDF.

LiSA aims to use an unique rendering algorithm but with easely interchangable BSDF.

You can see some examples of rendered images [here](#Some-images).

## Build and usage

### Linux
You need CUDA, OpenGL, GLFW, Xinerama and Xcursor to compile this project.

For Debian distribution run: 
```bash
sudo apt install nvidia-cuda-toolkit freeglut3-dev libglfw3-dev libxinerama-dev libxcursor-dev
```

Then:
```bash
mkdir build
cd build
cmake ../src
make
cd bin
./LiSA -s scene_file.rto -d (optional, display the rendering in real-time)
```

## Scene file
You need to provide all the necessary information here such as:
- materials
- meshes
- camera
- parameters of the rendering: the number of samples,
  the maximum number of times a ray can bounce, the size of the image,
  and the path of the image (in [ppm](https://fr.wikipedia.org/wiki/Portable_pixmap) format)

The syntax is quite free: 
- material names have the same constraints as C/C++ variables
- you can skip lines or add as many spaces as you want
- any text between ```/*``` and ```*/``` is treated as a comment

### Scene file example
```
material white {
  color = (1, 1, 1, 1)
  roughness = 1
}

material transparent {
  color = (1, 1, 1, 0)
  n = 1.5
}

material red {
  color = (1, 0, 0, 1)
  roughness = 1
}

material green {
  color = (0, 1, 0, 1)
  roughness = 1
}

material light {
  emit = true
  color = (1, 1, 1, 1)
}


mesh {
  obj_file = assets/objs/cornell_box/bot.obj
  material = white
}

mesh {
  obj_file = assets/objs/cornell_box/top.obj
  material = white
}

mesh {
  obj_file = assets/objs/cornell_box/back.obj
  material = white
}

mesh {
  obj_file = assets/objs/cornell_box/right.obj
  material = green
}

mesh {
  obj_file = assets/objs/cornell_box/left.obj
  material = red
}

mesh {
  obj_file = assets/objs/cornell_box/large_box.obj
  material = white
}

mesh {
  obj_file = assets/objs/cornell_box/small_box.obj
  material = transparent
}

mesh {
  obj_file = assets/objs/cornell_box/sphere.obj
  material = white
}

mesh {
  obj_file = assets/objs/cornell_box/light.obj
  material = light
}

camera {
  position = (-0.01, 0.015, 0.6)
  look_at = (-0.01, 0.015, 0)
  fov = 40
}

num_samples = 2000
num_bounces = 7

width = 2000
height = 2000

output_image = ../../images/cornel_box.ppm
```

## How it works
The main function creates an instance of a `SceneParser` which retrieves
all the information contained in the scene file. Then, it passes
the necessary parameters (such as the vertices, the materials, the rendering algorithm parameters)
to the GPU using Optix. The image is then rendered using the algorithm in `shader.cu` and the
lambertain BSDF (`bsdfs/lambertian.cu`).

## Features
- lambertian BSDF
- FOV can be chosen
- anti aliasing
- focal plane
- only your GPU memory limits the number of triangles per scene
- transparency/Refraction
- thanks to Optix, LiSA is fast

## Limitations
There are some limitations to LiSA:
- the obj file MUST be composed of triangles
- if you have too much triangles for your GPU, LiSA wont work

## Some images
![](img/cornel_box.png)

model from https://benedikt-bitterli.me/resources/
![](img/dragon_glass.png)

model from https://benedikt-bitterli.me/resources/
![](img/dragon_glass_yellow.png)

model from https://benedikt-bitterli.me/resources/
![](img/spaceship.png)

model from https://github.com/knightcrawler25/GLSL-PathTracer
![](img/tropical_island.png)

## TODO list
- [x] anti aliasing
- [x] focal plane
- [x] optimize the computation time
- [x] add transparency and refraction
- [x] pointer to materials (avoid copy it twice)
- [x] tabulation size in scene_builder.{cc, hh}
- [x] scene file extension
- [x] run arg
- [x] save image
- [x] scene file/sample/bounce selection
- [x] print time elapsed
- [x] weird circles with Lambertian BRDF
- [ ] add support for texture
- [ ] add the Disney's BRDF
- [ ] add GGX BRDF (http://cwyman.org/code/dxrTutors/tutors/Tutor14/tutorial14.md.html)
- [ ] more example scenes
- [ ] focal plane & Aperture (https://github.com/knightcrawler25/GLSL-PathTracer/blob/master/src/core/Camera.cpp, https://github.com/knightcrawler25/GLSL-PathTracer/blob/master/src/shaders/tile.glsl)
- [ ] generalizable shader
  - [x] BDRF
  - [x] BTDF
  - [ ] more light when more triangles?
  - [ ] refractive index of "out" material

## Acknowledgments
- [GLSL Path tracer](https://github.com/knightcrawler25/GLSL-PathTracer). An amazing renderer. I took a lot of .obj for testing LiSA from here. 
- [Scratchpixel 2.0](https://www.scratchapixel.com/) for their lessons about ray tracing, monte carlo, etc..
- [Tyro](https://wwwtyro.net/2018/02/25/caffeine.html) for the excellent tutorial and explanations
## License
[GNU v3](https://choosealicense.com/licenses/gpl-3.0/)
