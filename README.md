# LiSA

LiSA is a path tracing render engine developped in C++ using NVidia Optix.

LiSA runs in multiple CUDA cores and uses the Monte-Carlo rendering technique associated with a simple Lambertian BSDF.

You can see some examples of rendered images [here](#Some-images).

## Build and usage

### Linux
You need CUDA, OpenGL, GLFW, Xinerama and Xcursor to compile this project.

For Ubuntu or Debian distribution run: 
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

## How it works
- The number of passes corresponds to the number of times LiSA will compute the color of each pixel to average it.
- The output file is a [ppm](https://fr.wikipedia.org/wiki/Portable_pixmap) image.
- The number of samples correponds to the number of times a ray can bounce. Default is 3.
- You have to provide a file describing the scene you want LiSA to render. You can find some examples [below](#Scene-file-example) or in the folder ````assets````.
- In the scene file you need one camera, atleast one material and output dimension. See [Limitations](#Limitations) to learn more.
- For more detailed information see [DOCUMENTATION.md](DOCUMENTATION.md)

## Scene file
You need to provide all the necessary information here such as:
- The materials
- The meshes
- The camera
- Parameters of the rendering: the number of samples,
  the maximum number of times a ray can bounce, the size of the image,
  and the path of the image

The syntax is quite free: 
- Material names have the same constraints as C/C++ variables.
- You can skip lines or add as many spaces as you want.
- Any text between ```/*``` and ```*/``` is treated as a comment

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

## Features
- Lambertian BRDF.
- FOV can be chosen.
- Anti aliasing.
- Focal plane
- Only your GPU memory limits the number of triangles per scene.
- Transparency/Refraction
- Thanks to Optix, LiSA is fast

## Limitations
There are some limitations to LiSA:
- The obj file HAS to be composed of triangles
- If you have too much triangles for your GPU, LiSA wont work

## Some images
In order to use these scenes files, you to be inside the `img/` folder.


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
- [x] Anti aliasing.
- [x] Focal plane
- [x] Optimize the computation time.
- [x] Add transparency and refraction.
- [x] Pointer to Materials (avoid copy it twice)
- [x] Tabulation size in scene_builder.{cc, hh}
- [x] Scene file extension
- [x] Run arg
- [x] Save image
- [x] Scene file/sample/bounce selection
- [x] Print time elapsed
- [x] Weird circles with Lambertian BRDF
- [ ] Add support for texture.
- [ ] Add the Disney's BRDF
- [ ] More example scenes
- [ ] Focal plane & Aperture (https://github.com/knightcrawler25/GLSL-PathTracer/blob/master/src/core/Camera.cpp, https://github.com/knightcrawler25/GLSL-PathTracer/blob/master/src/shaders/tile.glsl)
- [ ] Generalizable shader
  - [x] BDRF
  - [x] BTDF
  - [ ] More light when more triangles?
  - [ ] Refractive index of "out" material

## Credits
- [GLSL Path tracer](https://github.com/knightcrawler25/GLSL-PathTracer). An amazing renderer. I took a lot of .obj for testing LiSA from here. 
- [Scratchpixel 2.0](https://www.scratchapixel.com/) for their lessons about ray tracing, monte carlo, etc..
- [Tyro](https://wwwtyro.net/2018/02/25/caffeine.html) for the excellent tutorial and explanations
## License
[GNU v3](https://choosealicense.com/licenses/gpl-3.0/)
