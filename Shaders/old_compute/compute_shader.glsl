#version 450 core

uniform mat4 projMatrix;
uniform vec3 lightPos;

layout(local_size_x = 20, local_size_y = 20) in;

layout(rgba32f, binding = 0) uniform image2D framebuffer;

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Light {
    vec3 position;
    vec4 color;
    float intensity;
};

struct Intersection {
    vec3 hitPoint;
    vec3 normal;
};


struct Sphere {
	vec3 center;
	float radius;
	vec4 color;
};

Sphere sphere1 = {
    vec3(-0.5, 0.5, -3.5),
    1.7,
    vec4(0.4, 0.2, 0.4, 1)
};

Sphere sphere2 = {
    vec3(1., 0.5, -1.),
    0.3,
    vec4(0, 1, 0, 1)
};

Sphere sphere3 = {
    vec3(0.75, 0.5, -20),
    5,
    vec4(0, 0, 1, 1)
};

const Sphere[] spheres = {
    sphere1,
    sphere2,
    sphere3
};

Light light = {
    lightPos,
    vec4(1, 1, 1, 1),
    1.0
};

const int NUM_SPHERES = 2;


bool intersectSphere(Ray ray, Sphere s, out float t0, out float t1, inout Intersection i) { 
    vec3 l = s.center - ray.origin; 
    float tca = dot(l, ray.dir); 
    if (tca < 0) return false; 
    float d2 = dot(l, l) - tca * tca;
    float radius2 = s.radius * s.radius;
    if (d2 > radius2) return false; 
    float thc = sqrt(radius2 - d2); 
    t0 = tca - thc; 
    t1 = tca + thc;

    i.hitPoint = ray.origin + t0*ray.dir;
    i.normal = normalize(i.hitPoint - s.center);

 
    return true; 
} 


vec2 intersectSpheres(Ray ray, inout Intersection intersection) {
    float minDist = 10e30;
    int idx = -1;

    for(int i = 0; i<NUM_SPHERES; i++) {
        float t0, t1;
        bool inter = intersectSphere(ray, spheres[i], t0, t1, intersection);

        if(inter && t0 < minDist) {
            idx = i;
            minDist = t0;
        }
    }
    return vec2 (minDist, idx);
}


vec2 intersectObjects(Ray ray, inout Intersection intersection) {
    vec2 i_spheres = intersectSpheres(ray, intersection);
    return vec2(0, i_spheres.y);
}



vec4 trace(Ray ray) {
    vec4 color = vec4(0.05, 0.05, 0.05, 1);//vec4(0.97, 0.97, 1, 1);
    Intersection intersection;
    vec2 res = intersectObjects(ray, intersection);

    if(res.y >= 0) {

        Ray to_light_ray = {
            intersection.hitPoint,
            normalize(light.position - intersection.hitPoint)
        };
        Intersection intersection2;
        vec2 res2 = intersectObjects(to_light_ray, intersection2);

        if(res2.y >= 0){
            color = vec4(0, 0, 0, 1);
       } else {
            vec3 LightDir = normalize(light.position - intersection.hitPoint);
            float diff = clamp(dot(intersection.normal, LightDir), 0.1, 1.0);
	        //float diff = max(dot(intersection.normal, LightDir), 0.0);
            if(res.x == 0)
                color = diff * light.color * light.intensity * spheres[int(res.y)].color;
       }
    }
    return color;
}

void main() {



    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(framebuffer);
    vec2 pos = vec2(pix) / vec2(size.x, size.y);
    vec3 eye = vec3(0.5, 0.5, 0.5);
    vec3 dir = normalize(vec3(pos, 0.01) - eye);
    Ray ray = {eye, dir};
    vec4 color = trace(ray);
    imageStore(framebuffer, pix, color);
}