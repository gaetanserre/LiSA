#version 450 core

uniform mat4 PVMatrix;
uniform vec3 eyePos;

uniform vec3 lightPos;
uniform int seed;
uniform int nb_frames;

layout(local_size_x = 20, local_size_y = 20) in;

layout(rgba32f, binding = 0) uniform image2D framebuffer;

vec2 RandomState = gl_GlobalInvocationID.xy + vec2(seed);

float random(){
    float r = fract(sin(dot(RandomState.xy ,vec2(12.9898,78.233))) * 43758.5453);
    RandomState += vec2(r);
    return r;
}

vec3 Rand3Normal() {
    return vec3(random(), random(), random());
}

struct Material {
    vec4 color;
    float roughness;
};

Material buildMaterial(vec4 color, float roughness) {
    Material m = {
        color,
        roughness
    };
    return m;
}

struct Intersection {
    float t;
    vec3 hitPoint;
    vec3 normal;
    Material material;
    bool hit;
};

Intersection buildIntersection() {
    Intersection i = {
        -1,
        vec3(-1),
        vec3(-1),
        buildMaterial(vec4(-1), -1),
        false
    };
    return i;
}

struct Ray {
    vec3 origin;
    vec3 dir;
};


struct Light {
    vec3 position;
    float radius;
    vec4 color;
    float intensity;
    vec4 ambient_color;
    float ambient_intensity;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Plane {
    vec3 position;
    vec3 normal;
    Material material;
};

Sphere sphere1 = {
    vec3(-1.0, 0.2, -2.3),
    1,
    buildMaterial(vec4(0.4, 0.2, 0.4, 1), 0.0)
};

Sphere sphere2 = {
    vec3(1, -0.6, -2.3),
    0.7,
    buildMaterial(vec4(0.1, 0.5, 0.4, 1), 1.)
};

Sphere sphere3 = {
    vec3(0.5, 0, 2),
    0.5,
    buildMaterial(vec4(0, 0.5, 0.7, 1), 1.)
};


Sphere[] spheres = {
    sphere1,
    sphere2,
    sphere3
};

Plane plane1 = {
    vec3(0, -1.3, 0),
    vec3(0, 1, 0),
    buildMaterial(vec4(1, 0, 0, 1), 1)
};

Plane[] planes = {
    plane1
};

Light light = {
    lightPos,
    0.40,
    vec4(1, 1, 1, 1),
    1,
    vec4(1, 1, 1, 1),
    0.05
};

const int NUM_SPHERES = 3;
const int NUM_PLANES = 1;

bool intersectSphere(Ray ray, Sphere s, inout Intersection i) { 
    vec3 l = s.center - ray.origin; 
    float tca = dot(l, ray.dir); 
    if (tca < 0) return false;
    float d2 = dot(l, l) - tca * tca;
    float radius2 = s.radius * s.radius;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2); 
    float t0 = tca - thc; 
    float t1 = tca + thc;

    i.hitPoint = ray.origin + t0 * ray.dir;
    i.normal = normalize(i.hitPoint - s.center);
    i.t = t0;
    i.material = s.material;
    i.hit = true;
    
    return true; 
}

float intersectSpheres(Ray ray, inout Intersection intersection) {
    float minDist = 10e30;

    for(int i = 0; i<NUM_SPHERES; i++) {
        bool inter = intersectSphere(ray, spheres[i], intersection);

        if(inter && intersection.t < minDist) {
            minDist = intersection.t;
        }
    }
    return minDist;
}

bool intersectPlane(Ray ray, Plane plane, inout Intersection i) {
	float d = -dot(plane.position, plane.normal);
	float v = dot(ray.dir, plane.normal);
	float t = -(dot(ray.origin, plane.normal) + d) / v;

    if(t > 0.0) {
        i.hitPoint = ray.origin + t * ray.dir;
        i.normal = plane.normal;
        i.t = t;
        i.material = plane.material;
        i.hit = true;
        return true;
    }
    return false;
}

float intersectPlanes(Ray ray, inout Intersection intersection) {
    float minDist = 10e30;

    for(int i = 0; i<NUM_PLANES; i++) {
        bool inter = intersectPlane(ray, planes[i], intersection);
        if(inter && intersection.t < minDist) {
            minDist = intersection.t;
        }
    }
    return minDist;
}

bool intersectLight(Ray ray) {
    Sphere light_s = {
        light.position,
        light.radius,
        buildMaterial(light.color, 0)
    };

    Intersection _i = buildIntersection();
    return intersectSphere(ray, light_s, _i);
}

Intersection intersectObjects(Ray ray) {
    Intersection intersection_spheres = buildIntersection();
    Intersection intersection_planes = buildIntersection();

    float dist_spheres = intersectSpheres(ray, intersection_spheres);
    float dist_planes = intersectPlanes(ray, intersection_planes);

    if(dist_spheres < dist_planes)
        return intersection_spheres;
    else 
        return intersection_planes;
}

vec4 trace(inout Ray ray) {
    vec4 mask = vec4(1);
    vec4 accumulator = vec4(0);
    for(int i = 0; i<10; i++) {
        Intersection intersection = intersectObjects(ray);
        if (!intersection.hit) {
            if(intersectLight(ray)) {
                return accumulator +=  light.color * mask;
            } else {
                return accumulator += light.ambient_color * light.ambient_intensity * mask;
            }
        } else {
            mask *= intersection.material.color;
            vec3 lp = light.position + Rand3Normal() * light.radius;
            vec3 lray = normalize(lp - intersection.hitPoint);
            Ray shadow_ray = {
                intersection.hitPoint + lray * 0.0001,
                lray
            };
            
            if(intersectLight(shadow_ray) && !intersectObjects(shadow_ray).hit) {
                float d = clamp(dot(intersection.normal, shadow_ray.dir), 0.0, 1.0);
                //d *= pow(asin(light.radius / distance(shadow_ray.origin, light.position)), 2.0);
                accumulator += d * light.intensity * light.color * mask;
            }

            vec3 nray = normalize(
                mix(
                    reflect(ray.dir, intersection.normal),
                    intersection.normal + Rand3Normal(),
                    intersection.material.roughness
                    )
                );
            Ray t_ray = {
                intersection.hitPoint + 0.0001 * nray,
                nray
            };
            ray = t_ray;
        }
    }
    return accumulator;
}

void main() {

    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(framebuffer);
    vec4 dir = vec4(2 * vec2(pix) / vec2(size.x, size.y) - 1, 1, 1);
    dir = PVMatrix * dir;
    dir = normalize(dir);

    Ray ray = {eyePos, vec3(dir)};
    vec4 n_color = trace(ray) / nb_frames;
    vec4 o_color = imageLoad(framebuffer, pix);
    imageStore(framebuffer, pix, n_color + o_color);
}