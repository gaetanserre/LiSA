#version 450 core

uniform mat4 PVMatrix;
uniform vec3 eyePos;

uniform int seed;
uniform int nb_frames;


uniform vec4 spheres[100];
uniform vec4 materials[100];
uniform int isLight;
uniform int NUM_SPHERES;

uniform samplerBuffer vertices_normals;
uniform int NUM_VERTICES;


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
    vec3 color;
    float roughness;
    bool emit;
    float emit_intensity;
};

Material buildMaterial(vec3 color, float roughness) {
    Material m = {
        color,
        roughness,
        false,
        0
    };
    return m;
}

Material buildLight(vec3 emit_color, float intensity) {
    Material m = {
        emit_color,
        0,
        true,
        intensity
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
        buildMaterial(vec3(-1), -1),
        false
    };
    return i;
}

struct Ray {
    vec3 origin;
    vec3 dir;
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

struct Triangle {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 n1;
    vec3 n2;
    vec3 n3;
    Material material;
};

Plane plane1 = {
    vec3(0, -0.1, 0),
    vec3(0, 1, 0),
    buildMaterial(vec3(0.33, 0.24, 0.18), 0.07)
};

Plane[] planes = {
    plane1
};

const vec3 ambient_color = vec3(1, 1, 1);
const float ambient_intensity = 0.05;
const int NUM_PLANES = 1;

Material transformMaterial(int idx) {
    if (isLight == idx)
        return buildLight(materials[idx].xyz, materials[idx].w);
    else
        return buildMaterial(materials[idx].xyz, materials[idx].w);
}

Sphere transformSpheres(int idx) {
    Material m = transformMaterial(idx);
    Sphere sphere = {
        spheres[idx].xyz,
        spheres[idx].w,
        m
    };
    return sphere;
}


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
    Intersection temp = buildIntersection();

    for(int i = 0; i<NUM_SPHERES; i++) {
        bool inter = intersectSphere(ray, transformSpheres(i), temp);

        if(inter && temp.t < minDist) {
            minDist = temp.t;
            intersection = temp;
        }
        temp = buildIntersection();
    }
    return minDist;
}


Triangle transformTriangle(int idx) {
    Material m = transformMaterial(NUM_SPHERES);
    vec3 p1 = vec3(texelFetch(vertices_normals, idx).x, texelFetch(vertices_normals, idx).y, texelFetch(vertices_normals, idx).z);
    vec3 p2 = vec3(texelFetch(vertices_normals, idx+1).x, texelFetch(vertices_normals, idx+1).y, texelFetch(vertices_normals, idx+1).z);
    vec3 p3 =  vec3(texelFetch(vertices_normals, idx+2).x, texelFetch(vertices_normals, idx+2).y, texelFetch(vertices_normals, idx+2).z);
    vec3 n1 = vec3(
        texelFetch(vertices_normals, NUM_VERTICES + idx).x,
        texelFetch(vertices_normals, NUM_VERTICES + idx).y,
        texelFetch(vertices_normals, NUM_VERTICES + idx).z
    );
    vec3 n2 = vec3(
        texelFetch(vertices_normals, NUM_VERTICES + idx+1).x,
        texelFetch(vertices_normals, NUM_VERTICES + idx+1).y,
        texelFetch(vertices_normals, NUM_VERTICES + idx+1).z
    );
    vec3 n3 = vec3(
        texelFetch(vertices_normals, NUM_VERTICES + idx+2).x,
        texelFetch(vertices_normals, NUM_VERTICES + idx+2).y,
        texelFetch(vertices_normals, NUM_VERTICES + idx+2).z
    );

    Triangle t = {
        p1,
        p2,
        p3,
        n1,
        n2,
        n3,
        m
    };
    return t;
}


bool intersectTriangle(Ray ray, Triangle triangle, inout Intersection i) {
    const float EPSILON = 0.0000001;
    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;
    vec3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = p2 - p1;
    edge2 = p3 - p1;
    h = cross(ray.dir, edge2);
    a = dot(edge1, h);

    if (a > - EPSILON && a < EPSILON) return false; //Rayon parallèle

    f = 1.0/a;
    s = ray.origin - p1;
    u = f * (dot(s, h));

    if (u < 0.0 || u > 1.0) return false;
    q = cross(s, edge1);
    v = f * dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) return false;

    float t = f * dot(edge2, q);
    if (t > EPSILON) {


        i.hitPoint = ray.origin + t * ray.dir;

        /***** Barycentric coordinates *****/
        vec3 N = cross(edge1, edge2);
        float denom = dot(N,N);
        vec3 C = cross(edge1, i.hitPoint - triangle.p2);
        float u = dot(N, C);
        C = cross(edge2, i.hitPoint - triangle.p3);
        float v = dot(N, C);

        u /= denom;
        v /= denom;

        vec3 normalHit = u*triangle.n2 + v * triangle.n3 + (1 - u - v)*triangle.n1;

        i.normal = normalHit;
        i.t = t;
        i.material = triangle.material;
        i.hit = true;

        return true;

    } else return false;
}

float intersectTriangles(Ray ray, inout Intersection intersection) {
    float minDist = 10e30;
    Intersection temp = buildIntersection();

    for(int i = 0; i<NUM_VERTICES; i+=3) {
        bool inter = intersectTriangle(ray, transformTriangle(i), temp);
        if(inter && temp.t < minDist) {
            minDist = temp.t;
            intersection = temp;
        }
        temp = buildIntersection();
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
    Intersection temp = buildIntersection();

    for(int i = 0; i<NUM_PLANES; i++) {
        bool inter = intersectPlane(ray, planes[i], temp);
        if(inter && temp.t < minDist) {
            minDist = temp.t;
            intersection = temp;
        }
        temp = buildIntersection();
    }
    return minDist;
}


Intersection intersectObjects(Ray ray) {
    Intersection intersection_spheres = buildIntersection();
    Intersection intersection_planes = buildIntersection();
    Intersection intersection_meshes = buildIntersection();

    float dist_spheres = intersectSpheres(ray, intersection_spheres);
    float dist_planes = intersectPlanes(ray, intersection_planes);
    float dist_meshes = intersectTriangles(ray, intersection_meshes);

    float m = min(dist_spheres, min(dist_planes, dist_meshes));

    if (m == dist_spheres) return intersection_spheres;
    if (m == dist_planes) return intersection_planes;
    if (m == dist_meshes) return intersection_meshes;
}

vec3 trace(Ray ray) {
    vec3 mask = vec3(1);
    vec3 accumulator = vec3(0);
    for(int i = 0; i<30; i++) {
        Intersection intersection = intersectObjects(ray);

        if(!intersection.hit) {
            return accumulator += ambient_color * ambient_intensity * mask;
        } else {
            if (intersection.material.emit) {
                return accumulator += intersection.material.color * intersection.material.emit_intensity * mask;
            } else {
                mask *= intersection.material.color;

                vec3 lp = spheres[isLight].xyz + Rand3Normal() * spheres[isLight].w;
                vec3 lray = normalize(lp - intersection.hitPoint);
                Ray shadow_ray = {
                    intersection.hitPoint + 0.0001 * lray,
                    lray
                };

                Intersection temp = intersectObjects(shadow_ray);

                if(temp.material.emit) {
                    float d = clamp(dot(intersection.normal, shadow_ray.dir), 0.0, 1.0);
                    //d *= pow(asin(light.radius / distance(shadow_ray.origin, light.position)), 2.0);
                    accumulator += d * temp.material.emit_intensity * temp.material.color * mask;
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
    vec3 n_color = trace(ray) / nb_frames;
    
    vec4 o_color = imageLoad(framebuffer, pix);
    imageStore(framebuffer, pix, vec4(n_color, 1) + o_color);
}
