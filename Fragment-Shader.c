#version 330 core



#ifdef GL_ES
precision mediump float;
#endif
out vec4 fragColor;

// Function to create a pseudo-random gradient vector
vec2 randomGradient(vec2 coord) {
    float random = fract(sin(dot(coord, vec2(127.1, 311.7))) * 43758.5453);
    float angle = random * 2.0 * 3.14159265359; // Random angle in radians
    return vec2(cos(angle), sin(angle));
}

// Fade function for smoothing
float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Linear interpolation
float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Perlin noise function
float perlinNoise(vec2 uv) {
    // Grid cell coordinates
    vec2 p0 = floor(uv);
    vec2 p1 = p0 + vec2(1.0, 0.0);
    vec2 p2 = p0 + vec2(0.0, 1.0);
    vec2 p3 = p0 + vec2(1.0, 1.0);

    // Local coordinates within the cell
    vec2 local = fract(uv);

    // Fade curves for interpolation
    vec2 fadeVals = vec2(fade(local.x), fade(local.y));

    // Compute dot products between gradient vectors and local position vectors
    float d0 = dot(randomGradient(p0), local - vec2(0.0, 0.0));
    float d1 = dot(randomGradient(p1), local - vec2(1.0, 0.0));
    float d2 = dot(randomGradient(p2), local - vec2(0.0, 1.0));
    float d3 = dot(randomGradient(p3), local - vec2(1.0, 1.0));

    // Interpolate the results along x and y axes
    float xInterp1 = lerp(d0, d1, fadeVals.x);
    float xInterp2 = lerp(d2, d3, fadeVals.x);
    return lerp(xInterp1, xInterp2, fadeVals.y);
}


uniform vec2 resolution;
uniform vec3 camera_pos;
uniform vec3 camera_dir;

uniform vec4 metaballs[64];
uniform int metaballcount;

uniform float time;

bool isObjectAt(vec3 point) {
    //if (point.y < -2){
    //    return true;

    //}
    float TotalDistance = 0;
    for (int i = 0; i < metaballcount; i++) {
        vec3 center = metaballs[i].xyz;
        float radius = metaballs[i].w;
        if (distance(point,center) > 0){
            TotalDistance += radius/(distance(point,center));
        }
        if (distance(point, center) <= radius) {
            //return true;
        }
    if (TotalDistance > metaballcount){
    return true;
    }
    }
    return false;
    
}

vec3 calculateNormal(vec3 point) {
    float eps = 0.01; // Small offset for finite difference calculation
    float value = 0.0;

    int closestIndex = 0;
    float closestDistance = distance(point,metaballs[closestIndex].xyz);

    // Evaluate the scalar field at the given point
    for (int i = 0; i < metaballcount; i++) {
        vec3 center = metaballs[i].xyz;
        float radius = metaballs[i].w;
        float dist = distance(point, center);
        if (dist < closestDistance){
            closestIndex = i;
            closestDistance = distance(point,metaballs[closestIndex].xyz);


        }
        if (dist > 0.0) {
            value += radius / dist;
        }
    }

    // Approximate the partial derivatives using finite differences
    float dx = 0.0, dy = 0.0, dz = 0.0;

    for (int i = 0; i < metaballcount; i++) {
        vec3 center = metaballs[i].xyz;
        float radius = metaballs[i].w;

        dx += (radius / distance(point + vec3(eps, 0.0, 0.0), center)) - (radius / distance(point - vec3(eps, 0.0, 0.0), center));
        dy += (radius / distance(point + vec3(0.0, eps, 0.0), center)) - (radius / distance(point - vec3(0.0, eps, 0.0), center));
        dz += (radius / distance(point + vec3(0.0, 0.0, eps), center)) - (radius / distance(point - vec3(0.0, 0.0, eps), center));
    }
    vec3 center = metaballs[closestIndex].xyz;
    float radius = metaballs[closestIndex].w;
    dx = (radius / distance(point + vec3(eps, 0.0, 0.0), center)) - (radius / distance(point - vec3(eps, 0.0, 0.0), center));
    dy = (radius / distance(point + vec3(0.0, eps, 0.0), center)) - (radius / distance(point - vec3(0.0, eps, 0.0), center));
    dz = (radius / distance(point + vec3(0.0, 0.0, eps), center)) - (radius / distance(point - vec3(0.0, 0.0, eps), center));

    // Combine the partial derivatives into the gradient
    vec3 gradient = vec3(dx, dy, dz) / (2.0 * eps);
    //vec3 gradient = vec3(dx, dy, dz);
    return normalize(gradient); // Normalize to get the unit normal
}


vec3 RotateOnX(vec3 Point,float angle){
    float rotatedY = (sin(angle) * Point.y)-(cos(angle) * Point.z);
    float rotatedZ = (cos(angle) * Point.y)+(sin(angle) * Point.z);
    return vec3(Point.x,rotatedY,rotatedZ);
}
vec3 RotateOnY(vec3 Point,float angle){
    float rotatedX = (sin(angle) * Point.x)-(cos(angle) * Point.z);
    float rotatedZ = (cos(angle) * Point.x)+(sin(angle) * Point.z);
    return vec3(rotatedX,Point.y,rotatedZ);
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution * 2.0 - 1.0;
    uv.y *= resolution.y / resolution.x;

    //uv.x += perlinNoise(uv);

    vec3 ray_origin = camera_pos;
    
    vec3 ray_dir = normalize(vec3(uv.x, uv.y, -1.0)); // Ray direction in view space
    //ray_dir.x += distance(randomGradient(uv+(1/time)),vec2(0,0));
    ray_dir = normalize(ray_dir);
    //ray_dir = vec3()r;
    float Cam_XAngle = camera_dir.x;
    float Cam_YAngle = camera_dir.y;
    ray_dir = RotateOnX(ray_dir,Cam_XAngle);
    ray_dir = RotateOnY(ray_dir,Cam_YAngle);
   // ray_dir = vec3((sin(Cam_XAngle)*uv.x)-(cos(Cam_XAngle)*uv.z),uv.y,(cos(Cam_XAngle)*uv.x)+(sin(Cam_XAngle)*uv.z));
    ray_dir = normalize(ray_dir);

    float max_distance = 50.0;
    float step_size = 0.1;
    vec3 current_pos = ray_origin;
    bool hit = false;
    //hit = true;
    float b = 0;
    vec3 normal = vec3(0,1,0);
    for (float t = 0.0; t < max_distance; t += step_size) {
        current_pos += ray_dir * step_size;
        b += 1/(max_distance+step_size);
        if (distance(vec3(3,2,-5),current_pos) < 1){
            hit = true;
            b = 0;
            ray_dir = vec3(0,0,0);
            break;
        }
        if (current_pos.y < -2){
            
            hit=true;
            normal = vec3(0,1,0);
            vec3 I = (ray_dir/normalize(ray_dir));
            vec3 N = (normal/normalize(normal));
            ray_dir = vec3(ray_dir.x,abs(ray_dir.y),ray_dir.z);
            //ray_dir = I - 2 * dot(I,N) * N;
            //b = 0;
            //b = 0;
            //break;


        }
        if (isObjectAt(current_pos)) {
            //b = 0;
            hit = true;
            //t++;
            //current_pos += ray_dir;
            normal = calculateNormal(current_pos);
            //normal = vec3(0,1,0);
            vec3 I = (ray_dir/normalize(ray_dir));
            vec3 N = (normal/normalize(normal));
            ray_dir = I - 2 * dot(I,N) * N;
            //ray_dir = vec3(0,1,0);
            //ray_dir.y += perlinNoise(uv);
            ray_dir = -normal;
            //ray_dir.y = -(ray_dir.y);
            //break;w
        }
    }

    if (hit) {
        b = 1-b;
        if (b > 0){
            //b = b*2;

        }
        fragColor = vec4(b, 0.0, 0.0, 1.0); // Red for hit
        //fragColor = vec4(1, 0.0, 0.0, 1.0); // Red for hit
        //fragColor = vec4(abs(ray_dir),1.0);
    } else {
        vec2 Noisepos = -vec2(camera_dir.y,camera_dir.x);
        //float noiseValue = perlinNoise(Noisepos+uv);
        float noiseValue = perlinNoise(uv);
        fragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue for miss
        fragColor = vec4(vec3(noiseValue * 0.5 + 0.5), 1.0);
    }
}