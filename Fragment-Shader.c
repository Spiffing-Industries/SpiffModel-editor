#version 330 core
out vec4 fragColor;

uniform vec2 resolution;
uniform vec3 camera_pos;
uniform vec3 camera_dir;

uniform vec4 metaballs[10];
uniform int metaballcount;

bool isObjectAt(vec3 point) {
    if (point.y < -2){
        return true;

    }
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

    vec3 ray_origin = camera_pos;
    vec3 ray_dir = normalize(vec3(uv.x, uv.y, -1.0)); // Ray direction in view space
    //ray_dir = vec3();
    float Cam_XAngle = camera_dir.x;
    ray_dir = RotateOnX(ray_dir,Cam_XAngle);
   // ray_dir = vec3((sin(Cam_XAngle)*uv.x)-(cos(Cam_XAngle)*uv.z),uv.y,(cos(Cam_XAngle)*uv.x)+(sin(Cam_XAngle)*uv.z));
    ray_dir = normalize(ray_dir);

    float max_distance = 100.0;
    float step_size = 0.1;
    vec3 current_pos = ray_origin;
    bool hit = false;
    float b = 0;
    for (float t = 0.0; t < max_distance; t += step_size) {
        current_pos += ray_dir * step_size;
        b += 1/(max_distance+step_size);
        if (isObjectAt(current_pos)) {
            hit = true;
            break;
        }
    }

    if (hit) {
        fragColor = vec4(1-b, 0.0, 0.0, 1.0); // Red for hit
    } else {
        fragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue for miss
    }
}