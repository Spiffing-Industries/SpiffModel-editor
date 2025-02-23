#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
//uniform sampler2D screenTexture;

uniform vec3 camera_dir;


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


void main()
{
    vec2 resolution = vec2(1.0,1.0);
    //vec2 uv = gl_FragCoord.xy / resolution * 2.0 - 1.0;
    vec2 uv = TexCoords.xy;
    uv.y *= resolution.y / resolution.x;

    
    vec3 ray_dir = normalize(vec3(uv.x, uv.y, -1.0)); // Ray direction in view space
    //ray_dir.x += distance(randomGradient(uv+(1/time)),vec2(0,0));
    ray_dir = normalize(ray_dir);
    float Cam_XAngle = camera_dir.x;
    float Cam_YAngle = camera_dir.y;
    ray_dir = RotateOnX(ray_dir,Cam_XAngle);
    ray_dir = RotateOnY(ray_dir,Cam_YAngle);
   // ray_dir = vec3((sin(Cam_XAngle)*uv.x)-(cos(Cam_XAngle)*uv.z),uv.y,(cos(Cam_XAngle)*uv.x)+(sin(Cam_XAngle)*uv.z));
    ray_dir = normalize(ray_dir);

    float noiseValue = perlinNoise(ray_dir.xy);
    fragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue for miss
    fragColor = vec4(vec3(noiseValue * 0.5 + 0.5), 1.0);

    //fragColor = vec4(ray_dir.xy,0.0,1.0);

    //fragColor = vec4(1.0,0,0,1.0);

    //fragColor = (texture(screenTexture, TexCoords));
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);

}
