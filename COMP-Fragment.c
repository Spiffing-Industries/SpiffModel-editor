#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform vec2 resolution;
uniform sampler2D screenTexture;
uniform sampler2D uiTexture;
uniform sampler2D skyTexture;
uniform sampler2D pauseTexture;
uniform sampler2D outline_mask;
uniform sampler2D gizmoTexture;




float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}


vec4 guassianBlur(sampler2D image, vec2 texelSize, float sigma, int radius) {
    vec3 result = vec3(0.0,0.0,0.0);
    float weightSum = 0.0;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            float weight = gaussian(length(vec2(x, y)), sigma);
            result += texture(image, TexCoords + offset).rgb * weight;
            weightSum += weight;
        }
    }

    result /= weightSum;
    return vec4(result, 1.0);
}



void main()
{
    fragColor = (texture(screenTexture, TexCoords));
    if (TexCoords.x > 0.5){
    fragColor = (texture(uiTexture, TexCoords));
    //fragColor = vec4(0,1.0,0,1.0);
    }
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);
    


    float UI_alpha =  (texture(uiTexture, TexCoords)).a;

    vec4 worldColor = texture(screenTexture, TexCoords);

    vec4 skyColor = texture(skyTexture, TexCoords);

    vec4 outlineMaskColor = texture(outline_mask, TexCoords);


    vec4 WorldAndSkyColor = (skyColor*(1-worldColor.a))+(worldColor*worldColor.a);
    if (worldColor.a == 0){
        worldColor = texture(skyTexture, TexCoords);
        //worldColor.g = 1;
        //worldColor = skyTexture;
    }
    //worldColor = texture(skyTexture, TexCoords);
    fragColor = (WorldAndSkyColor*(1-UI_alpha))+ (texture(uiTexture, TexCoords));
    fragColor = (fragColor*(1-texture(pauseTexture, TexCoords).a))+texture(pauseTexture, TexCoords);
    //fragColor = outlineMaskColor;
    vec2 texelSize = vec2(1/resolution.x,1/resolution.y);

    

    vec4 OutlineBlur1 = guassianBlur(outline_mask,texelSize,5.0,3);
    vec4 OutlineBlur2 = guassianBlur(outline_mask,texelSize,0.4,3);
    OutlineBlur2.a = 0.0;
    vec4 outlineColor = abs(OutlineBlur1-OutlineBlur2)*20;

    float outlineColorDistance = distance(outlineColor.xyz,vec3(0,0,0));
    outlineColor.a = outlineColorDistance;

    fragColor = (fragColor*(1-outlineColor.a)) + outlineColor;
    //fragColor = vec4(outlineColorDistance,0.0,0.0,1.0);


    //fragColor = texture(skyTexture, TexCoords);

    fragColor = texture(gizmoTexture, TexCoords);




}
