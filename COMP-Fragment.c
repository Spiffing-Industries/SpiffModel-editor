#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;
uniform sampler2D uiTexture;
uniform sampler2D skyTexture;
uniform sampler2D pauseTexture;
uniform sampler2D outline_mask;

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
    fragColor = outlineMaskColor;
    //fragColor = texture(skyTexture, TexCoords);




}
