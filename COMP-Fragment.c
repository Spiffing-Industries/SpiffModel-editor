#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;
uniform sampler2D uiTexture;

void main()
{
    fragColor = (texture(screenTexture, TexCoords));
    if (TexCoords.x > 0.5){
    fragColor = (texture(uiTexture, TexCoords));
    //fragColor = vec4(0,1.0,0,1.0);
    }
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);
    


    float UI_alpha =  (texture(uiTexture, TexCoords)).a;



    fragColor = ((texture(screenTexture, TexCoords))*(1-UI_alpha))+ (texture(uiTexture, TexCoords));

}
