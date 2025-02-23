#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;

void main()
{
    fragColor = vec4(0,0,0,0);
    if (TexCoords.y > 0.5) {
        fragColor = vec4(1.0,0.0,0.0,1.0);

    }


    //fragColor = (texture(screenTexture, TexCoords));
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);

}
