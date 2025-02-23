#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;

void main()
{
    vec2 tex_offset = 1.0 / textureSize(screenTexture, 0); // size of one texel
    vec4 result = texture(screenTexture, TexCoords).rgba * 0.227027;
    result += texture(screenTexture, TexCoords + vec2(tex_offset.x, 0.0)).rgba * 0.1945946;
    result += texture(screenTexture, TexCoords - vec2(tex_offset.x, 0.0)).rgba * 0.1945946;
    result += texture(screenTexture, TexCoords + vec2(0.0, tex_offset.y)).rgba * 0.1945946;
    result += texture(screenTexture, TexCoords - vec2(0.0, tex_offset.y)).rgba * 0.1945946;
    fragColor = vec4(result);


    //fragColor = (texture(screenTexture, TexCoords));
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);

}
