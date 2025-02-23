#version 330 core
in vec2 TexCoords;
out vec4 fragColor;
uniform sampler2D screenTexture;
uniform bool Enabled;
void main()
{

    /*
    float borderX = 0.1;
    float borderY = 0.1;
    fragColor = vec4(TexCoords,0.0,0);
    if (Enabled == true){ 
        if (TexCoords.x > 1-borderX){
            fragColor = vec4(1.0,0,0,0.5);
        }
        if (TexCoords.x < borderX){
            fragColor = vec4(1.0,0,0,0.5);
        }
        if (TexCoords.y > 1-borderY){
            fragColor = vec4(1.0,0,0,0.5);
        }
        if (TexCoords.y < borderY){
            fragColor = vec4(1.0,0,0,0.5);
        }
    //fragColor = vec4(TexCoords,0.0,0.5);
    }
    */


    float dx = TexCoords.x-0.5;
    float dy = TexCoords.y-0.5;
    fragColor = vec4(TexCoords,0.0,0);

    dx = abs(dx);
    dy = abs(dy);
    float power = 9.0;
    if (Enabled == true){ 
        fragColor = vec4(0.3,0.3,0.3,1.0);
    if ((pow(dx,power)+pow(dy,power)) > pow(0.4,power)){

        fragColor = vec4(0,0,0,0.5);
    }

    }
    //fragColor = (texture(screenTexture, TexCoords));
    //fragColor = vec4(gl_FragCoord.xy*vec2(400,300),0.0,1.0);

}
