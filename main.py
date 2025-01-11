import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import time
# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

#WINDOW_WIDTH = 1920
#WINDOW_HEIGHT = 1080
# Vertex shader (passes through vertex positions)
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment shader (performs ray tracing)
FRAGMENT_SHADER = """
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

void main() {
    vec2 uv = gl_FragCoord.xy / resolution * 2.0 - 1.0;
    uv.y *= resolution.y / resolution.x;

    vec3 ray_origin = camera_pos;
    vec3 ray_dir = normalize(vec3(uv.x, uv.y, -1.0)); // Ray direction in view space
    //ray_dir = vec3();
    float Cam_XAngle = camera_dir.x;
    ray_dir = vec3((sin(Cam_XAngle)*uv.x)-(cos(Cam_XAngle)*uv.z),uv.y,(cos(Cam_XAngle)*uv.x)+(sin(Cam_XAngle)*uv.z));
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
"""



for fragment_shader_file_path in ["Fragment-Shader.c","_internals/Fragment-Shader.c"]:
    try:
        with open(fragment_shader_file_path) as file:
            FRAGMENT_SHADER = file.read()
            break
    except FileNotFoundError as e:
        print(e)
        continue




def create_shader_program():
    return compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

def rotate_camera(camera_dir, yaw, pitch):
    # Rotate around Y-axis (yaw)
    rotation_matrix_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotate around X-axis (pitch)
    rotation_matrix_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Apply yaw and pitch rotations to the camera direction
    rotated_dir = np.dot(rotation_matrix_yaw, camera_dir)
    rotated_dir = np.dot(rotation_matrix_pitch, rotated_dir)

    return rotated_dir

def main():
    pygame.init()
    print("OpenGL Major Version",pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
    print("OpenGL Major Version",pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,2)
    print("OpenGL Major Version",pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
    print("OpenGL Major Version",pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    


    # Create a shader program
    shader = create_shader_program()

    # Define a fullscreen quad
    quad_vertices = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
        -1.0,  1.0, 0.0,
         1.0,  1.0, 0.0,
    ], dtype=np.float32)

    # Create a Vertex Buffer Object and Vertex Array Object
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    # Set the shader program and uniforms
    glUseProgram(shader)
    resolution_location = glGetUniformLocation(shader, "resolution")
    glUniform2f(resolution_location, WINDOW_WIDTH, WINDOW_HEIGHT)

    camera_pos_location = glGetUniformLocation(shader, "camera_pos")
    camera_dir_location = glGetUniformLocation(shader, "camera_dir")

    metaball_location = glGetUniformLocation(shader, "metaballs")
    metaball_count_location = glGetUniformLocation(shader, "metaballcount")

    lights_location = glGetUniformLocation(shader, "light_positions")
    lights_color_location = glGetUniformLocation(shader, "light_colors")
    lights_amount_location = glGetUniformLocation(shader, "light_count")

    object_meshes_location = glGetUniformLocation(shader, "ObjectsMeshes")
    object_id_location = glGetUniformLocation(shader, "ObjectID")
    object_id_list_location = glGetUniformLocation(shader, "ObjectIDList")
    object_count_location = glGetUniformLocation(shader, "ObjectCount")
    object_mesh_count_location = glGetUniformLocation(shader, "ObjectMeshCount")

    time_location = glGetUniformLocation(shader, "time")
    
    camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    camera_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    clock = pygame.time.Clock()

    sphere_data = np.array([
        [0.0, 0.0, -5.0, 2.0],  # Sphere 1: center (0,0,-5), radius 2
        [2, 0.0, -5.0, 1.0],  # Sphere 2: center (2,0,-5), radius 1
    ], dtype=np.float32)

    light_positions = np.array([[3.0,2.0,-5.0],[5.0,2.0,-5.0]], dtype=np.float32)
    lights_colors = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]], dtype=np.float32)

    ObjectMeshes = np.array([[0.0, 0.0, -3.0, 1.0],[2, 0.0, -3.0, 1.0]], dtype=np.float32)
    ObjectIDList = np.array([0], dtype=np.float32)
    ObjectIDS = np.array([0,0],dtype=np.float32)



    last_mouse_x, last_mouse_y = pygame.mouse.get_pos()

    running = True
    i = 0
    MouseGrabbed = False
    yaw = 0
    pitch = 0
    while running:
        i += 1
        sphere_data = np.array([
            [0.0, 0.0, -5.0, 1.0],  # Sphere 1: center (0,0,-5), radius 2
            [math.sin(math.radians(i)) * 10, 0.0, -5, 1.0],  # Sphere 2: center (2,0,-5), radius 1
        ], dtype=np.float32)

        #ObjectMeshes = np.array([
        #    [0.0, 0.0, -3.0, 1.0],  # Sphere 1: center (0,0,-5), radius 2
        #    [math.sin(math.radians(i)) * 10, 0.0, -3, 1.0],  # Sphere 2: center (2,0,-5), radius 1
        #    [0.0, 0.0, -2.0, 1.0],
        #    [0.0, 2.0, -3.0, 1.0]
        #], dtype=np.float32)
        #ObjectIDS = np.array([0,0,1,2],dtype=np.float32)
        #ObjectIDList = np.array([0,1,2], dtype=np.float32)
        """
        sphere_data = []
        for x in range(1):
            for y in range(-2,2):
                sphere_data.append([0,0.0,y+-5,1.0])
        sphere_data = np.array(sphere_data)
        """
        pygame.event.set_grab(MouseGrabbed)
        #if MouseGrabbed:
            #pygame.mouse.set_pos((WINDOW_WIDTH/2,WINDOW_HEIGHT/2))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    MouseGrabbed = not MouseGrabbed
            if event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_dx,mouse_dy = event.rel
                #if MouseGrabbed:
                 #   pygame.mouse.set_pos((WINDOW_WIDTH/2,WINDOW_HEIGHT/2))
        if MouseGrabbed:
            pygame.mouse.set_pos((WINDOW_WIDTH/2,WINDOW_HEIGHT/2))

        # Get mouse movement
        
        #print(mouse_x,mouse_y)
        if MouseGrabbed:
            pass
            #mouse_x = mouse_x - WINDOW_WIDTH/2
            #mouse_dy = mouse_y - WINDOW_HEIGHT/2
        else:
            mouse_dx = mouse_x - last_mouse_x
            mouse_dy = mouse_y - last_mouse_y
        last_mouse_x, last_mouse_y = mouse_x, mouse_y

        # Sensitivity for camera rotation
        sensitivity = 0.002

        # Update yaw and pitch
        yaw += mouse_dx * sensitivity*-1
        pitch += mouse_dy * sensitivity

        #print(camera_dir)

        # Apply the camera rotation
        camera_dir = np.array([pitch, yaw, 0], dtype=np.float32)

        # Movement controls
        keys = pygame.key.get_pressed()
        print(yaw)
        if keys[K_w]:
            #camera_pos += camera_dir * 0.1
            camera_pos[2]+= -math.sin(yaw)*0.1
            camera_pos[0]+= math.cos(yaw)*0.1
        if keys[K_s]:
            #camera_pos -= camera_dir * 0.1
            camera_pos[2]+= -math.sin(yaw)*-0.1
            camera_pos[0]+= math.cos(yaw)*-0.1
            #camera_pos[2]-= 0.1
        if keys[K_a]:
            #camera_pos[0] -= 0.1
            camera_pos[2]+= -math.sin(yaw+math.radians(90))*0.1
            camera_pos[0]+= math.cos(yaw+math.radians(90))*0.1
        if keys[K_d]:
            camera_pos[2]+= -math.sin(yaw-math.radians(90))*0.1
            camera_pos[0]+= math.cos(yaw-math.radians(90))*0.1
        if keys[K_q]:
            camera_pos[1] -= 0.1
        if keys[K_e]:
            camera_pos[1] += 0.1
        if keys[K_z]:
            pitch += 0.1
        camera_pos = camera_pos[:3]
        camera_dir = camera_dir[:3]

        #print(sphere_data)
        glUniform1f(time_location, time.time())
        try:
            #print(*camera_pos)
            #print("glUniform3f(camera_pos_location, *camera_pos)")
            glUniform3f(camera_pos_location, *camera_pos)
            #print("glUniform3f(camera_dir_location, *camera_dir)")
            glUniform3f(camera_dir_location, *camera_dir)
            glUniform1i(metaball_count_location, len(sphere_data))
            glUniform1i(lights_amount_location, len(light_positions))
            glUniform3fv(lights_location, len(light_positions),light_positions.flatten())
            glUniform3fv(lights_color_location, len(light_positions),lights_colors.flatten())
            glUniform4fv(metaball_location, len(sphere_data), sphere_data.flatten())

            glUniform4fv(object_meshes_location, len(ObjectMeshes), ObjectMeshes.flatten())
            glUniform1fv(object_id_location, len(ObjectIDS), ObjectIDS.flatten())
            glUniform1fv(object_id_list_location, len(ObjectIDList), ObjectIDList.flatten())

            glUniform1i(object_count_location, len(ObjectIDList))
            glUniform1i(object_mesh_count_location, len(ObjectIDS))
        except Exception as e:
            print(e)
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw the fullscreen quad
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
