#include "../headers/engine.h"

RayTracingEngine::RayTracingEngine(
                const char* window_name,
                int width,
                int heigth,
                const char* vshader_path,
                const char* fshader_path,
                const char* cshader_path
)
{
    this->window = createWindow(width, heigth, window_name);
    this->quad_Tex = createTex(width, heigth);
    this->Display_Prog = LoadVFShaders(vshader_path, fshader_path);
    this->Compute_Prog = LoadComputeShader(cshader_path);
    this->quad_VAO = getQuadVao();

    this->WIDTH = width;
    this->HEIGTH = heigth;
        
}

void RayTracingEngine::run(int nbFrames) {
	glm::vec3 lightPos = glm::vec3(1.75, 1, 0.5);

    GLuint uniformSeed = glGetUniformLocation(this->Compute_Prog, "seed");
	int seed = 0;

	GLuint uniformNbFrames = glGetUniformLocation(this->Compute_Prog, "nb_frames");
	int NbFrames = nbFrames;

	GLuint uniformPV = glGetUniformLocation(this->Compute_Prog, "PVMatrix");

	glm::mat4 projectionMatrix = glm::perspective(
		glm::radians(50.f),
		float(WIDTH) / float(HEIGTH),
		0.01f,
		10.f
	);

	GLuint uniformEyePos = glGetUniformLocation(this->Compute_Prog, "eyePos");
	glm::vec3 eye_pos = glm::vec3(0, 0, 0.5);
	glm::mat4 viewMatrix = glm::lookAt(
		eye_pos,
		glm::vec3(0.f, 0.f, 0.f),
		glm::vec3(0, 1, 0)
	);

	glm::mat4 PVMatrix = glm::inverse(projectionMatrix * viewMatrix);

    GLuint uniformSpheres = glGetUniformLocation(this->Compute_Prog, "spheres");
	GLuint uniformMaterials = glGetUniformLocation(this->Compute_Prog, "materials");
	GLuint uniformIsLight = glGetUniformLocation(this->Compute_Prog, "isLight");
    GLuint u_NUM_SPHERES = glGetUniformLocation(this->Compute_Prog, "NUM_SPHERES");
    glm::vec4 spheres[] = {glm::vec4(-1.0, 0.2, -2.3, 1), glm::vec4(1, -0.6, -2.3, 0.7), glm::vec4(0.5, 0, 2, 0.5), glm::vec4(lightPos, 0.40)};
	glm::vec4 materials[] = {glm::vec4(0.4, 0.2, 0.4, 0), glm::vec4(0.1, 0.5, 0.4, 1), glm::vec4(0, 0.5, 0.7, 1), glm::vec4(1)};
	int isLight = 3;
    int nb_spheres = 4;

    glUseProgram(this->Compute_Prog);

	glUniform1i(u_NUM_SPHERES, nb_spheres);
    glUniform4fv(uniformSpheres, nb_spheres, glm::value_ptr(spheres[0]));
	glUniform4fv(uniformMaterials, nb_spheres, glm::value_ptr(materials[0]));
	glUniform1i(uniformIsLight, isLight);

	glUniform1i(uniformSeed, seed);
	glUniform1i(uniformNbFrames, NbFrames);
	glUniformMatrix4fv(uniformPV, 1, GL_FALSE, glm::value_ptr(PVMatrix));
	glUniform3fv(uniformEyePos, 1, glm::value_ptr(eye_pos));


	glBindTexture(GL_TEXTURE_2D, this->quad_Tex);
	glBindImageTexture(0, this->quad_Tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	glDispatchCompute(this->WIDTH / 20, this->HEIGTH / 20, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

    bool print = false;
	printf("Starting rendering..\n");

	while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.97f, 0.97f, 1.f, 1.f);

		glUseProgram(this->Display_Prog);

		glBindTexture(GL_TEXTURE_2D, this->quad_Tex);

		glBindVertexArray(this->quad_VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glUseProgram(0);

		glfwSwapBuffers(this->window);
		glfwPollEvents();

		seed += 1;
		if (seed < NbFrames) {
			glUseProgram(this->Compute_Prog);

			glUniform1i(uniformSeed, seed);
			glUniform1i(uniformNbFrames, NbFrames);
			glUniformMatrix4fv(uniformPV, 1, GL_FALSE, glm::value_ptr(PVMatrix));

			glBindTexture(GL_TEXTURE_2D, this->quad_Tex);
			glBindImageTexture(0, this->quad_Tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			glDispatchCompute(this->WIDTH / 20, this->HEIGTH / 20, 1);
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
			glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
			glBindTexture(GL_TEXTURE_2D, 0);
			glUseProgram(0);
		}
		else if (!print) {
			printf("Done\n");
			print = true;
		}
	}

	glfwTerminate();
}