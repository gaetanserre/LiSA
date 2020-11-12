#include "../headers/engine.h"

RayTracingEngine::RayTracingEngine(
				char* scene_file_path,
                const char* window_name,
                int width,
                int heigth,
                const char* vshader_path,
                const char* fshader_path,
                const char* cshader_path
)
{	
	this->scene_builder = SceneBuilder (scene_file_path);
    this->window = createWindow(width, heigth, window_name);
    this->quad_Tex = createTex(width, heigth);
    this->Display_Prog = LoadVFShaders(vshader_path, fshader_path);
    this->Compute_Prog = LoadComputeShader(cshader_path);
    this->quad_VAO = getQuadVao();

    this->WIDTH = width;
    this->HEIGTH = heigth;
        
}

void RayTracingEngine::run(int nbFrames) {

    GLuint uniformSeed = glGetUniformLocation(this->Compute_Prog, "seed");
	int seed = 0;

	GLuint uniformNbFrames = glGetUniformLocation(this->Compute_Prog, "nb_frames");
	int NbFrames = nbFrames;


	glm::mat4 projectionMatrix = glm::perspective(
		glm::radians(50.f),
		float(WIDTH) / float(HEIGTH),
		0.01f,
		100.f
	);
	
	this->scene_builder.sendDataToShader(this->Compute_Prog, projectionMatrix);

    glUseProgram(this->Compute_Prog);

	glUniform1i(uniformSeed, seed);
	glUniform1i(uniformNbFrames, NbFrames);

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