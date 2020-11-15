#include "../headers/engine.h"

RayTracingEngine::RayTracingEngine(
				char* scene_file_path,
                const char* window_name,
                const char* vshader_path,
                const char* fshader_path,
                const char* cshader_path
)
{	
	this->scene_builder = SceneBuilder (scene_file_path, &this->WIDTH, &this->HEIGTH);
    this->window = createWindow(this->WIDTH, this->HEIGTH, window_name);
    this->quadTex = createTex(this->WIDTH, this->HEIGTH);
    this->Display_Prog = LoadVFShaders(vshader_path, fshader_path);
    this->Compute_Prog = LoadComputeShader(cshader_path);
    this->quad_VAO = getQuadVao();
        
}

void RayTracingEngine::run(int nbFrames, char* output_path, int nb_sample) {

    GLuint uniformSeed = glGetUniformLocation(this->Compute_Prog, "seed");

	GLuint uniformNbFrames = glGetUniformLocation(this->Compute_Prog, "nb_frames");

	glm::mat4 projectionMatrix = glm::perspective(
		glm::radians(50.f),
		float(WIDTH) / float(HEIGTH),
		0.01f,
		100.f
	);
	
	this->scene_builder.sendDataToShader(this->Compute_Prog, projectionMatrix);
	srand (time(NULL));


    glUseProgram(this->Compute_Prog);
	
	float random = ((float) rand()) / (float) RAND_MAX;
	glUniform1f(uniformSeed, random);
	glUniform1i(uniformNbFrames, nbFrames);
	glUniform1i(glGetUniformLocation(this->Compute_Prog, "nb_sample"), nb_sample);

	glBindTexture(GL_TEXTURE_2D, this->quadTex);
	glBindImageTexture(0, this->quadTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	glDispatchCompute(this->WIDTH / 10, this->HEIGTH / 10, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

    bool print = false;
	printf("Starting rendering..\n");
	int currentFrame = 0;

	while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {

		random = ((float) rand()) / (float) RAND_MAX;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.97f, 0.97f, 1.f, 1.f);

		glUseProgram(this->Display_Prog);

		glBindTexture(GL_TEXTURE_2D, this->quadTex);

		glBindVertexArray(this->quad_VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glUseProgram(0);

		glfwSwapBuffers(this->window);
		glfwPollEvents();

		currentFrame += 1;
		if (currentFrame < nbFrames) {
			glUseProgram(this->Compute_Prog);

			glUniform1f(uniformSeed, random);
			glUniform1i(uniformNbFrames, nbFrames);

			glBindTexture(GL_TEXTURE_2D, this->quadTex);
			glBindImageTexture(0, this->quadTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			glDispatchCompute(this->WIDTH / 10, this->HEIGTH / 10, 1);
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
			glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
			glBindTexture(GL_TEXTURE_2D, 0);
			glUseProgram(0);
		}
		else if (!print) {
			printf("Rendering finished.\n");
			if (output_path != NULL) {
				exportImage(this->WIDTH, this->HEIGTH, this->quadTex, output_path);
				printf("Exporting finished.\n");
			}
			print = true;
		}
	}

	glfwTerminate();
}