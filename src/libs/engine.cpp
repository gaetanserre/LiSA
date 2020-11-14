#include "../headers/engine.h"

RayTracingEngine::RayTracingEngine(char* scene_file_path,const char* cshader_path) {	
	this->scene_builder = SceneBuilder (scene_file_path, &this->WIDTH, &this->HEIGTH);
	initGL();
    this->quadTex = createTex(this->WIDTH, this->HEIGTH);
    this->Compute_Prog = LoadComputeShader(cshader_path);
}

void RayTracingEngine::run(int nbFrames, char* output_path) {

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
	
	glUniform1i(uniformNbFrames, NbFrames);

	glBindTexture(GL_TEXTURE_2D, this->quadTex);
	glBindImageTexture(0, this->quadTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	glDispatchCompute(this->WIDTH / 10, this->HEIGTH / 10, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

    bool print = false;
	printf("Starting rendering..\n");

	while (seed < nbFrames) {
		glUseProgram(this->Compute_Prog);

		glUniform1i(uniformSeed, seed);
		glUniform1i(uniformNbFrames, NbFrames);

		glBindTexture(GL_TEXTURE_2D, this->quadTex);
		glBindImageTexture(0, this->quadTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
		glDispatchCompute(this->WIDTH / 10, this->HEIGTH / 10, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);
		
		seed ++;
	}


	exportImage(this->WIDTH, this->HEIGTH, this->quadTex, output_path);
	printf("Rendering & exporting finished.\n");

	glfwTerminate();
}