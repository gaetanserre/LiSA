#include "../headers/export_image.hpp"


void exportImage(int WIDTH, int HEIGTH, GLuint texture, char* output_path) {
    ofstream image_ppm (output_path);
    if (image_ppm.is_open()) {
        image_ppm << "P3\n";
        image_ppm << WIDTH << " " << HEIGTH << "\n";
        image_ppm << "255\n";

        GLubyte* pixels = new GLubyte[WIDTH*HEIGTH*3];
		glBindTexture(GL_TEXTURE_2D, texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		GLuint r, g, b;
		int el_per_line = WIDTH * 3;
        int x, y;

        for (int i = HEIGTH-1; i>=0; i--) {
            for (int j = 0; j<WIDTH; j++) {
                x = j; y = i;
                int row = y * el_per_line; int col = x * 3;
                r = pixels[row + col];
                g = pixels[row + col + 1];
                b = pixels[row + col + 2];

                image_ppm << r << " " << g << " " << b;
                image_ppm << "  ";
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        image_ppm.close();
    } else {
        cerr << "Error in output file" << endl;
        exit(-1);
    }
}
