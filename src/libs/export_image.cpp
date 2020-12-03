#include "../headers/export_image.hpp"


void exportImage(int WIDTH, int HEIGTH, glm::vec3* image, char* output_path) {
    ofstream image_ppm (output_path);
    if (image_ppm.is_open()) {
        image_ppm << "P3\n";
        image_ppm << WIDTH << " " << HEIGTH << "\n";
        image_ppm << "255\n";


        for (int i = HEIGTH-1; i>=0; i--) {
            for (int j = 0; j<WIDTH; j++) {
                int idx = j + i*WIDTH;

                image_ppm << int(255*image[idx].r) << " " << int(255*image[idx].g) << " " << int(255*image[idx].b);
                image_ppm << "  ";
            }
            image_ppm << "\n";
        }
        image_ppm.close();
    } else {
        cerr << "Error in output file" << endl;
        exit(-1);
    }
}
