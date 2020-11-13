#include "../headers/export_image.h"


void exportImage(int WIDTH, int HEIGTH, char* output_path) {
    ofstream image_ppm (output_path);
    if (image_ppm.is_open()) {
        image_ppm << "P3\n";
        image_ppm << WIDTH << " " << HEIGTH << "\n";
        image_ppm << "255\n";

        for (int i = HEIGTH-1; i>=0; i--) {
            for(int j = 0; j<WIDTH; j++) {
                unsigned char pixel[4];
                glReadPixels(j, i, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixel);
                image_ppm << (int) pixel[0] << " " << (int) pixel[1] << " " << (int) pixel[2];
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