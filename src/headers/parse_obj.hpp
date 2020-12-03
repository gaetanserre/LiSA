#include "dependencies.hpp"
#include "cuda/3Dstructs.hpp"

vector<Triangle> parse_obj_file(string obj_file_path,
                                vector<glm::vec3> &vertices,
                                vector<glm::vec3> &normals,
                                int materialIdx
                    );
