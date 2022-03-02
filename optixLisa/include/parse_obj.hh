#include "structs.hh"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

void parse_obj(string obj_file_path,
               vector<float3> &vertices,
               vector<float3> &normals,
               vector<int> &mat_indices,
               int mat_idx);