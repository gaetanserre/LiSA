#include "structs.hh"
#include "parse_obj.hh"
#include <map>
#include <regex>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

class SceneParser {
  public:
    SceneParser(char* path, int &WIDTH, int &HEIGTH);

    vector<float3> vertices;
    vector<float3> normals;

    Camera camera;

    vector<Material> materials;
    vector<int> mat_indices;
  
  private:
    string mat_name = "([a-zA-Z0-9]|_)+";
    map<string, int> mat_name_idx;
    
    

    void build_materials(vector<string> materials_str);
    void build_meshes(vector<string> meshes_str);
    void build_camera(vector<string> camera_str);
};
