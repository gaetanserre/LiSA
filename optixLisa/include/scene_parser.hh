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
    SceneParser(char* path);
    RendererParams get_params() { return this->params; };
  
  private:
    string remove_comments(const string& file);
    regex search_vector(const string& begin, int n = 3);
    regex search_float(const string& begin);
    regex search_int(const string& begin);

    string search_param(const string& file, const string& param, bool is_string=false);
    void search_dim(const string &file);
    void build_materials(vector<string> materials_str);
    void build_meshes(vector<string> meshes_str);
    void build_camera(vector<string> camera_str);
    void mk_params();

    vector<float3> vertices;
    vector<float3> normals;

    vector<Material> materials;
    vector<int> mat_indices;

    unsigned int width, height;
    Camera camera;
    unsigned int num_samples;
    unsigned int num_bounces;

    string output_image;

    RendererParams params;

    string var_rgx = "([a-zA-Z0-9]|_)+";
    string path_rgx = "([a-zA-Z0-9]|_|\\/|.)+";
    map<string, int> mat_name_idx;
};
