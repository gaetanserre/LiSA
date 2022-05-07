#include "scene_parser.hh"
#include <sstream>
#include <utility>

string read_file(char* path) {
  string res;
  string line;
  ifstream scene_file (path);
  if (scene_file.is_open()) {
    while(getline(scene_file, line)) {
        res += line + "\n";
    }
    scene_file.close();
  } else {
    cerr << path << " not found" << endl;
    exit(1);
  }
  return res;
}

vector<string> match_reg(const string& str, const regex& r) {
  vector<string> res;
  std::smatch match;

  string::const_iterator searchStart(str.cbegin());

  while (regex_search(searchStart, str.cend(), match, r)) {
    res.push_back(match[0]);
    searchStart = match.suffix().first;
  }
  return res;
}


SceneParser::SceneParser(char* path) {
  regex Material_reg("material\\s+"+this->var_rgx+"+\\s*\\{\n*[^\\}]*");
  regex Meshes_reg ("mesh\\s*\\{\n*[^\\}]*");
  regex Camera_reg ("camera\\s*\\{\n*[^\\}]*");


  string file = read_file(path);

  file = this->remove_comments(file);

  this->width  = stoi(this->search_param(file, "width"));
  this->height = stoi(this->search_param(file, "height"));

  this->num_samples = stoi(this->search_param(file, "num_samples"));
  this->num_bounces = stoi(this->search_param(file, "num_bounces"));

  this->output_image = this->search_param(file, "output_image", true);

  this->build_materials(match_reg(file, Material_reg));
  this->build_meshes(match_reg(file, Meshes_reg));
  this->build_camera(match_reg(file, Camera_reg));
  this->mk_params();
}

string SceneParser::remove_comments(const string& file) {
  const regex comments_rgx("\\/\\*((.|\n)*?)\\*\\/");
  stringstream result;
  regex_replace(ostream_iterator<char>(result), file.begin(), file.end(), comments_rgx, "");
  return result.str();
}

regex SceneParser::search_vector(const string& begin, int n) {

  std::stringstream reg;
  reg << "\\s*=\\s*\\(";

  string reg_float = "\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*";
  for (int i = 0; i < n - 1; i++) {
    reg << reg_float << ",";
  }
  reg << reg_float << "\\)";

  regex res(begin + reg.str());
  return res;
}

regex SceneParser::search_float(const string& begin) {
  return regex(begin + "\\s*=\\s*([+-]?([0-9]*[.])?[0-9]+)");
}

regex SceneParser::search_int(const string& begin) {
  return regex(begin + "\\s*=\\s*([0-9]+)");
}

string SceneParser::search_param(const string& file, const string& param, bool is_string) {
  smatch match;
  const string s = std::move(file);
  regex param_rgx;
  if (is_string)
    param_rgx = regex(param + "\\s*=\\s*(" + this->path_rgx + ")");
  else
    param_rgx = this->search_int(param);
  if(regex_search(s.begin(), s.end(), match, param_rgx)) {
    return match[1];
  } else {
    cerr << "Param " << param << " not found.\n";
    exit(1);
  }
}


void SceneParser::build_materials(vector<string> materials_str) {
  auto throw_error_mat = [] (const string& error) {
      cerr << "Error in materials declaration: ";
      cerr << error << endl;
      exit(1);
  };

  if (materials_str.empty())
      throw_error_mat("no valid materials declared.");

  for (long unsigned int i = 0; i<materials_str.size(); i++) {
    const string s = materials_str[i];
    smatch match;

    /******* Search name *******/
    regex name_rgx ("\\s+(" + this->var_rgx + ")\\s*\\{");
    string name;
    if(regex_search(s.begin(), s.end(), match, name_rgx)) {
      name = match[1];
    }

    /******* Search light *******/
    regex light_rgx ("(emit\\s*=\\s*true)");
    bool is_light = regex_search(s.begin(), s.end(), match, light_rgx);

    /******* Search color *******/
    Material m;
    float3 color;
    float  alpha;
    regex color_rgx = search_vector("color", 4);
    if(regex_search(s.begin(), s.end(), match, color_rgx)){
      color = make_float3(stof(match[1]), stof(match[3]), stof(match[5]));
      alpha = stof(match[7]);
      if (color.x > 1 || color.y > 1 || color.z > 1) {
        color.x /= 255.0;
        color.y /= 255.0;
        color.z /= 255.0;
      }
    } else
      throw_error_mat("no valid color provided in material " + name + ".");

    /******* make material *******/
    if (is_light) {
      m = mk_material_emit(color);
    } else {
      regex alpha_rgx = search_float("(roughness | n)");
      if(regex_search(s.begin(), s.end(), match, alpha_rgx))
        m = mk_material_diffuse(color, alpha, stof(match[2]));
      else
        throw_error_mat("no valid roughness|refractive index in material " + name + ".");
    }
    /* We do not add the material if it has already been declared */
    if (this->mat_name_idx.find(name) == this->mat_name_idx.end()){
      this->materials.push_back(m);
      this->mat_name_idx[name] = (int) this->materials.size() - 1;
    }
  }
}

void SceneParser::build_meshes(vector<string> meshes_str) {
  auto throw_error_mesh = [] (const string& error) {
    cerr << "Error in mesh declaration: ";
    cerr << error << endl;
    exit(1);
  };

  for (long unsigned int i = 0; i <meshes_str.size(); i++) {
    const string s = meshes_str[i];
    smatch match;

    /******* Search material *******/
    regex mat_name_rgx ("material\\s*=\\s*(" + this->var_rgx + ")");
    int mat_idx;
    if(regex_search(s.begin(), s.end(), match, mat_name_rgx)) {
      if(this->mat_name_idx.find(match[1]) != this->mat_name_idx.end()) {
        mat_idx = this->mat_name_idx[match[1]];
      } else {
        throw_error_mesh((string) match[1] + " material not found");
      }
    } else {
      throw_error_mesh("no valid material provided");
    }

    /******* Search obj file *******/
    regex obj_file_rgx ("obj_file\\s*=\\s*(.+\\.obj)");
    if (regex_search(s.begin(), s.end(), match, obj_file_rgx)) {
      string path = match[1];
      parse_obj(path, this->vertices, this->normals, this->mat_indices, mat_idx);
    } else {
      throw_error_mesh("no valid obj_file provided");
    }
  }
}

void SceneParser::build_camera(vector<string> camera_str) {
  auto throw_error_camera = [] (const string& error) {
    cerr << "Error in camera declaration: ";
    cerr << error << endl;
    exit(1);
  };

  if (camera_str.size() != 1)
    throw_error_camera("no valid camera declared.");
  
  const string s = camera_str[0];
  float3 eye;
  float3 look_at;
  float fov;

  smatch match;

  /******* Search position *******/
  regex position_rgx = search_vector("position");
  if(regex_search(s.begin(), s.end(), match, position_rgx))
    eye = make_float3(stof(match[1]), stof(match[3]), stof(match[5]));
  else
    throw_error_camera("no valid position provided.");

  /******* Search lookAt *******/
  regex look_at_rgx = search_vector("look_at");
  if(regex_search(s.begin(), s.end(), match, look_at_rgx))
    look_at = make_float3(stof(match[1]), stof(match[3]), stof(match[5]));
  else
    throw_error_camera("no valid look_at provided.");

  /******* Search FOV *******/
  regex fov_rgx = search_float("fov");
  if(regex_search(s.begin(), s.end(), match, fov_rgx))
    fov = stof(match[1]);
  else
    throw_error_camera("no valid fov provided.");

  Camera c = {
    eye,
    look_at,
    fov
  };

  this->camera = c;
}

void SceneParser::mk_params() {
  this->params.vertices      = this->vertices.data();
  this->params.normals       = this->normals.data();
  this->params.materials     = this->materials.data();
  this->params.mat_indices   = this->mat_indices.data();
  this->params.num_vertices  = this->vertices.size();
  this->params.num_materials = this->materials.size();

  this->params.width        = this->width;
  this->params.height       = this->height;
  this->params.camera       = this->camera;
  this->params.num_samples  = this->num_samples;
  this->params.num_bounces  = this->num_bounces;
  this->params.output_image = this->output_image.c_str();

}