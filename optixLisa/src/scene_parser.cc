#include "scene_parser.hh"
#include <sstream>

string read_file(char* path) {
  string res = "";
  string line;
  ifstream scene_file (path);
  if (scene_file.is_open()) {
    while(getline(scene_file, line)) {
        res += line + "\n";
    }
    scene_file.close();
  } else {
    cerr << path << " not found" << endl;
    exit(-1);
  }
  return res;
}

void search_dim(string str, int &WIDTH, int &HEIGTH) {
  auto throwErrorDim = [] (string error) {
    cerr << "Error in output dimension declaration: ";
    cerr << error << endl;
    exit(-1);
  };

  smatch match;
  const string s = str;
  regex width_rgx ("output_width\\s*=\\s*([0-9]+)");
  if(regex_search(s.begin(), s.end(), match, width_rgx))
    WIDTH = stoi(match[1]);
  else 
    throwErrorDim("no valid width declared.");

  regex heigth_rgx ("output_heigth\\s*=\\s*([0-9]+)");
  if(regex_search(s.begin(), s.end(), match, heigth_rgx))
    HEIGTH = stoi(match[1]);
  else 
    throwErrorDim("no valid heigth declared.");
}


vector<string> match_reg(string str, regex r) {
  vector<string> res;
  std::smatch match;

  string::const_iterator searchStart(str.cbegin());

  while (regex_search(searchStart, str.cend(), match, r)) {
    res.push_back(match[0]);
    searchStart = match.suffix().first;
  }
  return res;
}

string remove_comments(string file) {
  const string s = file;
  const regex comments_rgx("\\/\\*((.|\n)*?)\\*\\/");

  stringstream result;
  regex_replace(ostream_iterator<char>(result), s.begin(), s.end(), comments_rgx, "");

  return result.str();
}


SceneParser::SceneParser(char* path, int &WIDTH, int &HEIGTH) {
  regex Material_reg("Material\\s+"+this->mat_name+"+\\s*\\{\n*[^\\}]*");
  regex Meshes_reg ("Mesh\\s*\\{\n*[^\\}]*");
  regex Camera_reg ("Camera\\s*\\{\n*[^\\}]*");


  string file = read_file(path);

  file = remove_comments(file);
  search_dim(file, WIDTH, HEIGTH);
  build_materials(match_reg(file, Material_reg));
  build_meshes(match_reg(file, Meshes_reg));
  build_camera(match_reg(file, Camera_reg));
}


regex search_vector(string begin, int n = 3) {

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

regex search_float(string begin) {
  regex r(begin + "\\s*=\\s*([+-]?([0-9]*[.])?[0-9]+)");
  return r;
}


void SceneParser::build_materials(vector<string> materials_str) {
  auto throw_error_mat = [] (string error) {
      cerr << "Error in materials declaration: ";
      cerr << error << endl;
      exit(-1);
  };

  if (materials_str.size() < 1)
      throw_error_mat("no valid materials declared.");

  for (long unsigned int i = 0; i<materials_str.size(); i++) {
    const string s = materials_str[i];
    smatch match;

    /******* Search name *******/
    regex name_rgx ("\\s+("+this->mat_name+") *\\{");
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
  auto throw_error_mesh = [] (string error) {
    cerr << "Error in mesh declaration: ";
    cerr << error << endl;
    exit(-1);
  };

  for (long unsigned int i = 0; i <meshes_str.size(); i++) {
    const string s = meshes_str[i];
    smatch match;

    /******* Search material *******/
    regex mat_name_rgx ("material\\s*=\\s*("+this->mat_name+")");
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
  auto throw_error_camera = [] (string error) {
    cerr << "Error in camera declaration: ";
    cerr << error << endl;
    exit(-1);
  };

  if (camera_str.size() != 1)
    throw_error_camera("no valid camera declared.");
  
  const string s = camera_str[0];
  float3 eye;
  float3 look_at;
  float fov;
  float2 focal_plane = make_float2(0.0f, 0.0f);

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

  /******* Search focal_plane *******/
  regex focal_rgx = search_float("focal_plane");
  if(regex_search(s.begin(), s.end(), match, focal_rgx)) {
    focal_plane.x = stof(match[1]);
    focal_plane.y = 1;
  }

  Camera c = {
    eye,
    look_at,
    fov,
    focal_plane
  };

  this->camera = c;
}