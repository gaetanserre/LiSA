#include "parse_obj.hh"
using namespace std;

size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
  size_t pos = txt.find(ch);
  size_t initialPos = 0;
  strs.clear();

  // Decompose statement
  while (pos != std::string::npos) {
    strs.push_back(txt.substr(initialPos, pos - initialPos));
    initialPos = pos + 1;

    pos = txt.find(ch, initialPos);
  }

  // Add the last one
  strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));

  return strs.size();
}

void parse_obj(string obj_file_path,
               vector<float3> &vertices,
               vector<float3> &normals,
               vector<int> &mat_indices,
               int mat_idx)
{
  cout << "Importing " << obj_file_path << "...\n";

  vector<float3> vertices_tmp;
  vector<float3> normals_tmp;

	ifstream obj_file(obj_file_path);
  if (obj_file.is_open()) {
    string line;
    int nb_triangles = 0;

    while (getline(obj_file, line)) {
      vector<string> line_splitted;
      split(line, line_splitted, ' ');
      
      if (line_splitted[0] == "v")
        vertices_tmp.push_back({stof(line_splitted[1]), stof(line_splitted[2]), stof(line_splitted[3])});
      else if (line_splitted[0] == "vn")
        normals_tmp.push_back({stof(line_splitted[1]), stof(line_splitted[2]), stof(line_splitted[3])});
  
      else if (line_splitted[0] == "f") {
        vector<string> idx;
        for (int i = 1; i < 4 ; i++) {
          split(line_splitted[i], idx, '/');
          vertices.push_back(vertices_tmp[stoi(idx[0])-1]);
          normals.push_back(normals_tmp[stoi(idx[2])-1]);
        }
        mat_indices.push_back(mat_idx);
        nb_triangles++;
      }
    }

    printf("Done. Imported %d triangles.\n", nb_triangles);

    obj_file.close();
      
  } else {
    cerr << obj_file_path << " not found." << endl;
    exit(-1);
  }
}
