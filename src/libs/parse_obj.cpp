#include "../headers/parse_obj.hpp"

size_t split(const std::string& txt, std::vector<std::string>& strs, char ch)
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

vector<Triangle> parse_obj_file(string obj_file_path,
                                vector<glm::vec3> &vertices,
                                vector<glm::vec3> &normals,
                                int materialIdx)
{
    cout << "Importing " << obj_file_path << endl;
    
    int oldV_size = vertices.size();
    int oldN_size = normals.size();

    vector<Triangle> triangles;

	ifstream obj_file(obj_file_path);
    if (obj_file.is_open()) {
        string line;

        while (getline(obj_file, line)) {
            vector<string> line_splitted;
            split(line, line_splitted, ' ');
            
            if (line_splitted[0] == "v") {
                vertices.push_back(
                    glm::vec3(
                        stof(line_splitted[1]),
                        stof(line_splitted[2]),
                        stof(line_splitted[3])
                    )
                );
            }
            else if (line_splitted[0] == "vn") {
                normals.push_back(
                    glm::vec3(
                        stof(line_splitted[1]),
                        stof(line_splitted[2]),
                        stof(line_splitted[3])
                    )
                );
            }
            else if (line_splitted[0] == "f") {
                vector<string> idx;

                split(line_splitted[1], idx, '/');
                int p1Idx = stoi(idx[0]) + oldV_size - 1;
                int n1Idx = stoi(idx[2]) + oldN_size - 1;

                split(line_splitted[2], idx, '/');
                int p2Idx = stoi(idx[0]) + oldV_size - 1;
                int n2Idx = stoi(idx[2]) + oldN_size - 1;

                split(line_splitted[3], idx, '/');
                int p3Idx = stoi(idx[0]) + oldV_size - 1;
                int n3Idx = stoi(idx[2]) + oldN_size - 1;

                Triangle t = {
                    p1Idx,
                    p2Idx,
                    p3Idx,
                    n1Idx,
                    n2Idx,
                    n3Idx,
                    materialIdx
                };
                triangles.push_back(t);
            }
        }

        cout << "Done : " << triangles.size() << " triangles" << endl;

        obj_file.close();
        
        return triangles;
        
    } else {
            cerr << obj_file_path << " not found." << endl;
            exit(-1);
    }
}
