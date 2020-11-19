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

void parse_obj_file(string obj_file_path, vector<glm::vec3> &vertices, vector<glm::vec3> &normals) {
    cout << "Importing " << obj_file_path << endl;
	vector<glm::vec3> vertices_t;
    vector<glm::vec3> normals_t;

	ifstream obj_file(obj_file_path);
    if (obj_file.is_open()) {
        string line;

        while (getline(obj_file, line)) {
            vector<string> line_splitted;
            split(line, line_splitted, ' ');
            
            if (line_splitted[0] == "v") {
                vertices_t.push_back(
                    glm::vec3(
                        stof(line_splitted[1]),
                        stof(line_splitted[2]),
                        stof(line_splitted[3])
                    )
                );
            }
            else if (line_splitted[0] == "vn") {
                normals_t.push_back(
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
                vertices.push_back(vertices_t[stoi(idx[0]) - 1]);
                normals.push_back(normals_t[stoi(idx[2]) - 1]);

                split(line_splitted[2], idx, '/');
                vertices.push_back(vertices_t[stoi(idx[0]) - 1]);
                normals.push_back(normals_t[stoi(idx[2]) - 1]);

                split(line_splitted[3], idx, '/');
                vertices.push_back(vertices_t[stoi(idx[0]) - 1]);
                normals.push_back(normals_t[stoi(idx[2]) - 1]);
            }
        }

        cout << "Done : " << vertices.size() << " vertices" << endl;

        obj_file.close();
        
    } else {
            cerr << obj_file_path << " not found." << endl;
            exit(-1);
    }
}
