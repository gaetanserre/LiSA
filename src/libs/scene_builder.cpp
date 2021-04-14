#include "../headers/scene_builder.hpp"

string readFile(char* path) {
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

void searchDim(string str, int* WIDTH, int* HEIGTH) {
    auto throwErrorDim = [] (string error) {
        cerr << "Error in output dimension declaration: ";
        cerr << error << endl;
        exit(-1);
    };

    smatch match;
    const string s = str;
    regex width_rgx ("output_width\\s*=\\s*([0-9]+)");
    if(regex_search(s.begin(), s.end(), match, width_rgx))
        *WIDTH = stoi(match[1]);
    else 
        throwErrorDim("no valid width declared.");

    regex heigth_rgx ("output_heigth\\s*=\\s*([0-9]+)");
    if(regex_search(s.begin(), s.end(), match, heigth_rgx))
        *HEIGTH = stoi(match[1]);
    else 
        throwErrorDim("no valid heigth declared.");
}


vector<string> matchReg(string str, regex r) {
    vector<string> res;
    std::smatch match;

    string::const_iterator searchStart( str.cbegin() );

    while ( regex_search( searchStart, str.cend(), match, r ) )
    {
        res.push_back(match[0]);
        searchStart = match.suffix().first;
    }

    return res;
}

string removeComments(string file) {
    const string s = file;
    const regex comments_rgx("\\/\\*((.|\n)*?)\\*\\/");

    stringstream result;
    regex_replace(ostream_iterator<char>(result), s.begin(), s.end(), comments_rgx, "");

    return result.str();
}

SceneBuilder::SceneBuilder() {
}


SceneBuilder::SceneBuilder(char* path, int* WIDTH, int* HEIGTH) {
    regex Material_reg("Material\\s+"+this->mat_name+"+\\s*\\{\n*[^\\}]*");
    regex Sphere_reg ("Sphere\\s*\\{\n*[^\\}]*");
    regex Meshes_reg ("Mesh\\s*\\{\n*[^\\}]*");
    regex Camera_reg ("Camera\\s*\\{\n*[^\\}]*");


    string file = readFile(path);

    file = removeComments(file);
    searchDim(file, WIDTH, HEIGTH);
    buildMaterials(matchReg(file, Material_reg));
    buildSpheres(matchReg(file, Sphere_reg));
    buildMeshes(matchReg(file, Meshes_reg));
    buildCamera(matchReg(file, Camera_reg));
}


regex searchVector(string begin) {
    regex res(begin + "\\s*=\\s*\\(\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*,\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*,\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*\\)");
    return res;
}

regex searchFloat(string begin) {
    regex r(begin + "\\s*=\\s*([+-]?([0-9]*[.])?[0-9]+)");
    return r;
}


void SceneBuilder::buildMaterials(vector<string> materials_str) {

    auto throwErrorMat = [] (string error) {
        cerr << "Error in materials declaration: ";
        cerr << error << endl;
        exit(-1);
    };

    if (materials_str.size() < 1)
        throwErrorMat("no valid materials declared.");

    for (int i = 0; i<materials_str.size(); i++) {
        const string s = materials_str[i];
        smatch match;

        /******* Search name *******/
        regex name_rgx ("\\s+("+this->mat_name+") *\\{");
        string name;
        if(regex_search(s.begin(), s.end(), match, name_rgx)) {
            this->materials_name.push_back(match[1]);
            name = match[1];
        }

        /******* Search light *******/
        regex light_rgx ("(light\\s*=\\s*true)");
        bool isLight = regex_search(s.begin(), s.end(), match, light_rgx);
        //this->matIsLight.push_back(isLight);

        /******* Search color *******/
        Material m;
        glm::vec3 color;
        regex color_rgx = searchVector("color");
        if(regex_search(s.begin(), s.end(), match, color_rgx)){
            color = glm::vec3(stof(match[1]), stof(match[3]), stof(match[5]));
            if (color[0] > 1 || color[1] > 1 || color[2] > 1) {
            	color[0] /= 255.0;
            	color[1] /= 255.0;
            	color[2] /= 255.0;
            }
        } else
            throwErrorMat("no valid color provided in material " + name + ".");

        /******* Search emit/roughness *******/
        regex alpha_rgx = searchFloat("(roughness|emit_intensity)");
        if(regex_search(s.begin(), s.end(), match, alpha_rgx)) {
            if (isLight)
                m = buildLight(color, stof(match[2]));
            else
                m = buildMaterial(color, stof(match[2]));
        }
        else
            throwErrorMat("no valid roughness|emit_intensity in material " + name + ".");
        
        this->materials.push_back(m);

    }
}


void SceneBuilder::buildSpheres(vector<string> spheres_str) {
    auto throwErrorSphere = [] (string error) {
        cerr << "Error in spheres declaration: ";
        cerr << error << endl;
        exit(-1);
    };

    vector<glm::vec4> materials_n;
    vector<bool> matIsLight_n;
    for (int i = 0; i<spheres_str.size(); i++) {
        Sphere sphere;
        const string s = spheres_str[i];
        smatch match;

        /******* Search material *******/
        regex mat_name_rgx ("material\\s*=\\s*("+this->mat_name+")");
        if(regex_search(s.begin(), s.end(), match, mat_name_rgx)) {
            bool found = false;
            for (int j = 0; j<this->materials_name.size(); j++) {
                if (match[1] == this->materials_name[j]) {
                    sphere.materialIdx = j;
                    if (this->materials[j].emit) {
                        if(this->idxLight == -1)
                            idxLight = spheres.size();
                        else
                            throwErrorSphere("Only one light is allowed.");
                    }
                    found = true;
                    break;
                }
            }
            if (not found) throwErrorSphere((string) match[1] + " material not found.");
        } else
            throwErrorSphere("no material provided.");

        /******* Search center *******/
        regex center_rgx = searchVector("center");
        if(regex_search(s.begin(), s.end(), match, center_rgx))
            sphere.center = glm::vec3(stof(match[1]), stof(match[3]), stof(match[5]));
        else
            throwErrorSphere("no valid center provided.");
        

        /******* Search radius *******/
        regex radius_rgx = searchFloat("radius");
        if(regex_search(s.begin(), s.end(), match, radius_rgx))
            sphere.radius = stof(match[1]);
        else
            throwErrorSphere("no valid radius provided.");

        this->spheres.push_back(sphere);
    }
}


void SceneBuilder::buildMeshes(vector<string> meshes_str) {
    auto throwErrorMeshe = [] (string error) {
        cerr << "Error in mesh declaration: ";
        cerr << error << endl;
        exit(-1);
    };

    for (int i = 0; i<meshes_str.size(); i++) {
        const string s = meshes_str[i];
        smatch match;

        /******* Search material *******/
        int materialIdx;
        regex mat_name_rgx ("material\\s*=\\s*("+this->mat_name+")");
        if(regex_search(s.begin(), s.end(), match, mat_name_rgx)) {
            bool found = false;
            for (int j = 0; j<this->materials_name.size(); j++) {
                if (match[1] == this->materials_name[j]) {
                    materialIdx = j;
                    if (this->materials[j].emit && this->idxLight != -1)
                        throwErrorMeshe("Only one light is allowed.");
                    found = true;
                    break;
                }
            }
            if (not found) throwErrorMeshe((string) match[1] + " material not found");
        } else throwErrorMeshe("no valid material provided");

        /******* Search obj file *******/
        regex obj_file_rgx ("obj_file\\s*=\\s*(.+\\.obj)");
        if (regex_search(s.begin(), s.end(), match, obj_file_rgx)) {
            string path = match[1];
            vector<Triangle> t = parse_obj_file(path, this->meshes_vertices, this->meshes_normals, materialIdx);
            this->triangles.insert(this->triangles.end(), t.begin(), t.end());

        } else throwErrorMeshe("no valid obj_file provided");
    }
}


void SceneBuilder::buildCamera(vector<string> camera_str) {
    auto throwErrorCamera = [] (string error) {
        cerr << "Error in camera declaration: ";
        cerr << error << endl;
        exit(-1);
    };

    if (camera_str.size() != 1)
        throwErrorCamera("no valid camera declared.");
    
    const string s = camera_str[0];
    glm::vec3 pos;
    glm::vec3 look_at;
    float fov;
    glm::vec2 focal_plane(0,0);

    smatch match;

    /******* Search position *******/
    regex position_rgx = searchVector("position");
    if(regex_search(s.begin(), s.end(), match, position_rgx))
        pos = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
    else
        throwErrorCamera("no valid position provided.");

    /******* Search lookAt *******/
    regex look_at_rgx = searchVector("look_at");
    if(regex_search(s.begin(), s.end(), match, look_at_rgx))
        look_at = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
    else
        throwErrorCamera("no valid look_at provided.");

    /******* Search FOV *******/
    regex fov_rgx = searchFloat("fov");
    if(regex_search(s.begin(), s.end(), match, fov_rgx))
        fov = stof(match[1]);
    else
        throwErrorCamera("no valid fov provided.");

    /******* Search focal_plane *******/
    regex focal_rgx = searchFloat("focal_plane");
    if(regex_search(s.begin(), s.end(), match, focal_rgx)) {
        focal_plane.x = stof(match[1]);
        focal_plane.y = 1;
    }

    Camera c = {
        pos,
        look_at,
        fov,
        focal_plane
    };

    this->camera = c;

}


void SceneBuilder::sendDataToCuda(CudaEngine *cudaEngine, int width, int heigth) {

	glm::mat4 viewMatrix = glm::lookAt(
		this->camera.pos,
		this->camera.look_at,
		glm::vec3(0, 1, 0)
	);

    glm::mat4 projection_matrix = glm::perspectiveFov(
		glm::radians(this->camera.fov),
		float(width),
		float(heigth),
		0.01f,
		100.f
	);


    glm::mat4 PVMatrix = glm::inverse(projection_matrix * viewMatrix);

    /***** Transform spheres and materials *****/

    (*cudaEngine).init(this->materials, this->spheres,
                       this->triangles,
                       this->meshes_vertices,
                       this->meshes_normals,
                       this->idxLight, PVMatrix,
                       this->camera.pos, this->camera.focal_plane);

}
