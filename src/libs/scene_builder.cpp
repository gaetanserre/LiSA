#include "../headers/scene_builder.h"

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

SceneBuilder::SceneBuilder() {
}


SceneBuilder::SceneBuilder(char* path) {
    regex Material_reg("Material\\s+[a-z0-9]+\\s*\\{\n*[^\\}]*");
    regex Sphere_reg ("Sphere\\s*\\{\n*[^\\}]*");
    regex Meshes_reg ("Mesh\\s*\\{\n*[^\\}]*");
    regex Camera_reg ("Camera\\s*\\{\n*[^\\}]*");


    string file = readFile(path);

    buildMaterials(matchReg(file, Material_reg));
    buildSpheres(matchReg(file, Sphere_reg));
    buildMeshes(matchReg(file, Meshes_reg));
    buildCamera(matchReg(file, Camera_reg));
}


vector<string> SceneBuilder::matchReg(string str, regex r) {
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

regex searchVector(string begin) {
    regex res(begin + "\\s*=\\s*\\(\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*,\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*,\\s*([+-]?([0-9]*[.])?[0-9]+)\\s*\\)");
    return res;
}

regex searchFloat(string begin) {
    regex r(begin + "\\s*=\\s*([+-]?([0-9]*[.])?[0-9]+)");
    return r;
}

void throwErrorMat() {
    cerr << "Error in materials declaration." << endl;
    exit(-1);
}

void SceneBuilder::buildMaterials(vector<string> materials_str) {
    if (materials_str.size() < 1)
        throwErrorMat();

    for (int i = 0; i<materials_str.size(); i++) {
        const string s = materials_str[i];
        smatch match;

        /******* Search name *******/
        regex name_rgx ("\\s+([a-z0-9]+) *\\{");
        if(regex_search(s.begin(), s.end(), match, name_rgx))
            this->materials_name.push_back(match[1]);

        /******* Search light *******/
        regex light_rgx ("(light\\s*=\\s*true)");
        if(regex_search(s.begin(), s.end(), match, light_rgx))
            matIsLight.push_back(true);
        else 
            matIsLight.push_back(false);

        /******* Search color *******/
        glm::vec4 mat(-1);
        regex color_rgx = searchVector("color");
        if(regex_search(s.begin(), s.end(), match, color_rgx))
            mat = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
        else
            throwErrorMat();

        /******* Search emit/roughness *******/
        regex alpha_rgx = searchFloat("(roughness|emit_intensity)");
        if(regex_search(s.begin(), s.end(), match, alpha_rgx))
            mat.w = stof(match[2]);
        else
            throwErrorMat();
        
        this->materials_temp.push_back(mat);

    }
}

void throwErrorSphere() {
    cerr << "Error in spheres declaration." << endl;
    exit(-1);
}

void SceneBuilder::buildSpheres(vector<string> spheres_str) {
    if (spheres_str.size() < 1)
        throwErrorMat();

    vector<glm::vec4> materials_n;
    vector<bool> matIsLight_n;
    for (int i = 0; i<spheres_str.size(); i++) {
        const string s = spheres_str[i];
        smatch match;

        /******* Search material *******/
        regex mat_name_rgx ("material\\s*=\\s*([a-z0-9]+)");
        if(regex_search(s.begin(), s.end(), match, mat_name_rgx)) {
            bool found = false;
            for (int j = 0; j<this->materials_name.size(); j++) {
                if (match[1] == this->materials_name[j]) {
                    materials_n.push_back(this->materials_temp[j]);
                    matIsLight_n.push_back(this->matIsLight[j]);
                    found = true;
                    break;
                }
            }
            if (not found) throwErrorSphere();
        }

        else
            throwErrorSphere();

        /******* Search center *******/
        glm::vec4 sphere(-1);
        regex center_rgx = searchVector("center");
        if(regex_search(s.begin(), s.end(), match, center_rgx))
            sphere = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
        else
            throwErrorSphere();
        

        /******* Search radius *******/
        regex radius_rgx = searchFloat("radius");
        if(regex_search(s.begin(), s.end(), match, radius_rgx))
            sphere.w = stof(match[1]);
        else
            throwErrorSphere();

        this->spheres.push_back(sphere);
    }

    this->materials = materials_n;
    this->matIsLight = matIsLight_n;

}

void throwErrorMeshes() {
    cerr << "Error in meshes declaration" << endl;
    exit(-1);
}

void SceneBuilder::buildMeshes(vector<string> meshes_str) {
    vector<glm::vec4> materials_n;
    for (int i = 0; i<meshes_str.size(); i++) {
        const string s = meshes_str[i];
        smatch match;

        regex mat_name_rgx ("material\\s*=\\s*([a-z0-9]+)");
        if(regex_search(s.begin(), s.end(), match, mat_name_rgx)) {
            bool found = false;
            for (int j = 0; j<this->materials_name.size(); j++) {
                if (match[1] == this->materials_name[j]) {
                    materials_n.push_back(this->materials_temp[j]);
                    found = true;
                    break;
                }
            }
            if (not found) throwErrorMeshes();
        }

        /******* Search obj file *******/
        regex obj_file_rgx ("obj_file\\s*=\\s*(.+\\.obj)");
        if (regex_search(s.begin(), s.end(), match, obj_file_rgx)) {
            vector<glm::vec3> vertices;
            vector<glm::vec3> normals;
            string path = match[1];
            parse_obj_file(path, vertices, normals);
            this->meshes_vertices.insert(meshes_vertices.end(), vertices.begin(), vertices.end());
            this->meshes_normals.insert(meshes_normals.end(), normals.begin(), normals.end());


        } else throwErrorMeshes();
        
    }

    this->materials.insert(materials.end(), materials_n.begin(), materials_n.end());
}

void throwErrorCamera() {
    cerr << "Error in camera declaration." << endl;
    exit(-1);
}

void SceneBuilder::buildCamera(vector<string> camera_str) {
    if (camera_str.size() != 1)
        throwErrorCamera();
    
    const string s = camera_str[0];
    glm::vec3 pos;
    glm::vec3 look_at;

    smatch match;

    /******* Search position *******/
    regex position_rgx = searchVector("position");
    if(regex_search(s.begin(), s.end(), match, position_rgx))
        pos = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
    else
        throwErrorCamera();

    /******* Search lookAt *******/
    regex look_at_rgx = searchVector("look_at");
    if(regex_search(s.begin(), s.end(), match, look_at_rgx))
        look_at = glm::vec4(stof(match[1]), stof(match[3]), stof(match[5]), -1);
    else
        throwErrorCamera();

    Camera c = {
        pos,
        look_at
    };

    this->camera = c;

}


void SceneBuilder::sendDataToShader(GLuint ComputeShaderProgram, glm::mat4 projection_matrix) {

	glm::vec3 eye_pos = glm::vec3(0, 0, 0.5);
	glm::mat4 viewMatrix = glm::lookAt(
		this->camera.pos,
		this->camera.look_at,
		glm::vec3(0, 1, 0)
	);


    glm::mat4 PVMatrix = glm::inverse(projection_matrix * viewMatrix);

    /***** Transform spheres and materials *****/
    glm::vec4 *spheres_a = &this->spheres[0];
    glm::vec4 *materials_a = &this->materials[0];
    int nb_spheres = this->spheres.size();

    int isLight = -1;
    for (int i = 0 ; i<matIsLight.size(); i++) {
        if (matIsLight[i]){
            isLight = i;
            break;
        }
    }


    /***** Transform meshes *****/
    float *vertices_normals = (float*) malloc(
        (this->meshes_vertices.size() + this->meshes_normals.size()) * 3 * sizeof(float))
    ;

    int count = 0;
    for (int i = 0; i<this->meshes_vertices.size(); i++) {
        vertices_normals[count] = this->meshes_vertices[i].x;
        vertices_normals[count+1] = this->meshes_vertices[i].y;
        vertices_normals[count+2] = this->meshes_vertices[i].z;
        count += 3;
    }

    for (int i = 0; i<this->meshes_normals.size(); i++) {
        vertices_normals[count] = this->meshes_normals[i].x;
        vertices_normals[count+1] = this->meshes_normals[i].y;
        vertices_normals[count+2] = this->meshes_normals[i].z;
        count += 3;
    }

    /***** Create TBO for meshes *****/
    GLuint tbo_vert_norm, tbo_tex_vert_norm;

    /***** Vertices & normals *****/  
    glGenBuffers(1, &tbo_vert_norm);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_vert_norm);
    glBufferData(GL_TEXTURE_BUFFER,
    (this->meshes_vertices.size() + this->meshes_normals.size()) * 3 * sizeof(float),
    vertices_normals, GL_STATIC_DRAW);

    glGenTextures(1, &tbo_tex_vert_norm);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tbo_tex_vert_norm);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo_vert_norm);


	GLuint uniformEyePos = glGetUniformLocation(ComputeShaderProgram, "eyePos");
    GLuint uniformPV = glGetUniformLocation(ComputeShaderProgram, "PVMatrix");
    GLuint uniformSpheres = glGetUniformLocation(ComputeShaderProgram, "spheres");
	GLuint uniformMaterials = glGetUniformLocation(ComputeShaderProgram, "materials");
	GLuint uniformIsLight = glGetUniformLocation(ComputeShaderProgram, "isLight");
    GLuint u_NUM_SPHERES = glGetUniformLocation(ComputeShaderProgram, "NUM_SPHERES");
    GLuint u_NUM_VERTICES = glGetUniformLocation(ComputeShaderProgram, "NUM_VERTICES");

    glUseProgram(ComputeShaderProgram);

    GLuint u_vertices_normals = glGetUniformLocation(ComputeShaderProgram, "vertices_normals");
    glUniform1i(u_vertices_normals, 0);


    glUniformMatrix4fv(uniformPV, 1, GL_FALSE, glm::value_ptr(PVMatrix));
	glUniform3fv(uniformEyePos, 1, glm::value_ptr(eye_pos));

    glUniform1i(u_NUM_SPHERES, nb_spheres);
    glUniform1i(u_NUM_VERTICES, this->meshes_vertices.size());
    glUniform4fv(uniformSpheres, nb_spheres, glm::value_ptr(spheres_a[0]));
	glUniform4fv(uniformMaterials, this->materials.size(), glm::value_ptr(materials_a[0]));
	glUniform1i(uniformIsLight, isLight);

    glUseProgram(0);

    free(vertices_normals);

}