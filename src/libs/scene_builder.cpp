#include "../headers/scene_builder.h"

SceneBuilder::SceneBuilder(string path) {
    regex Material_reg("Material\\s+[a-z1-9]+\\s*\\{\n*[^\\}]*");
    regex Sphere_reg ("Sphere\\s*\\{\n*[^\\}]*");
    regex Camera_reg ("Camera\\s*\\{\n*[^\\}]*");


    string file = readFile(path);

    buildMaterials(matchReg(file, Material_reg));
    matchReg(file, Sphere_reg);
    matchReg(file, Camera_reg);
}

string SceneBuilder::readFile(string path) {
    string res = "";
    string line;
    ifstream myfile (path);
    if (myfile.is_open()) {
        while(getline(myfile, line)) {
            res += line + "\n";
        }
        myfile.close();
    }
    return res;
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

    /*for(string s : res)
        cout << s;*/

    return res;
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
        /******* Search name *******/
        regex name_rgx ("\\s+([a-z1-9]+) *\\{");
        smatch match;
        if(regex_search(s.begin(), s.end(), match, name_rgx))
            materials_name.push_back(match[1]);

        /******* Search light *******/
        regex light_rgx ("(light\\s*=\\s*true)");
        if(regex_search(s.begin(), s.end(), match, light_rgx))
            matIsLight.push_back(true);
        else 
            matIsLight.push_back(false);

        /******* Search color *******/
        glm::vec4 mat(-1);
        regex color_rgx ("color\\s*=\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)");
        if(regex_search(s.begin(), s.end(), match, color_rgx))
            mat = glm::vec4(stoi(match[1]), stoi(match[2]), stoi(match[3]), -1);
        else
            throwErrorMat();

        /******* Search emit/roughness *******/
        regex alpha_rgx ("(roughness|emit_intensity)\\s*=\\s*([0-9](.[0-9]+)?)");
        if(regex_search(s.begin(), s.end(), match, alpha_rgx))
            mat.w = stof(match[2]);
        else
            throwErrorMat();
        
        materials.push_back(mat);

    }

    for (int i = 0; i<materials.size(); i++)
        cout << materials[i].x << " " << materials[i].y << " " << materials[i].z << " " << materials[i].w << endl;

    for (string s : materials_name)
        cout << s << endl;
    
    for (bool b : matIsLight)
        cout << b << endl;
}
