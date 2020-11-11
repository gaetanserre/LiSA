#include <regex>
#include <iostream>
#include <fstream>
using namespace std;

string readFile(string path) {
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

int main()
{   
    string s = readFile("scene.rto");
    regex rgx("Material\\s+[a-z]+\\s*\\{\n*[^\\}]*");
    std::smatch match;

    string::const_iterator searchStart( s.cbegin() );

    while ( regex_search( searchStart, s.cend(), match, rgx ) )
    {
        cout << match[0] << "." << endl;
        searchStart = match.suffix().first;
    }
}
