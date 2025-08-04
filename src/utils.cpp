#include <string>
#include <vector>
#include <set>
#include "utils.h"
using namespace std;

vector<string> split(const string &s, char val){
    string temp = "";
    vector<string> res;
    for(int i = 0;i<s.size();i++){
        if(s[i] == val){
            res.push_back(temp);
            temp = "";
            continue;
        }
        temp+=s[i];
    }
    if(temp!=""){res.push_back(temp);}
    return res;
}

string trim(const string &s, char val){
    string res = "";
    set<int> indices;
    for(int i = 0;i<s.size();i++){
        if(s[i] == ' '){
            indices.insert(i);
        }
        else{
            break;
        }
    }
    for(int i = s.size()-1;i>=0;i--){
        if(s[i] == ' '){
            indices.insert(i);
        }
        else{
            break;
        }
    }
    for(int i = 0;i<s.size();i++){
        if(indices.find(i)==indices.end()){
            res+=s[i];
        }
    }
    return res;
}
