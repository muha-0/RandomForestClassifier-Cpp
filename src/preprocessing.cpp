#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <cmath>
#include <tuple>
#include <algorithm>
#include "utils.h"
#include "random_utils.h"
#include "preprocessing.h"

using namespace std;

vector< vector<string> > read_csv(const string &filename){
    ifstream file(filename);  // path to your CSV file
    string line;
    vector< vector<string> > data;

    bool first_row = true;
    while (getline(file, line)) {
        if(first_row){
            first_row = false;
            continue;
        }
        stringstream ss(line);
        string cell;
        vector<string> row;

        // This function is not very general. It is tailored for the titanic dataset
        string temp = "";
        while (getline(ss, cell, ',')) {
            if(cell[0] == '"'){
                temp = cell;
                continue;
            }
            cell+=temp;
            temp = "";
            row.push_back(cell);
        }


        //Print the row
        /*for (const string& val : row) {
            cout << val << " ";
        }
        cout<<endl;*/
        data.push_back(row);

    }
    return data;
}

void check_if_imbalanced(const vector< vector<string> > &data, int index){
    unordered_map<string, int> mp;
    for(auto& example: data){
        if(mp.find(example[index]) == mp.end()){
            mp.insert({example[index],0});
        }
        mp[example[index]] += 1;
    }

    for(auto& pairr: mp){
        cout<<pairr.first<<": "<<pairr.second<<endl;
    }
}

void check_nulls(const vector<vector<string>> &data){
    int cnt = 0;
    for(int j = 0;j<data[0].size();j++){
        cout<<"Index "<<j<<": ";
        cnt = 0;
        for(int i = 0;i<data.size();i++){
            if(data[i][j] == ""){
                cnt++;
            }
        }
        cout<<cnt<<endl;
    }
}




void Mean::fillna(vector< vector<string> > &data, int index) {
    //calculate the mean first
    int m = 0;
    float sum = 0;
    vector<int> indices;
    for(int i = 0;i<data.size();i++){
        if(data[i][index] == ""){
            indices.push_back(i);
            continue;
        }
        sum+= stof(data[i][index]);
        m++;
    }
    if(m==0){return;}
    float mean = sum/m;

    //fill the missing values
    for(auto& i: indices){
        data[i][index] = to_string(mean);
    }
}



void Mode::fillna(vector< vector<string> > &data, int index) {
    //calculate the mode first
    unordered_map<string, int> freq;
    vector<int> indices;
    for(int i = 0;i<data.size();i++){
        if(data[i][index] == ""){
            indices.push_back(i);
            continue;
        }
        else if(freq.find(data[i][index]) == freq.end()){
            freq[data[i][index]] = 0;
        }
        freq[data[i][index]] ++;
    }
    if(freq.empty()){return;}
    int maxi = -1;
    string mode = "";
    for(auto& [key,value]: freq){
        if(value>maxi){
            maxi = value;
            mode = key;
        }
    }
    //fill the missing values
    for(auto& i: indices){
        data[i][index] = mode;
    }
}




Constant::Constant(const string &fill_value){
    this->fill_value = fill_value;
}
void Constant::fillna(vector<vector<string>>& data, int index) {
    for (int i = 0; i < data.size(); i++) {
        if (data[i][index] == "") {
            data[i][index] = fill_value;
        }
    }
}



void fillna(vector<vector<string>> &data, int index, Ifillna &method){
    method.fillna(data,index);
}

pair<vector< vector<float> >, vector<float>> prepare_the_titanic_dataset(vector< vector<string> > &data, int target_variable_index){
    vector<float> y;
    for(int i = 0;i<data.size();i++){
        y.push_back(stof(data[i][target_variable_index]));
    }

    vector< vector<float> > X;
    vector<float> x;
    //preprocess some columns first
    for(int i = 0;i<data.size();i++){
        for(int j = 0;j<data[i].size();j++){
            switch(j){
                case 3:
                    data[i][j] = trim(split(data[i][j], '.')[0], ' ');
                    break;
                case 10:
                    data[i][j] = data[i][j][0];
                    break;
                default:
                    break;
            }
        }
    }

    //Lets do one hot encoding and convert all to float

    unordered_map<int, set<string> > index_to_set;
    vector<int> indices_to_one_hot_encode = {3,10,11,4};

    for(auto &j: indices_to_one_hot_encode){
        index_to_set[j] = {};
        for(int i = 0;i<data.size();i++){
            index_to_set[j].insert(data[i][j]);
        }

    }



    for(int i = 0;i<data.size();i++){
        x.clear();
        for(int j = 0;j<data[0].size();j++){
            switch(j){
                case 2:
                case 5:
                case 6:
                case 7:
                case 9:
                    x.push_back(stof(data[i][j]));
                    break;
                case 3:
                case 10:
                case 11:
                case 4:
                    for(auto& val: index_to_set[j]){
                        if(data[i][j]==val){
                            x.push_back(1);
                        }
                        else{
                            x.push_back(0);
                        }
                    }
                    break;
                default:
                    //drop
                    break;
            }

        }
        X.push_back(x);
    }
    if(X.size() != y.size()){
        cout<<"Data Preparation Failed"<<endl;
    }
    return make_pair(X,y);
}

tuple<vector<vector<float>>, vector<vector<float>>, vector<float>, vector<float>> train_test_split(const vector<vector<float>> &X, const vector<float> &y, float test_size){
    vector<float> y_train;
    vector<float> y_test;
    vector<vector<float>> X_train;
    vector<vector<float>> X_test;


    vector<int> indices;
    for(int i = 0;i<X.size();i++){
        indices.push_back(i);
    }
    shuffle(indices.begin(), indices.end(), gen);
    int sz = floor(X.size()*test_size);
    for(int i = 0;i<X.size();i++){
        if(i<sz){
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }
        else{
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
    return make_tuple(X_train, X_test, y_train, y_test);
}
