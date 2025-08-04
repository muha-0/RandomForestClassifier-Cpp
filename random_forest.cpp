#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <random>
#include "random_utils.h"


using namespace std;

class BT{
private:
    BT* left = nullptr;
    BT* right = nullptr;
    tuple<int,bool,float> indicator;
    float major_class;

    float compute_entropy(const vector<float> &y, const vector<int> &indices){
        if(indices.size()<1){
            return 0.0f;
        }
        map<float, int> freq;
        for(auto& i:indices){
            freq[y[i]]++;
        }
        float entropy = 0.0f;
        for(auto& [key,value]: freq){
            float p = value / float(indices.size());

            if(p>0.0f){
               entropy += -p * log2(p);
            }

        }

        return entropy;
    }
public:
    BT(){

    }
    float predict(const vector<float> &x){
        int index = get<0> (indicator);
        bool is_binary = get<1> (indicator);
        if(is_binary){
            if(x[index] == 1.0){
                if(!right){
                    return major_class;
                }
                else{return right->predict(x);}
            }
            else{
                if(!left){
                    return major_class;
                }
                else{return left->predict(x);}
            }
        }
        else{
            float val = get<2> (indicator);
            if(x[index] > val){
                if(!right){
                    return major_class;
                }
                else{return right->predict(x);}
            }
            else{
                if(!left){
                    return major_class;
                }
                else{return left->predict(x);}
            }
        }
    }
    void fit(const vector<vector<float>> &X, const vector<float> &y, const vector<int>* indices, const vector<bool> &is_binary, int max_depth){
        //first we need to set the major class
        unordered_map<float, int>* freq = new unordered_map<float, int>();
        for(auto& i: *indices){
            (*freq)[y[i]]++;
        }
        if(freq->size()==1){
            //purely one class
            major_class = freq->begin()->first;
            delete freq;
            return;
        }
        int maxi = -1;
        for(auto& [key,value]:*freq){
            if(value > maxi){
                maxi = value;
                major_class = key;
            }
        }
        delete freq;
        if(max_depth<=0){
            return;
        }


        //Now decide on the feature subset we would use randomly
        vector<int>* features_indices = new vector<int>();
        vector<int>* v = new vector<int>();
        for(int i = 0;i<X[0].size();i++){
            v->push_back(i);
        }


        shuffle(v->begin(),v->end(),gen);
        int num_features = min(static_cast<int>(X[0].size()), static_cast<int>(sqrt(X[0].size())));
        for(int i = 0;i<num_features;i++){
            features_indices->push_back((*v)[i]);
        }
        delete v;
        //Continue and do split search
        int index_to_split = -1;
        float value_if_continious=0.0f;
        float best_info_gain = 0.0f;
        float cur_entropy = compute_entropy(y,*indices);
        for(auto& j:*features_indices){
            vector<int> left_indices;
            vector<int> right_indices;
            vector<int> best_left_indices;
            vector<int> best_right_indices;
            if(is_binary[j]){

                for(auto& i:*indices){
                    if(X[i][j] == 0.0f){
                        left_indices.push_back(i);
                    }
                    else{
                        right_indices.push_back(i);
                    }
                }
            }

            else {

                vector<pair<float, int>> value_index_pairs;
                for (auto& i : *indices) {
                    value_index_pairs.emplace_back(X[i][j], i);
                }

                // Sort by feature value
                sort(value_index_pairs.begin(), value_index_pairs.end());

                int num_candidates = 5;
                int step = max(1, int(value_index_pairs.size() / (num_candidates + 1)));

                for (int k = 1; k <= num_candidates; ++k) {
                    int split_pos = k * step;
                    if (split_pos >= value_index_pairs.size()) break;

                    float threshold = (value_index_pairs[split_pos - 1].first + value_index_pairs[split_pos].first) / 2.0f;

                    left_indices.clear();
                    right_indices.clear();

                    for (auto& [val, idx] : value_index_pairs) {
                        if (val <= threshold)
                            left_indices.push_back(idx);
                        else
                            right_indices.push_back(idx);
                    }

                    float pL = left_indices.size() / float(indices->size());
                    float pR = right_indices.size() / float(indices->size());

                    float info_gain = cur_entropy - (
                        pL * compute_entropy(y, left_indices) +
                        pR * compute_entropy(y, right_indices)
                    );

                    if (info_gain > best_info_gain) {
                        best_info_gain = info_gain;
                        index_to_split = j;
                        value_if_continious = threshold; // record threshold
                        best_left_indices = left_indices;
                        best_right_indices = right_indices;
                    }
                }
            }
            if(!is_binary[j]){left_indices = best_left_indices; right_indices = best_right_indices;}
            float pL = left_indices.size() / float(indices->size());
            float pR = right_indices.size() / float(indices->size());
            float info_gain = cur_entropy - (pL * compute_entropy(y,left_indices) + pR * compute_entropy(y,right_indices));

            if(info_gain>=best_info_gain){

                best_info_gain = info_gain;
                index_to_split = j;
            }
        }

        indicator = make_tuple(index_to_split, is_binary[index_to_split], value_if_continious);

        delete features_indices;
        vector<int>* left_indices = new vector<int>();
        vector<int>* right_indices = new vector<int>();

        if(is_binary[index_to_split]){
            for(auto& i:*indices){
                if(X[i][index_to_split] == 0.0f){
                    left_indices->push_back(i);
                }
                else{
                    right_indices->push_back(i);
                }
            }
        }

        else {
            for (auto& i : *indices) {
                if (X[i][index_to_split] <= value_if_continious)
                    left_indices->push_back(i);
                else
                    right_indices->push_back(i);
            }
        }

        left = new BT();
        right = new BT();
        if(left_indices->size()>1)
            left->fit(X,y,left_indices,is_binary,max_depth-1);
        if(right_indices->size()>1)
            right->fit(X,y,right_indices, is_binary,max_depth-1);

        delete left_indices;
        delete right_indices;

        return;

    }
    ~BT(){
        delete left;
        delete right;
    }
};

class RandomForestClassifier{
private:
    int n_estimators;
    int max_depth;
    vector<BT*> tree_ensemble;

    pair< vector<vector<float>>, vector<float>> sampling_with_replacement(const vector<vector<float>> &X, const vector<float> &y){
        int sz = X.size();
        vector<vector<float>> res_X;
        vector<float> res_y;
        std::uniform_int_distribution<> distrib(0, sz - 1);
        int random_index;
        while(sz--){
            random_index = distrib(gen);
            res_X.push_back(X[random_index]);
            res_y.push_back(y[random_index]);
        }
        return make_pair(res_X,res_y);
    }
    vector<bool> compute_is_binary(const vector<vector<float>> &X){
        int num_features = X[0].size();
        vector<bool> is_binary(num_features, false);
        for (int f = 0; f < num_features; f++) {
            unordered_set<float> unique_vals;
            for (int i = 0; i < X.size(); i++) {
                unique_vals.insert(X[i][f]);
                if (unique_vals.size() > 2) break;
            }
            is_binary[f] = unique_vals.size() == 2 && unique_vals.count(0.0f) && unique_vals.count(1.0f);
        }
        return is_binary;
    }
public:
    RandomForestClassifier(int n_estimators, int max_depth){
        this->n_estimators = max(n_estimators,1);
        this->max_depth= max(max_depth,1);
    }

    void fit(const vector<vector<float>> &X, const vector<float> &y){
        //Get a vector of all indices [0,n-1]
        vector<int> indices;
        for(int i = 0;i<X.size();i++){
            indices.push_back(i);
        }

        for(int i = 0;i<n_estimators;i++){
            auto sampled_dataset = sampling_with_replacement(X,y);
            vector<vector<float>> sampled_dataset_X = sampled_dataset.first;
            vector<float> sampled_dataset_y = sampled_dataset.second;
            vector<bool> is_binary = compute_is_binary(X);
            BT* tree = new BT();

            tree->fit(sampled_dataset_X, sampled_dataset_y, &indices, is_binary,max_depth);
            cout<<"Trained Tree number: "<<i<<endl;
            tree_ensemble.push_back(tree);
        }
    }
    vector<float> predict(const vector<vector<float>> &X){
        //Run predict on all trees and get them to vote
        unordered_map<float, int> predictions;
        vector<float> res;
        for(int i = 0;i<X.size();i++){
            predictions.clear();
            for(int j = 0;j<n_estimators;j++){
                float estimation = tree_ensemble[j]->predict(X[i]);
                predictions[estimation]++;
            }
            int maxi = -1;
            float winner = -1;
            for(const auto& [key,value]: predictions){
                if(value>maxi){
                    maxi = value;
                    winner = key;
                }
            }
            res.push_back(winner);
        }
        return res;
    }
    ~RandomForestClassifier(){
        for(auto& ptr: tree_ensemble){
            delete ptr;
        }
    }
};
