#include <iostream>
#include <vector>
#include <string>

#include "random_forest.cpp"
#include "preprocessing.h"
#include "evaluate.h"
using namespace std;


int main2(){
    vector <vector <string> > data = read_csv("pulsar_data_train.csv");
    check_nulls(data);
    //index 8 is the target variable
    Mean method;
    fillna(data,2,method);
    fillna(data,5,method);
    fillna(data,7,method);

    cout<<"====================================="<<endl;
    check_nulls(data);
    vector<vector<float>> X;
    vector<float> y;

    for(int i = 0;i<data.size();i++){
        vector<float> x;
        for(int j=0;j<data[0].size();j++){
            if(j==8){y.push_back(stof(data[i][j]));}
            else{x.push_back(stof(data[i][j]));}
        }
        if(!x.empty()){X.push_back(x);}
    }

    auto data_splitted = train_test_split(X,y,0.2);
    vector<vector<float>> X_train = get<0> (data_splitted);
    vector<vector<float>> X_test = get<1> (data_splitted);
    vector<float> y_train = get<2> (data_splitted);
    vector<float> y_test = get<3> (data_splitted);

    cout<<"====================================="<<endl;
    RandomForestClassifier model = RandomForestClassifier(100,10);
    model.fit(X_train,y_train);
    vector<float> y_pred = model.predict(X_test);
    evaluate(y_test,y_pred);
    return 0;
}

int main()
{
    string response = "";
    while(response!="1" && response != "2"){
        cout<<"Choose the dataset:\n1.Titanic Dataset\n2.Pulsar Star Prediction Dataset\n1 or 2? :";
        cin>>response;
    }
    if(response == "2"){
        return main2();
    }
    vector <vector <string> > data = read_csv("Titanic-Dataset.csv");
    //The target variable is of index 1
    int target_variable_index = 1;
    //Lets do some data inspections

    //First we will check if the dataset is imbalanced. Maybe it needs oversampling or undersampling
    cout<<"The value counts of the target variable:"<<endl;
    check_if_imbalanced(data, target_variable_index);
    //The output was 549 examples with label 0 and 342 with 1. which is good not bad.

    cout<<"====================================="<<endl;

    //Now lets clean the data, first check the nulls
    cout<<"Speaking in code this is df.isnull().sum():"<<endl;
    check_nulls(data);
    // 177 missing value for index 5 (age) and 687 for index 10 (Cabin)

    cout<<"====================================="<<endl;

    //Fill the nulls
    Mean method;
    fillna(data, 5, method); //Fill the age column with mean
    Constant method2("U"); //Fill the missing values with 'U' for Unknown. Mode is not helpful because >50% of the values are missing
    fillna(data, 10, method2);
    cout<<"df.isnull().sum():"<<endl;
    check_nulls(data);
    cout<<"It is clean now!!"<<endl;

    cout<<"====================================="<<endl;

    //Data preprocessing and one hot encoding/ defining labels and featues.
    /*
    index 0 (ID) -> drop
    index 1 (Survived) -> the target variable, convert to float
    index 2 (Pclass) -> convert to float
    index 3 (Name) -> take only the keywords (Mr, Mrs, ..) and one hot encode after (drop_first = False)
    index 4 (Sex) -> one hot encode (drop_first = False)
    index 5 (Age) -> covert to float
    index 6 (SibSp) -> convert to float
    index 7 (Parch) -> convert to float
    index 8 (Ticket) -> drop
    index 9 (Fare) -> convert to float
    index 10 (Cabin) -> take the first char only and one hot encode
    index 11 (Embarked) -> one hot encode
    */

    auto data_preprocessed = prepare_the_titanic_dataset(data, target_variable_index);
    vector< vector<float> > X = data_preprocessed.first;
    vector<float> y = data_preprocessed.second;
    cout<<"printing X.head():"<<endl;
    for(int i = 0;i<5;i++){
        for(int j = 0;j<X[0].size();j++){
            cout<<X[i][j]<<",";
        }
        cout<<endl;
    }

    cout<<"====================================="<<endl;

    auto data_splitted = train_test_split(X,y,0.2);
    vector<vector<float>> X_train = get<0> (data_splitted);
    vector<vector<float>> X_test = get<1> (data_splitted);
    vector<float> y_train = get<2> (data_splitted);
    vector<float> y_test = get<3> (data_splitted);
    cout<<"X_train.size() = "<<X_train.size()<<endl;
    cout<<"X_test.size() = "<<X_test.size()<<endl;
    cout<<"y_train.size() = "<<y_train.size()<<endl;
    cout<<"y_test.size() = "<<y_test.size()<<endl;

    cout<<"====================================="<<endl;


    //build the model and decide on n_estimators and max_depth
    RandomForestClassifier model = RandomForestClassifier(100,10);
    model.fit(X_train,y_train);
    cout<<"====================================="<<endl;

    //Testing
    cout<<"Testing..."<<endl;
    vector<float> y_predict = model.predict(X_test);
    evaluate(y_test,y_predict);
    return 0;
}
