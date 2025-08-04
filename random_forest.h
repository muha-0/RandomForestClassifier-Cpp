#pragma once
#include <vector>

class RandomForestClassifier {
public:
    // Constructor: specify number of trees and maximum tree depth
    RandomForestClassifier(int n_estimators, int max_depth);

    // Train the model on feature matrix X and labels y
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y);

    // Predict the class labels for unseen data X
    std::vector<float> predict(const std::vector<std::vector<float>>& X);

    // Destructor: cleans up dynamically allocated trees
    ~RandomForestClassifier();
};

