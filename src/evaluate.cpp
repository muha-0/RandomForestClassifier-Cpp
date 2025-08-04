#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include "evaluate.h"
using namespace std;

void evaluate(const vector<float> &y_test, const vector<float> &y_pred) {
    int correct = 0;
    set<float> unique_classes(y_test.begin(), y_test.end());

    unordered_map<float, int> TP, FP, FN;

    for (int i = 0; i < y_pred.size(); i++) {
        float true_label = y_test[i];
        float pred_label = y_pred[i];

        if (true_label == pred_label) {
            TP[true_label]++;
            correct++;
        } else {
            FP[pred_label]++;
            FN[true_label]++;
        }
    }

    float accuracy = (correct * 1.0f) / y_pred.size();
    cout << "Accuracy: " << accuracy << endl << endl;

    for (float cls : unique_classes) {
        int tp = TP[cls];
        int fp = FP[cls];
        int fn = FN[cls];

        float precision = (tp + fp) == 0 ? 0.0f : tp / float(tp + fp);
        float recall = (tp + fn) == 0 ? 0.0f : tp / float(tp + fn);
        float f1 = (precision + recall == 0.0f) ? 0.0f : 2 * precision * recall / (precision + recall);
        cout << "Class " << cls << ":" << endl;
        cout << "  Precision: " << precision << endl;
        cout << "  Recall: " << recall << endl;
        cout << "  F1 Score: " << f1 << endl << endl;
        cout << endl;
    }
}

