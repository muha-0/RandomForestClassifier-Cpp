#pragma once
#include <vector>

// Evaluates classification performance using precision, recall, F1, and accuracy.
// Accepts predicted labels and true labels (float encoded).
void evaluate(const std::vector<float>& y_test, const std::vector<float>& y_pred);

