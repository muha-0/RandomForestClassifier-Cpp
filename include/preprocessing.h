#pragma once
#include <vector>
#include <string>
#include <tuple>

// === CSV Processing ===
std::vector<std::vector<std::string>> read_csv(const std::string& filename);

// === Null Checks and Imbalance Inspection ===
void check_nulls(const std::vector<std::vector<std::string>>& data);
void check_if_imbalanced(const std::vector<std::vector<std::string>>& data, int index);

// === Missing Value Imputation ===
class Ifillna {
public:
    virtual void fillna(std::vector<std::vector<std::string>>& data, int index) = 0;
    virtual ~Ifillna() {}
};

class Mean : public Ifillna {
public:
    void fillna(std::vector<std::vector<std::string>>& data, int index) override;
};

class Mode : public Ifillna {
public:
    void fillna(std::vector<std::vector<std::string>>& data, int index) override;
};

class Constant : public Ifillna {
private:
    std::string fill_value;
public:
    Constant(const std::string& fill_value);
    void fillna(std::vector<std::vector<std::string>>& data, int index) override;
};

void fillna(std::vector<std::vector<std::string>>& data, int index, Ifillna& method);

// === Dataset Specific Preprocessing ===
std::pair<std::vector<std::vector<float>>, std::vector<float>>
prepare_the_titanic_dataset(std::vector<std::vector<std::string>>& data, int target_variable_index);

// === Train/Test Split ===
std::tuple<
    std::vector<std::vector<float>>,
    std::vector<std::vector<float>>,
    std::vector<float>,
    std::vector<float>
> train_test_split(const std::vector<std::vector<float>>& X, const std::vector<float>& y, float test_size);

