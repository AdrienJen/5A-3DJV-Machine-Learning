#include <vector>
using namespace std;
#pragma once
class PMC
{
public:

    PMC(const std::vector<int>& npl);
    std::vector<float> predict(const std::vector<float>& inputs, bool is_classification) const;
    void train(const std::vector<std::vector<float>>& X_train,
        const std::vector<std::vector<float>>& Y_train,
        bool is_classification,
        float alpha = 0.01,
        int nb_iter = 10000);


private:
    void _propagate(const std::vector<float>& inputs, bool is_classification);
    std::vector<std::vector<std::vector<float>>> W;
    std::vector<int> d;
    int L;
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> deltas;


};

