#include <vector>
using namespace std;
#pragma once
extern "C" class __declspec(dllexport)PMC
{
public:
    // constructor: Initialize the class and sets up the weight matrices (W), the number of neurons in each layer (d), the number of layers in the network (L), and the caches for the inputs (X) and error values (deltas).


    PMC(std::vector<int> npl);
    std::vector<float> predict(std::vector<float> inputs, bool is_classification);
    void train(std::vector<std::vector<float>> X_train, std::vector<std::vector<float>> Y_train, bool is_classification,
        float alpha , int nb_iter);
    //void train(std::vector<std::vector<float>> X_train, std::vector<std::vector<float>> Y_train, bool is_classification,
        //float alpha = 0.01, int nb_iter = 10000);


private:
    // weights
    vector<std::vector<std::vector<double>>> W;
    // number of neurons per layer
    vector<int> d;
    // number of layers
    int L;
    vector<std::vector<double>> X;
    vector<std::vector<double>> deltas;
    void _propagate(std::vector<float> inputs, bool is_classification);



};

