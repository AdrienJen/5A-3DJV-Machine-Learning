#include "PMC.h"
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "PMC.h"
#include <array>

PMC::PMC(const std::vector<int>& npl) : d(npl), L(npl.size() - 1) {
    // Initialisation des W
    for (int l = 0; l < d.size(); l++) {
        std::vector<std::vector<float>> layer_weights;
        if (l == 0) {
            continue;
        }
        for (int i = 0; i <= d[l - 1]; i++) {
            std::vector<float> node_weights;
            for (int j = 0; j <= d[l]; j++) {
                node_weights.push_back(j == 0 ? 0.0 : ((float)rand() / RAND_MAX) * 2 - 1);
            }
            layer_weights.push_back(node_weights);
        }
        W.push_back(layer_weights);
    }

    // Initialisation des X et des deltas
    for (int l = 0; l < d.size(); l++) {
        std::vector<float> layer_X, layer_deltas;
        for (int j = 0; j <= d[l]; j++) {
            layer_X.push_back(j == 0 ? 1.0 : 0.0);
            layer_deltas.push_back(0.0);
        }
        X.push_back(layer_X);
        deltas.push_back(layer_deltas);
    }
}


void PMC::_propagate(const std::vector<float>& inputs, bool is_classification) {
    for (int j = 1; j <= d[0]; j++) {
        X[0][j] = inputs[j - 1];
    }

    for (int l = 1; l < d.size(); l++) {
        for (int j = 1; j <= d[l]; j++) {
            float total = 0;
            for (int i = 0; i <= d[l - 1]; i++) {
                total += W[l][i][j] * X[l - 1][i];
            }
            X[l][j] = total;
            if (is_classification || l < L) {
                X[l][j] = tanh(total);
            }
        }
    }
}

std::vector<float> PMC::predict(const std::vector<float>& inputs, bool is_classification) const {
    // Appelle la fonction _propagate avec les entrées données et le booléen is_classification
    const_cast<PMC*>(this)->_propagate(inputs, is_classification);

    // Copie les valeurs de la dernière couche X[L][1:] dans un vecteur de sortie
    std::vector<float> output;
    for (int i = 1; i < this->X[this->L].size(); i++) {
        output.push_back(this->X[this->L][i]);
    }
    return output;
}






void PMC::train(const std::vector<std::vector<float>>& X_train,
    const std::vector<std::vector<float>>& Y_train,
    bool is_classification,
    float alpha,
    int nb_iter) {
    for (int it = 0; it < nb_iter; ++it) {
        int k = rand() % X_train.size();
        auto Xk = X_train[k];
        auto Yk = Y_train[k];

        _propagate(Xk, is_classification);
        for (int j = 1; j <= d[L]; ++j) {
            deltas[L][j] = X[L][j] - Yk[j - 1];
            if (is_classification) {
                deltas[L][j] *= (1 - X[L][j] * X[L][j]);
            }
        }

        for (int l = L - 1; l >= 1; --l) {
            for (int i = 1; i <= d[l - 1]; ++i) {
                float total = 0.0;
                for (int j = 1; j <= d[l]; ++j) {
                    total += W[l][i][j] * deltas[l][j];
                }
                deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * total;
            }
        }

        for (int l = 1; l < d.size(); ++l) {
            for (int i = 0; i <= d[l - 1]; ++i) {
                for (int j = 1; j <= d[l]; ++j) {
                    W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
}
