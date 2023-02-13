#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>

// Fonction pour séparer les données en plusieurs morceaux
std::vector<std::string> split(const std::string &line, char delimiter) {
    std::vector<std::string> split_line;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        split_line.push_back(item);
    }
    return split_line;
}

int main() {
    // Chargement des données à partir du fichier
    std::ifstream file("country-flag-database-from-wikipedia.csv");
    std::string line;
    std::getline(file, line); // Ignorer la première ligne qui contient les en-têtes
    std::vector<std::vector<double>> data;
    std::vector<std::string> labels;
    std::unordered_map<std::string, int> label_mapping;
    int label_count = 0;
    while (std::getline(file, line)) {
        auto split_line = split(line, ',');
        // Convertissez les données en double
        std::vector<double> row;
        for (int i = 0; i < split_line.size() - 1; ++i) {
            row.push_back(std::stod(split_line[i]));
        }
        data.push_back(row);
        // Ajouter l'étiquette à la liste des étiquettes
        const auto &label = split_line.back();
        if (label_mapping.count(label) == 0) {
            label_mapping[label] = label_count;
            ++label_count;
        }
        labels.push_back(label);
    }

    // Séparer les données en entrées et sorties
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;
    for (int i = 0; i < data.size(); ++i) {
        X.push_back(data[i]);
        y.push_back(std::vector<double>(label_count, 0));
        y.back()[label_mapping[labels[i]]] = 1;
    }

    // Initialiser les poids aléatoirement
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    std::vector<double> weights(X[0].size());
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = distribution(generator);
    }

    // Définir les hyperparamètres
    int num_iterations = 1000;
    double learning_rate = 0.01;

    // Boucle d'entraînement
    for (int i = 0; i < num_iterations; ++i) {
        // Calculer les prédictions
        std::vector<std::vector<double>> predictions;
        for (const auto &x : X) {
            double dot_product = 0;
            for (int j = 0; j < x.size(); ++j) {
                dot_product += x[j] * weights[j];
            }
            predictions.push_back({1.0 / (1.0 + std::exp(-dot_product))});
        }

        // Calculer les erreurs
        std::vector<std::vector<double>> errors;
        for (int j = 0; j < predictions.size(); ++j) {
            std::vector<double> prediction_error;
            for (int k = 0; k < predictions[j].size(); ++k) {
                prediction_error.push_back(predictions[j][k] - y[j][k]);
            }
            errors.push_back(prediction_error);
        }

        // Calculer les gradients pour chaque poids
        std::vector<double> gradients(weights.size());
        for (int j = 0; j < weights.size(); ++j) {
            for (int k = 0; k < X.size(); ++k) {
                for (int l = 0; l < errors[k].size(); ++l) {
                    gradients[j] += errors[k][l] * X[k][j];
                }
            }
        }

        // Mettre à jour les poids
        for (int j = 0; j < weights.size(); ++j) {
            weights[j] -= learning_rate * gradients[j];
        }
    }

    // Afficher les poids finaux
    for (const auto &weight : weights) {
        std::cout << weight << std::endl;
    }

    return 0;
}

