#include "RBF.h"
#include "RBF.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
using namespace std;
// Constructeur
RBF::RBF(int nInputs, int nOutputs, int nCenters, double spread)
{
    this->nInputs = nInputs;
    this->nOutputs = nOutputs;
    this->nCenters = nCenters;
    this->spread = spread;
    this->weights = vector<double>(nCenters * nOutputs);
    this->centers = vector<vector<double> >(nCenters, vector<double>(nInputs));
}

// Fonction pour initialiser les centres
void RBF::initCenters(vector<vector<double> > inputData)
{
    for (int i = 0; i < nCenters; i++)
    {
        int randomIndex = rand() % inputData.size();
        centers[i] = inputData[randomIndex];
    }
}

// Fonction pour calculer les activations des neurones cachés
vector<double> RBF::computeHiddenActivations(vector<double> input)
{
    vector<double> activations(nCenters);
    for (int i = 0; i < nCenters; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < nInputs; j++)
        {
            double delta = input[j] - centers[i][j];
            sum += delta * delta;
        }
        activations[i] = exp(-sum / (2 * spread * spread));
    }
    return activations;
}

// Fonction pour prédire la sortie à partir des activations des neurones cachés
vector<double> RBF::predict(vector<double> input)
{
    vector<double> hiddenActivations = computeHiddenActivations(input);
    vector<double> output(nOutputs);
    for (int i = 0; i < nOutputs; i++)
    {
        output[i] = 0.0;
        for (int j = 0; j < nCenters; j++)
        {
            output[i] += weights[j + i * nCenters] * hiddenActivations[j];
        }
    }
    return output;
}
// Fonction pour entraîner le réseau en utilisant l'algorithme de moindres carrés
void RBF::train(vector<vector<double> > inputData, vector<vector<double> > targetData, int nEpochs, double learningRate)
{
    initCenters(inputData);
    for (int epoch = 0; epoch < nEpochs; epoch++)
    {
        for (int i = 0; i < inputData.size(); i++)
        {
            vector<double> hiddenActivations = computeHiddenActivations(inputData[i]);
            vector<double> prediction = predict(inputData[i]);
            // Calculer les erreurs pour chaque sortie
            vector<double> errors(nOutputs);
            for (int j = 0; j < nOutputs; j++)
            {
                errors[j] = targetData[i][j] - prediction[j];
            }

            // Mettre à jour les poids
            for (int j = 0; j < nCenters; j++)
            {
                for (int k = 0; k < nOutputs; k++)
                {
                    weights[j + k * nCenters] += learningRate * errors[k] * hiddenActivations[j];
                }
            }
        }
    }
}
/*
* int main()
{
    int nInputs = 1;
    int nOutputs = 1;
    int nCenters = 5;
    double spread = 0.1;
    RBF network(nInputs, nOutputs, nCenters, spread);

    // Génération de données d'entraînement
    int nSamples = 100;
    std::vector<std::vector<double> > inputData(nSamples, std::vector<double>(nInputs));
    std::vector<std::vector<double> > targetData(nSamples, std::vector<double>(nOutputs));
    std::mt19937 generator(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nSamples; i++)
    {
        inputData[i][0] = dist(generator);
        targetData[i][0] = 2 * inputData[i][0] + 1;
    }

    // Entraîner le réseau
    int nEpochs = 1000;
    double learningRate = 0.01;
    network.train(inputData, targetData, nEpochs, learningRate);

    // Tester le réseau sur des données de test
    int nTestSamples = 100;
    std::vector<std::vector<double> > testInputData(nTestSamples, std::vector<double>(nInputs));
    for (int i = 0; i < nTestSamples; i++)
    {
        testInputData[i][0] = dist(generator);
    }
    double totalError = 0.0;
    for (int i = 0; i < nTestSamples; i++)
    {
        std::vector<double> prediction = network.predict(testInputData[i]);
        double target = 2 * testInputData[i][0] + 1;
        double error = std::abs(target - prediction[0]);
        totalError += error;
    }
    double avgError = totalError / nTestSamples;
    std::cout << "Average error: " << avgError << std::endl;

    return 0;
}
*/

