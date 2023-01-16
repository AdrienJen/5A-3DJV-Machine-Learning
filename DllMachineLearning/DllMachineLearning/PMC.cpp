#include "pch.h"
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "PMC.h"
#include <array>

// PMC class takes a single argument and a vector of integers
PMC::PMC(std::vector<int> npl) {
    // Vector 
    d = npl;
    // Size of vector 
    L = npl.size() - 1;

    // Initialize weight matrices
    // Loops through each layer in the network
    for (int l = 0; l < d.size(); l++) 
    {
        //Vector
        std::vector<std::vector<double>> layer_weights;
        // If current layer is in the input layer,loop skips
        if (l == 0) continue;
        // For each pair of neurons, push a random weight between -1 and 1 to the neuron_weights vector
        for (int i = 0; i < d[l - 1] + 1; i++) 
        {
            std::vector<double> neuron_weights;
            // For each neuron in the current layer, neuron_weights vector pushed to the layer_weights vector
            for (int j = 0; j < d[l] + 1; j++) 
            {
                neuron_weights.push_back((j == 0) ? 0.0 : (rand() / (double)RAND_MAX) * 2 - 1);
            }
            layer_weights.push_back(neuron_weights);
        }
        // Layer_weight pushed to W variable
        W.push_back(layer_weights);
    }

    // Initialize X and deltas
    // Loops through each layer in the network
    // For each layer, initializes a vector layer_X and a vector layer_deltas
    for (int l = 0; l < d.size(); l++) 
    {
        std::vector<double> layer_X, layer_deltas;

        //For each neuron in current layer,pushes 0 to the layer_deltas vector, 
        //push 1 if neuron is bias neuron else 0 to the layer_X vector
        
        for (int j = 0; j < d[l] + 1; j++) 
        {
            layer_deltas.push_back(0.0);
            layer_X.push_back((j == 0) ? 1.0 : 0.0);
        }
        //Push
        X.push_back(layer_X);
        deltas.push_back(layer_deltas);
    }


}
//Function propagate take two inputs, one vector float and one boolean
void PMC::_propagate(std::vector<float> inputs, bool is_classification)
{
    // For each layer, iterate through each neuron and calculate the total input for that neuron
    for (int j = 1; j < d[0] + 1; j++) 
    {
        X[0][j] = inputs[j - 1];
    }
    for (int l = 1; l < d.size(); l++) 
    {
        for (int j = 1; j < d[l] + 1; j++) 
        {
            float total = 0;
            for (int i = 0; i < d[l - 1] + 1; i++) 
            {
                total += W[l][i][j] * X[l - 1][i];
            }
            X[l][j] = total;
            //If is_classification or current layer is not the last layer
            if (is_classification || l < L) 
            {
                //neuron's value is passed to the function (tanh)
                X[l][j] = tanh(total);
                //tanh hyperbolic tangent
            }
        }
    }
}
std::vector<float> PMC::predict(std::vector<float> inputs, bool is_classification) 
{
    _propagate(inputs, is_classification);
    std::vector<float> output(X[L].begin() + 1, X[L].end());
    return output;
}

//Function Train  to train the PMC
// takes 5 arguments : X_train vector of input training, Y_train vector of output training, 
//is_classification boolean check if used for classfication or regression, alpha learning rate of the model
//nb_itter number of iteration

void PMC::train(std::vector<std::vector<float>> X_train, std::vector<std::vector<float>> Y_train, bool is_classification, float alpha, int nb_iter)
{
    // train loop of nb_iter itérations
    for (int it = 0; it < nb_iter; it++) 
    {
        // take random exemple in X_train
        int k = std::rand() % X_train.size();
        std::vector<float> Xk = X_train[k];
        std::vector<float> Yk = Y_train[k];

        // Propagate the selected example
        this->_propagate(Xk, is_classification);
        for (int j = 1; j < d[L] + 1; j++) 
        {
            deltas[L][j] = X[L][j] - Yk[j - 1];
            if (is_classification) 
            {
                deltas[L][j] = deltas[L][j] * (1 - X[L][j] * X[L][j]);
            }
        }

        for (int l = L - 1; l >= 1; l--) 
        {
            for (int i = 1; i < d[l - 1] + 1; i++) 
            {
                float total = 0.0;
                for (int j = 1; j < d[l] + 1; j++) 
                {
                    total += W[l][i][j] * deltas[l][j];
                }
                deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * total;
            }
        }

        for (int l = 1; l < d.size(); l++) 
        {
            for (int i = 0; i < d[l - 1] + 1; i++) 
            {
                for (int j = 1; j < d[l] + 1; j++) 
                {
                    W[l][i][j] += -alpha * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
}

void PMCNoLayer(std::vector<float> classes, std::vector<std::array<float, 2>> points)
{
    //PMC test without hidden layer on the linear model dataset
   // Creating an instance of the PMC class with 2 input neurons and 1 output neuron

    PMC model = PMC({ 2, 1 });

    // Creating empty lists for test points and colors
    std::vector<std::vector<float>> test_points;
    std::vector<std::string> test_colors;

    // Iterating through a 2D grid of points
    for (float row = 0; row < 300; row++) 
    {
        for (float col = 0; col < 300; col++) 
        {
            // Creating a point with x and y coordinates
            std::vector<float> p = { col / 100, row / 100 };

            // Getting the prediction of the model for the current point
            std::vector<float> prediction = model.predict(p, true);
            std::string c;

            // Assigning a color to the point based on the prediction
            if (prediction[0] >= 0) 
            {
                c = "lightcyan";//test
            }
            else 
            {
                c = "pink";//test
            }
            test_points.push_back(p);
            test_colors.push_back(c);
        }
    }
    // Training the model with the points and classes
    std::vector<std::vector<float>> Y_train;
    for (auto c : classes) 
    {
        Y_train.push_back({ c });
    }
    // model.train(points, std::vector<std::vector<float>>({ classes.begin(), classes.end() }), true);
     //not working

     // Iterating through a 2D grid of points
    for (float row = 0; row < 300; row++) 
    {
        for (float col = 0; col < 300; col++) 
        {
            // Creating a point with x and y coordinates
            std::vector<float> p = { col / 100, row / 100 };

            // Getting the prediction of the model for the current point
            std::vector<float> prediction = model.predict(p, true);
            std::string c;

            // Assigning a color to the point based on the prediction
            if (prediction[0] >= 0) 
            {
                c = "lightcyan";//test
            }
            else 
            {
                c = "pink";//test
            }
            test_points.push_back(p);
            test_colors.push_back(c);
        }
    }
}
void PMCXOR() {
 

    // Initialize data
    std::vector<std::array<float, 2>> xor_points = { {0, 0}, {1, 1}, {0, 1}, {1, 0} };
    std::vector<std::vector<float>> xor_classes = { {-1}, {-1}, {1}, {1} };


    // Create a PMC model with 2 input, 2 hidden, and 1 output layers
    PMC model = PMC({ 2, 2, 1 });

    // Train the model
   // model.train(xor_points, xor_classes, true);
    //problem with train

    std::vector<std::array<float, 2>> test_points;
    std::vector<std::string> test_colors;
    for (int row = 0; row < 300; row++) 
    {
        for (int col = 0; col < 300; col++) 
        {
            std::array<float, 2> p = { (col / 100.0f) - 1, (row / 100.0f) - 1 };
            //std::string c = model.predict(p, true)[0] >= 0 ? "lightcyan" : "pink";
            test_points.push_back(p);
            //test_colors.push_back(c);
        }
    }
    std::vector<std::array<float, 2>> xor_points = { {0, 0}, {1, 1}, {0, 1}, {1, 0} };
    std::vector<std::vector<float>> xor_classes = { {-1}, {-1}, {1}, {1} };
    PMC model = PMC({ 2, 2, 1 });

    // Train the model on the xor_points and xor_classes data with 100000 iterations and a learning rate of 0.01
    //model.train(xor_points, xor_classes, true, 100000, 0.01);*
    //pb with train

    std::vector<std::array<float, 2>> test_points;
    std::vector<std::string> test_colors;

    for (int row = 0; row < 300; row++) 
    {
        for (int col = 0; col < 300; col++) 
        {
            std::array<float, 2> p = { col / 100 - 1, row / 100 - 1 };
            //std::string c = model.predict(p, true)[0] >= 0 ? "lightcyan" : "pink";
            test_points.push_back(p);
            //test_colors.push_back(c);
        }
    }
}

void train_linear_function(PMC& model, std::vector<std::vector<float>> X_train, std::vector<std::vector<float>> Y_train, float alpha, int nb_iter) {
    // linear function with one input and one output
    model = PMC({ 1, 1 });
    // Train the model with provided data
    model.train(X_train, Y_train, false, alpha, nb_iter);
}
