// FlagMachineLearning.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include "PMC.h"

int main()
{
    // Définition des entrées et des sorties pour XOR
    std::vector<std::vector<float>> X_train = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<float>> Y_train = { {0}, {1}, {1}, {0} };

    // Définition de la structure de la PMC (2 entrées, 1 sortie)
    std::vector<int> npl = { 2, 3, 1 };

    // Création de la PMC
    PMC pmc(npl);

    // Entraînement de la PMC avec un nombre d'itérations, une alpha et en utilisant une classification
    int nb_iter = 10000;
    float alpha = 0.1;
    bool is_classification = false;
    pmc.train(X_train, Y_train, is_classification, alpha, nb_iter);

    // Test des prédictions
    std::cout << "XOR 0, 0 : " << pmc.predict({ 0, 0 }, is_classification)[0] << std::endl;
    std::cout << "XOR 0, 1 : " << pmc.predict({ 0, 1 }, is_classification)[0] << std::endl;
    std::cout << "XOR 1, 0 : " << pmc.predict({ 1, 0 }, is_classification)[0] << std::endl;
    std::cout << "XOR 1, 1 : " << pmc.predict({ 1, 1 }, is_classification)[0] << std::endl;

    return 0;
}

// Exécuter le programme : Ctrl+F5 ou menu Déboguer > Exécuter sans débogage
// Déboguer le programme : F5 ou menu Déboguer > Démarrer le débogage

// Astuces pour bien démarrer : 
//   1. Utilisez la fenêtre Explorateur de solutions pour ajouter des fichiers et les gérer.
//   2. Utilisez la fenêtre Team Explorer pour vous connecter au contrôle de code source.
//   3. Utilisez la fenêtre Sortie pour voir la sortie de la génération et d'autres messages.
//   4. Utilisez la fenêtre Liste d'erreurs pour voir les erreurs.
//   5. Accédez à Projet > Ajouter un nouvel élément pour créer des fichiers de code, ou à Projet > Ajouter un élément existant pour ajouter des fichiers de code existants au projet.
//   6. Pour rouvrir ce projet plus tard, accédez à Fichier > Ouvrir > Projet et sélectionnez le fichier .sln.
