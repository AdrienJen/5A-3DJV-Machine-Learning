#pragma once
#include <vector>
class RBF
{
public:
	int nInputs;
	int nOutputs;
	int nCenters;
	double spread;
	std::vector<double> weights;
	std::vector<std::vector<double> > centers;
private:
	RBF(int nInputs, int nOutputs, int nCenters, double spread);
	void initCenters(std::vector<std::vector<double> > inputData);
	std::vector<double> computeHiddenActivations(std::vector<double> input);
	std::vector<double> predict(std::vector<double> input);
	void train(std::vector<std::vector<double> > inputData, std::vector<std::vector<double> > targetData, int nEpochs, double learningRate);
};

