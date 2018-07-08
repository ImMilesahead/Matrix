#include <iostream>
#include "Matrix.h"
#include <vector>
#include <map>
#include <string>

namespace NN {
	std::pair<Matrix, Matrix> relu(const Matrix &A);
	Matrix reluBackwards(Matrix dA, Matrix cache);
	std::pair<Matrix, Matrix> sigmoid(const Matrix &Z);
	Matrix sigmoidBackwards(Matrix dA, Matrix cache);
	std::map<std::string, Matrix> initializeParameters(std::vector<int> layerDims);
	std::pair<Matrix, std::vector<Matrix> > linearForward(Matrix A, Matrix W, Matrix b);
	std::pair<Matrix, std::pair<std::vector<Matrix>, Matrix> > activateForward(Matrix Aprev, Matrix W, Matrix b, std::string activation);
	std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > forwardPropagation(Matrix X, std::map<std::string, Matrix> parameters);
	std::vector<Matrix> linearBackwards(Matrix dZ, std::vector<Matrix> cache);
	std::vector<Matrix> linearActivateBackwards(Matrix dA, std::pair<std::vector<Matrix>, Matrix> cache, std::string activation);
	std::map<std::string, Matrix> backPropagation(Matrix AL, Matrix Y, std::vector<std::pair<std::vector<Matrix>, Matrix> > caches);
	std::map<std::string, Matrix> updateParameters(std::map<std::string, Matrix> parameters, std::map<std::string, Matrix> grads, double learningRate);
	double compute_cost(Matrix AL, Matrix Y);
	std::pair<std::map<std::string, Matrix>, std::vector<double> > model(Matrix X, Matrix Y, std::vector<int> layerDims, double learningRate, int numIterations, bool printCost);
}
