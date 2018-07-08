#include <iostream>
#include "Matrix.h"
#include <vector>
#include <map>
#include <string>
#include "NeuralNetwork.h"

using namespace NN;

int main(){
	//			parameters						costs
	std::pair<std::map<std::string, Matrix>, std::vector<double> > paramCost;
	std::vector<int> layerDims;
	layerDims.push_back(2);
	layerDims.push_back(3);
	layerDims.push_back(1);
	int nX = 4;
	Matrix X(layerDims[0], nX);
	X.populate(0);

	X[0][1] = 1.0f;
	X[1][2] = 1.0f;
	X[0][3] = 1.0f;
	X[1][3] = 1.0f;

	Matrix Y(layerDims[layerDims.size()-1], nX);
	Y.populate(0);
	Y[0][1] = 1.0f;
	Y[0][2] = 1.0f;

	std::pair<std::map<std::string, Matrix>, std::vector<double> > r = model(X, Y, layerDims, 1.0f, 100000, true);
	std::map<std::string, Matrix> parameters = r.first;

	std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > fr;
	fr = forwardPropagation(X, parameters);
	Matrix output = fr.first;



	std::cout << X << std::endl;
	std::cout << Y << std::endl;
	std::cout << output << std::endl;

	for(int i = 1; i < layerDims.size(); ++i){
		std::cout << "W" << i << std::endl << parameters["W" + std::to_string(i)] << std::endl;
		std::cout << "b" << i << std::endl << parameters["b" + std::to_string(i)] << std::endl;

	}

	return 0;
}
