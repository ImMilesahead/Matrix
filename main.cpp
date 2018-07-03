#include <iostream>
#include "Matrix.h"
#include <vector>
#include <map>
#include <string>
#include <ctime>

float e = 2.71828f;

Matrix relu(const Matrix &A){
	return A * (A > 0);
}

Matrix sigmoid(const Matrix &A){
	return 1 / ( 1 + power(e, -A));
}

std::map<std::string, Matrix> initializeParameters(std::vector<int> layerDims){
	std::map<std::string, Matrix> parameters;
	for(int l = 1; l < layerDims.size(); ++l){
		Matrix W(layerDims[l-1], layerDims[l]);
		W.rand();
		W *= 0.01;
		parameters['W' + std::to_string(l)] = W;
		Matrix B(layerDims[l], 1);
		B.rand();
		B *= 0.01;
		parameters['b' + std::to_string(l)] = B;
	}
	return parameters;
}

int main(){
	std::clock_t t = clock();
	std::vector<int> layerDims;
	layerDims.push_back(2);
	layerDims.push_back(3);
	layerDims.push_back(1);
	std::map<std::string, Matrix> parameters = initializeParameters(layerDims);
	int nX = 4;

	Matrix X(layerDims[0], nX);
	X.populate(0);

	t = clock() - t;

	

	return 0;
}
