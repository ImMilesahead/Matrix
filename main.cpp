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
	std::cout << "Test\n";
	for(int l = 1; l < layerDims.size(); ++l){
		std::cout << l << std::endl;
		Matrix W(layerDims[l-1], layerDims[l]);
		W.rand();
		W = W * 0.01f;
		std::string index = "W" + std::to_string(l);
		std::cout << "Inserting " << W << " into " << index << std::endl;
		parameters[index] = W;
		Matrix B(layerDims[l], 1);
		B.rand();
		B = B * 0.01;
		parameters["b" + std::to_string(l)] = B;
		std::cout << "this is a test\n";
	}
	return parameters;
}

int main(){
	std::cout << "Hello, World!\n";
	std::clock_t t = clock();
	std::vector<int> layerDims;
	layerDims.push_back(2);
	layerDims.push_back(3);
	layerDims.push_back(1);
	std::cout << layerDims.size() << std::endl;
	std::map<std::string, Matrix> parameters = initializeParameters(layerDims);
	int nX = 4;
	std::cout << "Hello, again\n";

	Matrix X(layerDims[0], nX);
	X.populate(0);

	t = clock() - t;

	std::cout << parameters["W1"] << std::endl;

	return 0;
}
