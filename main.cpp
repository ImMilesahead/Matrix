#include <iostream>
#include "Matrix.h"
#include <vector>
#include <map>
#include <string>

float e = 2.71828f;


std::pair<Matrix, Matrix> relu(const Matrix &A){
	Matrix Z = A * (A > 0);
	std::pair<Matrix, Matrix> r;
	r.first = Z;
	r.second = A;
	return r;
}
Matrix reluBackwards(Matrix dA, Matrix cache){
	Matrix Z = cache;
	Matrix r = Z > 0;
	Matrix dZ = dA * r;
	return dZ;
}
std::pair<Matrix, Matrix> sigmoid(const Matrix &A){
	Matrix Z = 1 / ( 1 + power(e, -A));
	std::pair<Matrix, Matrix> r;
	r.first = Z;
	r.second = A;
	return r;
}
Matrix sigmoidBackwards(Matrix dA, Matrix cache){
	Matrix Z = cache;
	Matrix s = 1 / ( 1 + power(e, -Z));
	Matrix dZ = dA * s * (1 - s);
	return dZ;
}

std::map<std::string, Matrix> initializeParameters(std::vector<int> layerDims){
	std::map<std::string, Matrix> parameters;
	for(int l = 1; l < layerDims.size(); ++l){
		Matrix W(layerDims[l], layerDims[l-1]);
		W.rand();
		std::string index = "W" + std::to_string(l);
		parameters[index] = W;
		Matrix B(layerDims[l], 1);
		B.populate(0.0f);
		index = "b" + std::to_string(l);
		parameters[index] = B;
	}
	return parameters;
}

std::pair<Matrix, std::vector<Matrix> > linearForward(Matrix A, Matrix W, Matrix b){
	Matrix Z = W.dot(A) + b;
	std::vector<Matrix> cache;
	cache.push_back(A);
	cache.push_back(W);
	cache.push_back(b);
	std::pair<Matrix, std::vector<Matrix> > t;
	t.first = Z;
	t.second = cache;
	return t;
}
std::pair<Matrix, std::pair<std::vector<Matrix>, Matrix> > activateForward(Matrix Aprev, Matrix W, Matrix b, std::string activation){
	std::pair<Matrix, std::vector<Matrix> > t = linearForward(Aprev, W, b);
	Matrix Z = t.first;
	std::vector<Matrix> linearCahce = t.second;
	std::pair<Matrix, Matrix> r;
	if(activation == "SIGMOID"){
		r = sigmoid(Z);
	}else if(activation == "RELU"){
		r = relu(Z);
	}
	Matrix A = r.first;
	Matrix activationCache = r.second;
	std::pair<std::vector<Matrix>, Matrix> cache;
	cache.first = linearCahce;
	cache.second = activationCache;
	std::pair<Matrix, std::pair<std::vector<Matrix>, Matrix> > v;
	v.first = A;
	v.second = cache;
	return v;
}
//			AL			caches			linearCahce			activationCache
std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > forwardPropagation(Matrix X, std::map<std::string, Matrix> parameters){
	std::vector<std::pair<std::vector<Matrix>, Matrix> > caches;
	Matrix A = X;
	int L = parameters.size() / 2;
	for(int l = 1; l < L; ++l){
		Matrix A_prev = A;
		std::pair<Matrix, std::pair<std::vector<Matrix>, Matrix> > returned = activateForward(A_prev, parameters["W" + std::to_string(l)], parameters["b" + std::to_string(l)], "RELU");
		A = returned.first;
		std::pair<std::vector<Matrix>, Matrix> cache = returned.second;
		caches.push_back(cache);
	}
	std::pair<Matrix, std::pair<std::vector<Matrix>, Matrix> > returned = activateForward(A, parameters["W" + std::to_string(L)], parameters["b" + std::to_string(L)], "SIGMOID");
	A = returned.first;
	std::pair<std::vector<Matrix>, Matrix> cache = returned.second;
	caches.push_back(cache);
	std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > toReturn;
	toReturn.first = A;
	toReturn.second = caches;
	return toReturn;
}

std::vector<Matrix> linearBackwards(Matrix dZ, std::vector<Matrix> cache){
	Matrix A_prev = cache[0];
	Matrix W = cache[1];
	Matrix b = cache[2];
	int n = A_prev.getShape().n;
	Matrix dW = dZ.dot(A_prev.T()) / n;

	Matrix Wtrans = W.T();
	Matrix dA_prev = Wtrans.dot(dZ);

	Matrix temp(dZ.getShape().n, 1);
	temp.populate(1.0f);
	Matrix db = dZ.dot(temp) / n;
	std::vector<Matrix> r;
	r.push_back(dA_prev);
	r.push_back(dW);
	r.push_back(db);
	return r;
}

std::vector<Matrix> linearActivateBackwards(Matrix dA, std::pair<std::vector<Matrix>, Matrix> cache, std::string activation){
	std::vector<Matrix> linCache = cache.first;
	Matrix actCache = cache.second;
	Matrix dZ;
	if(activation == "SIGMOID"){
		dZ = sigmoidBackwards(dA, actCache);
	}else if(activation == "RELU"){
		dZ = reluBackwards(dA, actCache);
	}else{
		throw -1;
	}
	return linearBackwards(dZ, linCache);
}

std::map<std::string, Matrix> backPropagation(Matrix AL, Matrix Y, std::vector<std::pair<std::vector<Matrix>, Matrix> > caches){
	std::map<std::string, Matrix> grads;
	int L = caches.size();
	int nX = AL.getShape().n;
	Matrix dAL = - ((Y / AL) - ((1 - Y) / (1 - AL)));
	std::pair<std::vector<Matrix>, Matrix> currentCache = caches[L-1];
	std::vector<Matrix> t = linearActivateBackwards(dAL, currentCache, "SIGMOID");
	grads["dA" + std::to_string(L-1)] = t[0];
	grads["dW" + std::to_string(L)] = t[1];
	grads["db" + std::to_string(L)] = t[2];
	for(int l = L-2; l >= 0; --l){
		currentCache = caches[l];
		std::vector<Matrix> t = linearActivateBackwards(grads["dA" + std::to_string(l+1)], currentCache, "RELU");
		grads["dA" + std::to_string(l)] = t[0];
		grads["dW" + std::to_string(l + 1)] = t[1];
		grads["db" + std::to_string(l + 1)] = t[2];
	}
	return grads;
}

std::map<std::string, Matrix> updateParameters(std::map<std::string, Matrix> parameters, std::map<std::string, Matrix> grads, float learningRate){
	int L = parameters.size() / 2;
	for(int l = 1; l <= L; ++l){
		parameters["W" + std::to_string(l)] = parameters["W" + std::to_string(l)] - (grads["dW" + std::to_string(l)] * learningRate);
		parameters["b" + std::to_string(l)] = parameters["b" + std::to_string(l)] - (grads["db" + std::to_string(l)] * learningRate);
	}
	return parameters;
}

float compute_cost(Matrix AL, Matrix Y){
	int n = Y.getShape().n;
	Matrix preSum = Y * mlog(AL) + (1 - Y) * mlog(1 - AL);
	float cost = preSum.sum()/n;
	return -cost;
}

// 				parameters					costs
std::pair<std::map<std::string, Matrix>, std::vector<float> > model(Matrix X, Matrix Y, std::vector<int> layerDims, float learningRate, int numIterations, bool printCost){
	std::vector<float> costs;
	std::map<std::string, Matrix> parameters = initializeParameters(layerDims);
	for(int i = 0; i < numIterations; ++i){
		std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > ALCaches = forwardPropagation(X, parameters);
		Matrix AL = ALCaches.first;
		std::vector<std::pair<std::vector<Matrix>, Matrix> > caches = ALCaches.second;
		float cost = compute_cost(AL, Y);
		std::map<std::string, Matrix> grads = backPropagation(AL, Y, caches);
		parameters = updateParameters(parameters, grads, learningRate);
		if (i % 1000 == 0){
			if(printCost){
				std::cout << "Cost after " << i << ": " << cost << std::endl;
			}
			costs.push_back(cost);
			if(cost < 0.00001f){
				std::pair<std::map<std::string, Matrix>, std::vector<float> > temp;
				temp.first = parameters;
				temp.second = costs;
				return temp;
			}
		}
	}
	std::pair<std::map<std::string, Matrix>, std::vector<float> > temp;
	temp.first = parameters;
	temp.second = costs;
	return temp;
}

int main(){
	//			parameters						costs
	std::pair<std::map<std::string, Matrix>, std::vector<float> > paramCost;
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

	Matrix Y(layerDims[2], nX);
	Y.populate(0);
	Y[0][1] = 1.0f;
	Y[0][2] = 1.0f;

	std::pair<std::map<std::string, Matrix>, std::vector<float> > r = model(X, Y, layerDims, 0.5f, 100000, true);
	std::map<std::string, Matrix> parameters = r.first;

	std::pair<Matrix, std::vector<std::pair<std::vector<Matrix>, Matrix> > > fr;
	fr = forwardPropagation(X, parameters);
	Matrix output = fr.first;



	std::cout << X << std::endl;
	std::cout << Y << std::endl;
	std::cout << output << std::endl;

	return 0;
}
