CC=g++

Compile: main.cpp Matrix.cpp
	$(CC) -o main main.cpp Matrix.cpp --std=c++14

CNN: main.cpp Matrix.cpp NeuralNetwork.cpp
	$(CC) -o main main.cpp Matrix.cpp NeuralNetwork.cpp
