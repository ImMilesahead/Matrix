#include "Matrix.h"
#include <ctime>
#include <cstdlib>
#include <cmath>

// Helper functions
void makeTheseFuckersTheSame(Matrix &A, Matrix &B){
	Shape aShape = A.getShape();
	Shape bShape = B.getShape();
	if(aShape == bShape){
		return;
	}else if(aShape == Shape(bShape.m, 1)){
		A = B.makeThisFuckerTheSameShape(A);
	}else if(Shape(aShape.m, 1) == bShape){
		B = A.makeThisFuckerTheSameShape(B);
	}else if(aShape == Shape(1, bShape.n)){
		A = B.makeThisFuckerTheSameShape(A);
	}else if(Shape(1, aShape.n) == bShape){
		B = A.makeThisFuckerTheSameShape(B);
	}else{
		std::cout <<  "Can't makeThisFuckerTheSameShape with size " << aShape << " and " << bShape << "\n";
		throw -1;
	}
}
// These next 4 functions exist so that I don't have to optimize over 16 functions for cuda processing
// So it turned out being 10 functions to cudaify but that's better than 22
Matrix add(Matrix &A, Matrix &B){
	// Assume they're the correct shapes
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			newMatrix[i][j] = A[i][j] + B[i][j];
		}
	}
	return newMatrix;
}
Matrix sub(Matrix &A, Matrix &B){
	// Assume they're the correct shapes
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			newMatrix[i][j] = A[i][j] - B[i][j];
		}
	}
	return newMatrix;
}
Matrix mult(Matrix &A, Matrix &B){
	// Assume they're the correct shapes
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			newMatrix[i][j] = A[i][j] * B[i][j];
		}
	}
	return newMatrix;
}
Matrix div(Matrix &A, Matrix &B){
	// Assume they're the correct shapes
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			newMatrix[i][j] = A[i][j] / B[i][j];
		}
	}
	return newMatrix;
}
Matrix greaterThan(Matrix &A, Matrix &B){
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			if(A[i][j] > B[i][j]){
				newMatrix[i][j] = 1.0f;
			}else{
				newMatrix[i][j] = 0.0f;
			}
		}
	}
	return newMatrix;
}
Matrix lessThan(Matrix &A, Matrix &B){
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			if(A[i][j] < B[i][j]){
				newMatrix[i][j] = 1.0f;
			}else{
				newMatrix[i][j] = 0.0f;
			}
		}
	}
	return newMatrix;
}
Matrix greaterThanEqual(Matrix &A, Matrix &B){
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			if(A[i][j] >= B[i][j]){
				newMatrix[i][j] = 1.0f;
			}else{
				newMatrix[i][j] = 0.0f;
			}
		}
	}
	return newMatrix;
}
Matrix lessThanEqual(Matrix &A, Matrix &B){
	makeTheseFuckersTheSame(A, B);
	Shape aShape = A.getShape();
	Matrix newMatrix(aShape);
	for(int i = 0; i < aShape.m; ++i){
		for(int j = 0; j < aShape.n; ++j){
			if(A[i][j] <= B[i][j]){
				newMatrix[i][j] = 1.0f;
			}else{
				newMatrix[i][j] = 0.0f;
			}
		}
	}
	return newMatrix;
}
Matrix power(float base, Matrix p){
	Shape shape = p.getShape();
	Matrix newMatrix(shape);
	for(int i = 0; i < shape.m; ++i){
		for(int j = 0; j < shape.n; ++j){
			newMatrix[i][j] = std::pow(base, p.getData(i, j));
		}
	}
	return newMatrix;
}
Matrix trans(const Matrix &m){
	Shape mShape = m.getShape();
	Shape nShape(mShape.n, mShape.m);
	Matrix newMatrix(nShape);
	for(int i = 0; i < mShape.m; ++i){
		for(int j = 0; j < mShape.m; ++j){
			newMatrix[j][i] = m.getData(i, j);
		}
	}
	return newMatrix;
}

Matrix::Matrix(){

}
Matrix::Matrix(int m, int n){
	if(m < 1 || n < 1){
		throw -1;
	}
	this->shape = Shape(m, n);
	this->data = new float*[this->shape.m];
	for(int i = 0; i < this->shape.m; ++i){
		this->data[i] = new float[this->shape.n];
	}
}
Matrix::Matrix(Shape shape){
	this->shape = Shape(shape.m, shape.n);
	this->data = new float*[this->shape.m];
	for(int i = 0; i < this->shape.m; ++i){
		this->data[i] = new float[this->shape.n];
	}
}
Matrix::~Matrix(){
	for(int i = 0; i < this->shape.m; ++i){
		delete[] this->data[i];
	}
	delete[] this->data;
}
void Matrix::rand(){
	srand(time(0));
	for(int m = 0; m < this->shape.m; ++m){
		for(int n = 0; n < this->shape.n; ++n){
			this->data[m][n] = static_cast<float>(std::rand() % 2000) / 100 - 10;
		}
	}
}
void Matrix::randn(){

}
Matrix Matrix::transpose(){
	*this = this->T();
	return *this;
}
Matrix Matrix::T() const {
	return trans(*this);
}


void Matrix::populate(float init){
	for(int m = 0; m < this->shape.m; ++m){
		for(int n = 0; n < this->shape.n; ++n){
			this->data[m][n] = init;
		}
	}
}
Shape Matrix::getShape() const {
	return this->shape;
}
Matrix Matrix::dot(const Matrix &other) const {
	// Check to see if shape matches
	if(this->shape.n != other.shape.m){
		// Throw error
		std::cout << "Can't dot matricies of sizes " << this->getShape() << " and " << other.getShape() << "\n";
		throw -1;
	}
	// get new Shape
	Shape newShape = Shape(this->shape.m, other.shape.n);
	Matrix newMatrix(newShape);
	for(int nm = 0; nm < newShape.m; ++nm){
		for(int nn = 0; nn < newShape.n; ++nn){
			float value = 0;
			// calculate Value
			for(int i = 0; i < this->shape.n; ++i){
				value += this->data[nm][i] * other.data[i][nn];
			}
			newMatrix.data[nm][nn] = value;
		}
	}
	return newMatrix;
}
Matrix Matrix::makeThisFuckerTheSameShape(const Matrix &A) const {
	Matrix newMatrix(this->shape);
	if(A.shape.m == 1){
		for(int i = 0; i < this->shape.m; ++i){
			for(int j = 0; j < this->shape.n; ++j){
				newMatrix[i][j] = A.getData(0, j);
			}
		}
	}else if(A.shape.n == 1){
		for(int i = 0; i < this->shape.m; ++i){
			for(int j = 0; j < this->shape.n; ++j){
				newMatrix[i][j] = A.getData(i, 0);
			}
		}
	}else{
		std::cout <<  "Can't makeThisFuckerTheSameShape with size " << A.shape << "\n";
		throw -1;
	}
	return newMatrix;

}
float Matrix::getData(int m, int n) const {
	return this->data[m][n];
}

// Operators
Matrix Matrix::operator*=(const float &rhs){
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	*this = mult(A, B);
	return *this;
}
Matrix Matrix::operator/=(const float &rhs){
	*this = *this / rhs;
	return *this;
}
Matrix Matrix::operator-=(const float &rhs){
	*this = *this - rhs;
	return *this;
}
Matrix Matrix::operator+=(const float &rhs){
	*this = *this + rhs;
	return *this;
}
Matrix Matrix::operator*=(const Matrix &rhs){
	*this = *this * rhs;
	return *this;
}
Matrix Matrix::operator/=(const Matrix &rhs){
	*this = *this / rhs;
	return *this;
}
Matrix Matrix::operator-=(const Matrix &rhs){
	*this = *this - rhs;
	return *this;
}
Matrix Matrix::operator+=(const Matrix &rhs){
	*this = *this + rhs;
	return *this;
}
Matrix Matrix::operator*(const float &rhs) const{
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return mult(A, B);
}
Matrix Matrix::operator*(const Matrix &other) const {
	Matrix A = *this;
	Matrix B = other;
	return mult(A, B);
}
Matrix Matrix::operator/(const float &rhs) const{
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return div(A, B);
}
Matrix Matrix::operator/(const Matrix &other) const {
	Matrix A = *this;
	Matrix B = other;
	return div(A, B);
}
Matrix Matrix::operator-(const float &rhs) const{
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return sub(A, B);
}
Matrix Matrix::operator-(const Matrix &other) const {
	Matrix A = *this;
	Matrix B = other;return sub(A, B);
}
Matrix Matrix::operator+(const float &rhs) const{
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return add(A, B);
}
Matrix Matrix::operator+(const Matrix &other) const {
	Matrix A = *this;
	Matrix B = other;
	return add(A, B);
}
std::ostream& operator<<(std::ostream& os, const Shape& shape){
	os << "(" << shape.m << ", " << shape.n << ")";
	return os;
}
std::ostream& operator<<(std::ostream& os, const Matrix &m){
	os << "Matrix [";
	Shape shape = m.getShape();
	for(int mm = 0; mm < shape.m; ++mm){
		os << "[";
		for(int n = 0; n < shape.n; ++n){
			os << m.getData(mm, n);
			if(n == shape.n-1){
				os << "]";
			}else{
				os << ", ";
			}
		}
		if(mm == shape.m-1){
			continue;
		}
		os << ",\n";
	}
	os << "]\n";
	return os;
}
Matrix Matrix::operator>(const float &rhs) const {
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return greaterThan(A, B);
}
Matrix Matrix::operator>(const Matrix &rhs) const {
	Matrix A = *this;
	Matrix B = rhs;
	return greaterThan(A, B);
}
Matrix Matrix::operator<(const float &rhs) const {
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return lessThan(A, B);
}
Matrix Matrix::operator<(const Matrix &rhs) const {
	Matrix A = *this;
	Matrix B = rhs;
	return lessThan(A, B);
}
Matrix Matrix::operator>=(const float &rhs) const {
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return greaterThanEqual(A, B);
}
Matrix Matrix::operator>=(const Matrix &rhs) const {
	Matrix A = *this;
	Matrix B = rhs;
	return greaterThanEqual(A, B);
}
Matrix Matrix::operator<=(const float &rhs) const {
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return lessThanEqual(A, B);
}
Matrix Matrix::operator<=(const Matrix &rhs) const {
	Matrix A = *this;
	Matrix B = rhs;
	return lessThanEqual(A, B);
}
Matrix Matrix::operator-() const{
	Matrix B(this->shape);
	B.populate(0.0f);
	Matrix A = *this;
	return sub(B, A);
}
Matrix operator+(const float &lhs, const Matrix &rhs){
	return rhs + lhs;
}
Matrix operator+(const int &lhs, const Matrix &rhs){
	return rhs + lhs;
}
Matrix operator/(const float &lhs, const Matrix &rhs){
	Matrix A = rhs;
	Matrix B(rhs.getShape());
	B.populate(lhs);
	return B / A;
}
Matrix operator/(const int &lhs, const Matrix &rhs){
	return static_cast<float>(lhs) / rhs;
}
Matrix operator-(const float &lhs, const Matrix &rhs){
	// TODO is a not very effecient, still O(N) but more like O(2N)
	return -rhs + lhs;
}
Matrix operator-(const int &lhs, const Matrix &rhs){
	// TODO is not ver effecient currently runs in O(2N) could run in O(N)
	return -rhs + static_cast<float>(lhs);
}
Matrix operator*(const float &lhs, const Matrix &rhs){
	return rhs * lhs;
}
Matrix operator*(const int &lhs, const Matrix &rhs){
	return rhs * lhs;
}
float *Matrix::operator[](int x){
   return this->data[x];
}
