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
			newMatrix[i][j] = A[i][j] / (B[i][j] + 1e-6);
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
Matrix power(double base, Matrix p){
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
		for(int j = 0; j < mShape.n; ++j){
			newMatrix[j][i] = m.getData(i, j);
		}
	}
	return newMatrix;
}
Matrix mlog(const Matrix &A){
	Matrix newMatrix(A.getShape());
	Shape shape = A.getShape();
	for(int i = 0; i < shape.m; ++i){
		for(int j = 0; j < shape.n; ++j){
			newMatrix[i][j] = std::log(std::abs(A.getData(i, j)) + 0.0001);
		}
	}
	return newMatrix;
}
Matrix::Matrix(){
	std::srand(std::time(0));
}
Matrix::Matrix(int m, int n){
	std::srand(std::time(0));
	if(m < 1 || n < 1){
		throw -1;
	}
	this->shape = Shape(m, n);
	this->data = new double*[this->shape.m];
	for(int i = 0; i < this->shape.m; ++i){
		this->data[i] = new double[this->shape.n];
		for(int j = 0; j < this->shape.n; ++j){
			this->data[i][j] = 1.0f;
		}
	}
}
Matrix::Matrix(Shape shape){
	std::srand(std::time(0));
	this->shape = Shape(shape.m, shape.n);
	this->data = new double*[this->shape.m];
	for(int i = 0; i < this->shape.m; ++i){
		this->data[i] = new double[this->shape.n];
		for(int j = 0; j < shape.n; ++j){
			this->data[i][j] = 1.0f;
		}
	}
}
Matrix::~Matrix(){
	/*
	for(int i = 0; i < this->shape.m; ++i){
		if(this->data[i] != nullptr){
			delete[] this->data[i];
		}
	}
	if(this->data != nullptr){
		delete[] this->data;
	}
	*/
}
void Matrix::rand(){
	for(int m = 0; m < this->shape.m; ++m){
		for(int n = 0; n < this->shape.n; ++n){
			this->data[m][n] = static_cast<double>(std::rand() % 200) / 100 - 1 + 0.000000001;
		}
	}
}
void Matrix::randn(){
	for(int m = 0; m < this->shape.m; ++m){
		for(int n = 0; n < this->shape.n; ++n){
			this->data[m][n] = static_cast<double>(std::rand() % 250) / 100 + 0.000000001;
		}
	}
}
Matrix Matrix::transpose(){
	*this = this->T();
	return *this;
}
Matrix Matrix::T() const {
	return trans(*this);
}
void Matrix::populate(double init){
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
			double value = 0;
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
double Matrix::getData(int m, int n) const {
	return this->data[m][n];
}
double Matrix::sum(){
	double sum = 0.0f;
	for(int i = 0; i < this->shape.m; ++i){
		for(int j = 0; j < this->shape.n; ++j){
			sum += this->data[i][j];
		}
	}
	return sum;
/*	Matrix A(this->shape.n, 1);
	Matrix B(1, this->shape.m);
	A.populate(1.0f);
	B.populate(1.0f);
	Matrix C = this->dot(A);
	Matrix D = B.dot(C);
	return D[0][0];*/
}
// Operators
// OSTREAM
std::ostream& operator<<(std::ostream& os, const Shape& shape){
	os << "(" << shape.m << ", " << shape.n << ")";
	return os;
}
std::ostream& operator<<(std::ostream& os, const Matrix &m){
	os << "Matrix [";
	Shape shape = m.getShape();
	for(int mm = 0; mm < shape.m; ++mm){
		if(mm > 0)
			os << "        ";
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

// CONDITIONALS
Matrix Matrix::operator>(const double &rhs) const {
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
Matrix Matrix::operator<(const double &rhs) const {
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
Matrix Matrix::operator>=(const double &rhs) const {
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
Matrix Matrix::operator<=(const double &rhs) const {
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
// INDEX ACCESSOR
double *Matrix::operator[](int x){
   return this->data[x];
}
// ARITHMATIC
Matrix Matrix::operator*=(const double &rhs){
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	*this = mult(A, B);
	return *this;
}
Matrix Matrix::operator/=(const double &rhs){
	*this = *this / rhs;
	return *this;
}
Matrix Matrix::operator-=(const double &rhs){
	*this = *this - rhs;
	return *this;
}
Matrix Matrix::operator+=(const double &rhs){
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
Matrix Matrix::operator*(const double &rhs) const{
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
Matrix Matrix::operator/(const double &rhs) const{
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
Matrix Matrix::operator-(const double &rhs) const{
	Matrix A = *this;
	Matrix B(this->shape);
	B.populate(rhs);
	return sub(A, B);
}
Matrix Matrix::operator-(const Matrix &other) const {
	Matrix A = *this;
	Matrix B = other;return sub(A, B);
}
Matrix Matrix::operator+(const double &rhs) const{
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
Matrix Matrix::operator-() const{
	Matrix B(this->shape);
	B.populate(0.0f);
	Matrix A = *this;
	return sub(B, A);
}
// NON MEMBER ARITHMATIC
Matrix operator+(const double &lhs, const Matrix &rhs){
	return rhs + lhs;
}
Matrix operator+(const int &lhs, const Matrix &rhs){
	return rhs + lhs;
}
Matrix operator/(const double &lhs, const Matrix &rhs){
	Matrix A = rhs;
	Matrix B(rhs.getShape());
	B.populate(lhs);
	return B / A;
}
Matrix operator/(const int &lhs, const Matrix &rhs){
	Matrix A = rhs;
	Matrix B(rhs.getShape());
	B.populate(static_cast<double>(lhs));
	return B / A;
}
Matrix operator-(const double &lhs, const Matrix &rhs){
	// TODO is a not very effecient, still O(N) but more like O(2N)
	return -rhs + lhs;
}
Matrix operator-(const int &lhs, const Matrix &rhs){
	// TODO is not ver effecient currently runs in O(2N) could run in O(N)
	return -rhs + static_cast<double>(lhs);
}
Matrix operator*(const double &lhs, const Matrix &rhs){
	return rhs * lhs;
}
Matrix operator*(const int &lhs, const Matrix &rhs){
	return rhs * lhs;
}
