#ifndef MATRIX_H
#define MATRIX_H


#include <iostream>

struct Shape {
	Shape(int m, int n): m(m), n(n) {}
	Shape(){}
	int m = 0;
	int n = 0;
	friend std::ostream& operator<<(std::ostream&, const Shape&);
	bool operator==(const Shape &other) const {
		if(other.m == this->m && other.n == this->n)
			return true;
		return false;
	}
};

class Matrix {
 private:
	 Shape shape;
	double **data;
 public:
	 // Constructs nothing make sure to never create an empty matrix
	 Matrix();
	 // Constructs a matrix of zeros from the shape of another matrix
	 Matrix(Shape);
	 // Constructs a matrix of zeros given dimensions m, n
	 Matrix(int, int);
	 // Destructor
	 ~Matrix();
	 // assigns a random number betwee 1.00 and -1.00 to every value in the matrix
	 void rand();
	 // assigns a random number between 1.25 and -1.25 to every slot in the matrix
	 void randn();
	 // Populates the matrix with a single value
	 void populate(double);
	 // Returns the shape of the matrix Shape(m, n)
	 Shape getShape() const;
	 // Gets the value at m, n
	 double getData(int, int) const;
	 // Performs matrix multiplication with Nother matrix and returns the new matrix
	 Matrix dot(const Matrix&) const;
	 // Performs Broadcasting on another matrrix to make it the same shape as this one
	 Matrix makeThisFuckerTheSameShape(const Matrix &) const;
	 // Returns the transpose of a matrix
	 Matrix T() const;
	 // Transposes the current matrix and returns it
	 Matrix transpose();
	 // Totals all the values on the matrix
	 double sum();

	 friend std::ostream& operator<<(std::ostream&, const Matrix&);
	 double *operator[](int);
	 Matrix operator*=(const double&);
	 Matrix operator/=(const double&);
	 Matrix operator-=(const double&);
	 Matrix operator+=(const double&);
	 Matrix operator*=(const Matrix&);
	 Matrix operator/=(const Matrix&);
	 Matrix operator-=(const Matrix&);
	 Matrix operator+=(const Matrix&);
 	 Matrix operator*(const double &other) const;
 	 Matrix operator/(const double &other) const;
 	 Matrix operator-(const double &other) const;
 	 Matrix operator+(const double &other) const;
 	 Matrix operator*(const Matrix &other) const;
 	 Matrix operator/(const Matrix &other) const;
 	 Matrix operator-(const Matrix &other) const;
 	 Matrix operator+(const Matrix &other) const;
	 Matrix operator>(const double &rhs) const;
	 Matrix operator>(const Matrix &rhs) const;
	 Matrix operator<(const double &rhs) const;
	 Matrix operator<(const Matrix &rhs) const;
	 Matrix operator>=(const double &rhs) const;
	 Matrix operator>=(const Matrix &rhs) const;
	 Matrix operator<=(const double &rhs) const;
	 Matrix operator<=(const Matrix &rhs) const;

	 Matrix operator-() const;

 	 friend Matrix operator+(const double&, const Matrix&);
 	 friend Matrix operator+(const int&, const Matrix&);
 	 friend Matrix operator-(const double&, const Matrix&);
 	 friend Matrix operator-(const int&, const Matrix&);
 	 friend Matrix operator/(const double&, const Matrix&);
 	 friend Matrix operator/(const int&, const Matrix&);
 	 friend Matrix operator*(const double&, const Matrix&);
 	 friend Matrix operator*(const int&, const Matrix&);

};


void makeTheseFuckersTheSame(Matrix &, Matrix &);
Matrix add(Matrix&, Matrix&);
Matrix sub(Matrix&, Matrix&);
Matrix mult(Matrix&, Matrix&);
Matrix div(Matrix&, Matrix&);
Matrix power(double, Matrix);
Matrix trans(const Matrix &);
Matrix mlog(const Matrix &);


#endif	// MATRIX_H
