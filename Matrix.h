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
	 Matrix();
	 Matrix(Shape);
	 Matrix(int, int);
	 ~Matrix();
	 void rand();
	 void randn();
	 void populate(double);
	 Shape getShape() const;
	 double getData(int, int) const;
	 Matrix dot(const Matrix&) const;
	 Matrix makeThisFuckerTheSameShape(const Matrix &) const;
	 Matrix T() const;
	 Matrix transpose();
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
