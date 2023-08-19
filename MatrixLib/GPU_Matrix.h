#pragma once

#include <random>

class GPU_Matrix;
class CPU_Matrix;

struct _DUMY_ {
	int x = 0;
	int y = 0;
	int z = 0;
};

double mseLoss(const GPU_Matrix& output, const GPU_Matrix& expected);
double crossEntropy(const GPU_Matrix&, const GPU_Matrix&);
GPU_Matrix exp(const GPU_Matrix& A);
GPU_Matrix log(const GPU_Matrix& A);
GPU_Matrix tanh(const GPU_Matrix& A);
GPU_Matrix tanh_prime(const GPU_Matrix& A);
GPU_Matrix soft_max(const GPU_Matrix& A);
GPU_Matrix leakReLu(const GPU_Matrix& A, double slope);
GPU_Matrix leakReLu_prime(const GPU_Matrix& A, double slope);

class GPU_Matrix
{
	double* data;
	size_t row_;
	size_t column_;
public:

	GPU_Matrix(size_t row, size_t column)noexcept;
	GPU_Matrix(size_t row, size_t column,double val)noexcept;
	GPU_Matrix(const GPU_Matrix& other)noexcept;
	GPU_Matrix(GPU_Matrix&& other) noexcept;
	~GPU_Matrix() noexcept;

	GPU_Matrix& operator=(const GPU_Matrix& other) noexcept;
	GPU_Matrix& operator=(GPU_Matrix&& other) noexcept;

	GPU_Matrix operator+(const GPU_Matrix& other) const noexcept;
	GPU_Matrix expendetAdd(const GPU_Matrix& other) const noexcept;
	GPU_Matrix operator-(const GPU_Matrix& other) const noexcept;
	GPU_Matrix operator*(const GPU_Matrix& other) const noexcept;
	GPU_Matrix operator%(const GPU_Matrix& other) const noexcept;
	GPU_Matrix operator+(double val) const noexcept;
	GPU_Matrix operator-(double val) const noexcept;
	GPU_Matrix operator*(double val) const noexcept;
	GPU_Matrix operator/(double val) const noexcept;

	GPU_Matrix& operator-=(const GPU_Matrix& A) noexcept;
	GPU_Matrix& operator%=(const GPU_Matrix& A) noexcept;

	GPU_Matrix T() const noexcept;
	GPU_Matrix reduceSumROW() const noexcept;
	GPU_Matrix reduceSumCOL() const noexcept;

	size_t row() const noexcept;
	size_t col() const noexcept;
	void reshape(size_t new_row, size_t new_column) noexcept;
	size_t memorySize() const noexcept;
	size_t length()const noexcept;

	static GPU_Matrix Zero(size_t row, size_t column) noexcept;
	static GPU_Matrix One(size_t row, size_t column) noexcept;
	static GPU_Matrix Random(size_t row, size_t column,std::default_random_engine& eng) noexcept;

	CPU_Matrix copyToCPU() const noexcept;
	friend CPU_Matrix;
	friend GPU_Matrix exp(const GPU_Matrix& A);
	friend GPU_Matrix log(const GPU_Matrix& A);
	friend GPU_Matrix tanh(const GPU_Matrix& A);
	friend GPU_Matrix tanh_prime(const GPU_Matrix& A);
	friend GPU_Matrix soft_max(const GPU_Matrix& A);
	friend GPU_Matrix leakReLu(const GPU_Matrix& A,double slope);
	friend GPU_Matrix leakReLu_prime(const GPU_Matrix& A,double slope);
};

GPU_Matrix operator-(double a,const GPU_Matrix& A);