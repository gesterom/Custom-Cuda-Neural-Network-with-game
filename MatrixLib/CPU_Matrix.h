#pragma once

#include <random>
#include <functional>

class GPU_Matrix;
class CPU_Matrix;

double mseLoss(const CPU_Matrix& output, const CPU_Matrix& expected);
double max(const CPU_Matrix& A);
CPU_Matrix exp(const CPU_Matrix& A);
CPU_Matrix tanh(const CPU_Matrix& A);
CPU_Matrix tanh_prime(const CPU_Matrix& A);
CPU_Matrix soft_max(const CPU_Matrix& A);
CPU_Matrix leakReLu(const CPU_Matrix& A, double slope);
CPU_Matrix leakReLu_prime(const CPU_Matrix& A, double slope);

class CPU_Matrix
{
	double* data;
	size_t row_;
	size_t column_;
public:
	CPU_Matrix() noexcept;
	CPU_Matrix(size_t row, size_t column)noexcept;
	CPU_Matrix(const CPU_Matrix& other)noexcept;
	CPU_Matrix(CPU_Matrix&& other) noexcept;
	~CPU_Matrix() noexcept;

	CPU_Matrix& operator=(const CPU_Matrix& other) noexcept;
	CPU_Matrix& operator=(CPU_Matrix&& other) noexcept;
	class Row_t {
		CPU_Matrix& ref;
		size_t row;
	public:
		Row_t(CPU_Matrix& matrix, size_t n_row) noexcept;
		double operator[](size_t) const noexcept;
		double& operator[](size_t) noexcept;
	};
	class const_Roww_t {
		const CPU_Matrix& ref;
		size_t row;
	public:
		const_Roww_t(const CPU_Matrix& matrix, size_t n_row) noexcept;
		double operator[](size_t) const noexcept;
	};
	const const_Roww_t operator[](size_t) const noexcept;
	Row_t operator[](size_t) noexcept;

	CPU_Matrix operator+(const CPU_Matrix& other) const noexcept;
	CPU_Matrix expendetAdd(const CPU_Matrix& other) const noexcept;
	CPU_Matrix operator-(const CPU_Matrix& other) const noexcept;
	CPU_Matrix operator*(const CPU_Matrix& other) const noexcept;
	CPU_Matrix operator%(const CPU_Matrix& other) const noexcept;
	CPU_Matrix operator+(const double val) const noexcept;
	CPU_Matrix operator-(const double val) const noexcept;
	CPU_Matrix operator*(const double val) const noexcept;
	CPU_Matrix operator/(const double val) const noexcept;

	CPU_Matrix& operator-=(const CPU_Matrix& A) noexcept;
	CPU_Matrix& operator%=(const CPU_Matrix& A) noexcept;

	CPU_Matrix T() const;

	CPU_Matrix reduceSumROW() const noexcept;
	CPU_Matrix reduceSumCOL() const noexcept;

	void store(std::string) const;
	void load(std::string, std::default_random_engine& eng);

	size_t row() const noexcept;
	size_t col() const noexcept;
	void reshape(size_t new_row, size_t new_column) noexcept;
	size_t memorySize() const noexcept;
	size_t length()const noexcept;

	GPU_Matrix copyToGPU() const noexcept;

	static CPU_Matrix Zero(size_t row, size_t column) noexcept;
	static CPU_Matrix One(size_t row, size_t column) noexcept;
	static CPU_Matrix HotOne(size_t row, size_t column,size_t i,size_t j) noexcept;
	static CPU_Matrix Random(size_t row, size_t column, std::default_random_engine& eng) noexcept;

	CPU_Matrix operator>>(std::function<double(double)> func) const noexcept;
	CPU_Matrix& operator>>=(std::function<double(double)> func) noexcept;
	CPU_Matrix appendRow(const CPU_Matrix&);
	friend GPU_Matrix;
};

