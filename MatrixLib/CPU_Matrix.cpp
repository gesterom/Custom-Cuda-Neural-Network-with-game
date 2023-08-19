#include "pch.h"
#include "CPU_Matrix.h"

#include <assert.h>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "GPU_Matrix.h"
CPU_Matrix::CPU_Matrix() noexcept
{
	this->data = nullptr;
	this->column_ = 0;
	this->row_ = 0;
}

CPU_Matrix::CPU_Matrix(size_t row, size_t column) noexcept
{
	this->data = new double[row * column];
	this->row_ = row;
	this->column_ = column;
}

CPU_Matrix::CPU_Matrix(const CPU_Matrix& other) noexcept : CPU_Matrix(other.row(), other.col())
{
	memcpy(this->data, other.data, this->memorySize());
}

CPU_Matrix::CPU_Matrix(CPU_Matrix&& other) noexcept : data(other.data), row_(other.row_), column_(other.column_)
{
	other.data = nullptr;
	other.row_ = 0;
	other.column_ = 0;
}

CPU_Matrix::~CPU_Matrix() noexcept
{
	delete[] this->data;
}

CPU_Matrix& CPU_Matrix::operator=(const CPU_Matrix& other) noexcept
{
	if (this->row() * this->col() != other.row() * other.col()) {
		delete[] this->data;
		data = new double[other.length()];
	}
	this->row_ = other.row();
	this->column_ = other.col();
	memcpy(this->data, other.data, this->memorySize());
	return *this;
}

CPU_Matrix& CPU_Matrix::operator=(CPU_Matrix&& other) noexcept
{
	if (this->data == other.data) return *this;
	delete[] this->data;
	this->data = other.data;
	this->row_ = other.row();
	this->column_ = other.col();
	other.data = nullptr;
	other.row_ = 0;
	other.column_ = 0;
	return *this;
}

CPU_Matrix::Row_t::Row_t(CPU_Matrix& matrix, size_t n_row) noexcept : ref(matrix), row(n_row) {}

double CPU_Matrix::Row_t::operator[](size_t n_col) const noexcept
{
	return ref.data[this->row * ref.col() + n_col];
}

double& CPU_Matrix::Row_t::operator[](size_t n_col) noexcept
{
	return ref.data[this->row * ref.col() + n_col];
}

CPU_Matrix::const_Roww_t::const_Roww_t(const CPU_Matrix& matrix, size_t n_row) noexcept : ref(matrix), row(n_row) {}

double CPU_Matrix::const_Roww_t::operator[](size_t n_col) const noexcept
{
	return ref.data[this->row * ref.col() + n_col];
}

const CPU_Matrix::const_Roww_t CPU_Matrix::operator[](size_t n_row) const noexcept
{
	return const_Roww_t(*this, n_row);
}

CPU_Matrix::Row_t CPU_Matrix::operator[](size_t n_row) noexcept
{
	return Row_t(*this, n_row);
}

CPU_Matrix CPU_Matrix::operator+(const CPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] + other[i][j];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::expendetAdd(const CPU_Matrix& other) const noexcept
{
	assert(other.col() == 1);
	assert(this->row() == other.row());
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] + other[i][0];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator-(const CPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] - other[i][j];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator*(const CPU_Matrix& other) const noexcept
{
	assert(this->col() == other.row());
	size_t Z = this->col();
	CPU_Matrix res = CPU_Matrix::Zero(this->row(), other.col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < other.col(); j++) {
			for (int k = 0; k < Z; k++) {
				res[i][j] += (*this)[i][k] * other[k][j];
			}
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator%(const CPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	CPU_Matrix res = CPU_Matrix::Zero(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] * other[i][j];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator+(const double val) const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] + val;
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator-(const double val) const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] - val;
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator*(const double val) const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] * val;
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator/(const double val) const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j] / val;
		}
	}
	return res;
}

CPU_Matrix& CPU_Matrix::operator-=(const CPU_Matrix& A) noexcept
{

	assert(this->row() == A.row());
	assert(this->col() == A.col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			(*this)[i][j] -= A[i][j];
		}
	}
	return *this;
}

CPU_Matrix& CPU_Matrix::operator%=(const CPU_Matrix& A) noexcept
{
	assert(this->row() == A.row());
	assert(this->col() == A.col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			(*this)[i][j] *= A[i][j];
		}
	}
	return *this;
}

CPU_Matrix CPU_Matrix::T() const
{
	CPU_Matrix res(this->col(), this->row());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[j][i] = (*this)[i][j];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::reduceSumROW() const noexcept
{
	auto res = CPU_Matrix::Zero(1, this->col());
	for (int col = 0; col < this->col(); col++) {
		for (int row = 0; row < this->row(); row++) {
			res[0][col] += (*this)[row][col];
		}
	}
	return res;
}

CPU_Matrix CPU_Matrix::reduceSumCOL() const noexcept
{
	auto res = CPU_Matrix::Zero(this->row(), 1);
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][0] += (*this)[i][j];
		}
	}
	return res;
}

void CPU_Matrix::store(std::string filename) const
{
	std::ofstream file(filename);
	file << this->row() << " " << this->col() << "\n";
	for (size_t i = 0; i < this->row(); i++) {
		for (size_t j = 0; j < this->col(); j++) {
			file << (*this)[i][j] << " ";
		}
		file << "\n";
	}
}

void CPU_Matrix::load(std::string filename,std::default_random_engine& eng )
{
	std::ifstream file(filename);

	if (file.good() == false) (*this) = std::move(CPU_Matrix::Random(this->row(), this->col(),eng));

	file >> this->row_;
	file >> this->column_;
	delete[] this->data;
	this->data = new double[this->length()];
	for (size_t i = 0; i < this->row(); i++) {
		for (size_t j = 0; j < this->col(); j++) {
			file >> (*this)[i][j];
		}
	}
}

size_t CPU_Matrix::row() const noexcept
{
	return row_;
}

size_t CPU_Matrix::col() const noexcept
{
	return column_;
}

void CPU_Matrix::reshape(size_t new_row, size_t new_column) noexcept
{
	assert(this->row() * this->col() == new_row * new_column);
	this->row_ = new_row;
	this->column_ = new_column;
}

size_t CPU_Matrix::memorySize() const noexcept
{
	return this->length() * sizeof(double);
}

size_t CPU_Matrix::length() const noexcept
{
	return this->row() * this->col();
}

GPU_Matrix CPU_Matrix::copyToGPU() const noexcept
{
	GPU_Matrix res(this->row(), this->col());
	cudaMemcpy(res.data, this->data, this->memorySize(), cudaMemcpyHostToDevice);
	return res; // FUN
}

CPU_Matrix CPU_Matrix::Zero(size_t row, size_t column) noexcept
{
	CPU_Matrix res(row, column);
	for (size_t i = 0; i < res.length(); i++) {
		res.data[i] = 0;
	}
	return res;
}

CPU_Matrix CPU_Matrix::One(size_t row, size_t column) noexcept
{
	CPU_Matrix res(row, column);
	for (size_t i = 0; i < res.length(); i++) {
		res.data[i] = 1;
	}
	return res;
}

CPU_Matrix CPU_Matrix::HotOne(size_t row, size_t column, size_t i, size_t j) noexcept
{
	auto res = Zero(row,column);
	res[i][j] = 1;
	return res;
}

CPU_Matrix CPU_Matrix::Random(size_t row, size_t column, std::default_random_engine& eng) noexcept
{
	std::uniform_real_distribution<> dist(-1.0, 1.0);
	double weight_scale = std::sqrt(6.0 / (row + column));

	CPU_Matrix res(row, column);

	for (int i = 0; i < res.length(); i++) {
		res.data[i] = dist(eng) * weight_scale;
	}
	return res;
}

CPU_Matrix CPU_Matrix::operator>>(std::function<double(double)> func) const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = func((*this)[i][j]);
		}
	}
	return res;
}

CPU_Matrix& CPU_Matrix::operator>>=(std::function<double(double)> func) noexcept
{
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			(*this)[i][j] = func((*this)[i][j]);
		}
	}
	return *this;
}

CPU_Matrix CPU_Matrix::appendRow(const CPU_Matrix& other)
{
	assert(this->col() == other.col());
	CPU_Matrix res(this->row()+other.row(),this->col());
	for (int i = 0; i < this->row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i][j] = (*this)[i][j];
		}
	}
	for (int i = 0; i < other.row(); i++) {
		for (int j = 0; j < this->col(); j++) {
			res[i + this->row()][j] = other[i][j];
		}
	}
	return res;
}

double max(double a, double b) {
	if (a > b)return a;
	else return b;
}

double max(const CPU_Matrix& A)
{
	double max_ = A[0][0];
	A >> [&max_](double a) {max_ = max(max_, a); return a; };
	return max_;
}

CPU_Matrix max_Col(const CPU_Matrix& A) {
	CPU_Matrix res(1,A.col());
	for (int i = 0; i < A.row(); i++) {
		for(int j = 0 ; j < A.col();j++)
			if (i == 0) {
				res[i][j] = A[i][j];
			}
			else {
				res[i][j] = max(A[i][j],res[i][j]);
			}
	}
	return res;
}

CPU_Matrix subTVectorExp(const CPU_Matrix& A, const CPU_Matrix& B) {
	CPU_Matrix res(A.row(), A.col());
	for (int i = 0; i < A.row(); i++) {
		for (int j = 0; j < A.col(); j++)
		{
			res[i][j] = exp(A[i][j] - B[0][j]);
		}
	}
	return res;
}

CPU_Matrix divTVectorElemenWise(const CPU_Matrix& A, const CPU_Matrix& B) {
	CPU_Matrix res(A.row(), A.col());
	for (int i = 0; i < A.row(); i++) {
		for (int j = 0; j < A.col(); j++)
		{
			res[i][j] = A[i][j] / B[0][j];
		}
	}
	return res;
}

CPU_Matrix exp(const CPU_Matrix& A)
{
	return A >> [](double a) {return exp(a); };
}

CPU_Matrix tanh(const CPU_Matrix& A)
{
	return A >> [](double a) {return tanh(a); };
}

CPU_Matrix tanh_prime(const CPU_Matrix& A)
{
	return A >> [](double a) {return 1 - a * a; };
}

CPU_Matrix soft_max(const CPU_Matrix& A)
{
	CPU_Matrix max_A = max_Col(A);
	CPU_Matrix t = subTVectorExp(A,max_A);
	return divTVectorElemenWise(t , t.reduceSumROW() );
}

CPU_Matrix leakReLu(const CPU_Matrix& A, double slope)
{
	return A >> [slope](double a) {if (a > 0)return a; else return slope * a; };
}

CPU_Matrix leakReLu_prime(const CPU_Matrix& A, double slope)
{
	return A >> [slope](double a) -> double {
		if (a > 0) return 1;
		else return slope;
	};
}

double mseLoss(const CPU_Matrix& A, const CPU_Matrix& B) {
	auto t = (A - B);
	t = t % t;
	return (t.reduceSumCOL().reduceSumROW()[0][0] * 0.5) / A.row();
}