#include "pch.h"
#include "GPU_Matrix.h"

#include <assert.h>
#include <iostream>

#include "CPU_Matrix.h"

//#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)

#define blockDim (_DUMY_())
#define blockIdx (_DUMY_())
#define threadIdx (_DUMY_())


#endif

GPU_Matrix::GPU_Matrix(size_t row, size_t column) noexcept
{
	this->row_ = row;
	this->column_ = column;
	this->data = nullptr;
	auto err = cudaMalloc(&(this->data), this->memorySize());
	//std::cout<<"ADDres : "<<this->data<<std::endl; 
	if (err != cudaSuccess)
	{
		std::cout << cudaGetErrorName(err) << std::endl;
		std::cout << "\t" << cudaGetErrorString(err) << std::endl;
		exit(-3);
	}
}

GPU_Matrix::GPU_Matrix(const GPU_Matrix& other) noexcept : GPU_Matrix(other.row(), other.col())
{
	cudaMemcpy(this->data, other.data, other.memorySize(), cudaMemcpyDeviceToDevice);
}

GPU_Matrix::GPU_Matrix(GPU_Matrix&& other) noexcept : data(other.data), row_(other.row_), column_(other.column_)
{
	other.data = nullptr;
	other.row_ = 0;
	other.column_ = 0;
}
GPU_Matrix::~GPU_Matrix() noexcept
{
	cudaFree(this->data);
}

GPU_Matrix& GPU_Matrix::operator=(const GPU_Matrix& other) noexcept
{
	if (this->row() * this->col() != other.row() * other.col()) {
		cudaFree(this->data);
		cudaMalloc(&(this->data), other.memorySize());
	}
	this->row_ = other.row();
	this->column_ = other.col();
	cudaMemcpy(this->data, other.data, other.memorySize(), cudaMemcpyDeviceToDevice);
	return *this;
}

__global__ void memSetDouble(double* A, double value, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		A[index] = value;
	}
}

GPU_Matrix& GPU_Matrix::operator=(GPU_Matrix&& other) noexcept
{
	if (this->data == other.data) return *this;
	cudaFree(this->data);
	this->data = other.data;
	this->row_ = other.row();
	this->column_ = other.col();
	other.data = nullptr;
	other.row_ = 0;
	other.column_ = 0;
	return *this;
}

__global__ void addMatrix(double* A, double* B, double* C, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		C[index] = A[index] + B[index];
	}
}

GPU_Matrix GPU_Matrix::operator+(const GPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	GPU_Matrix res(this->row(), this->col());

	addMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, res.data, this->row(), this->col());

	return res;
}

__global__ void addVector_ToMatrix(double* A, double* B, double* C, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		C[index] = A[index] + B[blockIdx.x];
	}
}

GPU_Matrix GPU_Matrix::expendetAdd(const GPU_Matrix& other) const noexcept
{
	assert(other.col() == 1);
	assert(this->row() == other.row());
	GPU_Matrix res(this->row(), this->col());
	addVector_ToMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, res.data, this->row(), this->col());
	return res;
}

__global__ void subMatrix(double* A, double* B, double* C, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		C[index] = A[index] - B[index];
	}
}

GPU_Matrix GPU_Matrix::operator-(const GPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	GPU_Matrix res(this->row(), this->col());

	subMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, res.data, this->row(), this->col());

	return res;
}

__global__ void mulMatrix(double* A, double* B, double* C, size_t row, size_t Z, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	double res = 0;
	if (blockIdx.x < row && threadIdx.x < col) {
		for (size_t k = 0; k < Z; k++) {
			//C[index] += A[Z * blockIdx.x + (i + threadIdx.x) % Z] * B[column * ((i + blockIdx.x) % column) + threadIdx.x];
			res += A[blockIdx.x * Z + k] * B[k * col + threadIdx.x];
		}
	}
	C[index] = res;
	//for (int k = 0; k < Z; k++) {
	//	res[i][j] += (*this)[i][k] * other[k][j];
	//}
}

GPU_Matrix GPU_Matrix::operator*(const GPU_Matrix& B) const noexcept
{
	assert(this->col() == B.row());
	GPU_Matrix res = GPU_Matrix::Zero(this->row(), B.col());
	size_t Z = this->col();

	mulMatrix KERNEL_ARGS2(this->row(), B.col()) (this->data, B.data, res.data, this->row(), Z, B.col());

	return res;
}

__global__ void mulElementlMatrix(double* A, double* B, double* C, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		C[index] = A[index] * B[index];
	}
}

GPU_Matrix GPU_Matrix::operator%(const GPU_Matrix& other) const noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	GPU_Matrix res(this->row(), this->col());

	mulElementlMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, res.data, this->row(), this->col());

	return res;
}

__global__ void addValMatrix(double* A, double* B, double C, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		B[index] = A[index] + C;
	}
}

GPU_Matrix GPU_Matrix::operator+(double value) const noexcept
{
	GPU_Matrix res(this->row(), this->col());

	addValMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, res.data, value, this->row(), this->col());
	return res;
}

GPU_Matrix GPU_Matrix::operator-(double value) const noexcept
{
	GPU_Matrix res(this->row(), this->col());

	addValMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, res.data, -value, this->row(), this->col());
	return res;
}

__global__ void mulValMatrix(double* A, double B, double* C, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		C[index] = A[index] * B;
	}
}

GPU_Matrix GPU_Matrix::operator*(double value) const noexcept
{
	GPU_Matrix res(this->row(), this->col());

	mulValMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, value, res.data, this->row(), this->col());

	return res;
}

GPU_Matrix GPU_Matrix::operator/(double value) const noexcept
{
	GPU_Matrix res(this->row(), this->col());

	mulValMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, 1.0 / value, res.data, this->row(), this->col());

	return res;
}

__global__ void subAsigneMatrix(double* A, double* B, size_t row, size_t col) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * col) {
		A[index] -= B[index];
	}
}

GPU_Matrix& GPU_Matrix::operator-=(const GPU_Matrix& other) noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());

	subAsigneMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, this->row(), this->col());

	return *this;
}

__global__ void mulElementAsigneMatrix(double* A, double* B, size_t size) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		A[index] *= B[index];
	}
}

GPU_Matrix& GPU_Matrix::operator%=(const GPU_Matrix& other) noexcept
{
	assert(this->row() == other.row());
	assert(this->col() == other.col());
	mulElementAsigneMatrix KERNEL_ARGS2(this->row(), this->col()) (this->data, other.data, this->row() * this->col());
	return *this;
}

__global__ void transpose_Matrix(double* A, double* B, size_t row, size_t col) {
	size_t index_1 = blockIdx.x * col + threadIdx.x;
	size_t index_2 = blockIdx.x + row * threadIdx.x;
	if (index_1 < row * col && index_2 < row * col) {
		B[index_2] = A[index_1];
	}
}

GPU_Matrix GPU_Matrix::T() const noexcept
{
	GPU_Matrix res(this->col(), this->row());

	transpose_Matrix KERNEL_ARGS2(this->row(), this->col()) (this->data, res.data, this->row(), this->col());

	return res;
}


__global__ void reduceSumROW_Matrix(double* A, double* B, size_t row, size_t column) {
	if (threadIdx.x < column) {
		for (size_t j = 0; j < row; j++)
		{
			B[threadIdx.x] += A[j * column + threadIdx.x];
		}
	}
}

GPU_Matrix GPU_Matrix::reduceSumROW() const noexcept
{
	auto res = GPU_Matrix::Zero(1, this->col());

	reduceSumROW_Matrix KERNEL_ARGS2(1, this->col()) (this->data, res.data, this->row(), this->col());

	return res;
}

__global__ void reduceSumCOL_Matrix(double* A, double* B, size_t row, size_t column) {
	if (threadIdx.x < row) {
		for (size_t j = 0; j < column; j++)
		{
			B[threadIdx.x] += A[threadIdx.x * column + j];
		}
	}
}

GPU_Matrix GPU_Matrix::reduceSumCOL() const noexcept
{
	auto res = GPU_Matrix::Zero(this->row(), 1);

	reduceSumCOL_Matrix KERNEL_ARGS2(1, this->row()) (this->data, res.data, this->row(), this->col());

	return res;
}

size_t GPU_Matrix::row() const noexcept
{
	return row_;
}

size_t GPU_Matrix::col() const noexcept
{
	return column_;
}

void GPU_Matrix::reshape(size_t new_row, size_t new_column) noexcept
{
	assert(this->row() * this->col() == new_row * new_column);
	this->row_ = new_row;
	this->column_ = new_column;
}

size_t GPU_Matrix::memorySize() const noexcept
{
	return this->length() * sizeof(double);
}

size_t GPU_Matrix::length() const noexcept
{
	return this->row() * this->col();
}

CPU_Matrix GPU_Matrix::copyToCPU() const noexcept
{
	CPU_Matrix res(this->row(), this->col());
	cudaMemcpy(res.data, this->data, this->memorySize(), cudaMemcpyDeviceToHost);
	return res;
}

GPU_Matrix GPU_Matrix::Zero(size_t row, size_t column) noexcept
{
	return GPU_Matrix(row, column, 0.0);
}

GPU_Matrix GPU_Matrix::One(size_t row, size_t column) noexcept
{
	return GPU_Matrix(row, column, 1);
}

GPU_Matrix GPU_Matrix::Random(size_t row, size_t column, std::default_random_engine& eng) noexcept
{
	return CPU_Matrix::Random(row, column, eng).copyToGPU();
}


GPU_Matrix::GPU_Matrix(size_t row, size_t column, double val) noexcept : GPU_Matrix(row, column)
{
	memSetDouble KERNEL_ARGS2(row, column) (this->data, val, row, column);
}


__global__ void exp_Matrix(double* A, double* B, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		B[index] = exp(A[index]);
	}
}

GPU_Matrix exp(const GPU_Matrix& A)
{
	GPU_Matrix res(A.row(), A.col());
	exp_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, res.data, A.row(), A.col());
	return res;
}

__global__ void log_Matrix(double* A, double* B, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		B[index] = log(A[index]);
	}
}

GPU_Matrix log(const GPU_Matrix& A) {
	GPU_Matrix res(A.row(), A.col());
	log_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, res.data, A.row(), A.col());
	return res;
}

__global__ void tanh_Matrix(double* A, double* B, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		B[index] = tanh(A[index]);
	}
}

GPU_Matrix tanh(const GPU_Matrix& A) {
	GPU_Matrix res(A.row(), A.col());
	tanh_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, res.data, A.row(), A.col());
	return res;
}

GPU_Matrix tanh_prime(const GPU_Matrix& A)
{
	return (A % A * -1) + 1;
}

__global__ void softMax_Max_helper_Matrix(double* A, double* B, size_t row, size_t col) {
	B[threadIdx.x] = A[threadIdx.x];
	for (int i = 1; i < row; i++) {
		if (A[i * col + threadIdx.x] > B[threadIdx.x]) {
			B[threadIdx.x] = A[i * col + threadIdx.x];
		}
	}
}

__global__ void softMax_helper_Matrix(double* A, double* B, double* C, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		C[index] = A[index] / B[threadIdx.x];
	}
}

__global__ void softMax_helper_exp_Matrix(double* A, double* B, double* C, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		C[index] = exp(A[index] - B[threadIdx.x]);
	}
}

GPU_Matrix soft_max(const GPU_Matrix& A)
{
	GPU_Matrix max_A(1, A.col());
	softMax_Max_helper_Matrix KERNEL_ARGS2(max_A.row(), max_A.col()) (A.data, max_A.data, A.row(), A.col());
	GPU_Matrix t(A.row(), A.col());
	softMax_helper_exp_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, max_A.data, t.data, A.row(), A.col());

	GPU_Matrix sums = t.reduceSumROW();
	GPU_Matrix res(A.row(), A.col());

	softMax_helper_Matrix KERNEL_ARGS2(res.row(), res.col()) (t.data, sums.data, res.data, A.row(), A.col());
	return res;
}

__global__ void leakReLu_Matrix(double* A, double* B, double slope, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		if (A[index] > 0) {
			B[index] = A[index];
		}
		else {
			B[index] = A[index] * slope;
		}
	}
}

GPU_Matrix leakReLu(const GPU_Matrix& A, double slope) {
	GPU_Matrix res(A.row(), A.col());
	leakReLu_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, res.data, slope, A.row(), A.col());
	return res;
}

__global__ void leakReLu_prime_Matrix(double* A, double* B, double slope, size_t row, size_t column) {
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < row * column) {
		if (A[index] > 0) {
			B[index] = 1;
		}
		else {
			B[index] = slope;
		}
	}
}

GPU_Matrix leakReLu_prime(const GPU_Matrix& A, double slope) {
	GPU_Matrix res(A.row(), A.col());
	leakReLu_prime_Matrix KERNEL_ARGS2(A.row(), A.col()) (A.data, res.data, slope, A.row(), A.col());
	return res;
}

GPU_Matrix operator-(double a, const GPU_Matrix& A)
{
	return A * (-1) + a;
}

double mseLoss(const GPU_Matrix& A, const GPU_Matrix& expected) {
	auto t = (A - expected);
	t %= t;
	return (t.copyToCPU().reduceSumCOL().reduceSumROW()[0][0] * 0.5) / A.row();
}

double crossEntropy(const GPU_Matrix& A, const GPU_Matrix& expected) {
	return (expected * log(A) + (1 - expected) * log(1 - A)).reduceSumCOL().reduceSumROW().copyToCPU()[0][0];
}