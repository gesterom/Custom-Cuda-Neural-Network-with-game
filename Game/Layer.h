#pragma once

#include "Matrix.h"

typedef GPU_Matrix Matrix;

class LeakReLu_Layer
{
	public:
	Matrix weight;
	Matrix bias;
	double slope;
	Matrix Z;
	Matrix A;

	double learningRate = 0.01;
	LeakReLu_Layer(size_t input_size,size_t output_size,std::default_random_engine& eng,double slope = 0.01);
	void store(std::string name);
	void load(std::string name, std::default_random_engine& eng);
	Matrix forward(const Matrix& X);
	Matrix backward(const Matrix& nextLayerWeights, const Matrix& nextLayerError, const Matrix& prevLayerOutput);
};

class OutputSoftMax_Layer
{
public:
	Matrix weight;
	Matrix bias;
	Matrix Z;
	Matrix A;

	double learningRate = 0.01;
	OutputSoftMax_Layer(size_t input_size, size_t output_size, std::default_random_engine& eng);
	Matrix forward(const Matrix& X);
	Matrix backward(const Matrix& expected, const Matrix& prevLayerOutput);
};

class Tanh_Layer
{
public:
	Matrix weight;
	Matrix bias;
	Matrix Z;
	Matrix A;

	double learningRate = 0.01;
	Tanh_Layer(size_t input_size, size_t output_size, std::default_random_engine& eng);

	Matrix forward(const Matrix& X);
	void store(std::string);
	void load(std::string, std::default_random_engine& eng);
	Matrix lastLayerBackward(const Matrix& error, const Matrix& prevLayerOutput);
	Matrix backward(const Matrix& nextLayerWeights, const Matrix& nextLayerError, const Matrix& prevLayerOutput);
};
