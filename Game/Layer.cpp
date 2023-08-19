#include "Layer.h"

LeakReLu_Layer::LeakReLu_Layer(size_t input_size, size_t output_size, std::default_random_engine& eng, double _slope)
	:
	weight(Matrix::Random(output_size, input_size, eng)),
	bias(Matrix::Random(output_size, 1, eng)),
	Z(Matrix::Zero(output_size, 1)),
	A(Matrix::Zero(output_size, 1)),
	slope(_slope)
{
}

void LeakReLu_Layer::store(std::string name)
{
	this->weight.copyToCPU().store(name + ".weight");
	this->bias.copyToCPU().store(name + ".bias");
}

void LeakReLu_Layer::load(std::string name, std::default_random_engine& eng)
{
	CPU_Matrix w(this->weight.row(), this->weight.col());
	w.load(name + ".weight",eng);
	this->weight = w.copyToGPU();
	CPU_Matrix b(this->bias.row(), this->bias.col());
	b.load(name + ".bias",eng);
	this->bias = b.copyToGPU();
}

Matrix LeakReLu_Layer::forward(const Matrix& X) {
	auto W = weight * X;
	Z = W.expendetAdd(bias);
	A = leakReLu(Z, slope);
	return A;
}

Matrix LeakReLu_Layer::backward(const Matrix& nextLayerWeights, const Matrix& nextLayerError, const Matrix& prevLayerOutput)
{
	auto delta_l = (nextLayerWeights.T() * nextLayerError) % leakReLu_prime(this->A, slope);
	this->weight -= delta_l * prevLayerOutput.T() * learningRate / delta_l.col();
	this->bias -= delta_l.reduceSumCOL() * learningRate / delta_l.col();
	return delta_l;
}

Tanh_Layer::Tanh_Layer(size_t input_size, size_t output_size, std::default_random_engine& eng)
	:
	weight(Matrix::Random(output_size, input_size, eng)),
	bias(Matrix::Random(output_size, 1, eng)),
	Z(Matrix::Zero(output_size, 1)),
	A(Matrix::Zero(output_size, 1))
{
}

Matrix Tanh_Layer::forward(const Matrix& X)
{
	Z = (weight * X).expendetAdd(bias);
	A = tanh(Z);
	return A;
}

void Tanh_Layer::store(std::string name)
{
	this->weight.copyToCPU().store(name + ".weight");
	this->bias.copyToCPU().store(name + ".bias");
}

void Tanh_Layer::load(std::string name,std::default_random_engine& eng)
{
	CPU_Matrix w(this->weight.row(), this->weight.col());
	w.load(name + ".weight",eng);
	this->weight = w.copyToGPU();
	CPU_Matrix b(this->bias.row(), this->bias.col());
	b.load(name + ".bias",eng);
	this->bias = b.copyToGPU();

}

Matrix Tanh_Layer::lastLayerBackward(const Matrix& expected, const Matrix& prevLayerOutput)
{

	auto delta_l = (this->A - expected) % tanh_prime(this->A);
	this->weight -= delta_l * prevLayerOutput.T() * learningRate / delta_l.col();
	this->bias -= delta_l.reduceSumCOL() * learningRate / delta_l.col();
	return delta_l;
}

Matrix Tanh_Layer::backward(const Matrix& nextLayerWeights, const Matrix& nextLayerError, const Matrix& prevLayerOutput)
{
	auto delta_l = (nextLayerWeights.T() * nextLayerError) % tanh_prime(this->A);
	this->weight -= delta_l * prevLayerOutput.T() * learningRate / delta_l.col();
	this->bias -= delta_l.reduceSumCOL() * learningRate / delta_l.col();
	return delta_l;
}

OutputSoftMax_Layer::OutputSoftMax_Layer(size_t input_size, size_t output_size, std::default_random_engine& eng)
	:
	weight(Matrix::Random(output_size, input_size, eng)),
	bias(Matrix::Random(output_size, 1, eng)),
	Z(Matrix::Zero(output_size, 1)),
	A(Matrix::Zero(output_size, 1))
{
}

Matrix OutputSoftMax_Layer::forward(const Matrix& X)
{
	Z = (weight * X).expendetAdd(bias);
	A = soft_max(Z);
	return A;
}

Matrix OutputSoftMax_Layer::backward(const Matrix& expected, const Matrix& prevLayerOutput)
{
	auto delta_l = A - expected;
	this->weight -= delta_l * prevLayerOutput.T() * learningRate / delta_l.col();
	this->bias -= delta_l.reduceSumCOL() * learningRate / delta_l.col();
	return delta_l;
}
