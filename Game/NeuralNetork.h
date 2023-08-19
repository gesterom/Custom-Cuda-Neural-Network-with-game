#pragma once

#include <random>

#include "Player.h"
#include "Layer.h"

class NeuralNetwork
{
	std::default_random_engine& eng;
	std::vector<LeakReLu_Layer*> layers;
	Tanh_Layer outputLayer;
	double learningRate;
	public:
	NeuralNetwork(std::default_random_engine& eng,std::vector<int> layerSize,double learningRate = 0.01);
	~NeuralNetwork();
	EvalFunc getFunc();
	void store(std::string name);
	void load(std::string name);
	GPU_Matrix forwward(const GPU_Matrix& input);
	double train(const GPU_Matrix& input,const GPU_Matrix& expected);
};

