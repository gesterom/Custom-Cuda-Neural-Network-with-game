#include "NeuralNetork.h"
#include <cassert>

NeuralNetwork::NeuralNetwork(std::default_random_engine& eng_, std::vector<int> layerSize, double learningRate_)
	:
	eng(eng_),
	outputLayer(layerSize[layerSize.size() - 2], layerSize[layerSize.size() - 1], eng_),
	learningRate(learningRate_)
{
	assert(layerSize.size() > 2);
	for (int i = 0; i < layerSize.size() - 2; i++) {
		layers.push_back(new LeakReLu_Layer(layerSize[i], layerSize[i + 1], eng));
	}
}

NeuralNetwork::~NeuralNetwork()
{
	for (auto i : layers) {
		delete i;
	}
}

std::pair<int, int> maxArg(const CPU_Matrix& A) {
	//double max = A[0][0];
	int max_i = 0;
	int max_j = 0;
	for (int i = 0; i < A.row(); i++) {
		for (int j = 0; j < A.col(); j++) {
			if (A[i][j] > A[max_i][max_j]) {
				max_i = i;
				max_j = j;
			}
		}
	}
	return std::make_pair(max_i, max_j);
}

EvalFunc NeuralNetwork::getFunc()
{
	return [this](const GameState& state)-> EvalResult {
		if (state.gameOver() != GameResult::gameNotOver) {
			return simpleEval(state);
		}
		auto z = convertToMatrix(state);
		//z.reshape(25, 1);
		z.reshape(z.col(),1);
		auto res = this->forwward(z.copyToGPU());
		return EvalResult{ 0, res.copyToCPU()[0][0], -1 };
	};
}

GPU_Matrix NeuralNetwork::forwward(const GPU_Matrix& input)
{
	layers[0]->forward(input);
	for (int i = 1; i < this->layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->A);
	}
	outputLayer.forward(layers[layers.size() - 1]->A);
	return outputLayer.A;
}

void NeuralNetwork::store(std::string name) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->store(name + "_hiden_" + std::to_string(i));
	}
	outputLayer.store(name + "_output_");
}
void NeuralNetwork::load(std::string name) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->load(name + "_hiden_" + std::to_string(i),eng);
	}
	outputLayer.load(name + "_output_",eng);
}

double NeuralNetwork::train(const GPU_Matrix& input, const GPU_Matrix& rewards)
{
	this->forwward(input);
	auto A = this->outputLayer.A;
	auto R = rewards;
	auto expected = (A - (R % R) % A) + R;
	double loss = mseLoss(this->outputLayer.A, expected);
	auto delta = outputLayer.lastLayerBackward(expected, layers[layers.size() - 1]->A);
	delta = layers[layers.size() - 1]->backward(outputLayer.weight, delta, layers[layers.size() - 2]->A);
	for (size_t i = layers.size() - 2; i > 0; i--) {

		delta = layers[i]->backward(layers[i + 1]->weight, delta, layers[i - 1]->A);
	}
	layers[0]->backward(layers[1]->weight, delta, input);
	return loss;
}
