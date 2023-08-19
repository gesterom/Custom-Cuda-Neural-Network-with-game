// Game.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "Layer.h"
#include "NeuralNetork.h"

void trainv2(NeuralNetwork& net, std::default_random_engine& eng, double epsilon = 0.05, int numberOfGames = 5, int iterations = 32) {
	std::uniform_real_distribution<> dis(0, 1);
	int trainingDepth = 3;
	int consecciveLosses = 0;
	while (trainingDepth < 8) {
		MinMaxAlgorythm alg(eng, net.getFunc(), "network", trainingDepth);
		CPU_Matrix episode(0, 25 + 48 + 1);
		CPU_Matrix expected(0, 1);
		if (trainingDepth > 1) {
			numberOfGames = 5;
		}
		for (int gameID = 0; gameID < numberOfGames; gameID++) {
			GameState state;
			int turn = 0;
			while (state.gameOver() == GameResult::gameNotOver)
			{
				Move m;
				//std::cout << "gameID : " << gameID << "\n" << state.toString() << std::endl;
				EvalResult eval = alg.minmax(state);
				if (dis(eng) < epsilon) {
					m = state.generateMoveset(eng)[0];
				}
				else {
					m = eval.move;
				}
				//if (trainingDepth >= 2 || eval.mate != 0 ) {
				if (turn != 0) {
					for (auto mm : state.transposePositions()) {
						auto t = convertToMatrix(mm);
						//t.reshape(1, 25);
						episode = episode.appendRow(t);
						auto z = CPU_Matrix(1, 1);
						z[0][0] = eval.eval;
						expected = expected.appendRow(z);
					}
				}
				else {
					auto t = convertToMatrix(state);
					//t.reshape(1, 25);
					episode = episode.appendRow(t);
					auto z = CPU_Matrix(1, 1);
					z[0][0] = eval.eval;
					expected = expected.appendRow(z);
				}
				//}

				if (state.move(m) == state) { exit(-11); };
				state = state.move(m);
				turn++;
			}
			std::cout << "gameID : " << gameID << "\n" << state.toString() << std::endl;
		}
		std::cout << "data set size : " << episode.row() << std::endl;

		episode.store("episode");
		expected.store("expected");

		double loss = 10000;
		for (int i = 0; i < iterations; i++) {
			loss = net.train(episode.T().copyToGPU(), expected.T().copyToGPU());
			std::cout << "Loss : " << loss << " Iteration : " << i << std::endl;
			iterations = (int)loss + 16;
			//if (loss < 2.5) break;
			net.store("Network5/");
		}
		if (loss < 1) {
			consecciveLosses++;
			if (consecciveLosses > 5) {
				trainingDepth++;
				consecciveLosses = 0;
			}
		}
		else {
			consecciveLosses = 0;
		}
		std::cout << "consecciveLosses : " << consecciveLosses << " trainingDepth " << trainingDepth << std::endl;
	}
}

void train(NeuralNetwork& net, std::default_random_engine& eng, double epsilon = 0.05, int depth = 4, int numberOfSessions = 16, int numberOfGames = 16, int iterations = 10) {
	std::uniform_real_distribution<> dis(0, 1);
	for (int sesionID = 0; sesionID < numberOfSessions; sesionID++) {
		MinMaxAlgorythm alg(eng, net.getFunc(), "network", depth);
		CPU_Matrix episode(0, 25);
		CPU_Matrix expected = CPU_Matrix::Zero(0, 1);
		for (int gameID = 0; gameID < numberOfGames; gameID++) {
			GameState state;
			//std::vector<int> actions;
			while (state.gameOver() == GameResult::gameNotOver)
			{
				Move m;
				std::cout << "gameID : " << gameID << "\n" << state.toString() << std::endl;
				EvalResult eval = alg.minmax(state);
				if (dis(eng) < epsilon) {
					m = state.generateMoveset(eng)[0];
				}
				else {
					m = eval.move;
				}
				auto t = convertToMatrix(state);
				if (state.move(m) == state) { exit(-11); };
				state = state.move(m);


				net.forwward(t.copyToGPU());
				t.reshape(1, 25);
				episode = episode.appendRow(t);
				auto z = CPU_Matrix(1, 1);
				z[0][0] = eval.eval;
				expected = expected.appendRow(z);
			}
			std::cout << "gameID : " << gameID << "\n" << state.toString() << std::endl;
		}
		for (int i = 0; i < iterations; i++) {
			episode.store("episodes");
			expected.store("expected");
			double loss = net.train(episode.T().copyToGPU(), expected.T().copyToGPU());
			net.store("Network2/");
			std::cout << "Loss : " << loss << " SesionID : " << sesionID << " Iteration : " << i << std::endl;
		}
	}
}

void humanPlay(NeuralNetwork& net, std::default_random_engine& eng) {
	GameState state;
	MinMaxAlgorythm alg(eng, net.getFunc(), "network");
	//MinMaxAlgorythm alg(eng, simpleEval, "simple_1", 1);
	while (state.gameOver() == GameResult::gameNotOver)
	{
		std::cout << state.toString() << std::endl;
		std::cout << alg.minmax(state) << std::endl;
		int x, y;
		std::cin >> x >> y;
		state = state.move(x, y);
	}
}

double singielMatch(MinMaxAlgorythm& A, MinMaxAlgorythm& B) {
	GameState state;
	while (state.gameOver() == GameResult::gameNotOver)
	{
		if (state.firstPlayerTurn()) {
			auto t = A.minmax(state);
			state = state.move(t.move);
		}
		else {
			auto t = B.minmax(state);
			state = state.move(t.move);
		}
	}
	std::cout << A.name << " vs " << B.name << std::endl << state.toString();
	if (state.gameOver() == GameResult::FirstPlayerWin) {
		return 1;
	}
	else if (state.gameOver() == GameResult::SecondPlayerWin) {
		return -1;
	}
	else {
		return 0;
	}
}

void turnament(int numberOfMatches, std::vector<MinMaxAlgorythm> algs) {
	std::ofstream file("turnaments_v2", std::ios::app);
	file << std::setw(12) << " ";
	for (int i = 0; i < algs.size(); i++) {
		file << std::setprecision(5) << std::setw(16) << algs[i].name << " ";
	}
	file << "\n";
	for (int i = 0; i < algs.size(); i++) {
		file << std::setprecision(5) << std::setw(16) << algs[i].name << " ";
		for (int j = 0; j < algs.size(); j++) {
			double sum = 0;
			double winrate = 0;
			for (int k = 0; k < numberOfMatches; k++) {
				std::cout << "k : " << k << std::endl;
				auto r = singielMatch(algs[i], algs[j]);
				if (r == 1) {
					winrate += r;
				}
				sum += r;
			}
			file << std::setprecision(5) << std::setw(12) << sum / numberOfMatches << std::setprecision(3) << " [" << winrate / numberOfMatches * 100 << "%] " << std::flush;
		}
		file << "\n";
	}
}

double XOR_Test(double lr)
{
	std::random_device rg;
	std::default_random_engine eng(rg());
	NeuralNetwork* net = new NeuralNetwork(eng, { 2,2,2,1 }, lr);

	CPU_Matrix sameple = CPU_Matrix::Zero(2, 4);
	CPU_Matrix expectedValue = CPU_Matrix::Zero(1, 4);
	//net->load();
	for (int i = 0; i < 4; i++) {
		int i_1 = (i % 4) % 2;
		int i_2 = (i % 4) / 2;
		sameple[0][i] = i_1 * 0.5;
		sameple[1][i] = i_2 * 0.5;
		expectedValue[0][i] = (i_1 ^ i_2) * 0.5;
	}
	double res = 1000;
	int i=0;
	for (i = 0; res >= 0.08; i++)
	{
		res = net->train(sameple.copyToGPU(), expectedValue.copyToGPU());
		//std::cout <<"lr : "<<lr<<" loss : " << res << std::endl;
	}
	return i;
}

void lista4() {
	for(int i=10;i<21;i++){
		std::cout<< (float)i * 0.01 <<" -> " << XOR_Test((float)i * 0.01) << std::endl;
	}
}

int main(int argc, char** args)
{
	lista4();
	return 0;
	//std::random_device rg;
	std::default_random_engine eng(std::chrono::system_clock::now().time_since_epoch().count());
	if (argc > 1) {
		if (std::string("turnament") == args[1]) {
			std::vector<MinMaxAlgorythm> algs;
			//algs.emplace_back(MinMaxAlgorythm(eng, net.getFunc(), "network_2", 2));
			//algs.emplace_back(MinMaxAlgorythm(eng, net.getFunc(), "network_4", 4));
			//algs.emplace_back(MinMaxAlgorythm(eng, net.getFunc(), "network_5", 5));
			algs.emplace_back(MinMaxAlgorythm(eng, montecarloEval(eng, 100), "monte_100_4", 4));
			algs.emplace_back(MinMaxAlgorythm(eng, montecarloEval(eng, 100), "monte_100_5", 5));
			algs.emplace_back(MinMaxAlgorythm(eng, montecarloEval(eng, 256), "monte_256_4", 4));
			algs.emplace_back(MinMaxAlgorythm(eng, montecarloEval(eng, 256), "monte_256_5", 5));
			//algs.emplace_back(MinMaxAlgorythm(eng, simpleEval, "simple_3", 3));
			algs.emplace_back(MinMaxAlgorythm(eng, simpleEval, "simple_4", 4));
			algs.emplace_back(MinMaxAlgorythm(eng, simpleEval, "simple_5", 5));
			//algs.emplace_back(MinMaxAlgorythm(eng, simpleEval, "simple_6", 6));
			algs[0].use_cashe = false;
			algs[1].use_cashe = false;
			algs[2].use_cashe = false;
			algs[3].use_cashe = false;
			turnament(50, algs);
		}
		else if (std::string("train") == args[1]) {

			NeuralNetwork net(eng, { 25 + 48 + 1,200,300,200,100,500,1 }, 0.01);
			//net.load("Network5/");
			trainv2(net, eng, 0.95, 5, 16);
		}
	}

	std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
