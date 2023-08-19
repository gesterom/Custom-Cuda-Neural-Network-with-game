#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <string>

#include "Matrix.h"

enum class Space {
	FirstPlayer = 1,
	empty = 0,
	SecondPlayer = -1
};

enum class GameResult
{
	SecondPlayerWin = -1,
	gameNotOver = 0,
	FirstPlayerWin = 1,
	draw = 2,
};

typedef int Move;

class GameState
{
	//0x ffff fff f
	//0x ffff fff f
	uint64_t _board = 0;
	void set(int i, int j, Space s);
	void set(int i, Space s);
public:
	GameState();
	GameState(const GameState& other);
	GameState& operator=(const GameState& other);
	GameState(const CPU_Matrix& M);
	bool operator<(const GameState& other) const;
	bool operator==(const GameState& other) const;
	GameState(std::vector<Space> init);
	GameState move(int i, int j) const;
	GameState move(int i) const;
	GameState flipVertical() const;
	GameState flipHorizontal() const;
	GameState transpose() const;
	GameState rotate90()const;
	std::vector<GameState> transposePositions()const;
	Space get(int i, int j) const;
	Space get(int i) const;
	GameResult gameOver() const;
	bool firstPlayerTurn() const;
	std::string toString() const;
	std::vector<Move> generateMoveset(std::default_random_engine&) const;
};

CPU_Matrix convertToMatrix(const GameState& state);
