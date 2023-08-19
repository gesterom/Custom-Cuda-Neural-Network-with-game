#include "GameState.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>

GameState::GameState()
{
	_board = 0;
}

GameState::GameState(const GameState& other)
{
	this->_board = other._board;
}

GameState& GameState::operator=(const GameState& other)
{
	this->_board = other._board;
	return *this;
}

void setBit(uint64_t& u, int where, bool what) {
	if (what) {
		u = u | (0x1ll << where);
	}
	else {
		u = u & (~(0x1ll << where));
	}
}

GameState::GameState(const CPU_Matrix& M) : GameState()
{
	int counter = 0;
	for (int i = 0; i < 25; i++) {
		if (M[i][0] > 0.4) {
			setBit(_board, i, true);
			setBit(_board, i + 32, false);
			//_board[i] = 1;
			counter++;
		}
		else if (M[i][0] < -0.4) {
			setBit(_board, i, false);
			setBit(_board, i + 32, true);
			//_board[i + 32] = 1;
			counter--;
		}
		else {
			setBit(_board, i, false);
			setBit(_board, i + 32, false);
		}
	}
	if (counter == 1) {
		//_board[63] = 1;
		setBit(_board, 63, true);
	}
	else if (counter == 0) {
		//_board[63] = 0;
		setBit(_board, 63, false);
	}
	else {
		for (int i = 0; i < 25; i++) {
			std::cout << M[i][0] << " ";
		}
		std::cout << "\n";
		std::cout << "Counter : " << counter << std::endl;
		exit(4);
	}

}

bool GameState::operator<(const GameState& other) const
{
	return _board < other._board;
}

bool GameState::operator==(const GameState& other) const
{
	return this->_board == other._board;
}

GameState::GameState(std::vector<Space> init) : GameState()
{
	int counter = 0;
	for (int i = 0; i < 25; i++) {
		switch (init[i])
		{
		case Space::FirstPlayer:
			//_board[i] = 1;
			setBit(_board, i, true);
			setBit(_board, i + 32, false);
			counter++;
			break;
		case Space::empty:
			setBit(_board, i, false);
			setBit(_board, i + 32, false);
			break;
		case Space::SecondPlayer:
			//_board[i + 32] = 1;
			setBit(_board, i, false);
			setBit(_board, i + 32, true);
			counter--;
			break;
		}
	}
	if (counter == 1) {
		//_board[63] = 1;
		setBit(_board, 63, true);
	}
	else if (counter == 0) {
		//_board[63] = 0;
		setBit(_board, 63, false);
	}
	else {
		std::cout << "\n";
		std::cout << "Counter : " << counter << std::endl;
		exit(4);
	}
}

GameState GameState::move(int i, int j) const
{
	return this->move(i * 5 + j);
}

GameState GameState::move(int i) const
{
	GameState res = *this;
	assert(this->get(i) == Space::empty);
	if(this->get(i) != Space::empty) exit(-10);
	if (this->firstPlayerTurn()) {
		//res._board[63] = 1;
		setBit(res._board, 63, true);
		res.set(i, Space::FirstPlayer);
	}
	else {
		//res._board[63] = 0;
		setBit(res._board, 63, false);
		res.set(i, Space::SecondPlayer);
	}
	return res;
}

void GameState::set(int i, int j, Space s) {
	this->set(i * 5 + j, s);
}

void GameState::set(int i, Space s) {
	switch (s)
	{
	case Space::FirstPlayer:
		//this->_board[i] = 1;
		setBit(_board, i, true);
		//this->_board[i + 32] = 0;
		setBit(_board, i + 32, false);
		break;
	case Space::empty:
		//this->_board[i] = 0;
		setBit(_board, i, false);
		//this->_board[i + 32] = 0;
		setBit(_board, i + 32, false);
		break;
		break;
	case Space::SecondPlayer:
		//this->_board[i] = 0;
		setBit(_board, i, false);
		//this->_board[i + 32] = 1;
		setBit(_board, i + 32, true);
		break;
	default:
		exit(5);
		break;
	}
}

GameState GameState::flipVertical() const
{
	GameState res;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 5; j++) {
			res.set(4 - i, j, this->get(i, j));
		}
		for (int j = 0; j < 5; j++) {
			res.set(i, j, this->get(4 - i, j));
		}
	}
	return res;
}

GameState GameState::flipHorizontal() const
{
	GameState res;

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			res.set(i,j,this->get(4-i,j));
		}
	}
	return res;
}

GameState GameState::transpose() const
{
	return this->flipHorizontal().flipVertical();
}

GameState GameState::rotate90() const
{
	GameState res;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			res.set(i, j, this->get(4 - j, i));
		}
	}
	return res;
}

// Function to generate all transpositions of the board
std::vector<GameState> generate_transpositions(GameState board) {
	std::vector<GameState> transpositions;
	auto transpose = board.flipHorizontal();
	for (int i = 0; i < 4; i++) {
		transpositions.push_back(board);
		board = board.rotate90();
		transpositions.push_back(transpose);
		transpose = transpose.rotate90();
	}
	return transpositions;
}

std::vector<GameState> GameState::transposePositions() const
{
	return generate_transpositions(*this);
}

Space GameState::get(int i, int j) const
{
	return this->get(i * 5 + j);
}

Space GameState::get(int i) const
{
	bool a = ((_board & (0x1ll << i)) == 0);
	bool b = ((_board & (0x1ll << (i + 32))) == 0);

	if (a && b)
		return Space::empty;
	else if (!a && b) {
		return Space::FirstPlayer;
	}
	else if(a && ! b){
		return Space::SecondPlayer;
	}
	else {
		exit(-9);
	}
}

std::vector<std::array<Move, 4>> winingSpaces = {
	//horizontal
	{0,1,2,3},
	{1,2,3,4},
	{5,6,7,8},
	{6,7,8,9},
	{10,11,12,13},
	{11,12,13,14},
	{15,16,17,18},
	{16,17,18,19},
	{20,21,22,23},
	{21,22,23,24},

	//vertical
	{0,5,10,15},
	{5,10,15,20},
	{1,6,11,16},
	{5,11,16,21},
	{2,7,12,17},
	{7,12,17,22},
	{3,8,13,18},
	{8,13,18,23},
	{4,9,14,19},
	{9,14,19,24},

	//negative diagonal
	{0,6,12,18},
	{1,7,13,19},
	{6,12,18,24},
	{5,11,17,23},

	//positive diagonal
	{4,8,12,16},
	{3,7,11,15},
	{8,12,16,20},
	{9,13,17,21},
};

std::vector<std::array<Move, 3>> loseingSpaces = {
	//horizontal
	{0,1,2},
	{1,2,3},
	{2,3,4},
	{5,6,7},
	{6,7,8},
	{7,8,9},
	{10,11,12},
	{11,12,13},
	{12,13,14},
	{15,16,17},
	{16,17,18},
	{17,18,19},
	{20,21,22},
	{21,22,23},
	{22,23,24},

	//verrtical
	{0,5,10},
	{5,10,15},
	{10,15,20},
	{1,6,11},
	{6,11,16},
	{11,16,21},
	{2,7,12},
	{7,12,17},
	{12,17,22},
	{3,8,13},
	{8,13,18},
	{13,18,23},
	{4,9,14},
	{9,14,19},
	{14,19,24},

	//negative diagonal
	{0,6,12},
	{1,7,13},
	{2,8,14},
	{5,11,17},
	{6,12,18},
	{7,13,19},
	{10,16,22},
	{11,17,23},
	{12,18,24},

	//positive diagonal
	{2,6,10},
	{11,7,3},
	{12,8,4},
	{15,11,7},
	{16,12,8},
	{17,13,9},
	{20,16,12},
	{21,17,13},
	{22,18,14}
};

GameResult GameState::gameOver() const
{
	for (auto set : winingSpaces) {
		if (this->get(set[0]) != Space::empty &&
			this->get(set[0]) == this->get(set[1]) &&
			this->get(set[1]) == this->get(set[2]) &&
			this->get(set[2]) == this->get(set[3])) {
			return ((this->get(set[0]) == Space::FirstPlayer) ? (GameResult::FirstPlayerWin) : (GameResult::SecondPlayerWin));
		}
	}
	for (auto set : loseingSpaces) {
		if (this->get(set[0]) != Space::empty &&
			this->get(set[0]) == this->get(set[1]) &&
			this->get(set[1]) == this->get(set[2])) {
			return ((this->get(set[0]) == Space::FirstPlayer) ? (GameResult::SecondPlayerWin) : (GameResult::FirstPlayerWin));
		}
	}
	for (int i = 0; i < 25; i++) {
		if (this->get(i) == Space::empty) return GameResult::gameNotOver;
	}
	return GameResult::draw;
}

bool GameState::firstPlayerTurn() const
{
	return ((_board & (0x1ll << 63)) == 0);
}

std::vector<Move> GameState::generateMoveset(std::default_random_engine& eng) const
{
	std::vector<Move> res;
	for (int i = 0; i < 25; i++) {
		if (this->get(i) == Space::empty) {
			res.push_back(i);
		}
	}
	std::shuffle(res.begin(), res.end(), eng);
	return res;
}

CPU_Matrix convertToMatrix(const GameState& state)
{
	CPU_Matrix res(5, 5);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			double val = 0;
			switch (state.get(i, j))
			{
			case Space::FirstPlayer:
				val = 1;
				break;
			case Space::empty:
				val = 0;
				break;
			case Space::SecondPlayer:
				val = -1;
				break;
			default:
				//assert("Well pls dont");
				assert(false);
				break;
			}
			res[i][j] = val;
		}
	}
	CPU_Matrix res2(49,1);
	for (int i = 0 ; i < loseingSpaces.size();i++) {
		int x_1 = loseingSpaces[i][0] / 5;
		int y_1 = loseingSpaces[i][0] % 5;
		int x_2 = loseingSpaces[i][1] / 5;
		int y_2 = loseingSpaces[i][1] % 5;
		int x_3 = loseingSpaces[i][2] / 5;
		int y_3 = loseingSpaces[i][2] % 5;
		res2[i][0] = res[x_1][y_1] + res[x_2][y_2] + res[x_3][y_3];
		res2[i][0] /=3;
	}
	res2[48][0] = ((state.firstPlayerTurn()) ? (1) : (-1));
	res.reshape(25,1);
	res = res.appendRow(res2);
	res.reshape(1,25+48+1);
	return res;
}

std::string GameState::toString() const
{
	std::string res;
	res += "Move : ";
	res += ((this->firstPlayerTurn()) ? ("O\n") : ("X\n"));
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			res += (
				(this->get(i, j) == Space::empty ? " _"
					: this->get(i, j) == Space::FirstPlayer ? " O" : " X"));
		}
		res += "\n";
	}
	return res;
}
