#include "Player.h"
#include <iostream>
#include <cassert>

EvalResult::EvalResult() : mate(0), eval(0), move(-1)
{
}

EvalResult::EvalResult(const EvalResult& other)
{
	this->mate = other.mate;
	this->eval = other.eval;
	this->move = other.move;
}

EvalResult& EvalResult::operator=(const EvalResult& other)
{
	this->mate = other.mate;
	this->eval = other.eval;
	this->move = other.move;
	return *this;
}

EvalResult::EvalResult(int mate_, double eval_, int move_) : mate(mate_), eval(eval_), move(move_)
{
}

EvalResult EvalResult::min()
{
	return EvalResult(-1, -3, -1);
}

EvalResult EvalResult::max()
{
	return EvalResult(1, 3, -1);
}

bool EvalResult::operator<(const EvalResult& other) const
{
	if (this->mate == other.mate) return this->eval < other.eval;
	else if (this->mate < 0 && other.mate < 0) return this->mate > other.mate;
	else if (this->mate < 0) return true;
	else if (other.mate < 0) return false;
	else if (this->mate > 0 && other.mate > 0) return this->mate > other.mate;
	else if (this->mate > 0) return false;
	else if (other.mate > 0) return true;
	else return this->eval < other.eval;

}

bool EvalResult::operator>(const EvalResult& other) const
{
	if (this->mate == other.mate) return this->eval > other.eval;
	else if (this->mate > 0 && other.mate > 0) return this->mate < other.mate;
	else if (this->mate > 0) return true;
	else if (other.mate > 0) return false;
	else if (this->mate < 0 && other.mate < 0) return this->mate < other.mate;
	else if (this->mate < 0) return false;
	else if (other.mate < 0) return true;
	else return this->eval > other.eval;
}

EvalResult EvalResult::addMoveIfMate() const
{
	if (this->mate > 0) {
		return EvalResult(this->mate + 1, this->eval, this->move);
	}
	else if (this->mate < 0) {
		return EvalResult(this->mate - 1, this->eval, this->move);
	}
	else
	{
		return EvalResult(0, this->eval, this->move);
	}
}


std::ostream& operator<<(std::ostream& out, const EvalResult& e)
{
	out << "CPU Move : [" << e.move / 5 << "," << e.move % 5 << "]" << "{" << e.move << "}";
	out << " Eval: ";
	if (e.mate != 0) {
		out << "M_" << e.mate;
	}
	else {
		out << e.eval;
	}
	return out;
}

EvalResult max(const EvalResult& a, const EvalResult& b)
{
	if (a > b) return a;
	return b;
}

EvalResult min(const EvalResult& a, const EvalResult& b)
{
	if (a < b) return a;
	return b;
}

EvalResult simpleEval(const GameState& state)
{
	switch (state.gameOver())
	{
	case GameResult::gameNotOver:
		return EvalResult(0, 0, -1);
	case GameResult::FirstPlayerWin:
		return EvalResult(1, 1, -1);
	case GameResult::SecondPlayerWin:
		return EvalResult(-1, -1, -1);
	default:
		return EvalResult(0, 0, -1);
	}
	return EvalResult(0, 0, -1);
}
EvalFunc montecarloEval(std::default_random_engine& eng, int sampleSize)
{
	return [&eng, sampleSize](const GameState& state)->EvalResult {
		int sum = 0;
		double max_size = sampleSize;
		switch (state.gameOver())
		{
		case GameResult::SecondPlayerWin:
			return EvalResult(-1, -1);
			break;
		case GameResult::FirstPlayerWin:
			return EvalResult(1, 1);
			break;
		case GameResult::draw:
			return EvalResult(0, 0);
			break;

		case GameResult::gameNotOver:
			for (int i = 0; i < sampleSize; i++) {
				GameState workstate = state;
				while (workstate.gameOver() == GameResult::gameNotOver) {
					workstate = workstate.move(workstate.generateMoveset(eng)[0]);
				}
				if (workstate.gameOver() == GameResult::FirstPlayerWin) sum += 1;
				else if (workstate.gameOver() == GameResult::SecondPlayerWin) sum -= 1;
				else if (workstate.gameOver() == GameResult::draw) sum += 0;
			}
			return EvalResult(0, sum * 1.0 / sampleSize);
			break;
		default:
			return EvalResult(0, 0);
			break;
		}
		return EvalResult(0, 0);
	};
}

std::vector<Move> MinMaxAlgorythm::sortMoves(const GameState& s, std::vector<Move> moves, bool firstPlayer)
{
	for (auto m : moves) {
		auto it = this->cashe.find(s.move(m));
		if (it == this->cashe.end() || it->second.depth < 0)
			this->cashe[s.move(m)] = CasheEntry(0, this->eval(s.move(m)));
	}

	std::sort(moves.begin(), moves.end(), [this, s, firstPlayer](const Move& A, const Move& B) {
		if (firstPlayer) {
			return this->cashe[s.move(A)].eval > this->cashe[s.move(B)].eval;
		}
		else {
			return this->cashe[s.move(A)].eval < this->cashe[s.move(B)].eval;
		}
		});
	return moves;
}

MinMaxAlgorythm::MinMaxAlgorythm(std::default_random_engine& eng_, EvalFunc func, std::string name_, int depth)
	:
	eng(eng_),
	eval(func),
	name(name_),
	default_deph(depth)
{
}

EvalResult MinMaxAlgorythm::minmax(GameState state)
{
	return this->minmax(state, this->default_deph);
}

EvalResult MinMaxAlgorythm::minmax(GameState state, int depth, EvalResult alpha, EvalResult beta)
{
	auto it = cashe.find(state);
	if (it != cashe.end() && it->second.depth > depth && use_cashe) {
		auto r = it->second;
		//std::cout << " CachHit : " << it->second.depth << " = " << it->second.eval << std::endl;
		return it->second.eval;
	}
	if (depth == 0 || state.gameOver() != GameResult::gameNotOver) {
		return this->eval(state);
	}
	const int printConf = 2;
	if (depth > 1) {
		for (int i = 0; i < depth - 1 - 1; i++) {
			std::cout << "\t";
		}
		std::cout << "depth : " << depth << std::endl;
	}
	if (state.firstPlayerTurn()) {
		EvalResult value = EvalResult::min();
		auto moveset = state.generateMoveset(this->eng);
		if (depth > 1) {
			moveset = sortMoves(state, moveset, true);
		}
		for (auto move : moveset) {
			assert(state.get(move) == Space::empty);
			auto nextState = state.move(move);
			assert(nextState != state);
			if (nextState == state) {
				exit(-5);
			}
			auto t = minmax(nextState, depth - 1, alpha, beta);
			if (t > value) {
				value = t;
				value.move = move;
			}
			if (value > beta){
				break;
			}
			alpha = max(alpha, value);
		}
		cashe[state] = CasheEntry(depth, value.addMoveIfMate());
		return value.addMoveIfMate();
	}
	else {
		EvalResult value = EvalResult::max();
		auto moveset = state.generateMoveset(this->eng);
		if (depth > 1) {
			moveset = sortMoves(state, moveset, false);
		}
		for (auto move : moveset) {
			assert(state.get(move) == Space::empty);
			auto nextState = state.move(move);
			assert(nextState != state);
			if (nextState == state) {
				exit(-5);
			}
			auto t = minmax(nextState, depth - 1, alpha, beta);
			if (t < value) {
				value = t;
				value.move = move;
			}
			if (value < alpha) {
				break;
			}
			beta = min(beta, value);
		}
		cashe[state] = CasheEntry(depth, value.addMoveIfMate());
		return value.addMoveIfMate();
	}
}

CasheEntry::CasheEntry() : depth(-1), eval(EvalResult())
{
}

CasheEntry::CasheEntry(int d, EvalResult e) : depth(d), eval(e)
{
}

CasheEntry& CasheEntry::operator=(const CasheEntry& other)
{
	this->depth = other.depth;
	this->eval = other.eval;
	return *this;
}
