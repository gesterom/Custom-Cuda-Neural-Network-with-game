#pragma once



#include <functional>
#include "GameState.h"
#include <map>


struct EvalResult {
	int mate=0;
	double eval=0;
	int move=-1;
	EvalResult();
	EvalResult(const EvalResult& );
	EvalResult& operator=(const EvalResult&);
	EvalResult(int mate_,double eval_,int move_=-1);
	static EvalResult min();
	static EvalResult max();
	bool operator<(const EvalResult&) const;
	bool operator>(const EvalResult&) const;
	EvalResult addMoveIfMate() const;
};

std::ostream& operator<<(std::ostream& out, const EvalResult&);

EvalResult max(const EvalResult& a,const EvalResult& b);
EvalResult min(const EvalResult& a, const EvalResult& b);

typedef std::function<EvalResult(const GameState&)> EvalFunc;

EvalResult simpleEval(const GameState&);
EvalFunc montecarloEval(std::default_random_engine& eng, int sampleSize);

struct CasheEntry {
	int depth = -1;
	EvalResult eval;
	CasheEntry();
	CasheEntry(int,EvalResult);
	CasheEntry& operator=(const CasheEntry&);
};

struct MinMaxAlgorythm {
	EvalFunc eval;
	std::default_random_engine& eng;
	int default_deph;
	std::string name;
	std::map<GameState,CasheEntry> cashe;
	bool use_cashe = true;
	std::vector<Move> sortMoves(const GameState& s, std::vector<Move> moves, bool firstPlayer);
	MinMaxAlgorythm(std::default_random_engine& eng,EvalFunc func,std::string name,int depth = 4);
	EvalResult minmax(GameState state);
	private:
	EvalResult minmax(GameState state,int depth,EvalResult alpha = EvalResult::min(), EvalResult beta = EvalResult::max());
};
