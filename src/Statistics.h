#ifndef _STATISTICS_H_
#define _STATISTICS_H_

template<class Input = double, class Acc = Input, class Result = Acc>
class Statistics {
	std::function<Acc(const Input&, const Acc&)> add;
	std::function<Result(const Acc&, const unsigned int count)> end;
	Acc init;
	Acc acc;
	unsigned int count = 0;
public:
	Statistics(
		const Acc &acc = {}, 
		const std::function<Acc(const Input&, const Acc&)> &add = [](const Input& i, const Acc& acc){ return i + acc; },
		const std::function<Result(const Acc&, const unsigned int count)> &end = [](const Acc& acc, const unsigned int count){ return acc; }
	) : acc(acc), init(acc), add(add), end(end) {}

	void next(const Input& i) {
		acc = add(i, acc);
		++count;
	}
	Result get() const {
		return end(acc, count);
	}
	void reset() {
		acc = init;
		count = 0;
	}
	template<typename T = Input>
	void next(const std::vector<T>& arr) {
		for(const auto& v: arr)
			next(v);
	}
};

struct RMSPoint : public Statistics<Point, double> {
	RMSPoint() : Statistics<Point, double>(
		0,
		[&](auto& val, auto& acc){ return acc + (val^val); },
		[](auto& acc, auto count){ return std::sqrt(acc / count); }
	) {}
	static double calc(const std::vector<Point>& arr) {
		RMSPoint stat;
		stat.next(arr);
		return stat.get();
	}
};

#endif