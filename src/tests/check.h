#pragma once
/** RMS helpers matching python/tests/conftest.py. */

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Statistics.h"
#include "../mobj.h"

namespace grafen_test {

inline double rms(const std::vector<Point>& a) {
	return RMSPoint::calc(a);
}

inline double relative_rms(const std::vector<Point>& num, const std::vector<Point>& ref) {
	std::vector<Point> diff(num.size());
	for (size_t i = 0; i < num.size(); ++i)
		diff[i] = num[i] - ref[i];
	const double abs_err = rms(diff);
	const double denom = rms(ref);
	return denom == 0.0 ? abs_err : abs_err / denom;
}

inline double relative_rms(const Point& num, const Point& ref) {
	return relative_rms(std::vector<Point>{num}, std::vector<Point>{ref});
}

inline double check_relative_rms(
	const std::vector<Point>& num,
	const std::vector<Point>& ref,
	double tol,
	const std::string& label = "")
{
	std::vector<Point> diff(num.size());
	for (size_t i = 0; i < num.size(); ++i)
		diff[i] = num[i] - ref[i];
	const double abs_err = rms(diff);
	const double denom = rms(ref);
	const double rel = denom == 0.0 ? abs_err : abs_err / denom;
	const std::string prefix = label.empty() ? "" : label + ": ";
	std::cout << prefix << "RMS=" << abs_err << "  relative_RMS=" << rel
			  << "  (tol=" << tol << ")\n";
	if (!(rel < tol))
		throw std::runtime_error(prefix + "relative RMS " + std::to_string(rel) + " >= " + std::to_string(tol));
	return rel;
}

inline double check_relative_rms(
	const Point& num,
	const Point& ref,
	double tol,
	const std::string& label = "")
{
	return check_relative_rms(std::vector<Point>{num}, std::vector<Point>{ref}, tol, label);
}

inline void check_three_way(
	const std::vector<Point>& hardcoded,
	const std::vector<Point>& vtu,
	const std::vector<Point>& analytic,
	double tol_ana,
	const std::string& label,
	double tol_io = 1e-10)
{
	check_relative_rms(hardcoded, analytic, tol_ana, label + " hardcoded vs analytic");
	check_relative_rms(vtu, analytic, tol_ana, label + " vtu vs analytic");
	check_relative_rms(vtu, hardcoded, tol_io, label + " vtu vs hardcoded");
}

inline void check_three_way(
	const Point& hardcoded,
	const Point& vtu,
	const Point& analytic,
	double tol_ana,
	const std::string& label,
	double tol_io = 1e-10)
{
	check_three_way(
		std::vector<Point>{hardcoded},
		std::vector<Point>{vtu},
		std::vector<Point>{analytic},
		tol_ana,
		label,
		tol_io);
}

}  // namespace grafen_test
