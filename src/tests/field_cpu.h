#pragma once
/**
 * Field + demag for validation tests.
 * Default: host kernel (same formulas as calcField.cu).
 * With -DUSE_CUDA: gFieldSolver::getCUDAsolver (real GPU path, no MPI).
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../CG.h"
#include "../mobj.h"

#ifdef USE_CUDA
#include "../calcField.h"
#endif

namespace grafen_test {

/** CLI / main sets this; solve_magnetic(..., demag) uses it at call sites. */
inline bool demag = true;

constexpr double kFieldConst = -1.0 / (4.0 * M_PI);

inline void init_backend() {
#ifdef USE_CUDA
	if (!cuSolver::isCUDAavailable())
		throw std::runtime_error("USE_CUDA build but no CUDA device available");
	cuSolver::setDevice(0);
	std::cout << "Backend: CUDA (gFieldSolver)\n";
#else
	std::cout << "Backend: CPU\n";
#endif
	std::cout << "Demagnetization: " << (demag ? "on" : "off") << "\n";
}

#ifndef USE_CUDA

inline double tripleprod(const Point& o1, const Point& o2, const Point& o3) {
	return o1 ^ (o2 * o3);
}

inline Point intTrAn(const Point& p0, const Triangle<double>& t) {
	const Point a1 = t.p1 - p0;
	const Point a2 = t.p2 - p0;
	const Point a3 = t.p3 - p0;
	const double a1m = a1.eqNorm();
	const double a2m = a2.eqNorm();
	const double a3m = a3.eqNorm();
	const Point a12 = t.p2 - t.p1;
	const Point a23 = t.p3 - t.p2;
	const Point a31 = t.p1 - t.p3;
	const double a12m = a12.eqNorm();
	const double a23m = a23.eqNorm();
	const double a31m = a31.eqNorm();

	Point res = a31 * (std::log((a3m + a1m + a31m) / (a3m + a1m - a31m)) / a31m);
	res += a12 * (std::log((a1m + a2m + a12m) / (a1m + a2m - a12m)) / a12m);
	res += a23 * (std::log((a2m + a3m + a23m) / (a2m + a3m - a23m)) / a23m);

	const Point N = t.normal();
	res = N * res;
	res += N * (2.0 * std::atan2(
		tripleprod(a1, a2, a3),
		(a1m * a2m * a3m + a3m * (a1 ^ a2) + a2m * (a1 ^ a3) + a1m * (a2 ^ a3))));
	return res;
}

inline Point field_from_hexahedra(const Point& p0, const std::vector<HexahedronWid>& elems) {
	Point sum;
	for (const auto& h : elems) {
		for (int i = 0; i < HexahedronWid::nTriangles; ++i) {
			const auto tri = h.getTri(i);
			const Point g = intTrAn(p0, tri);
			if (std::isfinite(g.x) && std::isfinite(g.y) && std::isfinite(g.z))
				sum += g * (tri.normal() ^ h.dens);
		}
	}
	return sum * kFieldConst;
}

#else  // USE_CUDA

inline std::unique_ptr<gFieldSolver> make_cuda_solver(const std::vector<HexahedronWid>& elems) {
	if (elems.empty())
		throw std::runtime_error("CUDA field: empty mesh");
	return gFieldSolver::getCUDAsolver(elems.data(), elems.data() + elems.size(), false);
}

inline Point field_from_hexahedra(const Point& p0, const std::vector<HexahedronWid>& elems) {
	auto solver = make_cuda_solver(elems);
	return solver->solve(p0) * kFieldConst;
}

#endif  // USE_CUDA

inline std::vector<Point> field_at_points(
	const std::vector<Point>& points,
	const std::vector<HexahedronWid>& elems)
{
	std::vector<Point> out(points.size());
#ifdef USE_CUDA
	auto solver = make_cuda_solver(elems);
	for (size_t i = 0; i < points.size(); ++i)
		out[i] = solver->solve(points[i]) * kFieldConst;
#else
	for (size_t i = 0; i < points.size(); ++i)
		out[i] = field_from_hexahedra(points[i], elems);
#endif
	return out;
}

/** Magnetization for field calc: optionally solve I = I0 + K H_snd(I).
 *  With demag=false, dens stays I0. Mutates elems[*].dens. */
inline void solve_magnetic(
	std::vector<HexahedronWid>& elems,
	double kappa,
	double tol = 1e-3,
	int max_iter = 15,
	bool demag = true)
{
	if (!demag)
		return;

	const size_t n = elems.size();
	std::vector<Point> I0(n), centers(n);
	for (size_t i = 0; i < n; ++i) {
		I0[i] = elems[i].dens;
		centers[i] = elems[i].massCenter();
	}

	auto Op = [&](const std::vector<Point>& z) {
		for (size_t i = 0; i < n; ++i)
			elems[i].dens = z[i];
		std::vector<Point> Az(n);
#ifdef USE_CUDA
		auto solver = make_cuda_solver(elems);
		for (size_t i = 0; i < n; ++i)
			Az[i] = z[i] - solver->solve(centers[i]) * kFieldConst * kappa;
#else
		for (size_t i = 0; i < n; ++i)
			Az[i] = z[i] - field_from_hexahedra(centers[i], elems) * kappa;
#endif
		return Az;
	};

	CG<Point> cg(I0, I0, Op);
	cg.prepare();
	for (int it = 0; it < max_iter; ++it) {
		if (cg.getError() <= tol)
			break;
		cg.nextIter();
	}
	for (size_t i = 0; i < n; ++i)
		elems[i].dens = cg.x[i];
}

}  // namespace grafen_test
