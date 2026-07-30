#pragma once
/** Closed-form references (matches python/grafen/analytic.py). */

#include <algorithm>
#include <cmath>
#include <vector>

#include "../AssertException.h"
#include "../MagExperimentGenerator.h"
#include "../mobj.h"
#include "field_cpu.h"

namespace grafen_test {

inline Point sphere_magnetization(const Point& I0, double kappa) {
	return I0 / (1.0 + kappa / 3.0);
}

inline Point ellipsoid_magnetization(const Point& I0, double kappa, double req, double rpl) {
	if (std::abs(rpl - req) < 1e-15 * std::max({req, rpl, 1.0}))
		return sphere_magnetization(I0, kappa);
	Assert(rpl > req);

	const double m = rpl / req;
	const double s = std::sqrt(m * m - 1.0);
	const double lnPt = std::log((m + s) / (m - s));
	const double L = (1.0 / (m * m - 1.0)) * ((m / (2.0 * s)) * lnPt - 1.0);
	const double M = (m / (2.0 * (m * m - 1.0))) * (m - (1.0 / (2.0 * s)) * lnPt);
	return I0 / (Point(1.0) + Point(kappa).cmul(Point{M, M, L}));
}

/** SI exterior H of a uniformly magnetized sphere (matches polyhedron kernel). */
inline Point sphere_field_exterior(const Point& center, double R, const Point& J, const Point& p) {
	const Point rvec = p - center;
	const double r = rvec.eqNorm();
	if (r <= R * (1.0 + 1e-9))
		return -J / 3.0;
	return (rvec * (3.0 * (J ^ rvec)) / std::pow(r, 5) - J / (r * r * r)) * (R * R * R / 3.0);
}

inline std::vector<Point> sphere_field_exterior(
	const Point& center,
	double R,
	const Point& J,
	const std::vector<Point>& pts)
{
	std::vector<Point> out(pts.size());
	for (size_t i = 0; i < pts.size(); ++i)
		out[i] = sphere_field_exterior(center, R, J, pts[i]);
	return out;
}

inline std::vector<Point> cuboid_field_uniform(
	const Volume& bounds,
	const Point& magnetization,
	const std::vector<Point>& pts)
{
	std::vector<HexahedronWid> one;
	cubeGen(Volume{
		{bounds.x.lower, bounds.x.upper, 1},
		{bounds.y.lower, bounds.y.upper, 1},
		{bounds.z.lower, bounds.z.upper, 1}},
		magnetization, one);
	return field_at_points(pts, one);
}

}  // namespace grafen_test
