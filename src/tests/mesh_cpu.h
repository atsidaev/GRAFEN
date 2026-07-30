#pragma once
/** Mesh helpers: cubeGen from MagExperimentGenerator; ellipsoidGen from elFieldCU. */

#include <functional>
#include <vector>

#include "../MagExperimentGenerator.h"
#include "../mobj.h"

namespace grafen_test {

inline void ellipsoidGen(
	const Ellipsoid& e,
	int nl,
	int nB,
	int nR,
	const Point J,
	std::vector<HexahedronWid>& hsi)
{
	limits ll{0, M_PI_2, nl}, lB{0, M_PI_2, nB}, lReq{0, e.Req, nR}, lRpl{0, e.Rpl, nR};
	hsi.resize(0);

	const auto makeQuadrangle = [&](const Ellipsoid& ee, int Bi, int li) {
		return Quadrangle{
			ee.getPoint(lB.at(Bi + 1), ll.at(li + 1)),
			ee.getPoint(lB.at(Bi), ll.at(li + 1)),
			ee.getPoint(lB.at(Bi + 1), ll.at(li)),
			ee.getPoint(lB.at(Bi), ll.at(li)),
		};
	};

	for (int ri = 0; ri < static_cast<int>(lReq.n); ++ri)
		for (int li = 0; li < static_cast<int>(ll.n); ++li)
			for (int Bi = 0; Bi < static_cast<int>(lB.n); ++Bi) {
				Quadrangle internal;
				if (ri > 0) {
					Ellipsoid ei{lReq.at(ri), lRpl.at(ri)};
					internal = makeQuadrangle(ei, Bi, li);
				}
				Ellipsoid ee{lReq.at(ri + 1), lRpl.at(ri + 1)};
				Quadrangle external = makeQuadrangle(ee, Bi, li);
				hsi.push_back(Hexahedron{external, internal, J});
			}

	const auto dbl = [&](const std::function<void(Hexahedron&)>& f) {
		std::vector<Hexahedron> tmp(hsi.begin(), hsi.end());
		for (auto& h : tmp)
			f(h);
		hsi.insert(hsi.end(), tmp.begin(), tmp.end());
	};
	dbl([](Hexahedron& h) { h.mirrorX(); });
	dbl([](Hexahedron& h) { h.mirrorY(); });
	dbl([](Hexahedron& h) { h.mirrorZ(); });
}

inline std::vector<HexahedronWid> sphere_mesh(double R, int nl, int nb, int nr, const Point& I0) {
	std::vector<HexahedronWid> hsi;
	ellipsoidGen(Ellipsoid{R, R}, nl, nb, nr, I0, hsi);
	return hsi;
}

inline std::vector<HexahedronWid> ellipsoid_mesh(
	double req, double rpl, int nl, int nb, int nr, const Point& I0)
{
	std::vector<HexahedronWid> hsi;
	ellipsoidGen(Ellipsoid{req, rpl}, nl, nb, nr, I0, hsi);
	return hsi;
}

inline std::vector<HexahedronWid> cube_mesh(const Volume& v, const Point& I0) {
	std::vector<HexahedronWid> hsi;
	cubeGen(v, I0, hsi);
	return hsi;
}

inline void translate_mesh(std::vector<HexahedronWid>& elems, const Point& offset) {
	for (auto& e : elems)
		e += offset;
}

inline std::vector<HexahedronWid> merge_meshes(
	const std::vector<HexahedronWid>& a,
	const std::vector<HexahedronWid>& b)
{
	std::vector<HexahedronWid> out = a;
	out.insert(out.end(), b.begin(), b.end());
	return out;
}

inline Point mean_magnetization(const std::vector<HexahedronWid>& elems, size_t from = 0, size_t to = 0) {
	if (to == 0)
		to = elems.size();
	Point s;
	const size_t n = to - from;
	for (size_t i = from; i < to; ++i)
		s += elems[i].dens;
	return n ? s / static_cast<double>(n) : s;
}

}  // namespace grafen_test
