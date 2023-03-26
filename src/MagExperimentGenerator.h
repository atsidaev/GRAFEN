#ifndef _MAG_EXPERIMENT_GENERATOR_H_
#define _MAG_EXPERIMENT_GENERATOR_H_

#include "MagExperiment.h"
#include "inputParser.h"

struct MagExperimentGenerator {
    virtual MagExperiment generate(const InputParser& args) = 0;
    virtual void processResult(MagExperiment& model) = 0;
    virtual ~MagExperimentGenerator() {}
};

// MagExperimentGenerator tools

#include <vector>

template <class VAlloc>
void cubeGen(const Volume &v, const Point J, std::vector<HexahedronWid, VAlloc> &hsi) {
	hsi.resize(v.x.n * v.y.n * v.z.n);

	for (int zi = 0; zi < v.z.n; ++zi)
		for (int yi = 0; yi < v.y.n; ++yi)
			for (int xi = 0; xi < v.x.n; ++xi) {
				Quadrangle cur{
					Point{v.x.at(xi+1), v.y.at(yi+1), 0}, 
					Point{v.x.at(xi+1), v.y.at(yi), 0}, 
					Point{v.x.at(xi), v.y.at(yi+1), 0}, 
					Point{v.x.at(xi), v.y.at(yi), 0}, 
				};

				const int ind = (zi*v.y.n + yi)*v.x.n + xi;
				hsi[ind] = Hexahedron{
					cur + Point{0, 0, v.z.at(zi + 1)},
					cur + Point{0, 0, v.z.at(zi)},
					J };
			}	
}

std::vector<Point> gridGen(const limits& xlim, const limits& ylim, const double z = 0) {
	std::vector<Point> res(xlim.n * ylim.n);

	for (int xi = 0; xi < xlim.n; ++xi)
		for (int yi = 0; yi < ylim.n; ++yi)
			res[xi*ylim.n + yi] = Point{ xlim.atWh(xi), ylim.atWh(yi), z };

	return res;
}


#endif