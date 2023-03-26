#ifndef _EXPERIMENT_TWO_CUBOIDS_H_
#define _EXPERIMENT_TWO_CUBOIDS_H_

#include <iostream>
#include "../MagExperimentGenerator.h"
#include "../Statistics.h"

struct TwoCuboids : MagExperimentGenerator {
    virtual MagExperiment generate(const InputParser& inp) {
        // Setup default params
        int nx = 80, ny = 40, nz = 40;
		double K = 2;
		double K2 = K;
		double HprimeX = 14, HprimeY = 14, HprimeZ = 35; //~40A/m

        // Read input params        
        inp.checkUnknownParams({"nx", "ny", "nz", "K", "K2"});
		inp.parseIfExists("nx", nx);
		inp.parseIfExists("ny", ny);
		inp.parseIfExists("nz", nz);
		inp.parseIfExists("K", K); K2 = K;
		inp.parseIfExists("K2", K2);

		const Point Hprime = { HprimeX, HprimeY, HprimeZ };
        MagExperiment exp;

        int bodyCount = 0;
        const auto& genCuboid = [&](const double K, const Point& offset = {}) {
            MagExperiment exp;
			const auto I0 = Hprime * K;

            // Model
            Volume vol{{-10, 10, nx},{-5, 5, ny}, {-15, -5, nz}};
            vol += offset;
            cubeGen(vol, I0, exp.elements);
            exp.setKappa(K);
            exp.setI0(I0);

            // Calc "effect" of the rest model on this cube
            exp.fieldTasks.push_back(MagExperiment::FiledTask{
                exp.getElementCenters(), // field points
                "effect_on_cube_" + std::to_string(bodyCount + 1) + ".dat", // file name
                [K](const Point& val, auto) { return val * K; },     // transform before save
                {0, exp.size()},  // exclude this body from effect field calculation
            });
            
            ++bodyCount;
            return exp;
        };

        // Generate two cubes
        exp.append(genCuboid(K, Point(0, 0, 0)));
        exp.append(genCuboid(K, Point(10, 0, -12)));

        // Calculate field on surface (without demag)
        exp.fieldTasksBefore.push_back(MagExperiment::FiledTask{
            gridGen({-12, 22, nx*2}, {-7, 7, ny*2}),
            "field_no_demag_z0.dat",
            {[](auto p, auto) { return p; }},
            {},
            [](const std::vector<Point>& field) {
                std::cout << "field_no_demag_z0 RMS: " << RMSPoint::calc(field) << std::endl;
            }
        });

        // Calculate field on surface (with demag)
        exp.fieldTasks.push_back(MagExperiment::FiledTask{
            gridGen({-12, 22, nx*2}, {-7, 7, ny*2}),
            "field_z0.dat",
            {[](auto p, auto) { return p; }},
            {},
            [](const std::vector<Point>& field) {
                std::cout << "field_z0 RMS: " << RMSPoint::calc(field) << std::endl;
            }
        });

        return exp;
    }

    virtual void processResult(MagExperiment& model) {
		// Dump secondary magnetization
		dumpVectorField<HexahedronWid>(
			model.elements,
			"two_cuboids_Isnd.dat",
			[](const HexahedronWid& e, const size_t) { return e.massCenter(); },
			[&I0 = model.I0](const HexahedronWid& e, const size_t idx) { return e.dens - I0[idx]; }
		);
        std::cout << "Calc two cube model: Done" << std::endl;
    }

};

#endif
