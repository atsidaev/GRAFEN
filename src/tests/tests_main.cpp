/**
 * CPU/CUDA validation: analytic vs hardcoded mesh vs VTU-loaded mesh.
 * Build: make -C src/tests
 *        make -C src/tests GPU=1
 * Options: --no-demag | --demag 0|1
 */
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "analytic.h"
#include "check.h"
#include "field_cpu.h"
#include "mesh_cpu.h"
#include "model_vtu.h"

using namespace grafen_test;

static int g_failed = 0;

#define RUN_TEST(fn)                                                       \
	do {                                                                   \
		try {                                                              \
			std::cout << "=== " #fn " ===\n";                              \
			fn();                                                          \
			std::cout << "PASS\n";                                         \
		} catch (const std::exception& e) {                                \
			std::cout << "FAIL: " << e.what() << "\n";                     \
			++g_failed;                                                    \
		}                                                                  \
	} while (0)

static const Point H_PRIME{14.0, 14.0, 35.0};

static void test_sphere_magnetization_matches_analytic() {
	const double K = 2.0, R = 10.0;
	const Point I0 = H_PRIME * K;
	auto elems = sphere_mesh(R, 4, 4, 2, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_sphere_I.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	const Point i_ref = sphere_magnetization(I0, K);
	check_three_way(mean_magnetization(elems), mean_magnetization(elems_vtu), i_ref, 0.05, "sphere magnetization");
	Assert(mean_magnetization(elems).eqNorm() < I0.eqNorm() * 0.95);
}

static void test_sphere_exterior_field_matches_dipole() {
	const double K = 2.0, R = 10.0;
	const Point I0 = H_PRIME * K;
	auto elems = sphere_mesh(R, 4, 4, 2, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_sphere_H.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	const Point i_mean = mean_magnetization(elems);
	const Point i_ref = sphere_magnetization(I0, K);

	const std::vector<Point> pts{
		{0.0, 0.0, 25.0},
		{20.0, 0.0, 20.0},
		{0.0, 30.0, 0.0},
		{-15.0, 15.0, 15.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	const auto h_ana = sphere_field_exterior(Point{}, R, i_ref, pts);
	check_three_way(h_hard, h_vtu, h_ana, 0.08, "sphere field vs analytic dipole");

	const auto h_dip_mean = sphere_field_exterior(Point{}, R, i_mean, pts);
	check_relative_rms(h_hard, h_dip_mean, 0.08, "sphere field vs mean-I dipole");
}

static void test_two_spheres_far_apart_magnetization() {
	const double K = 2.0, R = 5.0, SEP = 40.0;
	const Point I0 = H_PRIME * K;
	auto c1 = sphere_mesh(R, 3, 3, 2, I0);
	auto c2 = sphere_mesh(R, 3, 3, 2, I0);
	translate_mesh(c2, Point{SEP, 0.0, 0.0});
	const size_t n1 = c1.size();
	auto elems = merge_meshes(c1, c2);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_two_spheres_I.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);

	const Point i_ref = sphere_magnetization(I0, K);
	check_three_way(
		mean_magnetization(elems, 0, n1),
		mean_magnetization(elems_vtu, 0, n1),
		i_ref,
		0.12,
		"two spheres I1");
	check_three_way(
		mean_magnetization(elems, n1, elems.size()),
		mean_magnetization(elems_vtu, n1, elems_vtu.size()),
		i_ref,
		0.12,
		"two spheres I2");
	check_relative_rms(
		mean_magnetization(elems, 0, n1),
		mean_magnetization(elems, n1, elems.size()),
		0.05,
		"two spheres I1 vs I2");
}

static void test_two_spheres_exterior_field_superposition() {
	const double K = 2.0, R = 5.0, SEP = 40.0;
	const Point I0 = H_PRIME * K;
	auto c1 = sphere_mesh(R, 3, 3, 2, I0);
	auto c2 = sphere_mesh(R, 3, 3, 2, I0);
	const Point offset{SEP, 0.0, 0.0};
	translate_mesh(c2, offset);
	auto elems = merge_meshes(c1, c2);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_two_spheres_H.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	const Point i_ref = sphere_magnetization(I0, K);

	const std::vector<Point> pts{
		{SEP / 2.0, 0.0, 30.0},
		{0.0, 25.0, 0.0},
		{SEP, 0.0, 25.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	auto h_ana = sphere_field_exterior(Point{}, R, i_ref, pts);
	const auto h2 = sphere_field_exterior(offset, R, i_ref, pts);
	for (size_t i = 0; i < h_ana.size(); ++i)
		h_ana[i] += h2[i];
	check_three_way(h_hard, h_vtu, h_ana, 0.15, "two spheres field");
}

static void test_cube_uniform_field_matches_analytic_cuboid() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume bounds{{-5.0, 5.0, 4}, {-2.0, 2.0, 2}, {-2.0, 2.0, 2}};
	auto elems = cube_mesh(bounds, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_cube_H.vtu");

	const std::vector<Point> pts{
		{0.0, 0.0, 6.0},
		{8.0, 0.0, 0.0},
		{0.0, 5.0, 5.0},
		{-6.0, 3.0, 4.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	const Volume oneCell{{-5.0, 5.0, 1}, {-2.0, 2.0, 1}, {-2.0, 2.0, 1}};
	const auto h_ana = cuboid_field_uniform(oneCell, I0, pts);
	check_three_way(h_hard, h_vtu, h_ana, 1e-3, "cube field vs analytic cuboid");
}

static void test_cube_demagnetization_reduces_magnetization() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume bounds{{-5.0, 5.0, 4}, {-2.0, 2.0, 3}, {-2.0, 2.0, 3}};
	auto elems = cube_mesh(bounds, I0);
	auto dens0 = elems;
	auto elems_vtu = roundtrip_vtu(elems, K, "t_cube_demag.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);

	const Point i_mean = mean_magnetization(elems);
	Assert((i_mean - I0).eqNorm() / I0.eqNorm() > 1e-4);
	Assert(i_mean.eqNorm() < I0.eqNorm());
	check_three_way(i_mean, mean_magnetization(elems_vtu), i_mean, 1e-10, "cube demag I mean");

	const std::vector<Point> pts{{0.0, 0.0, 6.0}, {8.0, 0.0, 3.0}};
	const auto h0 = field_at_points(pts, dens0);
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	Assert(relative_rms(h_hard, h0) > 1e-4);
	check_three_way(h_hard, h_vtu, h_hard, 1e-10, "cube demag field");
}

static void test_two_cubes_uniform_field_is_superposition() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume B1{{-2.0, 2.0, 3}, {-1.0, 1.0, 2}, {-1.0, 1.0, 2}};
	const Point OFFSET{20.0, 0.0, 0.0};
	auto c1 = cube_mesh(B1, I0);
	auto c2 = cube_mesh(B1, I0);
	translate_mesh(c2, OFFSET);
	auto elems = merge_meshes(c1, c2);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_two_cubes_H.vtu");

	const std::vector<Point> pts{
		{10.0, 0.0, 5.0},
		{0.0, 4.0, 4.0},
		{20.0, 0.0, 4.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	Volume B2 = B1;
	B2 += OFFSET;
	auto h_ana = cuboid_field_uniform(B1, I0, pts);
	const auto h2 = cuboid_field_uniform(B2, I0, pts);
	for (size_t i = 0; i < h_ana.size(); ++i)
		h_ana[i] += h2[i];
	check_three_way(h_hard, h_vtu, h_ana, 1e-3, "two cubes field");
}

static void test_two_cubes_far_demag_nearly_independent() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume B1{{-2.0, 2.0, 3}, {-1.0, 1.0, 2}, {-1.0, 1.0, 2}};
	const Point OFFSET{20.0, 0.0, 0.0};
	auto c1 = cube_mesh(B1, I0);
	auto c2 = cube_mesh(B1, I0);
	translate_mesh(c2, OFFSET);
	const size_t n1 = c1.size();

	auto single = c1;
	solve_magnetic(single, K, 1e-3, 15, demag);
	auto pair = merge_meshes(c1, c2);
	auto pair_vtu = roundtrip_vtu(pair, K, "t_two_cubes_demag.vtu");
	solve_magnetic(pair, K, 1e-3, 15, demag);
	solve_magnetic(pair_vtu, K, 1e-3, 15, demag);

	const Point ref = mean_magnetization(single);
	check_three_way(
		mean_magnetization(pair, 0, n1),
		mean_magnetization(pair_vtu, 0, n1),
		ref,
		0.05,
		"two cubes I1 vs single");
	check_three_way(
		mean_magnetization(pair, n1, pair.size()),
		mean_magnetization(pair_vtu, n1, pair_vtu.size()),
		ref,
		0.05,
		"two cubes I2 vs single");
}

static void test_ellipsoid_magnetization_matches_analytic() {
	const double K = 2.0, REQ = 10.0, RPL = 20.0;
	const Point I0 = H_PRIME * K;
	auto elems = ellipsoid_mesh(REQ, RPL, 4, 4, 2, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_ellipsoid.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);

	const Point i_ref = ellipsoid_magnetization(I0, K, REQ, RPL);
	check_three_way(
		mean_magnetization(elems),
		mean_magnetization(elems_vtu),
		i_ref,
		0.10,
		"ellipsoid magnetization");
	Assert((i_ref.z / I0.z) > (i_ref.x / I0.x));
	const Point i_mean = mean_magnetization(elems);
	Assert((i_mean.z / I0.z) > (i_mean.x / I0.x) * 0.9);
}

static void test_vtu_model_roundtrip() {
	const Point I0{2.8, 2.8, 7.0};
	const Volume bounds{{-5.0, 5.0, 3}, {-2.0, 2.0, 2}, {-2.0, 2.0, 2}};
	auto elems = cube_mesh(bounds, I0);
	std::vector<double> kappa(elems.size(), 0.2);
	const std::string path = "grafen_test_model_roundtrip.vtu";
	save_model_vtu(path, elems, kappa);
	std::vector<HexahedronWid> loaded;
	std::vector<double> kappa2;
	load_model_vtu(path, loaded, kappa2);
	Assert(models_almost_equal(elems, loaded, kappa, kappa2));
	std::remove(path.c_str());
}

static void test_sphere_no_demag() {
	const double K = 2.0, R = 10.0;
	const Point I0 = H_PRIME * K;
	auto elems = sphere_mesh(R, 4, 4, 2, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_sphere_nodemag.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	check_three_way(mean_magnetization(elems), mean_magnetization(elems_vtu), I0, 1e-12, "sphere no-demag magnetization");

	const std::vector<Point> pts{
		{0.0, 0.0, 25.0},
		{20.0, 0.0, 20.0},
		{0.0, 30.0, 0.0},
		{-15.0, 15.0, 15.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	const auto h_ana = sphere_field_exterior(Point{}, R, I0, pts);
	check_three_way(h_hard, h_vtu, h_ana, 0.08, "sphere no-demag field");
}

static void test_ellipsoid_no_demag() {
	const double K = 2.0, REQ = 10.0, RPL = 20.0;
	const Point I0 = H_PRIME * K;
	auto elems = ellipsoid_mesh(REQ, RPL, 4, 4, 2, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_ellipsoid_nodemag.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	check_three_way(mean_magnetization(elems), mean_magnetization(elems_vtu), I0, 1e-12, "ellipsoid no-demag magnetization");
}

static void test_cube_no_demag() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume bounds{{-5.0, 5.0, 4}, {-2.0, 2.0, 2}, {-2.0, 2.0, 2}};
	auto elems = cube_mesh(bounds, I0);
	auto elems_vtu = roundtrip_vtu(elems, K, "t_cube_nodemag.vtu");
	solve_magnetic(elems, K, 1e-3, 15, demag);
	solve_magnetic(elems_vtu, K, 1e-3, 15, demag);
	check_three_way(mean_magnetization(elems), mean_magnetization(elems_vtu), I0, 1e-12, "cube no-demag magnetization");

	const std::vector<Point> pts{
		{0.0, 0.0, 6.0},
		{8.0, 0.0, 0.0},
		{0.0, 5.0, 5.0},
		{-6.0, 3.0, 4.0},
	};
	const auto h_hard = field_at_points(pts, elems);
	const auto h_vtu = field_at_points(pts, elems_vtu);
	const Volume oneCell{{-5.0, 5.0, 1}, {-2.0, 2.0, 1}, {-2.0, 2.0, 1}};
	const auto h_ana = cuboid_field_uniform(oneCell, I0, pts);
	check_three_way(h_hard, h_vtu, h_ana, 1e-3, "cube no-demag field");
}

int main(int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--no-demag" || a == "-noDemag")
			demag = false;
		else if (a == "--demag" && i + 1 < argc)
			demag = std::string(argv[++i]) != "0";
		else if (a == "-h" || a == "--help") {
			std::cout << "Usage: " << argv[0] << " [--no-demag | --demag 0|1]\n";
			return 0;
		} else {
			std::cerr << "Unknown option: " << a << "\n";
			return 2;
		}
	}

	try {
		init_backend();
	} catch (const std::exception& e) {
		std::cerr << e.what() << "\n";
		return 2;
	}

	RUN_TEST(test_sphere_magnetization_matches_analytic);
	RUN_TEST(test_sphere_exterior_field_matches_dipole);
	RUN_TEST(test_two_spheres_far_apart_magnetization);
	RUN_TEST(test_two_spheres_exterior_field_superposition);
	RUN_TEST(test_cube_uniform_field_matches_analytic_cuboid);
	RUN_TEST(test_cube_demagnetization_reduces_magnetization);
	RUN_TEST(test_two_cubes_uniform_field_is_superposition);
	RUN_TEST(test_two_cubes_far_demag_nearly_independent);
	RUN_TEST(test_ellipsoid_magnetization_matches_analytic);
	RUN_TEST(test_vtu_model_roundtrip);
	RUN_TEST(test_sphere_no_demag);
	RUN_TEST(test_ellipsoid_no_demag);
	RUN_TEST(test_cube_no_demag);

	if (g_failed) {
		std::cout << g_failed << " test(s) failed\n";
		return 1;
	}
	std::cout << "All tests passed\n";
	return 0;
}
