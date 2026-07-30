/**
 * CPU/CUDA validation tests matching python/tests (analytic vs numeric + RMS print).
 * Build: make -C src/tests
 *        make -C src/tests GPU=1
 * Options: --no-demag | --demag 0|1
 */
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "analytic.h"
#include "check.h"
#include "field_cpu.h"
#include "mesh_cpu.h"

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

// --- sphere ---
static void test_sphere_magnetization_matches_analytic() {
	const double K = 2.0, R = 10.0;
	const Point I0 = H_PRIME * K;
	auto elems = sphere_mesh(R, 4, 4, 2, I0);
	solve_magnetic(elems, K, 1e-3, 15, demag);
	const Point i_ref = sphere_magnetization(I0, K);
	const Point i_mean = mean_magnetization(elems);
	check_relative_rms(i_mean, i_ref, 0.05, "sphere magnetization");
	Assert(i_mean.eqNorm() < I0.eqNorm() * 0.95);
}

static void test_sphere_exterior_field_matches_dipole() {
	const double K = 2.0, R = 10.0;
	const Point I0 = H_PRIME * K;
	auto elems = sphere_mesh(R, 4, 4, 2, I0);
	solve_magnetic(elems, K, 1e-3, 15, demag);
	const Point i_mean = mean_magnetization(elems);
	const Point i_ref = sphere_magnetization(I0, K);

	const std::vector<Point> pts{
		{0.0, 0.0, 25.0},
		{20.0, 0.0, 20.0},
		{0.0, 30.0, 0.0},
		{-15.0, 15.0, 15.0},
	};
	const auto h_num = field_at_points(pts, elems);
	const auto h_ana = sphere_field_exterior(Point{}, R, i_ref, pts);
	check_relative_rms(h_num, h_ana, 0.08, "sphere field vs analytic dipole");

	const auto h_dip_mean = sphere_field_exterior(Point{}, R, i_mean, pts);
	check_relative_rms(h_num, h_dip_mean, 0.08, "sphere field vs mean-I dipole");
}

// --- two spheres ---
static void test_two_spheres_far_apart_magnetization() {
	const double K = 2.0, R = 5.0, SEP = 40.0;
	const Point I0 = H_PRIME * K;
	auto c1 = sphere_mesh(R, 3, 3, 2, I0);
	auto c2 = sphere_mesh(R, 3, 3, 2, I0);
	translate_mesh(c2, Point{SEP, 0.0, 0.0});
	const size_t n1 = c1.size();
	auto elems = merge_meshes(c1, c2);
	solve_magnetic(elems, K, 1e-3, 15, demag);

	const Point i1 = mean_magnetization(elems, 0, n1);
	const Point i2 = mean_magnetization(elems, n1, elems.size());
	const Point i_ref = sphere_magnetization(I0, K);
	check_relative_rms(i1, i_ref, 0.12, "two spheres I1 vs analytic");
	check_relative_rms(i2, i_ref, 0.12, "two spheres I2 vs analytic");
	check_relative_rms(i1, i2, 0.05, "two spheres I1 vs I2");
}

static void test_two_spheres_exterior_field_superposition() {
	const double K = 2.0, R = 5.0, SEP = 40.0;
	const Point I0 = H_PRIME * K;
	auto c1 = sphere_mesh(R, 3, 3, 2, I0);
	auto c2 = sphere_mesh(R, 3, 3, 2, I0);
	const Point offset{SEP, 0.0, 0.0};
	translate_mesh(c2, offset);
	auto elems = merge_meshes(c1, c2);
	solve_magnetic(elems, K, 1e-3, 15, demag);
	const Point i_ref = sphere_magnetization(I0, K);

	const std::vector<Point> pts{
		{SEP / 2.0, 0.0, 30.0},
		{0.0, 25.0, 0.0},
		{SEP, 0.0, 25.0},
	};
	const auto h_num = field_at_points(pts, elems);
	auto h_ana = sphere_field_exterior(Point{}, R, i_ref, pts);
	const auto h2 = sphere_field_exterior(offset, R, i_ref, pts);
	for (size_t i = 0; i < h_ana.size(); ++i)
		h_ana[i] += h2[i];
	check_relative_rms(h_num, h_ana, 0.15, "two spheres field vs analytic");
}

// --- cube ---
static void test_cube_uniform_field_matches_analytic_cuboid() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume bounds{{-5.0, 5.0, 4}, {-2.0, 2.0, 2}, {-2.0, 2.0, 2}};
	auto elems = cube_mesh(bounds, I0);

	const std::vector<Point> pts{
		{0.0, 0.0, 6.0},
		{8.0, 0.0, 0.0},
		{0.0, 5.0, 5.0},
		{-6.0, 3.0, 4.0},
	};
	const auto h_num = field_at_points(pts, elems);
	const Volume oneCell{{-5.0, 5.0, 1}, {-2.0, 2.0, 1}, {-2.0, 2.0, 1}};
	const auto h_ana = cuboid_field_uniform(oneCell, I0, pts);
	check_relative_rms(h_num, h_ana, 1e-3, "cube field vs analytic cuboid");
}

static void test_cube_demagnetization_reduces_magnetization() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume bounds{{-5.0, 5.0, 4}, {-2.0, 2.0, 3}, {-2.0, 2.0, 3}};
	auto elems = cube_mesh(bounds, I0);
	auto dens0 = elems;
	solve_magnetic(elems, K, 1e-3, 15, demag);

	const Point i_mean = mean_magnetization(elems);
	Assert((i_mean - I0).eqNorm() / I0.eqNorm() > 1e-4);
	Assert(i_mean.eqNorm() < I0.eqNorm());

	const std::vector<Point> pts{{0.0, 0.0, 6.0}, {8.0, 0.0, 3.0}};
	const auto h0 = field_at_points(pts, dens0);
	const auto h = field_at_points(pts, elems);
	Assert(relative_rms(h, h0) > 1e-4);
}

// --- two cubes ---
static void test_two_cubes_uniform_field_is_superposition() {
	const double K = 0.2;
	const Point I0 = H_PRIME * K;
	const Volume B1{{-2.0, 2.0, 3}, {-1.0, 1.0, 2}, {-1.0, 1.0, 2}};
	const Point OFFSET{20.0, 0.0, 0.0};
	auto c1 = cube_mesh(B1, I0);
	auto c2 = cube_mesh(B1, I0);
	translate_mesh(c2, OFFSET);
	auto elems = merge_meshes(c1, c2);

	const std::vector<Point> pts{
		{10.0, 0.0, 5.0},
		{0.0, 4.0, 4.0},
		{20.0, 0.0, 4.0},
	};
	const auto h_num = field_at_points(pts, elems);
	Volume B2 = B1;
	B2 += OFFSET;
	auto h_ana = cuboid_field_uniform(B1, I0, pts);
	const auto h2 = cuboid_field_uniform(B2, I0, pts);
	for (size_t i = 0; i < h_ana.size(); ++i)
		h_ana[i] += h2[i];
	check_relative_rms(h_num, h_ana, 1e-3, "two cubes field vs analytic");
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
	solve_magnetic(pair, K, 1e-3, 15, demag);

	check_relative_rms(
		mean_magnetization(pair, 0, n1),
		mean_magnetization(single),
		0.05,
		"two cubes I1 vs single");
	check_relative_rms(
		mean_magnetization(pair, n1, pair.size()),
		mean_magnetization(single),
		0.05,
		"two cubes I2 vs single");
}

// --- ellipsoid ---
static void test_ellipsoid_magnetization_matches_analytic() {
	const double K = 2.0, REQ = 10.0, RPL = 20.0;
	const Point I0 = H_PRIME * K;
	auto elems = ellipsoid_mesh(REQ, RPL, 4, 4, 2, I0);
	solve_magnetic(elems, K, 1e-3, 15, demag);

	const Point i_ref = ellipsoid_magnetization(I0, K, REQ, RPL);
	const Point i_mean = mean_magnetization(elems);
	check_relative_rms(i_mean, i_ref, 0.10, "ellipsoid magnetization");
	Assert((i_ref.z / I0.z) > (i_ref.x / I0.x));
	Assert((i_mean.z / I0.z) > (i_mean.x / I0.x) * 0.9);
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

	if (demag) {
		RUN_TEST(test_sphere_magnetization_matches_analytic);
		RUN_TEST(test_sphere_exterior_field_matches_dipole);
		RUN_TEST(test_two_spheres_far_apart_magnetization);
		RUN_TEST(test_two_spheres_exterior_field_superposition);
		RUN_TEST(test_cube_uniform_field_matches_analytic_cuboid);
		RUN_TEST(test_cube_demagnetization_reduces_magnetization);
		RUN_TEST(test_two_cubes_uniform_field_is_superposition);
		RUN_TEST(test_two_cubes_far_demag_nearly_independent);
		RUN_TEST(test_ellipsoid_magnetization_matches_analytic);
	} else {
		// Uniform-I field checks only (no demag CG)
		RUN_TEST(test_cube_uniform_field_matches_analytic_cuboid);
		RUN_TEST(test_two_cubes_uniform_field_is_superposition);
	}

	if (g_failed) {
		std::cout << g_failed << " test(s) failed\n";
		return 1;
	}
	std::cout << "All tests passed\n";
	return 0;
}
