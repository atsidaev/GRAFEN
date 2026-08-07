/**
 * VTU → Surfer GRD magnetic field (no demagnetization), CUDA gFieldSolver.
 *
 * Mirrors python/tools/vtu_to_grd.py:
 *   ./vtu_to_grd model.vtu out.grd --x=-20,20,10 --y=-20,20,10 -z 0 -H 2 2 2 -k 1
 *
 * Magnetization: with -H, I0 = κ H' (κ from -k or VTU); else use VTU cell I.
 */

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Grid/Grid.h"
#include "../calcField.h"
#include "../mobj.h"
#include "../tests/model_vtu.h"

namespace {

constexpr double kFieldConst = -1.0 / (4.0 * M_PI);

struct AxisSpec {
	double lo = 0;
	double hi = 0;
	int n = 0;
	double step = 0;
};

struct Args {
	std::string vtu;
	std::string grd;
	std::string x_spec;
	std::string y_spec;
	int n_col = -1;
	int n_row = -1;
	double z = 0.0;
	std::string component = "hz";
	bool have_H = false;
	Point Hprime{0, 0, 0};
	bool have_kappa = false;
	double kappa = 0.0;
	int device = 0;
};

void usage(const char* prog) {
	std::cerr
		<< "Usage: " << prog << " model.vtu out.grd --x=lo,hi[,step] --y=lo,hi[,step] [options]\n"
		<< "\n"
		<< "  CUDA field (no demag) from VTU → Surfer 7 binary GRD.\n"
		<< "\n"
		<< "Grid:\n"
		<< "  --x=lo,hi,step     or  --x=lo,hi with --nCol=N\n"
		<< "  --y=lo,hi,step     or  --y=lo,hi with --nRow=N\n"
		<< "  -z Z               observation height (default 0)\n"
		<< "\n"
		<< "Magnetization (no demag):\n"
		<< "  -H Hx Hy Hz        inducing field; I0 = κ H' (ignores VTU I)\n"
		<< "  -k KAPPA           susceptibility (with -H; else κ from VTU)\n"
		<< "  (without -H: use magnetization I stored in VTU)\n"
		<< "\n"
		<< "Output:\n"
		<< "  --component hx|hy|hz|bt   scalar in GRD (default hz; bt=|H|)\n"
		<< "  --device ID               CUDA device (default 0)\n";
}

std::vector<double> split_csv(const std::string& s) {
	std::vector<double> out;
	std::string cur;
	for (size_t i = 0; i <= s.size(); ++i) {
		if (i == s.size() || s[i] == ',' || s[i] == ';') {
			if (!cur.empty()) {
				out.push_back(std::stod(cur));
				cur.clear();
			}
		} else if (!std::isspace(static_cast<unsigned char>(s[i]))) {
			cur.push_back(s[i]);
		}
	}
	return out;
}

AxisSpec parse_axis(const std::string& spec, int n_opt, const char* name) {
	auto parts = split_csv(spec);
	if (parts.size() != 2 && parts.size() != 3)
		throw std::runtime_error(std::string(name) + ": expect lo,hi or lo,hi,step");
	AxisSpec a;
	a.lo = parts[0];
	a.hi = parts[1];
	if (a.hi <= a.lo)
		throw std::runtime_error(std::string(name) + ": upper must be > lower");
	if (parts.size() == 3) {
		if (n_opt >= 0)
			throw std::runtime_error(std::string(name) + ": give step or nCol/nRow, not both");
		a.step = parts[2];
		if (a.step <= 0)
			throw std::runtime_error(std::string(name) + ": step must be > 0");
		a.n = static_cast<int>(std::llround((a.hi - a.lo) / a.step)) + 1;
		if (a.n < 2)
			throw std::runtime_error(std::string(name) + ": need at least 2 nodes");
		a.hi = a.lo + a.step * (a.n - 1);
	} else {
		if (n_opt < 2)
			throw std::runtime_error(std::string(name) + ": lo,hi needs --nCol/--nRow");
		a.n = n_opt;
		a.step = (a.hi - a.lo) / double(a.n - 1);
	}
	return a;
}

bool starts_with(const std::string& s, const char* pfx) {
	const size_t n = std::strlen(pfx);
	return s.size() >= n && s.compare(0, n, pfx) == 0;
}

std::string strip_key_eq(const std::string& arg, const char* key) {
	// "--x=..." or "-x=..." → value after '='
	const std::string k1 = std::string("--") + key + "=";
	const std::string k2 = std::string("-") + key + "=";
	if (starts_with(arg, k1.c_str()))
		return arg.substr(k1.size());
	if (starts_with(arg, k2.c_str()))
		return arg.substr(k2.size());
	return {};
}

Args parse_args(int argc, char** argv) {
	if (argc < 2) {
		usage(argv[0]);
		std::exit(2);
	}
	Args a;
	std::vector<std::string> positionals;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			usage(argv[0]);
			std::exit(0);
		}
		if (auto v = strip_key_eq(arg, "x"); !v.empty()) {
			a.x_spec = v;
			continue;
		}
		if (auto v = strip_key_eq(arg, "y"); !v.empty()) {
			a.y_spec = v;
			continue;
		}
		if (auto v = strip_key_eq(arg, "nCol"); !v.empty()) {
			a.n_col = std::stoi(v);
			continue;
		}
		if (auto v = strip_key_eq(arg, "nRow"); !v.empty()) {
			a.n_row = std::stoi(v);
			continue;
		}
		if (auto v = strip_key_eq(arg, "component"); !v.empty()) {
			a.component = v;
			continue;
		}
		if (auto v = strip_key_eq(arg, "device"); !v.empty()) {
			a.device = std::stoi(v);
			continue;
		}
		if (arg == "--x" || arg == "-x") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --x");
			a.x_spec = argv[i];
			continue;
		}
		if (arg == "--y" || arg == "-y") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --y");
			a.y_spec = argv[i];
			continue;
		}
		if (arg == "--nCol") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --nCol");
			a.n_col = std::stoi(argv[i]);
			continue;
		}
		if (arg == "--nRow") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --nRow");
			a.n_row = std::stoi(argv[i]);
			continue;
		}
		if (arg == "-z" || arg == "--z") {
			if (++i >= argc)
				throw std::runtime_error("missing value for -z");
			a.z = std::stod(argv[i]);
			continue;
		}
		if (starts_with(arg, "-z=")) {
			a.z = std::stod(arg.substr(3));
			continue;
		}
		if (arg == "-k" || arg == "--kappa") {
			if (++i >= argc)
				throw std::runtime_error("missing value for -k");
			a.have_kappa = true;
			a.kappa = std::stod(argv[i]);
			continue;
		}
		if (starts_with(arg, "-k=")) {
			a.have_kappa = true;
			a.kappa = std::stod(arg.substr(3));
			continue;
		}
		if (arg == "--component") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --component");
			a.component = argv[i];
			continue;
		}
		if (arg == "--device") {
			if (++i >= argc)
				throw std::runtime_error("missing value for --device");
			a.device = std::stoi(argv[i]);
			continue;
		}
		if (arg == "-H" || arg == "--Hprime") {
			// -H Hx Hy Hz   or   -H Hx,Hy,Hz
			if (++i >= argc)
				throw std::runtime_error("missing value for -H");
			std::string v = argv[i];
			if (v.find(',') != std::string::npos) {
				auto p = split_csv(v);
				if (p.size() != 3)
					throw std::runtime_error("-H expects Hx,Hy,Hz");
				a.Hprime = Point{p[0], p[1], p[2]};
			} else {
				if (i + 2 >= argc)
					throw std::runtime_error("-H expects Hx Hy Hz");
				a.Hprime = Point{std::stod(argv[i]), std::stod(argv[i + 1]), std::stod(argv[i + 2])};
				i += 2;
			}
			a.have_H = true;
			continue;
		}
		if (arg[0] == '-')
			throw std::runtime_error("unknown option: " + arg);
		positionals.push_back(arg);
	}
	if (positionals.size() != 2)
		throw std::runtime_error("need positional: model.vtu out.grd");
	a.vtu = positionals[0];
	a.grd = positionals[1];
	if (a.x_spec.empty() || a.y_spec.empty())
		throw std::runtime_error("--x and --y are required");
	if (a.have_kappa && !a.have_H)
		throw std::runtime_error("-k/--kappa is only used with -H");
	return a;
}

double component_value(const Point& h, const std::string& c) {
	if (c == "hx" || c == "x")
		return h.x;
	if (c == "hy" || c == "y")
		return h.y;
	if (c == "hz" || c == "z")
		return h.z;
	if (c == "bt" || c == "b" || c == "abs")
		return h.eqNorm();
	throw std::runtime_error("unknown --component (use hx|hy|hz|bt)");
}

}  // namespace

int main(int argc, char** argv) {
	try {
		Args args = parse_args(argc, argv);
		AxisSpec ax = parse_axis(args.x_spec, args.n_col, "--x");
		AxisSpec ay = parse_axis(args.y_spec, args.n_row, "--y");

		if (!cuSolver::isCUDAavailable())
			throw std::runtime_error("no CUDA device available");
		cuSolver::setDevice(args.device);
		std::cout << "CUDA device " << args.device
				  << " (" << cuSolver::getGPUnum() << " GPU(s))\n";

		std::vector<HexahedronWid> elems;
		std::vector<double> kappa;
		grafen_test::load_model_vtu(args.vtu, elems, kappa);
		if (elems.empty())
			throw std::runtime_error("empty VTU model");

		double z_lo = elems[0].p[0].z, z_hi = elems[0].p[0].z;
		for (const auto& e : elems) {
			for (int i = 0; i < 8; ++i) {
				z_lo = std::min(z_lo, e.p[i].z);
				z_hi = std::max(z_hi, e.p[i].z);
			}
		}
		if (args.z >= z_lo - 1e-9 && args.z <= z_hi + 1e-9)
			std::cerr << "warning: observation z=" << args.z
					  << " intersects model Z=[" << z_lo << "," << z_hi << "]\n";

		if (args.have_H) {
			for (size_t i = 0; i < elems.size(); ++i) {
				const double k = args.have_kappa ? args.kappa : kappa[i];
				elems[i].dens = args.Hprime * k;
			}
		} else {
			bool any = false;
			for (const auto& e : elems)
				if (e.dens.eqNorm() > 0) {
					any = true;
					break;
				}
			if (!any)
				std::cerr << "warning: VTU I is all zeros; pass -H Hx Hy Hz\n";
		}

		// Surfer / Grid order: row = y, col = x  →  data[row*nCol + col]
		Grid grid(ay.n, ax.n, ax.lo, ay.lo, ax.step, ay.step);
		auto solver = gFieldSolver::getCUDAsolver(
			elems.data(), elems.data() + elems.size(), false);

		for (int iy = 0; iy < ay.n; ++iy) {
			const double y = ay.lo + ay.step * iy;
			for (int ix = 0; ix < ax.n; ++ix) {
				const double x = ax.lo + ax.step * ix;
				const Point h = solver->solve(Point{x, y, args.z}) * kFieldConst;
				grid.at(ix, iy) = component_value(h, args.component);
			}
		}

		if (!grid.Write(args.grd))
			throw std::runtime_error("failed to write " + args.grd);

		std::cout << "Wrote " << args.grd << "  " << grid.nCol << "x" << grid.nRow
				  << "  X=[" << ax.lo << "," << ax.hi << "] dx=" << ax.step
				  << "  Y=[" << ay.lo << "," << ay.hi << "] dy=" << ay.step
				  << "  z=" << args.z << "  component=" << args.component;
		if (args.have_H)
			std::cout << "  I0=k*H' H=(" << args.Hprime.x << "," << args.Hprime.y << ","
					  << args.Hprime.z << ") kappa="
					  << (args.have_kappa ? std::to_string(args.kappa) : "from VTU");
		else
			std::cout << "  I from VTU";
		std::cout << "  zmin=" << grid.zMin << " zmax=" << grid.zMax << "\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "vtu_to_grd: " << ex.what() << "\n";
		return 1;
	}
}
