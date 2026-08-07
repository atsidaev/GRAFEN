#pragma once
/**
 * VTK UnstructuredGrid (.vtu) save/load for GRAFEN hex models.
 * Matches python/grafen/model_io.py (GRAFEN corner order <-> VTK_HEXAHEDRON).
 */

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mobj.h"

namespace grafen_test {

// VTK local node v uses GRAFEN corner kGrafenToVtk[v] (same as Python _GRAFEN_TO_VTK)
inline constexpr int kGrafenToVtk[8] = {7, 5, 4, 6, 3, 1, 0, 2};
// GRAFEN corner g comes from VTK node kVtkToGrafen[g] (argsort of above)
inline constexpr int kVtkToGrafen[8] = {6, 5, 7, 4, 2, 1, 3, 0};

inline constexpr int kVtkHexahedron = 12;

inline void save_model_vtu(
	const std::string& path,
	const std::vector<HexahedronWid>& elems,
	const std::vector<double>& kappa)
{
	if (kappa.size() != elems.size())
		throw std::runtime_error("save_model_vtu: kappa size must match elements");

	const size_t n = elems.size();
	std::ostringstream out;
	out.precision(16);
	out << std::scientific;
	out << "<?xml version=\"1.0\"?>\n";
	out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	out << "  <UnstructuredGrid>\n";
	out << "    <Piece NumberOfPoints=\"" << (n * 8) << "\" NumberOfCells=\"" << n << "\">\n";
	out << "      <Points>\n";
	out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
	for (const auto& e : elems)
		for (int i = 0; i < 8; ++i)
			out << e.p[i].x << " " << e.p[i].y << " " << e.p[i].z << " ";
	out << "\n        </DataArray>\n      </Points>\n";
	out << "      <Cells>\n";
	out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ";
	for (size_t c = 0; c < n; ++c)
		for (int v = 0; v < 8; ++v)
			out << (c * 8 + kGrafenToVtk[v]) << " ";
	out << "\n        </DataArray>\n";
	out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ";
	for (size_t c = 1; c <= n; ++c)
		out << (c * 8) << " ";
	out << "\n        </DataArray>\n";
	out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ";
	for (size_t c = 0; c < n; ++c)
		out << kVtkHexahedron << " ";
	out << "\n        </DataArray>\n      </Cells>\n";
	out << "      <CellData Scalars=\"kappa\" Vectors=\"I\">\n";
	out << "        <DataArray type=\"Float64\" Name=\"I\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
	for (const auto& e : elems)
		out << e.dens.x << " " << e.dens.y << " " << e.dens.z << " ";
	out << "\n        </DataArray>\n";
	out << "        <DataArray type=\"Float64\" Name=\"kappa\" format=\"ascii\">\n          ";
	for (double k : kappa)
		out << k << " ";
	out << "\n        </DataArray>\n      </CellData>\n";
	out << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";

	std::ofstream f(path);
	if (!f)
		throw std::runtime_error("save_model_vtu: cannot write " + path);
	f << out.str();
}

namespace detail {

inline std::vector<double> parse_doubles(const std::string& body) {
	std::istringstream iss(body);
	std::vector<double> v;
	double x;
	while (iss >> x)
		v.push_back(x);
	return v;
}

inline std::string data_array_named(const std::string& xml, const std::string& name) {
	const std::string tag = "Name=\"" + name + "\"";
	size_t pos = 0;
	while (true) {
		size_t da = xml.find("<DataArray", pos);
		if (da == std::string::npos)
			throw std::runtime_error("VTU: DataArray not found: " + name);
		size_t endOpen = xml.find('>', da);
		std::string open = xml.substr(da, endOpen - da);
		if (open.find(tag) != std::string::npos) {
			size_t close = xml.find("</DataArray>", endOpen);
			return xml.substr(endOpen + 1, close - endOpen - 1);
		}
		pos = endOpen + 1;
	}
}

}  // namespace detail

inline void load_model_vtu(
	const std::string& path,
	std::vector<HexahedronWid>& elems,
	std::vector<double>& kappa)
{
	std::ifstream f(path);
	if (!f)
		throw std::runtime_error("load_model_vtu: cannot open " + path);
	std::ostringstream ss;
	ss << f.rdbuf();
	const std::string xml = ss.str();

	size_t pointsSec = xml.find("<Points>");
	size_t pointsEnd = xml.find("</Points>");
	if (pointsSec == std::string::npos || pointsEnd == std::string::npos)
		throw std::runtime_error("VTU: missing Points");
	std::string pointsXml = xml.substr(pointsSec, pointsEnd - pointsSec);
	size_t pda = pointsXml.find("<DataArray");
	size_t pOpen = pointsXml.find('>', pda);
	size_t pClose = pointsXml.find("</DataArray>", pOpen);
	auto pts = detail::parse_doubles(pointsXml.substr(pOpen + 1, pClose - pOpen - 1));
	if (pts.size() % 3 != 0)
		throw std::runtime_error("VTU: bad points size");

	auto conn = detail::parse_doubles(detail::data_array_named(xml, "connectivity"));
	auto types = detail::parse_doubles(detail::data_array_named(xml, "types"));
	auto I = detail::parse_doubles(detail::data_array_named(xml, "I"));
	auto K = detail::parse_doubles(detail::data_array_named(xml, "kappa"));

	if (conn.size() % 8 != 0)
		throw std::runtime_error("VTU: connectivity not multiple of 8");
	const size_t n = conn.size() / 8;
	if (types.size() != n)
		throw std::runtime_error("VTU: types size mismatch");
	for (double t : types)
		if (static_cast<int>(t) != kVtkHexahedron)
			throw std::runtime_error("VTU: only hex cells (type 12) supported");
	if (I.size() != n * 3)
		throw std::runtime_error("VTU: I size mismatch");
	if (K.size() != n)
		throw std::runtime_error("VTU: kappa size mismatch");

	elems.resize(n);
	kappa.resize(n);
	for (size_t c = 0; c < n; ++c) {
		Hexahedron h;
		for (int g = 0; g < 8; ++g) {
			const size_t pid = static_cast<size_t>(conn[c * 8 + kVtkToGrafen[g]]);
			if (pid * 3 + 2 >= pts.size())
				throw std::runtime_error("VTU: point id out of range");
			h.p[g] = Point{pts[pid * 3], pts[pid * 3 + 1], pts[pid * 3 + 2]};
		}
		h.dens = Point{I[c * 3], I[c * 3 + 1], I[c * 3 + 2]};
		elems[c] = HexahedronWid{h};
		kappa[c] = K[c];
	}
}

inline bool models_almost_equal(
	const std::vector<HexahedronWid>& a,
	const std::vector<HexahedronWid>& b,
	const std::vector<double>& ka,
	const std::vector<double>& kb,
	double tol = 1e-12)
{
	if (a.size() != b.size() || ka.size() != kb.size() || a.size() != ka.size())
		return false;
	for (size_t i = 0; i < a.size(); ++i) {
		for (int c = 0; c < 8; ++c) {
			if (std::abs(a[i].p[c].x - b[i].p[c].x) > tol) return false;
			if (std::abs(a[i].p[c].y - b[i].p[c].y) > tol) return false;
			if (std::abs(a[i].p[c].z - b[i].p[c].z) > tol) return false;
		}
		if (std::abs(a[i].dens.x - b[i].dens.x) > tol) return false;
		if (std::abs(a[i].dens.y - b[i].dens.y) > tol) return false;
		if (std::abs(a[i].dens.z - b[i].dens.z) > tol) return false;
		if (std::abs(ka[i] - kb[i]) > tol) return false;
	}
	return true;
}

/** Save elems to VTU and load a copy (kappa filled with K). Temp file is removed. */
inline std::vector<HexahedronWid> roundtrip_vtu(
	const std::vector<HexahedronWid>& elems,
	double K,
	const std::string& path = "grafen_test_roundtrip.vtu")
{
	std::vector<double> kappa(elems.size(), K);
	save_model_vtu(path, elems, kappa);
	std::vector<HexahedronWid> loaded;
	std::vector<double> kappa2;
	load_model_vtu(path, loaded, kappa2);
	std::remove(path.c_str());
	return loaded;
}

}  // namespace grafen_test
