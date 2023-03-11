#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include "mobj.h"
#include "calcField.h"
#include "Dat.h"
#include "inputParser.h"
#include "Stopwatch.h"
#include "Quadrangles.h"
#include "MPIwrapper.h"
#include "MPIpool.h"
#include "CG.h"
#include "AssertException.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;


//get amount of quadrangles for nx*ny*nz discretization
int getQdAm(const int nx, const int ny, const int nz) {
	const int n_2 = 3 * nx*ny*nz;
	return n_2 + nx*ny + nx*nz + ny*nz;
}

//get amount of hexahedrons for nx*ny*nz discretization
int getHexAm(const int nx, const int ny, const int nz) {
	return nx*ny*nz;
}

//x (East direction) to Gaus-Kruger
double xToGK(const double x, const double l0) {
	const int zone = int(l0 / toRad(6.)) + 1;
	return x + (zone*1e3 + 5e2);
}

//Gaus-Kruger to x (East direction)
double xFromGK(const double x, const double l0) {
	const int zone = int(l0 / toRad(6.)) + 1;
	return x - (zone*1e3 + 5e2);
}

//estimate approximate buffer size for the Hexahedrons that can't be replaced by singular source
int triBufferSize(const limits &Nlim, const limits &Elim, const limits &Hlim, const double r) {
	auto f = [&](const limits &lim)->int {return (int)ceil(1.62*r*double(lim.n) / lim.width()); };
	const int v1 = f(Nlim)*f(Elim)*f(Hlim);
	const int v2 = Nlim.n*Elim.n*Hlim.n;
	if(v2 < 40000) return v2;
	return std::min(v1, v2);
}

struct Circle {
	Point2D center;
	double r;
	double distToCenter(const Point &p) const {
		return (center - p).eqNorm();
	}
	bool isIn(const Point &p) const {
		return distToCenter(p) < r;
	}
	Point2D toBorder(const Point &p) const {
		return center + (p - center) * r / (p - center).eqNorm();
	}
};
struct Cylinder : public Circle {
	double h;
	Cylinder(const Circle& c, const double h) : Circle(c), h(h) {}
};
struct QuadrangleRef {
	Point& p1;
	Point& p2;
	Point& p3;
	Point& p4;
	Point J;
	double k;
	bool inner = false;
	bool isCrossing(const Circle& s) const {
		return s.isIn(p1) || s.isIn(p2) || s.isIn(p3) || s.isIn(p4);
	}
	bool isIn(const Circle& s) const {
		return s.isIn(p1) && s.isIn(p2) && s.isIn(p3) && s.isIn(p4);
	}
	operator Quadrangle() const {
		return { p1, p2, p3, p4 };
	}
};

template <class VAlloc>
void wellGen(const Volume &v, const Cylinder &well, const Point Hprime, const double Koutter, const vector<double> Kinner,
		vector<HexahedronWid, VAlloc> &hsi, vector<double> &K) {

	Assert(Kinner.size() == v.z.n);

	const Point Joutter = Hprime*Koutter;
	//const Point J0inner = Hprime*Kinner;

	//make flat mesh
	vector<Point> mesh((v.x.n+1) * (v.y.n+1));
	const limits xLim = { v.x.lower - v.x.dWh() / 2.,  v.x.upper + v.x.dWh() / 2., v.x.n };
	const limits yLim = { v.y.lower - v.y.dWh() / 2.,  v.y.upper + v.y.dWh() / 2., v.y.n };
	for (int yi = 0; yi < yLim.n + 1; ++yi)
		for (int xi = 0; xi < xLim.n + 1; ++xi)
			mesh[yi*(xLim.n+1) + xi] = Point(xLim.at(xi), yLim.at(yi), 0.);
	
	//make lateral Qadrangles
	vector<QuadrangleRef> qrs;
	for (int yi = 0; yi < yLim.n; ++yi)
		for (int xi = 0; xi < xLim.n; ++xi)
			qrs.push_back({ mesh[(yi + 1)*(xLim.n+1) + xi + 1], mesh[(yi + 1)*(xLim.n+1) + xi], mesh[yi*(xLim.n+1) + xi + 1], mesh[yi*(xLim.n+1) + xi], Joutter, Koutter });
	
	//make lateral circle in mesh
	for (QuadrangleRef& q : qrs) {
		if (q.isIn(well))
			q.inner = true;
	}
	for (QuadrangleRef& q : qrs) {
		if (!q.inner && q.isCrossing(well)) {
			if (well.isIn(q.p1)) q.p1 = well.toBorder(q.p1);
			if (well.isIn(q.p2)) q.p2 = well.toBorder(q.p2);
			if (well.isIn(q.p3)) q.p3 = well.toBorder(q.p3);
			if (well.isIn(q.p4)) q.p4 = well.toBorder(q.p4);
		}
	}

	//make hexahedrons
	hsi.resize(xLim.n*yLim.n*v.z.n);
	K.resize(xLim.n*yLim.n*v.z.n);
	
//#pragma omp parallel for
	for (int zi = 0; zi < v.z.n; ++zi)
		for (int yi = 0; yi < yLim.n; ++yi)
			for (int xi = 0; xi < xLim.n; ++xi) {
				const QuadrangleRef& cur = qrs[yi*xLim.n + xi];
				const double Kval = cur.inner ? Kinner[zi] : cur.k;
				const Point Jval = cur.inner ? Hprime*Kinner[zi] : cur.J;

				const int ind = (zi*yLim.n + yi)*xLim.n + xi;
				hsi[ind] = Hexahedron{
					(Quadrangle)cur + Point{0, 0, v.z.at(zi)},
					(Quadrangle)cur + Point{0, 0, v.z.at(zi + 1)},
					Jval };
				K[ind] = Kval;
			}

	auto makeOuterHex = [&Joutter](const Point& llt, const Point& rub) { //left low top, right upper bottom
		return Hexahedron{ {
			{rub.x, rub.y, llt.z},
			{ rub.x, llt.y, llt.z },
			{ llt.x, rub.y, llt.z },
			llt,
			rub,
			{ rub.x, llt.y, rub.z },
			{ llt.x, rub.y, rub.z },
			{llt.x, llt.y, rub.z}
			}, Joutter };
	};
	
	//add 4 external quadrangles
	const double l = 1e8;
	hsi.push_back(makeOuterHex({ xLim.lower, - l, v.z.upper}, { l, yLim.lower, v.z.lower }));
	hsi.push_back(makeOuterHex({ xLim.upper, yLim.lower, v.z.upper }, { l, l, v.z.lower }));
	hsi.push_back(makeOuterHex({- l, yLim.upper, v.z.upper }, { xLim.upper, l, v.z.lower }));
	hsi.push_back(makeOuterHex({- l, - l, v.z.upper }, { xLim.lower, yLim.upper, v.z.lower }));
	K.push_back(Koutter);
	K.push_back(Koutter);
	K.push_back(Koutter);
	K.push_back(Koutter);
}

template <class VAlloc>
void cubeGen(const Volume &v, const Point J, vector<HexahedronWid, VAlloc> &hsi) {
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

template <class VAlloc>
void ellipsoidGen(const Ellipsoid &e, const int nl, const int nB, const int nR, const Point J, vector<HexahedronWid, VAlloc> &hsi) {
	limits ll{0, M_PI_2, nl}, lB{0, M_PI_2, nB}, lReq{0, e.Req, nR}, lRpl{0, e.Rpl, nR};
	// hsi.resize(ll.n * lB.n * lReq.n);
	hsi.resize(0);
	const int riStart = 0;

	const auto makeQuadrangle = [&](const Ellipsoid &e, const int Bi, const int li) {
		return Quadrangle{
			e.getPoint(lB.at(Bi+1), ll.at(li+1)),
			e.getPoint(lB.at(Bi), ll.at(li+1)),
			e.getPoint(lB.at(Bi+1), ll.at(li)),
			e.getPoint(lB.at(Bi), ll.at(li)),
		};
	};

	for(int ri = riStart; ri < lReq.n; ++ri)
		for(int li = 0; li < ll.n; ++li)
			for(int Bi = 0; Bi < lB.n; ++Bi) {
				Quadrangle internal;
				if(ri > 0) {
					Ellipsoid ei{lReq.at(ri), lRpl.at(ri)};
					internal = makeQuadrangle(ei, Bi, li);
				}
				Ellipsoid ee{lReq.at(ri+1), lRpl.at(ri+1)};
				Quadrangle external = makeQuadrangle(ee, Bi, li);
				// const int ind = (ri*ll.n + li)*lB.n + Bi;
				// hsi[ind] = Hexahedron{ external, internal, J };
				hsi.push_back(Hexahedron{ external, internal, J });
			}
			

	const auto dbl = [&](const std::function<void(Hexahedron&)> &f) {
		vector<Hexahedron> tmp(hsi.begin(), hsi.end());
		for(auto& h: tmp) f(h);
		hsi.insert(hsi.end(), tmp.begin(), tmp.end());
	};
	dbl([](Hexahedron& h){ h.mirrorX(); });
	dbl([](Hexahedron& h){ h.mirrorY(); });
	dbl([](Hexahedron& h){ h.mirrorZ(); });
}

template <class VAlloc>
void makeCloud(const vector<HexahedronWid, VAlloc> &hsi, const string datFname) {
	Dat2D<> dat;
	for (const auto& h : hsi)
		if(h.dens.x != 0)
			for (const auto& p : h.p)
				dat.es.push_back({ {p.x, p.y}, p.z });
	dat.write(datFname);
}

template <class VAlloc>
void makeBln(const vector<HexahedronWid, VAlloc> &hsi, const string datFname, const bool isIn = false) {
	Dat2D<> dat;
	for (const auto& h : hsi)
		if (isIn? (h.dens.x == 0) : (h.dens.x != 0)) {
			/*
			dat.es.push_back({ { 5, 0 }, 0 });
			dat.es.push_back({ { h.p[0].x, h.p[0].y }, 0 });
			dat.es.push_back({ { h.p[1].x, h.p[1].y }, 0 });
			dat.es.push_back({ { h.p[3].x, h.p[3].y }, 0 });
			dat.es.push_back({ { h.p[2].x, h.p[2].y }, 0 });
			dat.es.push_back({ { h.p[0].x, h.p[0].y }, 0 });
			*/
			for (int i = 0; i < 2; ++i) {
				const auto t = h.getTri(i);
				//const Triangle t = h.getTri(i);
				dat.es.push_back({ { 4, 0 }, 0 });
				dat.es.push_back({ { t.p1.x, t.p1.y }, 0 });
				dat.es.push_back({ { t.p2.x, t.p2.y }, 0 });
				dat.es.push_back({ { t.p3.x, t.p3.y }, 0 });
				dat.es.push_back({ { t.p1.x, t.p1.y }, 0 });
			}
		}
	dat.write(datFname);
}

class VolumeMod : public Volume {
public:
	template<typename... Args>
	VolumeMod(const Volume& v) : Volume(v) {}
	void save(const string& fname) const {
		Dat3D<Point> dat;
		for (int k = 0; k < z.n; ++k)
			for (int j = 0; j < y.n; ++j)
				for (int i = 0; i < x.n; ++i) {
					const Point p0{ x.atWh(i), y.atWh(j), z.atWh(k) };
					dat.es.push_back({ { p0.x, p0.y, p0.z }, Point() });
				}
	}
};

Point intHexTr__(const Point &p0, const HexahedronWid &h);

void hexTest() {
	// Hexahedron h({
	// 	{ 20, 20, 0 },{ 20, -20, 0 },{ -20, 20, 0 },{ -20, -20, 0 },
	// 	{ 20, 20, -4 },{ 20, -20, -4 },{ -20, 20, -4 },{ -20, -20, -4 }
	// }, Point{ 14, 14, 35 }*0.2);

	Hexahedron h({
		{ 2, 1, 0 },{ 2, -1, 0 },{ -2, 1, 0 },{ -2, -1, 0 },
		{ 2, 1, -2 },{ 2, -1, -2 },{ -2, 1, -2 },{ -2, -1, -2 }
	}, Point{ 14, 14, 35 }*0.02);

	auto hw = HexahedronWid(h);
	const double H = -0.25;


	cout << "Solving..." << endl;
	Dat3D<Point> dat;
	auto l = limits{ -25 + 0.00001, 25 + 0.00001, 40 };
	for (double i = 0; i < l.n; ++i) {
		for (int j = 0; j < l.n; ++j) {
			const Point p0{ l.atWh(j), l.atWh(i), H };
			const Point res = (-intHexTr__(p0, hw)
					 + (hw.isIn(p0)? hw.dens * (4.*M_PI / 3.) : Point())
				) / (4 * M_PI);
			dat.es.push_back({ { p0.x, p0.y, p0.z }, res });
		}
	}
	
	dat.write("cubeFieldSZ_IN.dat");
	cout << "Done." << endl;
}

template<class Input = double, class Acc = Input, class Result = Acc>
class Statistics {
	std::function<Acc(const Input&, const Acc&)> add;
	std::function<Result(const Acc&, const unsigned int count)> end;
	Acc init;
	Acc acc;
	unsigned int count = 0;
public:
	Statistics(
		const Acc &acc = {}, 
		const std::function<Acc(const Input&, const Acc&)> &add = [](const Input& i, const Acc& acc){ return i + acc; },
		const std::function<Result(const Acc&, const unsigned int count)> &end = [](const Acc& acc, const unsigned int count){ return acc; }
	) : acc(acc), init(acc), add(add), end(end) {}

	void next(const Input& i) {
		acc = add(i, acc);
		++count;
	}
	Result get() const {
		return end(acc, count);
	}
	void reset() {
		acc = init;
		count = 0;
	}
};

Point field_sphere_H(const double R, const Point J, const Point p) {
	const Point M0 = J * (4. * M_PI) * R*R*R / 3.;
	const double mu = M0.eqNorm();
	const Point n = M0.norm();
	const double r = p.eqNorm();
	const Point H = ( p * 3. * (n^p) / (r*r*r*r*r) - n / (r*r*r) ) * mu;
	return H;
}

double field_sphere_H_in_Hz(const double R, const double Hprim_z, const double K, const Point p0) {
	const double s = p0.x*p0.x + p0.y*p0.y + p0.z*p0.z;
	const double dr = sqrt(s*s*s*s*s);
	
	return K / (K+3) * Hprim_z * R*R*R * (2*p0.z*p0.z - p0.x*p0.x - p0.y*p0.y)/dr;
}

template<typename T, typename TS>
double eqNorm(const vector<T> &v, const std::function<TS(const T&)> &f) {
	double sum = 0;
	for(auto &e: v) {
		auto t = f(e);
		sum += t ^ t;
	}
	return std::sqrt(sum);
}

Point magnetization_J_theor_ellipsoid(const Ellipsoid &e, const Point J0, const double K) {
	if(e.Rpl == e.Req) {	// Sphere
		return J0 / (1. + K/3.);
	}

	Assert(e.Rpl > e.Req);

	const double m = e.Rpl / e.Req;
	// cout << m << endl;
	const double lnPt = log((m + sqrt(m*m - 1))/(m - sqrt(m*m - 1)));
	// cout << lnPt << endl;
	const double L = (1. / (m*m - 1.)) * (
		(m / (2. * sqrt(m*m - 1))) * lnPt - 1 
	);
	// cout << L << endl;
	const double M = (m / (2. * (m*m - 1.))) * (
		m - (1. / (2. * sqrt(m*m - 1))) * lnPt 
	);
	// cout << M << endl;
	const Point I = J0 / (Point(1.) + Point(K).cmul({M, M, L}));
	return I;
}

class WellDemagCluster : public MPIwrapper {
public:
	int triBufferSize = 0;

	WellDemagCluster(const vector<int> &gpuIdMap = {}) : MPIwrapper() {
		if (gridSize < 2) throw std::runtime_error("You must run at least 2 MPI processes.");
		if (root != 0) throw std::runtime_error("Root process must have rank = 0.");
		const auto lid = localId();
		const int devId = !isRoot() && std::get<1>(lid) ? std::get<0>(lid) - 1 : std::get<0>(lid);
		const int mappedDevId = devId < gpuIdMap.size()? gpuIdMap[devId] : devId;
		cuSolver::setDevice(mappedDevId);
	}

	void runExample(int argc, char *argv[]) {
		InputParser inp(argc, argv);

		// double ellipEq = 10, ellipPol = 20;
		// int nl = 30, nB = 80, nR = 1;

		double ellipEq = 10, ellipPol = 10;
		int nl = 30, nB = 25, nR = 10;

		double K = 2;
		double K2 = K;
		double HprimeX = 14, HprimeY = 14, HprimeZ = 35; //~40A/m

		double ellipsoidOuterDistanceY = 1;

		inp.parseIfExists("ellipEq", ellipEq);
		inp.parseIfExists("ellipPol", ellipPol);
		inp.parseIfExists("nl", nl);
		inp.parseIfExists("nB", nB);
		inp.parseIfExists("nR", nR);
		inp.parseIfExists("K", K);
		K2 = K;
		inp.parseIfExists("K2", K2);
		inp.parseIfExists("HprimeX", HprimeX);
		inp.parseIfExists("HprimeY", HprimeY);
		inp.parseIfExists("HprimeZ", HprimeZ);
		inp.parseIfExists("d", ellipsoidOuterDistanceY);

		const Ellipsoid e(ellipEq, ellipPol);
		const Point Hprime = { HprimeX, HprimeY, HprimeZ };

		const auto ellipsoidModelGenerator = [&](vector<HexahedronWid> &hsi, vector<double> &Kmodel, vector<Point> &I0out){
			const auto I0 = Hprime * K;
			// const auto Ipres = I0 / (1. + K/3.);	// Use known precise I
			// ellipsoidGen(e, nl, nB, nR, Ipres, hsi);
			ellipsoidGen(e, nl, nB, nR, I0, hsi);
			Kmodel.assign(hsi.size(), K);
			I0out.assign(hsi.size(), I0);
		};
		const auto cubeModelGenerator = [&](vector<HexahedronWid> &hsi, vector<double> &Kmodel, vector<Point> &I0out) {
			const auto I0 = Hprime * K;
			cubeGen(Volume{{-10, 10, nl},{-5, 5, nB}, {-5, 5, nR}}, I0, hsi);
			Kmodel.assign(hsi.size(), K);
			I0out.assign(hsi.size(), I0);
		};

		const auto twoEllipsoidsModelGenerator = [&](vector<HexahedronWid> &hsi, vector<double> &Kmodel, vector<Point> &I0out){
			// Generate first ellipsoid
			const auto I0_1 = Hprime * K;
			ellipsoidGen(e, nl, nB, nR, I0_1, hsi);
			Kmodel.assign(hsi.size(), K);
			I0out.assign(hsi.size(), I0_1);

			// Generate second ellipsoid
			vector<HexahedronWid> hsi2;
			const auto I0_2 = Hprime * K2;
			ellipsoidGen(e, nl, nB, nR, I0_2, hsi2);
			// Translate second ellipsoid
			for(auto &q: hsi)
				q += Point{e.Req + ellipsoidOuterDistanceY, 0, 0};
			// Append elements
			hsi.insert(hsi.end(), hsi2.begin(), hsi2.end());
			const vector<double> K2_v(hsi2.size(), K2);
			Kmodel.insert(Kmodel.end(), K2_v.begin(), K2_v.end());
			const vector<Point> I0_2_v(hsi2.size(), I0_2);
			I0out.insert(I0out.end(), I0_2_v.begin(), I0_2_v.end());
		};

		const auto createCudaSolver = [&](const vector<HexahedronWid>& hsi, const bool transpose) {
			return gFieldSolver::getCUDAsolver(&*hsi.cbegin(), &*hsi.cend(), transpose);
			// const double replDist = 3;
			// return gFieldSolver::getCUDAreplacingSolver(&*hsi.cbegin(), &*hsi.cend(), replDist, nl*nB*nR*8);
		};

		Stopwatch tmr;
		tmr.start();
		vector<HexahedronWid> hsi = demagCG<HexahedronWid>(twoEllipsoidsModelGenerator, createCudaSolver);

		// Calculate the effect of the rest of the model on the first ellipsoid
		const int firstEllipsoidElementsN = isRoot()? nR * nl * nB * 8 : 0;
		{
			vector<HexahedronWid> firstEllipsoid(hsi.cbegin(), hsi.cbegin() + firstEllipsoidElementsN);
			vector<HexahedronWid> restModel(hsi.cbegin() + firstEllipsoidElementsN, hsi.cend());
			const auto solver = createNodeSolver<HexahedronWid>(createCudaSolver, restModel.cbegin(), restModel.cend());
			const vector<double> K_firstEllipsoid(firstEllipsoid.size(), K);
			vector<Point> J = J_inElements<HexahedronWid>(solver, firstEllipsoid.cbegin(), firstEllipsoid.cend(), K_firstEllipsoid.cbegin());

			const double JeffectNorm = eqNorm<Point, Point>(J, [](auto &v){ return v; });
			const double JselfNorm = eqNorm<HexahedronWid, Point>(firstEllipsoid, [](auto &e){ return e.dens; });

			if(isRoot()) cout << "Rest model Effect: " << JeffectNorm / JselfNorm << endl;
		}


		if(!isRoot()) return;
		cout << "Total time: " << tmr.stop() << "sec." << endl;

		hsi.resize(firstEllipsoidElementsN); // Drop everything from the model except the first ellipsoid

		const int layersN = hsi.size() / (nl*nB*8);
		Assert(layersN == nR);
		cout << "Layers: " << layersN << endl;

		// Mean dens for model
		Statistics<HexahedronWid, Point> meanStat(
			Point{},
			[](auto& h, auto& acc){ return h.dens + acc; },
			[](auto& acc, auto count){ return acc / count; }
		);
		// Mean dens by layer
		std::vector meanStatLayers(layersN, Statistics<HexahedronWid, Point>(
			Point{},
			[](auto& h, auto& acc){ return h.dens + acc; },
			[](auto& acc, auto count){ return acc / count; }
		));
		const int inSz = nl*nB;
		for(int part = 0; part < 8; ++part)
			for(int layer = 0; layer < layersN; ++layer) 
				for(int i = 0; i < inSz; ++i) {
					const int idx = ((part * layersN) + layer) * inSz + i;
					meanStatLayers[layer].next(hsi[idx]);
				}
		for(auto& h: hsi) meanStat.next(h);
		const Point mean = meanStat.get();

		const auto I0 = Hprime * K;
		const auto Ipres = magnetization_J_theor_ellipsoid(e, I0, K);

		// RMS dens for model
		Statistics<Point, double> rmsStat(
			0,
			[&](auto& diff, auto& acc){
				return acc + (diff^diff); 
			},
			[](auto& acc, auto count){ return std::sqrt(acc / count); }
		);
		// RMS dens by layer
		std::vector rmsStatLayers(layersN, Statistics<Point, double>(
			0,
			[&](auto& diff, auto& acc){
				return acc + (diff^diff); 
			},
			[](auto& acc, auto count){ return std::sqrt(acc / count); }
		));
		for(int part = 0; part < 8; ++part)
			for(int layer = 0; layer < layersN; ++layer) 
				for(int i = 0; i < inSz; ++i) {
					const int idx = ((part * layersN) + layer) * inSz + i;
					rmsStatLayers[layer].next(hsi[idx].dens - Ipres);
				}
		for(auto& h: hsi) rmsStat.next(h.dens - Ipres);
		const double rms = rmsStat.get();

		// Testing, testing 1,2,3...

		cout << "Ellipsoid presice I = " << Ipres  << " | demag_rel_err = " << (Ipres-I0).eqNorm()/Ipres.eqNorm() << endl;
		cout << "Jmean= " << mean << " | Jmean_err_pres= " << (Ipres-mean)/Ipres << " ~ " << (Ipres-mean).eqNorm()/Ipres.eqNorm() << " | rms_err_pres= " << rms << " ~ " << rms/Ipres.eqNorm() << endl << endl;

		for(int layer = 0; layer < layersN; ++layer) {
			cout << layer << ": "  << "Jmean= " << meanStatLayers[layer].get() << " | rms_err= " << rmsStatLayers[layer].get() << endl;
		}

		// const double fieldElipHeight = 1;
		// const Point p0{0, 0, e.Req + fieldElipHeight};
		// cout << "Sphere presice Hsnd_z = " << field_sphere_H_in_Hz(e.Req, Hprime.z, K, p0) << endl;
		// const Point Hprec = field_sphere_H(e.Req, Ipres, p0) / (4.*M_PI);
		// cout << "Sphere presice H = " << Hprec << endl;
		// const Point noDemagHprec = field_sphere_H(e.Req, I0, p0) / (4.*M_PI);
		// cout << "Sphere presice H (no demag) = " << noDemagHprec  << " | rel_err = " << (Hprec-noDemagHprec).eqNorm()/Hprec.eqNorm() << endl;


		// {
		// 	Dat3D<Point> dd;
		// 	dd.es.resize(hsi.size());
		// 	std::transform(hsi.cbegin(), hsi.cend(), dd.es.begin(), [](const HexahedronWid &h) -> Dat3D<Point>::Element {
		// 		const auto p = h.massCenter();
		// 		return {{p.x, p.y, p.z}, h.dens};
		// 	});
		// 	dd.write("elip_J_int.dat");
		// }

		const auto solver = createCudaSolver(hsi, false);


		const auto &fOnDat = [&](Dat3D<Point> &res) {
			for (auto &i : res) 
				i.val = -solver->solve({ i.p.x, i.p.y, i.p.z }) / (4 * M_PI);
		};

		// {
		// 	Dat3D<Point> dd;
		// 	const double t = 0.001;
		// 	for (double x = -15-t; x < 15; x += 0.2)
		// 		for (double y = -15-t; y < 15; y += 0.2)
		// 			dd.es.push_back({{ x, y, e.Req/2 + t}});
		// 	fOnDat(dd);
		// 	dd.write("elip_f_in_out_no_J3.dat");
		// 	return;
		// }

		// {
		// 	Dat3D<Point> dd;
		// 	const double t = 0.007;
		// 	for (double x = -15-t; x < 15; x += 0.2)
		// 		for (double y = -15-t; y < 15; y += 0.2)
		// 			dd.es.push_back({{ x, y, e.Req/2 + t}});

		// 	fOnDat(dd);
		// 	dd.write("cube_f_in_out.dat");

		// 	for(auto &el: dd.es) el.p.z = e.Req + 1;
		// 	fOnDat(dd);
		// 	dd.write("cube_f_out.dat");
		// }

		// {
		// 	Dat3D<Point> dd;
		// 	for(int part = 0; part < 4; ++part)
		// 		for(int layer = 0; layer < layersN; ++layer) 
		// 			for(int li = 0; li < nl; ++li) {
		// 				const int Bi = 0;
		// 				const int idx = ((part * layersN) + layer) * inSz + (li * nB) + Bi;
		// 				const auto c = hsi[idx].massCenter();
		// 				dd.es.push_back({{c.x, c.y, c.z}, hsi[idx].dens});
		// 			}
		// 	dd.write("elip_J_in.dat");
		// }
		
		{
			Dat3D<Point> dd;
			for(int i = 0; i < hsi.size(); ++i) {
				const auto c = hsi[i].massCenter();
				dd.es.push_back({{c.x, c.y, c.z}, hsi[i].dens});
			}
			dd.write("ball_J_all_in.dat");
		}
	}

private:

	vector<Point> fieldInPoints(
		const std::unique_ptr<gFieldSolver> &solver,		// Valid on all nodes
		vector<Point>::const_iterator fieldPointsBegin,
		vector<Point>::const_iterator fieldPointsEnd,
		const bool logging = false
	) {	// result.size() == fieldPointsEnd - fieldPointsBegin
		const int nodeBatchSize = 1024;
		vector<Point> fieldPoints(fieldPointsBegin, fieldPointsEnd);
		vector<Point> field(fieldPoints.size());
		MPIpool<Point, Point> pool(*this, fieldPoints, field, nodeBatchSize);
		pool.logging = logging;

		int taskCount = 0;
		if (!isRoot()) {
			while (1) {
				const vector<Point> task = pool.getTask();
				if (!task.size()) break;
				if(pool.logging) cout << "Task accepted " << taskCount++ << " size: " << task.size() << endl;
				vector<Point> result(task.size());
				const double fieldConst = -(1. / (4. * M_PI));
				for (int i = 0; i < task.size(); ++i)
					result[i] = solver->solve(task[i]) * fieldConst;
				pool.submit(result);
			}
		} else {
			if(pool.logging) cout << "Result gather ok" << endl;
			return field; // root
		}
		return {}; // non-roots
	}

	template<class ClosedShape>
	vector<Point> J_inElements(
		const std::unique_ptr<gFieldSolver> &solver,					// Valid on all nodes
		typename vector<ClosedShape>::const_iterator fieldPointsBegin,
		typename vector<ClosedShape>::const_iterator fieldPointsEnd,
		vector<double>::const_iterator K_inFieldPointsBegin,			// (K_inFieldPointsEnd - K_inFieldPointsBegin) == (fieldPointsEnd - fieldPointsBegin)
		const bool logging = false
	) {	// result.size() == fieldPointsEnd - fieldPointsBegin
		vector<Point> fieldPoints(fieldPointsEnd - fieldPointsBegin);
		std::transform(fieldPointsBegin, fieldPointsEnd, fieldPoints.begin(), [](const ClosedShape &h) {return h.massCenter();});
		vector<Point> field = fieldInPoints(solver, fieldPoints.begin(), fieldPoints.end(), logging);
		std::transform(field.cbegin(), field.cend(), K_inFieldPointsBegin, field.begin(), [](const Point f, const double K) { return f * K; });
		return field; // return J = K*H
	}

	template<class ClosedShape>
	const std::unique_ptr<gFieldSolver> createNodeSolver(
		const std::function<std::unique_ptr<gFieldSolver>(const vector<ClosedShape>&, const bool)> createCudaSolver,
		const typename vector<ClosedShape>::const_iterator modelBegin,
		const typename vector<ClosedShape>::const_iterator modelEnd,
		const bool transpose = false
	) {
		vector<ClosedShape> model(modelBegin, modelEnd);
		Bcast(model);
		return createCudaSolver(model, transpose);
	}

	template<class ClosedShape>
	vector<ClosedShape> demagCG(
		const std::function<void(vector<ClosedShape>&, vector<double>&, vector<Point>&)> &modelGenerator,
		const std::function<std::unique_ptr<gFieldSolver>(const vector<ClosedShape>&, const bool)> createCudaSolver
	) {
		if(isRoot()) cout << "Generating model..." << endl;
		vector<ClosedShape> hsi;
		vector<double> K;
		vector<Point> J0;
		if(isRoot()) modelGenerator(hsi, K, J0);
		if(isRoot()) cout << "Model size: " << hsi.size() << endl;
		// return hsi;

		const auto OpCGt = [&hsi, &K, &createCudaSolver, this](const vector<Point> x = {}, const bool transpose = false) -> vector<Point> {
			const bool logging = false;
			auto model{ hsi };
			for(size_t i = 0; i < model.size(); ++i) model[i].dens = x[i];

			const auto solver = createNodeSolver(createCudaSolver, model.begin(), model.end(), transpose);
			vector<Point> J = J_inElements<ClosedShape>(solver, model.cbegin(), model.cend(), K.cbegin(), logging);

			std::transform(J.cbegin(), J.cend(), x.cbegin(), J.begin(), [](const Point J, const Point x) { return x - J; });
			return J; // return x - J
		};

		vector<Point> x0(hsi.size());
		std::transform(hsi.cbegin(), hsi.cend(), x0.begin(), [](const ClosedShape &h) {return h.dens;});
		CG<Point> cg{J0, x0, OpCGt};
		if(isRoot()) cout << "Demag Solving..." << endl;

		// Non-roots will do calculations here
		if (!isRoot()) {
			while (1) {
				bool cont = false;
				Bcast(cont);
				if (!cont) break;
				OpCGt();
			}
			cout << "Done." << endl;
			return {};
		}

		// Root only code path
		
		const double eps = 1e-4;
		const int maxIter = 10;
		double err = 1, prvErr = err*100;

		cout << "CG: preparing first iter" << endl;
		{
			bool cont = true;
			Bcast(cont);
			cg.prepare();
		}

		for (int it = 0; it < maxIter && err > eps; ++it) {
			cout << "Iter: " << it << endl;
			bool cont = true;
			Bcast(cont);
			cg.nextIter();
			prvErr = err;
			err = cg.getError();
			
			const auto errDiff = err - prvErr;
			cout << "Err: " << err << "(" << (errDiff>0?"+":"") << errDiff << ")" << " at iter: " << it << endl;
		}

		// Stop non-roots
		bool cont = false;
		Bcast(cont);

		// Copy updated J back into the model
		for (int i = 0; i < hsi.size(); ++i) hsi[i].dens = cg.x[i];

		return hsi;
	}
};

int main(int argc, char *argv[]) {
	bool isRoot = true;
	try {
		WellDemagCluster().runExample(argc, argv);
		return 0;
	}
	catch (std::exception &ex) {
		if(isRoot) std::cerr << "Global exception: " << ex.what() << endl;
		return 1;
	}
	cout << "Done" << endl;
	return 0;
}
