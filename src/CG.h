#ifndef _CG_CLASS_H
#define _CG_CLASS_H

#include <vector>
#include <functional>

#include "AssertException.h"

template<typename T>
struct CG {
    const std::vector<T> &b;
    const std::function<std::vector<T>(const std::vector<T>&)> Op;

    bool ready = false;
    std::vector<T> r;
    std::vector<T> z;
    std::vector<T> x;

    CG(const std::vector<T> &b, const std::vector<T> &x0, const std::function<std::vector<T>(const std::vector<T>&)> &Op): b(b), x(x0), Op(Op) {}

    // x = ax + by
    static void ax_plus_by(const double a, std::vector<T>& x, const double b, const std::vector<T>& y) {
        std::transform(x.cbegin(), x.cend(), y.cbegin(), x.begin(), [&a, &b](const auto& x, const auto &y){ return x*a + y*b; });
    }
    static double dot(const std::vector<T>& a, const std::vector<T>& b) {
        return std::transform_reduce(a.cbegin(), a.cend(), b.cbegin(), 0., 
            [](const auto& a, const auto& b){ return a + b; }, 
            [](const auto& a, const auto& b){ return a ^ b; }
        );
    }

    double getError() const {
        return sqrt(dot(r, r) / dot(b, b));
    }

    void prepare() {
        r = Op(x);	//r0 = Ax
        Assert(x.size() == r.size());
        ax_plus_by(-1, r, 1, b); //r0 = b - Ax
        z = r;
        ready = true;
    }

    void nextIter() {
        if(!ready) throw std::runtime_error("CG: call prepare() before iter()");
        const auto Az = Op(z);
        const double r_dot = dot(r, r);
        const double alpha = r_dot / dot(Az, z);
        ax_plus_by(1, x, alpha, z); 		// x = x_prv + alpha*z_prv
        ax_plus_by(1, r, -alpha, Az);		// r = r_prv - alpha*Az_prv
        const double beta = dot(r, r) / r_dot;
        ax_plus_by(beta, z, 1, r);  		// z = r + beta*z_prv
    }

};

#endif
