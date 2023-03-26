#ifndef _MAG_MODEL_H_
#define _MAG_MODEL_H_

#include <functional>
#include <vector>
#include "mobj.h"
#include "AssertException.h"
#include "Dat.h"

template <typename ClosedShape>
struct MagModel {
    // Elements of the whole model (all bodies)
    std::vector<ClosedShape> elements;    // elements[i].dens are calculated
    std::vector<double> Kappa;  // const
    std::vector<Point> I0;     // const

    size_t size() const { return elements.size(); }
    void setKappa(const double K) { Kappa.assign(size(), K); }
    void setI0(const Point& I) {
        I0.assign(size(), I);
        for(auto& e: elements) e.dens = I;
    }

    void append(const MagModel& model) {
        elements.insert(elements.end(), model.elements.cbegin(), model.elements.cend());
        Kappa.insert(Kappa.end(), model.Kappa.cbegin(), model.Kappa.cend());
        I0.insert(I0.end(), model.I0.cbegin(), model.I0.cend());
        check();
    }

    std::vector<Point> getElementCenters() const {
        std::vector<Point> centers(elements.size());
        std::transform(elements.cbegin(), elements.cend(), centers.begin(), [](const auto &h) {return h.massCenter();});
        return centers;
    }

    void check() const {
        Assert(elements.size() == Kappa.size());
        Assert(elements.size() == I0.size());
    }
};

template<typename T>
void dumpVectorField(const std::vector<T> &data, const std::string &fname, const std::function<Point(const T&, const size_t)> &point = [](const T& v, const size_t){ return v; }, const std::function<Point(const T&, const size_t)> &value = [](const T& v, const size_t){ return v; }) {
	Dat3D<Point> dd;
	for(size_t i = 0; i < data.size(); ++i) {
		const auto c = point(data[i], i);
		dd.es.push_back({{c.x, c.y, c.z}, value(data[i], i)});
	}
	dd.write(fname);
}

struct MagExperiment : MagModel<HexahedronWid> {
    struct FiledTask {
        std::vector<Point> points;      // const
        std::string fname;              // const
        std::function<Point(const Point&, size_t)> postProcess {[](auto p, auto) { return p; }}; // const
        struct Range {
            size_t from = 0;
            size_t to = 0;
            size_t length() const { return to - from; }
            Range& operator+= (const size_t offset) {
                from += offset;
                to += offset;
                return *this;
            }
        };
        Range exclude;                  // const
        std::function<void(const std::vector<Point>&)> processResult {[](auto) { }};
        std::vector<Point> field;       // calculated and post-processed

        void setAndSaveToFile(std::vector<Point>&& field_) {
            field = std::move(field_);
            for(size_t i = 0; i < field.size(); ++i)
                field[i] = postProcess(field[i], i);

            processResult(field);

            if(!fname.empty()) {
                dumpVectorField<Point>(
                    field,
                    fname,
                    [&points = this->points](const Point&, const size_t idx) { return points[idx]; }
                );
            }
        }
    };

    std::vector<FiledTask> fieldTasksBefore;
    std::vector<FiledTask> fieldTasks;

    void append(const MagExperiment& exp) {
        const size_t elementsOffset = elements.size();
        MagModel::append(exp);
        const size_t newTasksOffset = fieldTasks.size();
        fieldTasks.insert(fieldTasks.end(), exp.fieldTasks.cbegin(), exp.fieldTasks.cend());
        for(auto t = fieldTasks.begin() + newTasksOffset; t != fieldTasks.end(); ++t)
            t->exclude += elementsOffset;
    }
};


#endif