
// This program was created by the Equelle compiler from SINTEF.

#include "equelle/EquelleRuntimeCPU.hpp"

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);
void ensureRequirements(const equelle::EquelleRuntimeCPU& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {
    using namespace equelle;
    ensureRequirements(er);

    // ============= Generated code starts here ================

    const Scalar cfl = er.inputScalarWithDefault("cfl", double(0.9));
    const Scalar g = er.inputScalarWithDefault("g", double(9.81));
    const CollOfScalar h0 = er.inputCollectionOfScalar("h0", er.allCells());
    const CollOfScalar hu0 = er.inputCollectionOfScalar("hu0", er.allCells());
    const CollOfScalar hv0 = er.inputCollectionOfScalar("hv0", er.allCells());
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q0 = makeArray(h0, hu0, hv0);
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q = q0;
    auto compute_flux = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& ql, const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& qr, const CollOfScalar& l, const CollOfVector& n) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar hl = std::get<0>(ql);
        const CollOfScalar hul = std::get<1>(ql);
        const CollOfScalar hvl = std::get<2>(ql);
        const CollOfScalar hr = std::get<0>(qr);
        const CollOfScalar hur = std::get<1>(qr);
        const CollOfScalar hvr = std::get<2>(qr);
        const Scalar pl = double(0.7);
        const Scalar pr = double(0.9);
        const CollOfScalar cl = er.sqrt((g * hl));
        const CollOfScalar cr = er.sqrt((g * hr));
        const Scalar am = double(0);
        const Scalar ap = double(0);
        const std::tuple<Scalar, Scalar, Scalar> f_flux = makeArray(double(0.9), double(0.9), double(0.9));
        const std::tuple<Scalar, Scalar, Scalar> g_flux = makeArray(double(0.8), double(0.8), double(0.8));
        const std::tuple<Scalar, Scalar, Scalar> central_upwind_correction = makeArray(double(0.9), double(0.9), double(0.9));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> flux = makeArray(er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()));
        const CollOfScalar max_wave_speed = er.operatorExtend(double(0.8), er.allFaces());
        return makeArray(std::get<0>(flux), std::get<1>(flux), std::get<2>(flux), max_wave_speed);
    };
    auto reconstruct_plane = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        return makeArray(er.operatorExtend(double(0), er.allCells()), er.operatorExtend(double(0), er.allCells()));
    };
    const CollOfFace ifs = er.interiorFaces();
    const CollOfCell first = er.firstCell(ifs);
    const CollOfCell second = er.secondCell(ifs);
    const std::tuple<CollOfScalar, CollOfScalar> slopes = reconstruct_plane(q);
    const CollOfVector n = er.normal(ifs);
    const CollOfVector ip = er.centroid(ifs);
    const CollOfVector first_to_ip = (ip - er.centroid(first));
    const CollOfVector second_to_ip = (ip - er.centroid(second));
    const CollOfScalar l = er.norm(ifs);
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q1 = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), first), er.operatorOn(std::get<1>(q), er.allCells(), first), er.operatorOn(std::get<2>(q), er.allCells(), first));
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q2 = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), second), er.operatorOn(std::get<1>(q), er.allCells(), second), er.operatorOn(std::get<2>(q), er.allCells(), second));
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar, CollOfScalar> flux_and_max_wave_speed = compute_flux(q1, q2, l, n);
    const Scalar min_area = double(0.9);
    const Scalar max_wave_speed = double(0.8);
    const Scalar dt = (cfl * (min_area / (double(6) * max_wave_speed)));
    er.output("q0", std::get<0>(q));
    er.output("q1", std::get<1>(q));
    er.output("q2", std::get<2>(q));

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
