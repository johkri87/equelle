#define BOOST_TEST_NO_MAIN

#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <vector>
#include <numeric>
#include <fstream>
#include <tuple>

#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/CartesianGrid.hpp"





/**
 * Test that we can solve the heat equation
 */
BOOST_AUTO_TEST_CASE( heatEquation ) {
    int dim_x = 30;
    int dim_y = 50;
    int ghostWidth = 1;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );
    equelle::CartesianGrid::CartesianCollectionOfScalar u = grid.inputCellScalarWithDefault( "u", 1.0 );

    const double k = 1.0; //Material specific heat diffusion constant
    const double dx = 1.0;//5.0 / static_cast<double>(dim_x);
    const double dy = 1.0;//5.0 / static_cast<double>(dim_y);
    const double dt = 1.0;

    const float a = k * dt / (dx*dy);

    double t_end = 100.0;
    double t = 0.0;

    equelle::CartesianGrid::CellRange allCells = grid.allCells();
    equelle::CartesianGrid::CartesianCollectionOfScalar u0 = u;


    //Our stencil
    auto f = [&] (int i, int j) {
        grid.cellAt( i, j, u ) = grid.cellAt( i+0, j+0, u0 ) +
                                 a * 1.0/8.0 * ( grid.cellAt( i+0, j-1, u0 ) +
                                                 grid.cellAt( i+0, j+1, u0 ) +
                                                 grid.cellAt( i-1, j+0, u0 ) +
                                                 grid.cellAt( i+1, j+0, u0 ) -
                                             4.0*grid.cellAt( i+0, j+0, u0 ) );
    };

    while (t < t_end) {
        allCells.execute(f);
        /*
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                f(i, j);
            }
        }
        */
        t = t + dt;
        u0 = u;

        std::stringstream filename;
        std::showpoint(filename);
        filename << "waveheights_" << boost::format("%011.5f") % t << ".csv";

        std::ofstream file(filename.str());
        grid.dumpGridCells( u, file );
    }
}

inline double sign(double& a) {
    /**
      * The following works by bit hacks. In non-obfuscated code, something like
      *  float r = ((int&)a & 0x7FFFFFFF)!=0; //set r to one or zero
      *  (int&)r |= ((int&)a & 0x80000000);   //Copy sign bit of a
      *  return r;
      */
    return (signed((int&)a & 0x80000000) >> 31 ) | ((int&)a & 0x7FFFFFFF)!=0;
}

/**
 * @return min(a, b, c), {a, b, c} > 0
 *         max(a, b, c), {a, b, c} < 0
 *         0           , otherwise
 */
inline double minmod(double a, double b, double c) {
    return 0.25
        *sign(a)
        *(sign(a) + sign(b))
        *(sign(b) + sign(c))
        *std::min( std::min(std::abs(a), std::abs(b)), std::abs(c) );
}

inline double derivative(const double& left,
        const double& center,
        const double& right,
        const double& dx,
        const double& theta) {
    return minmod(theta*(center-left)/dx,
            0.5f*(right-left),
            theta*(right-center));
}

inline double fluxFunc(double a) {
    return a;
}

inline double centralUpwind(double a_max, double a_min,
        double fm, double fp,
        double um, double up) {
    return ((a_max*fm - a_min*fp) + a_max*a_min*(up-um))/(a_max-a_min);
}

/**
 * Test that we can solve the wave equation with a second order stencil
 */
BOOST_AUTO_TEST_CASE( heatEquation_2nd_order ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 2;

    typedef equelle::CartesianGrid::Face Face;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y ),  ghostWidth );
    equelle::CartesianGrid::CartesianCollectionOfScalar u = grid.inputCellScalarWithDefault( "u", 1.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar udx = grid.inputCellScalarWithDefault( "udx", 0.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar udy = grid.inputCellScalarWithDefault( "udy", 0.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar up = grid.inputFaceScalarWithDefault( "up", 0.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar um = grid.inputFaceScalarWithDefault( "um", 0.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar f = grid.inputFaceScalarWithDefault( "f", 0.0 );

    const double k = 1.0; //Material specific heat diffusion constant
    const double dx = 1.0;//5.0 / static_cast<double>(dim_x);
    const double dy = 1.0;//5.0 / static_cast<double>(dim_y);
    const double dt = 1.0;

    const float a = k * dt / (dx*dy);
    const double theta  = 1.3;

    double t_end = 100.0;
    double t = 0.0;

    equelle::CartesianGrid::CartesianCollectionOfScalar u0 = u;
    while (t < t_end) {
        //Generate a minmod limited derivative along x
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                grid.cellAt( i, j, udx ) = derivative(
                                                        grid.cellAt( i-1, j, u ),
                                                        grid.cellAt( i, j, u ),
                                                        grid.cellAt( i+1, j, u ),
                                                        dx, theta);
            }
        }

        //Generate a minmod limited derivative along y
        for( int j = -1; j <= dim_y; ++j ) {
            for( int i = -1; i <= dim_x; ++i ) {
                grid.cellAt( i, j, udy ) = derivative(
                                                        grid.cellAt( i, j-1, u ),
                                                        grid.cellAt( i, j, u ),
                                                        grid.cellAt( i, j+1, u ),
                                                        dy, theta);
            }
        }

        //Evaluate um/up for each internal face in x
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x-1; ++i ) {
                grid.faceAt( i, j, Face::posX, um ) = grid.cellAt( i, j, u )
                        + 0.5 * grid.cellAt( i, j, udx );
                grid.faceAt( i, j, Face::posX, up ) = grid.cellAt( i+1, j, u )
                        - 0.5 * grid.cellAt( i+1, j, udx );
            }
        }

        //Evaluate um/up for each internal face in y
        for( int j = 0; j < dim_y-1; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                grid.faceAt( i, j, Face::posY, um ) = grid.cellAt( i, j, u )
                        + 0.5 * grid.cellAt( i, j, udy );
                grid.faceAt( i, j, Face::posY, up ) = grid.cellAt( i, j+1, u )
                        - 0.5 * grid.cellAt( i, j+1, udy );
            }
        }

        //Evaluate flux for each *internal* face in x
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 1; i < dim_x; ++i ) {
                //Only bogus fluxes here, but models how we should do something like shallow water
                //((a_max*fm - a_min*fp) + a_max*a_min*(up-um))/(a_max-a_min);

                const double ul = 0.0f;
                const double ur = 0.0f;

                double a_max = 0.5;
                double a_min = -0.5;

                const double fl = grid.faceAt( i, j, Face::negX, um );
                const double fr = grid.faceAt( i, j, Face::negX, up );

                grid.faceAt( i, j, Face::negX, f ) = (fl - fr) / dx;
                        /*centralUpwind(a_max, a_min,
                        fl, fr, ul, ur);*/
            }
        }


        //Evaluate flux for each *internal* face in y
        for( int j = 1; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                //Only bogus fluxes here

                const double ul = 0.0f;
                const double ur = 0.0f;

                double a_max = 0.5;
                double a_min = -0.5;

                const double fl = grid.faceAt( i, j, Face::negY, um );
                const double fr = grid.faceAt( i, j, Face::negY, up );

                grid.faceAt( i, j, Face::negY, f ) = (fl - fr) / dy;
                        /*centralUpwind(a_max, a_min,
                        fl, fr, ul, ur);*/
            }
        }

        //Sum face fluxes for all cells
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                grid.cellAt( i, j, u ) = grid.cellAt( i, j, u0 ) +
                                         a * 1.0/4.0 * ( grid.faceAt(i, j, Face::negX, f) -
                                                 grid.faceAt(i, j, Face::posX, f) +
                                                 grid.faceAt(i, j, Face::negY, f) -
                                                 grid.faceAt(i, j, Face::posY, f));
            }
        }
        t = t + dt;

        /*
        std::stringstream filename;
        std::showpoint(filename);
        filename << "waveheights_" << boost::format("%011.5f") % t << ".csv";

        std::ofstream file(filename.str());
        grid.dumpGridCells( u, file );
        file << std::endl;
        grid.dumpGridFaces( f, Face::negX, file );
        file << std::endl;
        grid.dumpGridFaces( f, Face::negY, file );
        */

        u0 = u;
    }
}
