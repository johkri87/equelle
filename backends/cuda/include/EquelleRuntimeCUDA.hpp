/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HEADER_INCLUDED

#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/grid/GridManager.hpp>

#include <vector>
#include <string>
#include <map>
#include <tuple>

// Including device code
// This should be independent from the rest of the host code
//      and especially from any c++11 code.
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"
#include "CollOfVector.hpp"
#include "DeviceGrid.hpp"
#include "equelleTypedefs.hpp"
#include "SeqOfScalar.hpp"
#include "CudaMatrix.hpp"
#include "DeviceHelperOps.hpp"
#include "LinearSolver.hpp"



namespace equelleCUDA {



    // Array Of {X} Collection Of Scalar:
    /// For 1 CollOfScalar
    template <typename T>
    std::tuple<T> makeArray( const T& t) 
    {
        return std::tuple<T> {t};
    }

    /// For 2 CollOfScalar
    template <typename T>
    std::tuple<T, T> makeArray(const T& t1,
                               const T& t2) 
    {
        return std::tuple<T,T> {t1, t2};
    }

    /// For 3 CollOfScalar
    template <typename T>
    std::tuple<T, T, T> makeArray( const T& t1,
                                   const T& t2,
                                   const T& t3 ) 
    {
        return std::tuple<T, T, T> {t1, t2, t3};
    }

    /// For 4 CollOfScalar
    template <typename T>
    std::tuple<T,T,T,T> makeArray( const T& t1,
                                   const T& t2,
                                   const T& t3,
                                   const T& t4 ) 
    {
        return std::tuple<T,T,T,T> {t1, t2, t3, t4};
    }
    




// The Equelle runtime class.
// Contains methods corresponding to Equelle built-ins to make
// it easy to generate C++ code for an Equelle program.
//! Equelle runtime class for the CUDA back-end
/*!
  This class defines the main user interface towards the generated code.
  
  Most functions are designed to match the equivalen function as defined by 
  the Equelle language. 
  
*/
class EquelleRuntimeCUDA
{
public:
    /// Constructor.
    EquelleRuntimeCUDA(const Opm::ParameterGroup& param);

    /// Destructor:
    ~EquelleRuntimeCUDA();

    // ----------- Topology and geometry related ------------
    /// All cells in the grid
    CollOfCell allCells() const;
    /// All boundary cells in the grid
    CollOfCell boundaryCells() const;
    /// All interior cells in the grid
    CollOfCell interiorCells() const;
    /// All faces in the grid
    CollOfFace allFaces() const;
    /// All boundary faces in the grid
    CollOfFace boundaryFaces() const;
    /// All interior faces in the grid
    CollOfFace interiorFaces() const;
    /// First cells to a set of faces
    /*
      The orientation of a face is given by its normal vector,
      and the normal vector then points from the first to the second cell.
    */
    CollOfCell firstCell(CollOfFace faces) const;
    /// Second cells to a set of faces  
    /*
      The orientation of a face is given by its normal vector,
      and the normal vector then points from the first to the second cell.
    */
    CollOfCell secondCell(CollOfFace faces) const;
    /// Natural size (length/area/volume) of grid sets
    template <int codim>
    CollOfScalar norm(const CollOfIndices<codim>& set) const;
    /// 2-norm of a set of vectors
    CollOfScalar norm(const CollOfVector& vectors) const;
    CollOfScalar norm(const CollOfScalar& scalars) const;
    /// Centroid positions of a set of cells/faces
    template <int codim>
    CollOfVector centroid(const CollOfIndices<codim>& set) const;
    /// Normal vectors of faces
    CollOfVector normal(const CollOfFace& faces) const;


    // ---------- Operators and math functions.---------------
    /// Dot product between two vectors
    CollOfScalar dot(const CollOfVector& v1, const CollOfVector& v2) const;
    /// Not generated by front-end -> not implemented
    CollOfScalar negGradient(const CollOfScalar& cell_scalarfield) const;
    /// Not generated by front-end -> not implemented
    CollOfScalar interiorDivergence(const CollOfScalar& face_fluxes) const;
    

    // ----------- Operators and math functions ----------------
    /// Discrete gradient across internal faces
    /*!
      For each face, computes the different between a value on first cell and 
      second cell.
    */
    CollOfScalar gradient(const CollOfScalar& cell_scalarfield) const;
    /// Gradient using matrix-vector product
    /*!
      Slow evaluation of the gradient function where the matrix from
      DeviceHelperOps is used. Use rather the gradient function
      \sa gradient
     */
    CollOfScalar gradient_matrix(const CollOfScalar& cell_scalarfield) const;
    /// Discrete divergence in every cell
    /*!
      Computes the directional sum of values on all faces for every cell.
    */
    CollOfScalar divergence(const CollOfScalar& fluxes) const;  

    template <typename T, typename U, typename V>
    V multiplyAdd(const T& a, const U& b, const V& c);

    /// Divergence using matrix-vector product
    /*!
      Slow evaluation of the divergence function where the matrix fro
      DeviceHelperOps is used. Use rather the divergence function
      \sa divergence
    */
    CollOfScalar divergence_matrix(const CollOfScalar& fluxes) const;
    /// Check for empty grid members in the input
    /*!
      Example: FirstCell(BoundaryFaces()) gives us the cell inside the domain 
      or an empty cell that would be outside of the domain. isEmpty returns true for
      these empty cells.
     */
    template<int codim>
    CollOfBool isEmpty(const CollOfIndices<codim>& set) const;
    

    // ---------------- EXTEND and ON operators ------------------------
    /// Extend a subset to a set by inserting zeros
    template<int codim>
    CollOfScalar operatorExtend(const CollOfScalar& data_in,
                                const CollOfIndices<codim>& from_set,
                                const CollOfIndices<codim>& to_set) const;
    
    /// Extend a scalar to a uniform collection on the given set.
    template<int codim>
    CollOfScalar operatorExtend(const Scalar& data, const CollOfIndices<codim>& set);
    
    /// On operator 
    /*!
      returns elements in data_in corresponding to to_set when data_in is defined 
      on from_set.
    */
    template<int codim>
    CollOfScalar operatorOn(const CollOfScalar& data_in,
                            const CollOfIndices<codim>& from_set,
                            const CollOfIndices<codim>& to_set);
    
    // Implementation of the Equelle keyword On for CollOfIndices<>
    /// On operator
    /*!
      returns elements in in_data corresponding to to_set when data_in is defined
      on from_set.
    */
    template<int codim_data, int codim_set>
    CollOfIndices<codim_data> operatorOn( const CollOfIndices<codim_data>& in_data,
                                          const CollOfIndices<codim_set>& from_set,
                                          const CollOfIndices<codim_set>& to_set);
    
    /// Element-wise trinary if operator 
    /*!
      return_value[i] = predicate[i] ? iftrue[i] : iffalse[i]
    */
    CollOfScalar trinaryIf( const CollOfBool& predicate,
                            const CollOfScalar& iftrue,
                            const CollOfScalar& iffalse) const;
    //! Element-wise trinary if operator
    /*!
      return_value[i] = predicate[i] ? iftrue[i] : iffalse[i]
    */
    template <int codim>
    CollOfIndices<codim> trinaryIf( const CollOfBool& predicate,
                                    const CollOfIndices<codim>& iftrue,
                                    const CollOfIndices<codim>& iffalse) const;
    
    /// Reductions.
    /// Smallest value in a CollOfScalar
    Scalar minReduce(const CollOfScalar& x) const;
    /// Largest value in a CollOfScalar
    Scalar maxReduce(const CollOfScalar& x) const;
    /// Sum of the elements in a CollOfScalar
    Scalar sumReduce(const CollOfScalar& x) const;
    /// Product of all elements in a CollOfScalar
    Scalar prodReduce(const CollOfScalar& x) const;
    
    // Special functions:
    /// Element-wise square root.
    CollOfScalar sqrt(const CollOfScalar& x) const;
    
    /// For implicit methods
    /*!
      Let rescomp be the a residual function taking the primary variable as input.
      The function uses a Newton method to find (and return) the CollOfScalar u such that
      resomp(u) = 0, where u_initialguess is the initial guess. 
     */
    template <class ResidualFunctor> 
    CollOfScalar newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalar& u_initialguess);
    
    //    template <int Num>
    //    std::array<CollOfScalarCPU, Num> newtonSolveSystem(const std::array<typename ResCompType<Num>::type, Num>& rescomp,
    //                                                    const std::array<CollOfScalarCPU, Num>& u_initialguess);
    
    // -------------------------- Output ---------------------.
    /// Write a scalar value to standart output.
    void output(const String& tag, Scalar val) const;
    /// Write a Collection Of Scalar to standard output or file
    /*!
      Where the variable is written depends on the parameter output_to_file={true, false}
      from the parameter file. Default is false.
    */
    void output(const String& tag, const CollOfScalar& coll);
    
    // ---------------------- Input -------------------------
    /// Get value name from parameter file or use default value
    Scalar inputScalarWithDefault(const String& name,
                                  const Scalar default_value);
    
    /// Reads a index list from file
    /*!
      Requires the indices to be sorted.
    */
    template <int codim>
    CollOfIndices<codim> inputDomainSubsetOf( const String& name,
                                              CollOfIndices<codim> superset);
    /// Reads a Collection Of Scalar from file named name
    /*!
      The scalars need to be sorted with respect to the index of the element
      each of them is defined on.
    */
    template <int codim>
    CollOfScalar inputCollectionOfScalar(const String& name,
                                         const CollOfIndices<codim>& coll);
    /// Reads a list of scalars to be stored on the CPU's memory.
    SeqOfScalar inputSequenceOfScalar(const String& name);
    
    
    /// Ensuring requirements that may be imposed by Equelle programs.
    void ensureGridDimensionMin(const int minimum_grid_dimension) const;

    
    // ------- FUNCTIONS ONLY FOR TESTING -----------------------

    /// For testing
    UnstructuredGrid getGrid() const;
    /// For testing
    CudaMatrix getGradMatrix() const { return devOps_.grad();};
    /// For testing
    CudaMatrix getDivMatrix() const { return devOps_.div();};
    /// For testing
    CudaMatrix getFulldivMatrix() const {return devOps_.fulldiv();};
    /// For testing
    Scalar twoNormTester(const CollOfScalar& val) const { return twoNorm(val); };

    // ------------ PRIVATE MEMBERS -------------------------- //
private:
      
    /// Norms.
    Scalar twoNorm(const CollOfScalar& vals) const;
    
    /// Data members.
    std::unique_ptr<Opm::GridManager> grid_manager_;
    const UnstructuredGrid& grid_;
    equelleCUDA::DeviceGrid dev_grid_;
    mutable DeviceHelperOps devOps_;
    LinearSolver solver_;
    Opm::LinearSolverFactory serialSolver_;
    bool output_to_file_;
    int verbose_;
    const Opm::ParameterGroup& param_;
    std::map<std::string, int> outputcount_;
    // For newtonSolve().
    int max_iter_;
    double abs_res_tol_;

    CollOfScalar serialSolveForUpdate(const CollOfScalar& residual) const;


};

} // namespace equelleCUDA


// Include the implementations of template members.
#include "EquelleRuntimeCUDA_impl.hpp"

#endif // EQUELLERUNTIMECUDA_HEADER_INCLUDED
