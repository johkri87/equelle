#ifndef EQUELLE_CUSPARSE_MANAGER_HEADER_INCLUDED
#define EQUELLE_CUSPARSE_MANAGER_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "CudaMatrix.hpp"

namespace equelleCUDA
{
class CudaMatrix;
class CusparseManager
{
public:
    static CudaMatrix matrixMultiply(const CudaMatrix& A, const CudaMatrix& B);
    static CudaMatrix matrixMultiply2(const CudaMatrix& A, const CudaMatrix& B);
    static CudaMatrix matrixAddition(const CudaMatrix& lhs, const CudaMatrix& rhs);
    static CudaMatrix matrixSubtraction(const CudaMatrix& lhs, const CudaMatrix& rhs);
    static CudaMatrix precondILU(const CudaMatrix& A);
    static CudaArray biCGStab_ILU_public(const CudaMatrix& A, const int maxit, const CudaArray& x, const double tol);
private:
    CusparseManager();
    ~CusparseManager();

    CudaMatrix gemm2(const CudaMatrix& A, const CudaMatrix& B, const CudaMatrix& C, const double* alpha, const double* beta);
    CudaMatrix gemm(const CudaMatrix& lhs, const CudaMatrix& rhs);
    CudaMatrix geam(const CudaMatrix& lhs, const CudaMatrix& rhs, const double* alpha, const double* beta);
    CudaMatrix precond_ilu(const CudaMatrix& A);
    CudaArray biCGStab_ILU(const CudaMatrix& A, const int maxit, const CudaArray& x_in, const double tol);
    static CusparseManager& instance();

    // cuSPARSE  and CUDA variables
    cusparseHandle_t cusparseHandle_;
    csrgemm2Info_t gemm2Info_;
    //csrgeam2Info_t geam2Info_;
    cusparseStatus_t sparseStatus_;
    cudaError_t cudaStatus_;
    void* buffer_;
    size_t currentBufferSize_;
};
}

#endif