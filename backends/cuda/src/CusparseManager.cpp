#include "CusparseManager.hpp"
#include <time.h>

using namespace equelleCUDA;

CusparseManager::CusparseManager()
    : buffer_(NULL),
      currentBufferSize_(0)
{
    std::cout << "CusparseManager constructed." << std::endl;
    // Set up cuSPARSE
    cusparseCreate(&cusparseHandle_);
    cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
    cusparseCreateCsrgemm2Info(&gemm2Info_);
}

CusparseManager::~CusparseManager()
{
    std::cout << "CusparseManager destroyed." << std::endl;

    if (buffer_) {
        cudaFree(buffer_);
    }
    cusparseDestroy(cusparseHandle_);
    cusparseDestroyCsrgemm2Info(gemm2Info_);
}

/// Using the Meyers singleton pattern.
CusparseManager& CusparseManager::instance()
{
    static CusparseManager s;
    return s;
}

// gemm2 is slower then gemm for simple Matrix-Matrix multiplication.
// However, we keep this for testing and profiling purposes.
CudaMatrix CusparseManager::matrixMultiply2(const CudaMatrix& A, const CudaMatrix& B)
{
    double alpha = 1.0;
    return instance().gemm2(A, B, CudaMatrix(), &alpha, NULL);
}

CudaMatrix CusparseManager::matrixMultiply(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    return instance().gemm(lhs, rhs);
}

// gemm2, as opposed to gemm, does not call cudaFree implicitly.
CudaMatrix CusparseManager::gemm2(const CudaMatrix& A, const CudaMatrix& B, const CudaMatrix& C, const double* alpha, const double* beta)
{
    CudaMatrix out;
    int innerSize = out.confirmMultSize(A, B);

    // Allocate buffer
    size_t newBufferSize;
    out.sparseStatus_ = cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha,
                                     A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                                     B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                                     beta,
                                     C.description_, C.nnz_, C.csrRowPtr_, C.csrColInd_,
                                     gemm2Info_, &newBufferSize);
    out.checkError_("cusparseDcsrgemm2_bufferSizeExt() in CusparseManager::gemm2()");
    if (newBufferSize > currentBufferSize_) {
        if (buffer_ != NULL) {
            out.cudaStatus_ = cudaFree(buffer_);
            out.checkError_("cusparseDcsrgemm2() in CusparseManager::gemm2()");
        }
        out.cudaStatus_ = cudaMalloc(&buffer_, newBufferSize);
        out.checkError_("cudaMalloc(&buffer_, newBufferSize) in CusparseManager::gemm2()");
        currentBufferSize_ = newBufferSize;
    }

    // Allocate row pointer
    out.cudaStatus_ = cudaMalloc((void**)&out.csrRowPtr_, sizeof(int)*(out.rows_+1));
    out.checkError_("cudaMalloc((void**)&out.csrRowPtr_, sizeof(int)*(out.rows_+1)) in CusparseManager::gemm2()");

    // Compute NNZ
    int* nnzTotalDevHostPtr = &out.nnz_;
    out.sparseStatus_ = cusparseXcsrgemm2Nnz(cusparseHandle_,
                         out.rows_, out.cols_, innerSize,
                         A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                         B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                         C.description_, C.nnz_, C.csrRowPtr_, C.csrColInd_,
                         out.description_, out.csrRowPtr_,
                         nnzTotalDevHostPtr, gemm2Info_, buffer_);
    out.checkError_("cusparseXcsrgemm2Nnz() in CusparseManager::gemm2()");
    if (NULL != nnzTotalDevHostPtr) {
        out.nnz_ = *nnzTotalDevHostPtr;
    } else
    {
        int baseC;
        out.cudaStatus_ = cudaMemcpy(&out.nnz_, out.csrRowPtr_+out.rows_, sizeof(int), cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(&out.nnz_, out.csrRowPtr_+out.rows_, sizeof(int), cudaMemcpyDeviceToHost) in CusparseManager::gemm2()");
        out.cudaStatus_ = cudaMemcpy(&baseC, out.csrRowPtr_, sizeof(int), cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(&baseC, out.csrRowPtr_, sizeof(int), cudaMemcpyDeviceToHost) in CusparseManager::gemm2()");
        out.nnz_ -= baseC;
    }

    // Allocate memory for output matrix
    out.cudaStatus_ = cudaMalloc((void**)&out.csrColInd_, sizeof(int)*out.nnz_);
    out.checkError_("cudaMalloc((void**)&out.csrColInd_, sizeof(int)*out.nnz_) in CusparseManager::gemm2()");
    out.cudaStatus_ = cudaMalloc((void**)&out.csrVal_, sizeof(double)*out.nnz_);
    out.checkError_("cudaMalloc((void**)&out.csrVal_, sizeof(double)*out.nnz_) in CusparseManager::gemm2()");
    
    // Perform the gemm2 operation
    // D = alpha ∗ A ∗ B + beta ∗ C
    out.sparseStatus_ = cusparseDcsrgemm2(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha, 
                      A.description_, A.nnz_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, 
                      B.description_, B.nnz_, B.csrVal_, B.csrRowPtr_, B.csrColInd_,
                      beta,
                      C.description_, C.nnz_, C.csrVal_, C.csrRowPtr_, C.csrColInd_,
                      out.description_, out.csrVal_, out.csrRowPtr_, out.csrColInd_,
                      gemm2Info_, buffer_);
    out.checkError_("cusparseDcsrgemm2() in CusparseManager::gemm2()");
    return out;
}



CudaMatrix CusparseManager::gemm(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
// Create an empty matrix. Need to set rows, cols, nnz, and allocate arrays!
    CudaMatrix out;
    // Legal matrix sizes depend on whether the matrices are transposed or not!
    int innerSize = out.confirmMultSize(lhs, rhs);

    // Addition in two steps
    //    1) Find nonzero pattern of output
    //    2) Multiply matrices.

    // 1) Find nonzero pattern of output
    // Allocate rowPtr:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in CusparseManager::gemm()");

    // The following code for finding number of non-zeros is
    // taken from the Nvidia cusparse documentation, section 9.2
    // Only additions are the error checking.
    int *nnzTotalDevHostPtr = &out.nnz_;
    out.checkError_("cusparseSetPointerMode() in CusparseManager::gemm()");
    out.sparseStatus_ = cusparseXcsrgemmNnz( cusparseHandle_, 
                         lhs.operation_, rhs.operation_,
                         out.rows_, out.cols_, innerSize,
                         lhs.description_, lhs.nnz_,
                         lhs.csrRowPtr_, lhs.csrColInd_,
                         rhs.description_, rhs.nnz_,
                         rhs.csrRowPtr_, rhs.csrColInd_,
                         out.description_,
                         out.csrRowPtr_, nnzTotalDevHostPtr);
    out.checkError_("cusparseXcsrgemmNnz() in CusparseManager::gemm()");
    if ( nnzTotalDevHostPtr != NULL ) {
        out.nnz_ = *nnzTotalDevHostPtr;
    } else {
        int baseC;
        out.cudaStatus_ = cudaMemcpy(&out.nnz_, out.csrRowPtr_ + out.rows_,
                         sizeof(int), cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(out.csrRowPtr_ + out.rows_) in CusparseManager::gemm()");
        out.cudaStatus_ = cudaMemcpy(&baseC, out.csrRowPtr_, sizeof(int),
                         cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(baseC) in CusparseManager::gemm()");
        out.nnz_ -= baseC;
    }

     // Allocate the other two arrays:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrVal_, out.nnz_*sizeof(double));
    out.checkError_("cudaMalloc(out.csrVal_) in CusparseManager::gemm()");
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrColInd_, out.nnz_*sizeof(int));
    out.checkError_("cudaMalloc(out.csrColInd_) in CusparseManager::gemm()");
    
    // 2) Multiply the matrices:
    out.sparseStatus_ = cusparseDcsrgemm(cusparseHandle_,
                     lhs.operation_, rhs.operation_,
                     out.rows_, out.cols_, innerSize,
                     lhs.description_, lhs.nnz_,
                     lhs.csrVal_, lhs.csrRowPtr_, lhs.csrColInd_,
                     rhs.description_, rhs.nnz_,
                     rhs.csrVal_, rhs.csrRowPtr_, rhs.csrColInd_,
                     out.description_,
                     out.csrVal_, out.csrRowPtr_, out.csrColInd_);
    out.checkError_("cusparseDcsrgemm() in CusparseManager::gemm()");
    
    return out;
}