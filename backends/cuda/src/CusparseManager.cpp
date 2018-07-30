#include "CusparseManager.hpp"

using namespace equelleCUDA;

CusparseManager::CusparseManager()
    : buffer_(NULL)
{
    std::cout << "CusparseManager constructed." << std::endl;
    // Set up cuSPARSE
    sparseStatus_ = cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
    cusparseCreateCsrgemm2Info(&gemm2Info_);
}

CusparseManager::~CusparseManager()
{
    std::cout << "CusparseManager destroyed." << std::endl;

    if (buffer_) {
        cudaFree(buffer_);
    }

    cusparseDestroyCsrgemm2Info(gemm2Info_);
}

/// Using the Meyers singleton pattern.
CusparseManager& CusparseManager::instance()
{
    static CusparseManager s;
    return s;
}

CudaMatrix CusparseManager::matrixMultiply(const CudaMatrix& A, const CudaMatrix& B)
{
    double alpha = 1.0;
    return instance().gemm2(A, B, CudaMatrix(), &alpha, NULL);
}

CudaMatrix CusparseManager::gemm2(const CudaMatrix& A, const CudaMatrix& B, const CudaMatrix& C, double* alpha, double* beta)
{
    CudaMatrix out;
    /*
    CudaMatrix out;
    int innerSize = out.confirmMultSize(A, B);

    // Allocate buffer
    size_t bufferSize;
    cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha,
                                     A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                                     B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                                     beta,
                                     B.description_, 0, NULL, NULL,
                                     gemm2Info, &bufferSize);

    

    out.cudaStatus_ = cudaMalloc(&buffer_, bufferSize);
    out.cudaStatus_ = cudaMalloc((void**)&out.csrRowPtr_, sizeof(int)*(out.rows_+1));
    std::cout << "Buffer size: " << bufferSize << std::endl;

    // Compute NNZ
    int* nnzTotalDevHostPtr = &out.nnz_;
    out.sparseStatus_ = cusparseXcsrgemm2Nnz(cusparseHandle_,
                         out.rows_, out.cols_, innerSize,
                         A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                         B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                         C.description_, C.nnz_, C.csrRowPtr_, C.csrColInd_,
                         out.description_, out.csrRowPtr_,
                         nnzTotalDevHostPtr, gemm2Info_, buffer_);
    if (NULL != nnzTotalDevHostPtr) {
        out.nnz_ = *nnzTotalDevHostPtr;
    }else
    {
        int baseC;
        cudaMemcpy(&out.nnz_, out.csrRowPtr_+out.rows_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, out.csrRowPtr_, sizeof(int), cudaMemcpyDeviceToHost);
        out.nnz_ -= baseC;
    }

    // Allocate memory for output matrix
    cudaMalloc((void**)&out.csrColInd_, sizeof(int)*out.nnz_);
    cudaMalloc((void**)&out.csrVal_, sizeof(double)*out.nnz_);
    
    // Perform the gemm2 operation
    // D = alpha ∗ A ∗ B + beta ∗ C
    out.sparseStatus_ = cusparseDcsrgemm2(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha, 
                      A.description_, A.nnz_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, 
                      B.description_, B.nnz_, B.csrVal_, B.csrRowPtr_, B.csrColInd_,
                      beta,
                      C.description_, C.nnz_, C.csrVal_, C.csrRowPtr_,
                      out.description_, out.csrVal_, out.csrRowPtr_, out.csrColInd_,
                      gemm2Info_, buffer_);
                      */
    return out;
}