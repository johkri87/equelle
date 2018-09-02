#include "CusparseManager.hpp"
#include <time.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cublas_v2.h>

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


CudaMatrix CusparseManager::matrixMultiply(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    return instance().gemm(lhs, rhs);
}

// gemm2 is slower then gemm for simple Matrix-Matrix multiplication.
// However, we keep this for testing and profiling purposes.
CudaMatrix CusparseManager::matrixMultiply2(const CudaMatrix& A, const CudaMatrix& B)
{
    double alpha = 1.0;
    return instance().gemm2(A, B, CudaMatrix(), &alpha, NULL);
}

CudaMatrix CusparseManager::matrixAddition(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    double alpha = 1.0;
    double beta = 1.0;
    return instance().geam(lhs, rhs, &alpha, &beta);
}

CudaMatrix CusparseManager::matrixSubtraction(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    double alpha = 1.0;
    double beta = -1.0;
    return instance().geam(lhs, rhs, &alpha, &beta);
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


CudaMatrix CusparseManager::geam(const CudaMatrix& lhs, const CudaMatrix& rhs, const double* alpha, const double* beta)
{
    // Create an empty matrix. Need to set rows, cols, nnz, and allocate arrays!
    CudaMatrix out;
    out.rows_ = lhs.rows_;
    out.cols_ = lhs.cols_;

    // Addition in two steps
    //    1) Find nonzero pattern of output
    //    2) Add matrices.

    // 1) Find nonzero pattern:
    // Allocate rowPtr:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in CusparseManager::geam()");

    int *nnzTotalDevHostPtr = &out.nnz_;
    out.sparseStatus_ = cusparseXcsrgeamNnz( cusparseHandle_, out.rows_, out.cols_,
                         lhs.description_, lhs.nnz_,
                         lhs.csrRowPtr_, lhs.csrColInd_,
                         rhs.description_, rhs.nnz_,
                         rhs.csrRowPtr_, rhs.csrColInd_,
                         out.description_, out.csrRowPtr_,
                         nnzTotalDevHostPtr);
    out.checkError_("cusparseXcsrgeamNnz() in CusparseManager::geam()");
    if ( nnzTotalDevHostPtr != NULL) {
        out.nnz_ = *nnzTotalDevHostPtr;
    } else {
        out.cudaStatus_ = cudaMemcpy( &out.nnz_, out.csrRowPtr_ + out.rows_,
                                      sizeof(int), cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(out.csrRowPtr_ + rows_) in CusparseManager::geam()");
        int baseC;
        out.cudaStatus_ = cudaMemcpy( &baseC, out.csrRowPtr_, sizeof(int),
                                      cudaMemcpyDeviceToHost);
        out.checkError_("cudaMemcpy(&baseC) in CusparseManager::geam()");
        out.nnz_ -= baseC;
    }

    // Allocate the other two arrays:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrVal_, out.nnz_*sizeof(double));
    out.checkError_("cudaMalloc(out.csrVal_) in CusparseManager::geam()");
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrColInd_, out.nnz_*sizeof(int));
    out.checkError_("cudaMalloc(out.csrColInd_) in CusparseManager::geam()");
    
    // 2) Add matrices
    // Need to create alpha and beta:
    out.sparseStatus_ = cusparseDcsrgeam(cusparseHandle_, out.rows_, out.cols_,
                     alpha,
                     lhs.description_, lhs.nnz_,
                     lhs.csrVal_, lhs.csrRowPtr_, lhs.csrColInd_,
                     beta,
                     rhs.description_, rhs.nnz_,
                     rhs.csrVal_, rhs.csrRowPtr_, rhs.csrColInd_,
                     out.description_,
                     out.csrVal_, out.csrRowPtr_, out.csrColInd_);
    out.checkError_("cusparseDcsrgeam() in CusparseManager::geam()");

    return out;
}

CudaMatrix CusparseManager::precond_ilu(const CudaMatrix& A)
{
    CudaMatrix out = A;
    cusparseSolveAnalysisInfo_t analysisInfo;
    cusparseCreateSolveAnalysisInfo(&analysisInfo);
    out.sparseStatus_ = cusparseDcsrsv_analysis(cusparseHandle_, 
                        out.operation_,
                        out.rows_, 
                        out.nnz_, 
                        out.description_,
                        out.csrVal_, 
                        out.csrRowPtr_,
                        out.csrColInd_, 
                        analysisInfo);
    out.checkError_("cusparseDcsrsv_analysis() in CusparseManager::precond_ilu()");
    cudaDeviceSynchronize();
    out.sparseStatus_ = cusparseDcsrilu0(cusparseHandle_,
                 out.operation_, 
                 out.rows_, 
                 out.description_, 
                 out.csrVal_,
                 out.csrRowPtr_, 
                 out.csrColInd_,  
                 analysisInfo);
    cudaDeviceSynchronize();
    out.checkError_("cusparseDcsrilu0() in CusparseManager::precond_ilu()");
    cusparseDestroySolveAnalysisInfo(analysisInfo);
    return out;
}

CudaArray CusparseManager::biCGStab_ILU_public(const CudaMatrix& A, const int maxit, const CudaArray& x, const double tol)
{
    return instance().biCGStab_ILU(A,maxit,x,tol);
}


CudaArray CusparseManager::biCGStab_ILU(const CudaMatrix& A, const int maxit, const CudaArray& x_in, const double tol)
{
    std::cout << 1 << std::endl;
    CudaMatrix m = A;
    CudaArray x_out = x_in;

    cusparseSolveAnalysisInfo_t analysisInfo_u;
    cusparseSolveAnalysisInfo_t analysisInfo_l;
    cusparseCreateSolveAnalysisInfo(&analysisInfo_u);
    cusparseCreateSolveAnalysisInfo(&analysisInfo_l);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    std::cout << 2 << std::endl;
    double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
    double nrmr, nrmr0;
    rho = 0.0;
    double zero = 0.0;
    double one  = 1.0;
    double mone = -1.0;
    int i = 0;
    int j = 0;
    int n = A.rows_;
    int nnz = A.nnz_;
    std::cout << 3 << std::endl;
    double* r = 0;
    double* t = 0;
    double* s = 0;
    double* rw = 0;
    double* p = 0;
    double* x = x_out.data();
    double* f = 0;
    double* pw = 0;
    double* v = 0;

    std::cout << 4 << std::endl;
    cudaMalloc(&r, n*sizeof(double));
    cudaMalloc(&t, n*sizeof(double));
    cudaMalloc(&s, n*sizeof(double));
    cudaMalloc(&rw, n*sizeof(double));
    cudaMalloc(&p, n*sizeof(double));
    cudaMalloc(&f, n*sizeof(double));
    cudaMalloc(&v, n*sizeof(double));
    cudaMalloc(&pw, n*sizeof(double));


    thrust::fill(thrust::device, r, r+n, 0.0);
    thrust::fill(thrust::device, t, t+n, 0.0);
    thrust::fill(thrust::device, s, s+n, 0.0);
    thrust::fill(thrust::device, rw, rw+n, 0.0);
    thrust::fill(thrust::device, p, p+n, 0.0);
    thrust::fill(thrust::device, f, f+n, 0.0);
    thrust::fill(thrust::device, v, v+n, 0.0);
    thrust::fill(thrust::device, pw, pw+n, 0.0);
    cudaDeviceSynchronize();

    std::cout << 7 << std::endl;
    cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_UNIT);
    cusparseDcsrsv_analysis(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,m.rows_,nnz,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_l);
    cudaDeviceSynchronize();

    std::cout << 8 << std::endl;
    cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseDcsrsv_analysis(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,m.rows_,nnz,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_u);
    cudaDeviceSynchronize();
    std::cout << 9 << std::endl;
    m.sparseStatus_ = cusparseDcsrilu0(cusparseHandle_,
                 m.operation_, 
                 m.rows_, 
                 m.description_, 
                 m.csrVal_,
                 m.csrRowPtr_, 
                 m.csrColInd_,  
                 analysisInfo_l);
    cudaDeviceSynchronize();
    std::cout << A << std::endl;
    std::cout << m << std::endl;
    m.checkError_("cusparseDcsrilu0() in CusparseManager::precond_ilu()");
    std::cout << 10 << std::endl;
    //compute initial residual r0=b-Ax0 (using initial guess in x)

    // Residual r er output. Linjene under er -Ax0
    cusparseDcsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, A.description_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, x, &zero, r);
    cublasDscal(cublasHandle, n, &mone, r, 1);
    cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1);
    std::cout << 11 << std::endl;
    //copy residual r into r^{\hat} and p
    cublasDcopy(cublasHandle, n, r, 1, rw, 1);
    cublasDcopy(cublasHandle, n, r, 1, p, 1); 
    cublasDnrm2(cublasHandle, n, r, 1, &nrmr0);
    std::cout << 12 << std::endl;
    for (i=0; i<maxit; ){
        rhop = rho;
        cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho);

        if (i > 0){
            beta= (rho/rhop) * (alpha/omega);
            negomega = -omega;
            cublasDaxpy(cublasHandle,n, &negomega, v, 1, p, 1);
            cublasDscal(cublasHandle,n, &beta, p, 1);
            cublasDaxpy(cublasHandle,n, &one, r, 1, p, 1);
        }
        //preconditioning step (lower and upper triangular solve)

        cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_UNIT);
        cusparseDcsrsv_solve(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_l,p,t);

        cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_UPPER);
        cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseDcsrsv_solve(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_u,t,pw);


        //matrix-vector multiplication

        cusparseDcsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, A.description_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, pw, &zero, v);

        cublasDdot(cublasHandle,n, rw, 1, v, 1,&temp);
        alpha= rho / temp;
        negalpha = -(alpha);
        cublasDaxpy(cublasHandle,n, &negalpha, v, 1, r, 1);
        cublasDaxpy(cublasHandle,n, &alpha,    pw, 1, x, 1);
        cublasDnrm2(cublasHandle, n, r, 1, &nrmr);

        if (nrmr < tol*nrmr0){
            j=5;
            break;
        }

        //preconditioning step (lower and upper triangular solve)
        cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_UNIT);
        cusparseDcsrsv_solve(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_l,r,t);

        cusparseSetMatFillMode(m.description_,CUSPARSE_FILL_MODE_UPPER);
        cusparseSetMatDiagType(m.description_,CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseDcsrsv_solve(cusparseHandle_,CUSPARSE_OPERATION_NON_TRANSPOSE,n, &one,m.description_,m.csrVal_,m.csrRowPtr_,m.csrColInd_,analysisInfo_u,t,s);

        //matrix-vector multiplication

        cusparseDcsrmv(cusparseHandle_, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, A.description_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, s, &zero, t);

        cublasDdot(cublasHandle,n, t, 1, r, 1,&temp);
        cublasDdot(cublasHandle,n, t, 1, t, 1,&temp2);
        omega = temp / temp2;
        negomega = -(omega);
        cublasDaxpy(cublasHandle,n, &omega, s, 1, x, 1);
        cublasDaxpy(cublasHandle,n, &negomega, t, 1, r, 1);

        cublasDnrm2(cublasHandle,n, r, 1,&nrmr);

        if (nrmr < tol*nrmr0){
            i++;
            j=0;
            break;
        }
        i++;
    }  
    cusparseDestroySolveAnalysisInfo(analysisInfo_u);
    cusparseDestroySolveAnalysisInfo(analysisInfo_l);
    cudaFree(r);
    cudaFree(t);
    cudaFree(s);
    cudaFree(rw);
    cudaFree(p);
    cudaFree(f);
    cudaFree(v);
    cudaFree(pw);
    cublasDestroy(cublasHandle);

    return x_out;
    /*CudaMatrix out = instance.precondILU(A);

    for (i=0; i<maxit; ){
        rhop = rho;
        checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));

        if (i > 0){
            beta= (rho/rhop) * (alpha/omega);
            negomega = -omega;
            checkCudaErrors(cublasDaxpy(cublasHandle,n, &negomega, v, 1, p, 1));
            checkCudaErrors(cublasDscal(cublasHandle,n, &beta, p, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle,n, &one, r, 1, p, 1));
        }

        checkCudaErrors(cublasDdot(cublasHandle,n, rw, 1, v, 1,&temp));
        alpha= rho / temp;
        negalpha = -(alpha);
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &negalpha, v, 1, r, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle,n, &alpha,        pw, 1, x, 1));
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

        if (nrmr < tol*nrmr0){
            j=5;
            break;
        }

        cublasDdot(cublasHandle,n, t, 1, r, 1,&temp);
        cublasDdot(cublasHandle,n, t, 1, t, 1,&temp2);
        omega= temp / temp2;
        negomega = -(omega);
        cublasDaxpy(cublasHandle,n, &omega, s, 1, x, 1);
        cublasDaxpy(cublasHandle,n, &negomega, t, 1, r, 1);

        cublasDnrm2(cublasHandle,n, r, 1,&nrmr);

        if (nrmr < tol*nrmr0){
            i++;
            j=0;
            break;
        }
        i++;
    }*/
}

CudaMatrix CusparseManager::precondILU(const CudaMatrix& A)
{
    return instance().precond_ilu(A);
}