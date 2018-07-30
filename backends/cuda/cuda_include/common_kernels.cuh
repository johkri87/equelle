#ifndef EQUELLE_KERNELS_HEADER_INCLUDED
#define EQUELLE_KERNELS_HEADER_INCLUDED
namespace equelleCUDA {
namespace equelleKernels {
__global__ void multiplication_kernel(double* out, const double* rhs, const int size);
__global__ void multiplication_kernel(double* out, const double rhs, const int size);
__global__ void addition_kernel(double* out, const double* rhs, const int size);
__global__ void division_kernel(double* out, const double* rhs, const int size);
__global__ void division_kernel(double* out, const double rhs, const int size);
__global__ void division_kernel(const double lhs, double* out, const int size);
__global__ void negate_kernel(double* out, const int size);
__global__ void square_kernel(double* out, const int size);
__global__ void sqrt_kernel(double* out, const int size);
__global__ void reciprocal_kernel(double* out, const int size);
__global__ void inverse_squared_kernel(double* out, const int size);
__global__ void abs_kernel(double* out, const int size);
}
}
#endif // EQUELLE_KERNELS_HEADER_INCLUDED