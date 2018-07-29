#include "common_kernels.cuh"
#include "device_functions.cuh"

using namespace equelleCUDA;

__global__ void equelleKernels::multiplication_kernel(double* out, const double* rhs, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] * rhs[index];
    }
}

__global__ void equelleKernels::multiplication_kernel(double* out, const double rhs, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] * rhs;
    }
}

__global__ void equelleKernels::addition_kernel(double* out, const double* rhs, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] + rhs[index];
    }
}

__global__ void equelleKernels::division_kernel(double* out, const double* rhs, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] / rhs[index];
    }
}

__global__ void equelleKernels::division_kernel(const double lhs, double* out, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = lhs / out[index];
    }
}

__global__ void equelleKernels::division_kernel(double* out, const double rhs, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] / rhs;
    }
}

__global__ void equelleKernels::negate_kernel(double* out, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = -out[index];
    }
}

__global__ void equelleKernels::square_kernel(double* out, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = out[index] * out[index];
    }
}

__global__ void equelleKernels::sqrt_kernel(double* out, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = __dsqrt_rn(out[index]);
    }
}

__global__ void equelleKernels::reciprocal_kernel(double* out, const int size)
{
    const int index = myID();
    if ( index < size ) {
        out[index] = __drcp_rn(out[index]);
    }
}

__global__ void equelleKernels::abs_kernel(double* out, const int size)
{
    const int i = myID();
    if ( i < size ) {
        out[i] = fabs(out[i]);
    }
}