#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include <opm/common/ErrorMacros.hpp>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>

#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "device_functions.cuh"

using namespace equelleCUDA;

// --------------------------------------------
//                     EXTEND
// --------------------------------------------

CollOfScalar wrapDeviceGrid::extendToFull( const CollOfScalar& in_data,
					   const thrust::device_vector<int>& from_set,
					   const int full_size) {
    // Create a vector of size number_of_faces_:
    CudaArray val(full_size);
    // Extend values
    thrust::fill(thrust::device,val.data(),val.data()+full_size, 0.0);
    thrust::scatter(thrust::device, in_data.data(), in_data.data()+in_data.size(), from_set.begin(), val.data());
    if (in_data.useAutoDiff() ) {
       // Set up output matrix der
       CudaMatrix tempMat(in_data.derivative()); // Move the rvalue from derivative() into a temp object
       CudaMatrix der(full_size, tempMat.cols(), tempMat.nnz());
       
       // Copy csrColInd, csrVal and fill csrRowPtr with zeroes
       thrust::copy(thrust::device, tempMat.csrColInd(), tempMat.csrColInd()+tempMat.nnz(), der.csrColInd());
       thrust::copy(thrust::device, tempMat.csrVal(), tempMat.csrVal()+tempMat.nnz(), der.csrVal());
       thrust::fill(thrust::device,der.csrRowPtr(),der.csrRowPtr()+der.rows()+1, 0.0);
       cudaDeviceSynchronize();
       
       // Map values in set being extended to the new domain
       thrust::scatter(thrust::device, tempMat.csrRowPtr()+1, tempMat.csrRowPtr()+tempMat.rows()+1, from_set.begin(), der.csrRowPtr()+1);
       cudaDeviceSynchronize();
       
       // Fill in the gaps of the rowPtr
       // {0, 0, 2, 0, 0, 4, 0, 5} becomes 
       // {0, 0, 2, 2, 2, 4, 4, 5}
       thrust::maximum<int> binary_op;
       thrust::inclusive_scan(thrust::device, der.csrRowPtr(), der.csrRowPtr()+der.rows()+1, der.csrRowPtr(), binary_op);
       cudaDeviceSynchronize();
       return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}


// NOTE: Do not use this kernel. It is extremely slow and is kept for documentation purposes.
// It's slow mainly because of the inefficient global memory access, which is not coalesced (gathered into single calls).
// Build the csr pointer by filling in values for the empty areas in the new extended domain.
/*__global__ void wrapDeviceGrid::extendToFullKernel_buildRowPtr( int* outData,
                              const int* from_set,
                              const int from_size,
                              const int* inData,
                              const int full_size)
{

    const int outIndex = myID();
    if ( outIndex < from_size-1 ) {
        int diff = from_set[outIndex+1] - from_set[outIndex];

        // Fill in gaps
        #pragma unroll
        for( int i = 1; i < diff+1; i++ ){
            outData[from_set[outIndex]+i] = inData[outIndex+1];
        }

        // Fill in last part
        if ( outIndex == from_size-2 ){
            int lastDiff = (full_size-from_set[from_size-1]);
            #pragma unroll
            for( int i = 1; i <= lastDiff; i++ ){
                outData[from_set[from_size-1]+i] = inData[from_size];
            }
        }
    }
}*/


// ------------------------------------------------
//                ON for CollOfScalar
// ------------------------------------------------

CollOfScalar wrapDeviceGrid::onFromFull( const CollOfScalar& inData,
					 const thrust::device_vector<int>& to_set ) {

    // inData is a full set, so position is its index
    // to_set is indices which we get the input from.
    // out will be same size as to_set.

    // setup how many threads/blocks we need:
    kernelSetup s(to_set.size());

    // Create the output vector:
    CudaArray val(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    wrapDeviceGrid::onFromFullKernel<<<s.grid, s.block>>>(val.data(),
							  to_set_ptr,
							  to_set.size(),
							  inData.data());
    if ( inData.useAutoDiff() ) {
	CudaMatrix onMatrix(to_set, inData.size());
	return CollOfScalar(std::move(val), std::move(onMatrix * inData.derivative()));
    }
    else { // no AutoDiff
	return CollOfScalar(std::move(val));
    }    
}

CollOfScalar wrapDeviceGrid::extendToSubset( const CollOfScalar& inData,
                         const thrust::device_vector<int>& from_set,
                         const thrust::device_vector<int>& to_set,
                         const int full_size) {
    CollOfScalar temp_full = extendToFull( inData, from_set, full_size);
    return onFromFull(temp_full, to_set);

}

CollOfScalar wrapDeviceGrid::onFromSubset( const CollOfScalar& inData,
					   const thrust::device_vector<int>& from_set,
					   const thrust::device_vector<int>& to_set,
					   const int full_size) {
    
    CollOfScalar temp_full = extendToFull(inData, from_set, full_size);
    return onFromFull(temp_full, to_set);
}



__global__ void wrapDeviceGrid::onFromFullKernel( double* outData,
						  const int* to_set,
						  const int to_size,
						  const double* inData)
{
    const int toIndex = myID();
    if ( toIndex < to_size ) {
	outData[toIndex] = inData[to_set[toIndex]];
    }
}
						  

// -----------------------------------------------
//              ON for CollOfIndices
// -----------------------------------------------

thrust::device_vector<int> wrapDeviceGrid::onFromFullIndices( const thrust::device_vector<int>& inData,
							      const thrust::device_vector<int>& to_set ) {

    // inData is a full set, so position is its index
    // to_set is indices which we get the input from.
    // out will be same size as to_set.

    // setup how many threads/blocks we need:
    kernelSetup s(to_set.size());

    // Create the output vector:
    thrust::device_vector<int> out(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    const int* inData_ptr = thrust::raw_pointer_cast( &inData[0] );
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    wrapDeviceGrid::onFromFullKernelIndices<<<s.grid, s.block>>>(out_ptr,
								 to_set_ptr,
								 to_set.size(),
								 inData_ptr);
    return out;
}



thrust::device_vector<int> wrapDeviceGrid::onFromSubsetIndices( const thrust::device_vector<int>& inData,
								const thrust::device_vector<int>& from_set,
								const thrust::device_vector<int>& to_set,
								const int full_size) {
    
    thrust::device_vector<int> temp_full = extendToFullIndices(inData, from_set, full_size);
    return onFromFullIndices(temp_full, to_set);
}




__global__ void wrapDeviceGrid::onFromFullKernelIndices( int* outData,
							 const int* to_set,
							 const int to_size,
							 const int* inData)
{
    const int toIndex = myID();
    if ( toIndex < to_size ) {
	outData[toIndex] = inData[to_set[toIndex]];
    }
}


thrust::device_vector<int> wrapDeviceGrid::extendToFullIndices( const thrust::device_vector<int>& in_data,
								const thrust::device_vector<int>& from_set,
								const int full_size) {
    // setup how many threads/blocks we need:
    kernelSetup s(full_size);

    // create a vector of size number_of_faces_:
    thrust::device_vector<int> out(full_size);
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const int* in_data_ptr = thrust::raw_pointer_cast( &in_data[0] );
    const int* from_ptr = thrust::raw_pointer_cast( &from_set[0]);
    wrapDeviceGrid::extendToFullKernelIndices_step1<<<s.grid, s.block>>>( out_ptr,
									  full_size);
    wrapDeviceGrid::extendToFullKernelIndices_step2<<<s.grid, s.block>>>( out_ptr,
									  from_ptr,
									  from_set.size(),
									  in_data_ptr);
    
      
    return out;
}



// EXTEND TO FULL FOR INDICES DONE IN 2 STEPS

__global__ void wrapDeviceGrid::extendToFullKernelIndices_step1( int* outData,
								 const int full_size)
{
    const int outIndex = myID();
    if ( outIndex < full_size) {
	outData[outIndex] = 0;
    }
}

__global__ void wrapDeviceGrid::extendToFullKernelIndices_step2( int* outData,
								 const int* from_set,
								 const int from_size,
								 const int* inData)
{
    const int outIndex = myID();
    if ( outIndex < from_size ) {
	outData[from_set[outIndex]] = inData[outIndex];
    }
}

