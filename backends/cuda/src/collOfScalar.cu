
#include <string>
#include <iostream>
#include <vector>

#include <opm/common/ErrorMacros.hpp>

#include "CollOfScalar.hpp"
#include "CudaArray.hpp"
#include "CudaMatrix.hpp"

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

using namespace equelleCUDA;
using namespace wrapCudaArray;

CollOfScalar::CollOfScalar() 
    : val_(),
      der_(),
      autodiff_(false)
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size)
    : val_(size),
      der_(),
      autodiff_(false)
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size, const double value)
    : val_(size, value),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const CudaArray& val)
    : val_(val),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : val_(host_vec),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}


// Primary variable constructor
CollOfScalar::CollOfScalar(const CollOfScalar& val, const bool primaryVariable)
    : val_(val.val_),
      der_(val.size()),
      autodiff_(true)
{
    // It makes no sence to use this constructor with primaryVariable = false,
    // so we check that it is used correctly:
    if ( !primaryVariable ) {
        OPM_THROW(std::runtime_error, "Trying to create a primary variable with primaryVarible = " << primaryVariable );
    }
}

// Constructor from CudaArray and CudaMatrix
CollOfScalar::CollOfScalar(const CudaArray& val, const CudaMatrix& der)
    : val_(val),
      der_(der),
      autodiff_(true)
{
    // Intentionally left empty
}

// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll)
    : val_(coll.val_),
      der_(coll.der_),
      autodiff_(coll.autodiff_)
{
    // Intentionally left emtpy
}

// Move constructor
CollOfScalar::CollOfScalar(CollOfScalar&& coll)
    : val_(std::move(coll.val_)),
      der_(std::move(coll.der_)),
      autodiff_(coll.autodiff_) 
{
    // Intentionally left emtpy
    //std::cout << "In CollOfScalar move constructor." << std::endl;
}


// Move constructor from CudaArray and CudaMatrix.
// Both val and der are moved.
CollOfScalar::CollOfScalar(CudaArray&& val, CudaMatrix&& der) noexcept
    : val_(std::move(val)),
      der_(std::move(der)),
      autodiff_(true)
{
}

// Move constructor from CudaArray and CudaMatrix
// Only der is moved. val is copied.
CollOfScalar::CollOfScalar(const CudaArray& val, CudaMatrix&& der)
    : val_(val),
      der_(std::move(der)),
      autodiff_(true)
{
}

// Move constructor from CudaArray and CudaMatrix
// Only der is moved. val is copied.
CollOfScalar::CollOfScalar(CudaArray&& val, const CudaMatrix& der)
    : val_(std::move(val)),
      der_(der),
      autodiff_(true)
{
}


// Assignment copy operator
CollOfScalar& CollOfScalar::operator=(const CollOfScalar& other)
{
    // Protect against self assignment:
    if (this != &other) {
        val_ = other.val_;
        autodiff_ = other.autodiff_;
        if ( autodiff_ ) {
            der_ = other.der_;
        }
    }
    return *this;
}

// Assignment move operator
CollOfScalar& CollOfScalar::operator=(CollOfScalar&& other)
{
    //std::cout << "In CollOfScalar move assignment operator" << std::endl;
    val_ = std::move(other.val_);
    autodiff_ = other.autodiff_;
    if ( autodiff_ ) {
        der_ = std::move(other.der_);
    }
    return *this;
}
   
CollOfScalar::~CollOfScalar()
{
    // Intentionally left blank as val_ knows how to delete itself.
}

// Norm
CollOfScalar CollOfScalar::norm() const
{
    CollOfScalar out(this->val_.abs());
    return out;
}

// Member functions that only have to return val_'s function:
const double* CollOfScalar::data() const
{
    return val_.data();
}

double* CollOfScalar::data()
{
    return val_.data();
}

bool CollOfScalar::useAutoDiff() const
{
    return autodiff_;
}

kernelSetup CollOfScalar::setup() const
{
    return val_.setup();
}

std::vector<double> CollOfScalar::copyToHost() const
{
    return val_.copyToHost();
}

hostMat CollOfScalar::matrixToHost() const
{
    if ( !autodiff_ ) {
        OPM_THROW(std::runtime_error, "Trying to copy empty matrix to host\n");
    }
    return der_.toHost();
}

int CollOfScalar::size() const
{
    return val_.size();
}

CudaMatrix CollOfScalar::derivative() const
{
    return der_;
}

CudaArray CollOfScalar::value() const
{
    return val_;
}

// Reduction
double CollOfScalar::reduce(const EquelleReduce reduce) const
{
    // Copy to a device vector?
    thrust::device_vector<double> vec(this->size());
    double* vec_ptr = thrust::raw_pointer_cast( &vec[0] );

    cudaError_t stat = cudaMemcpy( vec_ptr, this->data(), 
                   this->size()*sizeof(double), 
                   cudaMemcpyDeviceToDevice);
    if ( stat != cudaSuccess ) {
        OPM_THROW(std::runtime_error, "Error in cudaMemcpy in CollOfScalar::reduce(const EquelleReduce) with EquelleReduce = " << reduce);
    }
    
    double result = 0;
    if ( reduce == SUM ) {
        result = thrust::reduce(vec.begin(), vec.end(), 
                    (double) 0, thrust::plus<double>());
    }
    else if ( reduce == PRODUCT ) {
        result = thrust::reduce(vec.begin(), vec.end(),
                    (double) 1.0, thrust::multiplies<double>());
    }
    else if ( reduce == MAX ) {
        double init = -1.0*std::numeric_limits<double>::max();
        result = thrust::reduce( vec.begin(), vec.end(), init, thrust::maximum<double>());
    }
    else if ( reduce == MIN ) {
        double init = std::numeric_limits<double>::max();
        result = thrust::reduce( vec.begin(), vec.end(), init, thrust::minimum<double>());
    }
    return result;
} // reduce


// ------- ARITHMETIC OPERATIONS --------------------
CollOfScalar equelleCUDA::multiplyAdd(const CollOfScalar& a, const Scalar b,
                                      const CollOfScalar& c)
{
    CudaArray val = multiplyAdd(a.val_, b, c.val_);
    if (a.autodiff_ || c.autodiff_) {
        CudaMatrix der = multiplyAdd(a.der_, b, c.der_);
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::multiplyAdd(const CollOfScalar& a, const CollOfScalar& b,
                                      const CollOfScalar& c)
{
    CudaArray val = multiplyAdd(a.val_, b.val_, c.val_);
    if (a.autodiff_ || b.autodiff_ || c.autodiff_) {
        CudaMatrix diag_u(a.val_);
        CudaMatrix diag_v(b.val_);
        CudaMatrix der = multiplyAdd(diag_v, a.der_, diag_u*b.der_) + c.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator+ (const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    //CudaArray val = lhs.val_ + rhs.val_;
    CudaArray val = lhs.val_ + rhs.val_;
    if (lhs.autodiff_ || rhs.autodiff_) {
        CudaMatrix der = lhs.der_ + rhs.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    CudaArray val = lhs.val_ - rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
        CudaMatrix der = lhs.der_ - rhs.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

//CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
//    CollOfScalar out;
//    out.val_ = lhs.val_ - rhs.val_;
//    if ( lhs.autodiff_ || rhs.autodiff_ ) {
//  out.autodiff_ = true;
//  out.der_ = lhs.der_ - rhs.der_;
//    }
//    return out;
//}


CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    CudaArray val = lhs.val_ * rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
        // (u*v)' = u'*v + v'*u = diag(v)*u' + diag(u)*v'
        // where u = lhs and v = rhs
        CudaMatrix diag_u(lhs.val_);
        CudaMatrix diag_v(rhs.val_);
        CudaMatrix der = diag_v*lhs.der_ + diag_u*rhs.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    CudaArray val = lhs.val_ / rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
        // (u/v)' = (u'*v - v'*u)/(v^2)
        // where u = lhs and v = rhs
        CudaMatrix diag_u(lhs.val_); // D1 
        CudaMatrix diag_v(rhs.val_); // D2
        CudaMatrix inv_v_squared( 1.0/(rhs.val_ * rhs.val_));
        CudaMatrix der = inv_v_squared*( diag_v*lhs.der_ - diag_u*rhs.der_);
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator*(const Scalar lhs, const CollOfScalar& rhs)
{
    CudaArray val = lhs * rhs.val_;
    if ( rhs.autodiff_ ) {
        CudaMatrix der = lhs * rhs.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const Scalar rhs)
{
    return ( rhs * lhs);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const Scalar rhs)
{
    CudaArray val = lhs.val_ / rhs;
    if ( lhs.autodiff_ ) {
        // (u/a)' = (u'*a)/(a^2) = u'/rhs
        // where u = lhs and v = rhs
        CudaMatrix der = lhs.der_/rhs;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
    //return ( (1/rhs) * lhs);
}

CollOfScalar equelleCUDA::operator/(const Scalar lhs, const CollOfScalar& rhs)
{
    CudaArray val = lhs / rhs.val_;
    if ( rhs.autodiff_ ) {
        // (a/u)' = - (a/u^2)*u'
        // where a = lhs and u = rhs
        CudaMatrix diag_u_squared(lhs/(rhs.val_ * rhs.val_));
        CudaMatrix der = -diag_u_squared*rhs.der_;
        return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg){
    CollOfScalar out(arg);
    out.val_ = -out.val_;
    if ( out.autodiff_ ) {
        out.der_ = -out.der_;
    }
    return out;
}

CollOfScalar& equelleCUDA::operator-(CollOfScalar&& arg) {
    arg.val_ = std::move(-arg.val_);
    if ( arg.autodiff_ ) {
        arg.der_ = std::move(-arg.der_);
    }
    return arg;
}

/*CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg) {
    return -arg;
}*/

//  >
CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    return ( lhs.val_ > rhs.val_ );
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const Scalar rhs)
{
    return ( lhs.val_ > rhs );
}

CollOfBool equelleCUDA::operator>(const Scalar lhs, const CollOfScalar& rhs)
{
    return ( lhs > rhs.val_ );
}


// <
CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    // if   a < b   then b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const Scalar rhs)
{
    // if  a < b  then   b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const Scalar lhs, const CollOfScalar& rhs)
{
    // if   a < b   then b > a
    return rhs > lhs;
}


// >=
CollOfBool equelleCUDA::operator>=(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    return ( lhs.val_ >= rhs.val_ );
}

CollOfBool equelleCUDA::operator>=(const CollOfScalar& lhs, const Scalar rhs)
{
    return ( lhs.val_ >= rhs );
}

CollOfBool equelleCUDA::operator>=(const Scalar lhs, const CollOfScalar& rhs)
{
    return ( lhs >= rhs.val_ );
}


// <= 
CollOfBool equelleCUDA::operator<=(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    // if   a <= b   then b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const CollOfScalar& lhs, const Scalar rhs)
{
    // if  a <= b  then   b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const Scalar lhs, const CollOfScalar& rhs)
{
    // if   a <= b   then b >= a
    return rhs >= lhs;
}


// ==
CollOfBool equelleCUDA::operator==(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    return ( lhs.val_ == rhs.val_ );
}

CollOfBool equelleCUDA::operator==(const CollOfScalar& lhs, const Scalar rhs)
{
    return ( lhs.val_ == rhs );
}

CollOfBool equelleCUDA::operator==(const Scalar lhs, const CollOfScalar& rhs)
{
    return (rhs == lhs);
}


// !=
CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const CollOfScalar& rhs)
{
    return ( lhs.val_ != rhs.val_ );
}

CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const Scalar rhs)
{
    return ( lhs.val_ != rhs );
}

CollOfBool equelleCUDA::operator!=(const Scalar lhs, const CollOfScalar& rhs)
{
    return (rhs != lhs);
}



// Matrix * CollOfScalar
CollOfScalar equelleCUDA::operator*(const CudaMatrix& mat, const CollOfScalar& coll)
{
    if ( coll.useAutoDiff() ) {
        return CollOfScalar( mat * coll.value(), mat * coll.derivative() );
    }
    return CollOfScalar( mat * coll.value() );
}