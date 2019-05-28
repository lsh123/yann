/*
 * types.h
 *
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

namespace yann {

typedef unsigned char Byte;
typedef double Value;

// We use RowMajor layout (i.e. in memory Matrix is represented as:
//   <row 0><row 1>...<row N>
// If layout changes then the batch (see below) should be changed as well
// and bunch of tests should be updated.
typedef Eigen::Matrix<Value, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Matrix<Value, 1, Eigen::Dynamic, Eigen::RowMajor> Vector;

typedef Eigen::Ref<const Matrix> RefConstMatrix;
typedef Eigen::Ref<Matrix> RefMatrix;
typedef Eigen::Ref<const Vector> RefConstVector;
typedef Eigen::Ref<Vector> RefVector;

typedef Eigen::Map<Matrix> MapMatrix;
typedef Eigen::Map<const Matrix> MapConstMatrix;

typedef Eigen::Map<Vector> MapVector;
typedef Eigen::Map<const Vector> MapConstVector;

typedef Matrix::Index MatrixSize;


// A batch of vectors is represented by a Matrix where each vector is a row (since
// we use RowMajor layout. We define helper types to distinguish between a batch
// and regular matrix. However the memory layouts between the two must be the same
// since we sometime look at the vector as a matrix and vice versa.
typedef Matrix          VectorBatch;
typedef RefMatrix       RefVectorBatch;
typedef RefConstMatrix  RefConstVectorBatch;
typedef Eigen::Map<VectorBatch> MapVectorBatch;
typedef Eigen::Map<const VectorBatch> MapConstVectorBatch;

// Note these helpers will need to change for a different memory layout.
inline void resize_batch(VectorBatch & vb, const MatrixSize & batch_size, const MatrixSize & other_size)
{
  vb.resize(batch_size, other_size);
}
inline MatrixSize get_batch_size(const RefConstVectorBatch & vb)
{
  return vb.rows();
}
inline MatrixSize get_batch_item_size(const RefConstVectorBatch & vb)
{
  return vb.cols();
}
inline void plus_batches(VectorBatch & mm, const Vector & vv)
{
  mm.rowwise() += vv;
}
#define get_batch_const(vb, pos)  ((vb).row(pos))
#define get_batch(vb, pos)        ((vb).row(pos))

// Helpers
template<typename T1, typename T2>
inline bool is_same_size(const T1 & m1, const T2 & m2) {
  return m1.rows() == m2.rows() && m1.cols() == m2.cols();
}

}; // namespace yann

// Eigen has a very fast matrix multiplication but it allocates internal
// buffers to speed up the computations which prevents the no malloc testing.
// We use this macro to switch between fast and slow versions.
#ifdef EIGEN_RUNTIME_NO_MALLOC
#define YANN_FAST_MATRIX_PRODUCT(aa, bb)   ((aa).lazyProduct((bb)))
#else   /* EIGEN_RUNTIME_NO_MALLOC */
#define YANN_FAST_MATRIX_PRODUCT(aa, bb)   ((aa) * (bb))
#endif  /* EIGEN_RUNTIME_NO_MALLOC */

// This is required for std::vector<Matrix> and std::vector<Vector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(yann::Matrix)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(yann::Vector)

//
// Operator overwrites in std:: namespace
//
namespace std {
  ostream& operator<<(ostream & os, const yann::RefConstMatrix & mm);
  ostream& operator<<(ostream & os, yann::RefMatrix mm);
  ostream& operator<<(ostream & os, const yann::Matrix & mm);
  istream& operator>>(istream & is, yann::Matrix & mm);

  ostream& operator<<(ostream & os, const yann::RefConstVector & vv);
  ostream& operator<<(ostream & os, yann::RefVector vv);
  ostream& operator<<(ostream & os, const yann::Vector & vv);
  istream& operator>>(istream & is, yann::Vector & vv);
}; // namespace std

#endif /* TYPES_H_ */
