/*
 * timer.cpp
 *
 */
#include <cstdlib>

#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/types.h"

using namespace std;
using namespace yann;

#define DEFAULT_PRECISION   8

namespace yann {

// writes (<data1>,<data2>,...<dataN>)
inline static void write_data(ostream & os, const yann::RefConstVector & vv) {
  os << std::setprecision(DEFAULT_PRECISION) << "(";
  for(auto ii = 0; ii < vv.size(); ++ii) {
    if(ii > 0) {
      os << "," << vv(ii);
    } else {
      os << vv(ii);
    }
  }
  os << ")";
}

// reads (<data1>,<data2>,...<dataN>)
inline static void read_data(istream & is, yann::RefVector vv) {
  char ch;

  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }

  yann::Value val;
  for(yann::MatrixSize ii = 0; ii < vv.size() && !is.fail(); ++ii) {
    if(!(is >> val >> ch)) {
      return;
    }
    if(ii < vv.size() - 1 && ch != ',') {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return;
    } else if(ii == vv.size() - 1 && ch != ')') {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return;
    }
    vv(ii) = val;
  }
}

// writes [<vector size>](<data1>,<data2>,...<dataN>)
template<typename T>
inline static ostream& write_vector(ostream & os, const T & vv)
{
  os << "[" << vv.size() << "]";
  write_data(os, vv);
  return os;
}

// reads [<vector size>](<data1>,<data2>,...<dataN>)
template<typename T>
inline static istream& read_vector(istream & is, T & vv)
{
  yann::MatrixSize size;
  char ch;

  if(is >> ch && ch != '[') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> size >> ch && ch != ']') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }

  if(!is.fail()) {
    vv.resize(size);
    read_data(is, vv);
  }
  return is;
}

// writes [<rows-num>,<cols-num>]((<row0 col0>,<row0 col1>,..)(<row1 col0>,<row1 col1>,..)..)
template<typename T>
inline static ostream& write_matrix(ostream & os, const T & mm)
{
  os << "[" << mm.rows() << "," << mm.cols() << "](";
  for(auto ii = 0; ii < mm.rows(); ++ii) {
    if(ii > 0) {
      os << ",";
    }
    write_data(os, mm.row(ii));
  }
  os << ")";
  return os;
}

// reads [<rows-num>,<cols-num>]((<row0 col0>,<row0 col1>,..)(<row1 col0>,<row1 col1>,..)..)
template<typename T>
inline static istream& read_matrix(istream & is, T & mm)
{
  yann::MatrixSize rows, cols;
  char ch;

  if(is >> ch && ch != '[') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> rows >> ch && ch != ',') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> cols >> ch && ch != ']') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is.fail()) {
    return is;
  }
  mm.resize(rows, cols);
  for(yann::MatrixSize ii = 0; ii < rows && !is.fail(); ++ii) {
    read_data(is, mm.row(ii));

    if(ii < rows - 1 && is >> ch && ch != ',') {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return is;
    }
  }
  if(is >> ch && ch != ')') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }

  return is;
}

}; // namespace yann

namespace std {



/**
 * Output operator for matrixes, generates format:
 *
 * [<rows-num>,<cols-num>]((<row0 col0>,<row0 col1>,..)(<row1 col0>,<row1 col1>,..)..)
 *
 */
ostream& operator<<(ostream & os, const yann::RefConstMatrix & mm)
{
  return write_matrix(os, mm);
}

ostream& operator<<(ostream & os, yann::RefMatrix mm)
{
  return write_matrix(os, mm);
}
ostream& operator<<(ostream & os, const yann::Matrix & mm)
{
  return write_matrix(os, mm);
}

/**
 * Output operator for matrixes, expected format:
 *
 * [<rows-num>,<cols-num>]((<row0 col0>,<row0 col1>,..)(<row1 col0>,<row1 col1>,..)..)
 *
 */
istream& operator>>(istream & is, yann::Matrix & mm)
{
  return read_matrix(is, mm);
}

/**
 * Output operator for vectors, expected format:
 *
 * [<vector size>](<data1>,<data2>,...<dataN>)
 *
 * For example for vector (1,2,3) the output would be
 *
 * [3](1,2,3)
 */
ostream& operator<<(ostream & os, const yann::RefConstVector & vv)
{
  return write_vector(os, vv);
}
ostream& operator<<(ostream & os, RefVector vv)
{
  return write_vector(os, vv);
}
ostream& operator<<(ostream & os, const yann::Vector & vv)
{
  return write_vector(os, vv);
}


/**
 * Input operator for vectors, expected format:
 *
 * [<vector size>](<data1>,<data2>,...<dataN>)
 */
istream& operator>>(istream & is, yann::Vector & vv)
{
  return read_vector(is, vv);
}

}; // namespace std
