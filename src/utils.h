/*
 * utils.h
 *
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <boost/assert.hpp>

#include "types.h"

#define DBG(var) \
  { \
    std::stringstream tmp;  \
    tmp << std::setprecision(10) << #var << "=" << (var); \
    print_debug(__FILE__, __LINE__, tmp.str()); \
  }

namespace yann {


////////////////////////////////////////////////////////////////////////////////////////////////
//
// debug helper
//
void print_debug(const std::string & filename, int line,
            const std::string & message);

////////////////////////////////////////////////////////////////////////////////////////////////
//
// helper templates
//
template<typename T>
inline void plus_scalar(VectorBatch & mm, const T & val)
{
  mm.array() *= val;
}

}; // namespace yann

// Overwrites from std:: namespace
namespace std {

template<typename T>
ostream& operator<<(ostream& os, const vector<T> & vv)
{
  os << "[" << vv.size() << "](";
  for (auto it = vv.begin(); it != vv.end(); ++it) {
    if (it != vv.begin()) {
      os << ", " << (*it);
    } else {
      os << (*it);
    }
  }
  os << ")";
  return os;
}
}; // namespace std

#endif /* UTILS_H_ */
