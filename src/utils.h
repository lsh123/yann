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

// YANN_CHECKs are enabled by default
#if defined (YANN_ENABLE_CHECKS) || ! defined (YANN_DISABLE_CHECKS)

// TODO: replace BOOST_VERIFY
#define YANN_CHECK( x )         BOOST_VERIFY ( x )
#define YANN_CHECK_EQ( x, y )   BOOST_VERIFY ( (x) == (y) )
#define YANN_CHECK_NE( x, y )   BOOST_VERIFY ( (x) != (y) )
#define YANN_CHECK_LE( x, y )   BOOST_VERIFY ( (x) <= (y) )
#define YANN_CHECK_LT( x, y )   BOOST_VERIFY ( (x) < (y) )
#define YANN_CHECK_GE( x, y )   BOOST_VERIFY ( (x) >= (y) )
#define YANN_CHECK_GT( x, y )   BOOST_VERIFY ( (x) > (y) )

#else /* defined (YANN_ENABLE_CHECKS) || ! defined (YANN_DISABLE_CHECKS) */

#define YANN_CHECK( x )
#define YANN_CHECK_EQ( x, y )
#define YANN_CHECK_NE( x, y )
#define YANN_CHECK_LE( x, y )
#define YANN_CHECK_LT( x, y )
#define YANN_CHECK_GE( x, y )
#define YANN_CHECK_GT( x, y )

#endif /* defined (YANN_ENABLE_CHECKS) || ! defined (YANN_DISABLE_CHECKS) */

// YANN_SLOW_CHECKs are disabled by default
#if defined (YANN_ENABLE_SLOW_CHECKS) && ! defined (YANN_DISABLE_SLOW_CHECKS)

#define YANN_SLOW_CHECK( x )         YANN_CHECK ( x )
#define YANN_SLOW_CHECK_EQ( x, y )   YANN_CHECK_EQ ( x, y )
#define YANN_SLOW_CHECK_NE( x, y )   YANN_CHECK_NE ( x, y )
#define YANN_SLOW_CHECK_LE( x, y )   YANN_CHECK_LE ( x, y )
#define YANN_SLOW_CHECK_LT( x, y )   YANN_CHECK_LT ( x, y )
#define YANN_SLOW_CHECK_GE( x, y )   YANN_CHECK_GE ( x, y )
#define YANN_SLOW_CHECK_GT( x, y )   YANN_CHECK_GT ( x, y )

#else /* defined (YANN_ENABLE_SLOW_CHECKS) && ! defined (YANN_DISABLE_SLOW_CHECKS) */

#define YANN_SLOW_CHECK( x )
#define YANN_SLOW_CHECK_EQ( x, y )
#define YANN_SLOW_CHECK_NE( x, y )
#define YANN_SLOW_CHECK_LE( x, y )
#define YANN_SLOW_CHECK_LT( x, y )
#define YANN_SLOW_CHECK_GE( x, y )
#define YANN_SLOW_CHECK_GT( x, y )

#endif /* defined (YANN_ENABLE_SLOW_CHECKS) && ! defined (YANN_DISABLE_SLOW_CHECKS) */


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
