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
#include <exception>

#include <boost/assert.hpp>

#include "core/types.h"

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
// read/write helpers for <name>:<object> format
//
inline void read_char(std::istream & is, const char & expected)
{
  char ch;
  if(is >> ch && ch != expected) {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    throw std::runtime_error(
        std::string("expected char: '") + expected +
        std::string("', actual char: '") + ch + std::string("'")
    );
  }
}

template<typename ObjectType>
inline void read_object(std::istream & is, const std::string & name, ObjectType & obj)
{
  for(size_t pos = 0; pos < name.size(); ++pos) {
    read_char(is, name[pos]);
  }
  read_char(is, ':');

  is >> obj;

  if(is.fail()) {
    throw std::runtime_error(std::string("failed to read \"") + name + std::string("\""));
  }
}

template<typename ObjectType>
inline std::ostream & write_object(std::ostream & os, const std::string & name, const ObjectType & obj)
{
  os << name << ":" << obj;
  return os;
}

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

// format: [<num>](<elem0>,<elem1>, ...)
template<typename T>
ostream& operator<<(ostream& os, const vector<T> & vv)
{
  os << "[" << vv.size() << "](";
  bool write_comma = false;
  for(const auto & elem: vv) {
    if(write_comma) {
      os << ",";
    }
    os << elem;
    write_comma = true;
  }
  os << ")";
  return os;
}

// format: [<num>](<elem0>,<elem1>, ...)
template<typename T>
istream& operator>>(istream& is, vector<T> & vv)
{
  char ch;
  size_t num;
  if(is >> ch && ch != '[') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  is >> num;
  if(is.fail()) {
    return is;
  }
  if(is >> ch && ch != ']') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return is;
  }
  vv.reserve(vv.size() + num);

  while(!is.fail()) {
    T elem;
    is >> elem;
    if(is.fail()) {
      return is;
    }
    vv.push_back(elem);
    if(is >> ch && ch != ',' && ch != ')') {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return is;
    }
    if(ch == ')') {
      break;
    }
  }

  return is;
}

// format: (<elem1>,<elem2>)
template<typename T1, typename T2>
ostream& operator<<(ostream& os, const pair<T1, T2> & pp)
{
    os << "(" << pp.first << "," << pp.second << ")";
    return os;
}

}; // namespace std

#endif /* UTILS_H_ */
