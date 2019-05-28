/*
 * utils.cpp
 *
 */
#include <cstdlib>
#include <ctime>

#include <libgen.h> // for basename()

#include <boost/assert.hpp>

#include "utils.h"

using namespace std;
using namespace yann;


void yann::print_debug(const std::string & filename, int line,
                       const std::string & message)
{
  string fullpath(filename); // copy since basename may modify it
  std::string name = basename(const_cast<char*>(fullpath.c_str()));
  std::cerr << "DEBUG: " << name << ":" << line << "  " << message << endl;
}
