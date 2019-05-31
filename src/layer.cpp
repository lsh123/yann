/*
 * nnlayer.cpp
 *
 */
#include <exception>
#include <string>
#include <stdexcept>

#include "utils.h"
#include "layer.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Layer implementation
//
ostream& std::operator<<(ostream & os, const Layer & layer)
{
  layer.write(os);
  return os;
}

istream& std::operator>>(istream & is, Layer & layer)
{
  layer.read(is);
  return is;
}

std::string yann::Layer::get_info() const
{
  ostringstream oss;
  print_info(oss);
  return oss.str();
}

void yann::Layer::print_info(std::ostream & os) const
{
  os << get_name()
    << "[" << get_input_size() << " -> " << get_output_size() << "]"
  ;
}


void yann::Layer::read(istream & is)
{
  string name;
  char ch;

  if(is >> ch && ch != '[') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  while(!is.fail()) {
    if(!(is >> ch)) {
      is.putback(ch);
      is.setstate(std::ios_base::failbit);
      return;
    }

    if(ch == ']') {
      break;
    }
    name += ch;
  }

  if(get_name().compare(name) != 0) {
    throw invalid_argument(
        "invalid file format: expected name: " + get_name()
        + " actual name: " + name);
  }
}

void yann::Layer::write(ostream & os) const
{
  os << "[" << get_name() << "]";
}

