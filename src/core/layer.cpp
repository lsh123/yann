/*
 * nnlayer.cpp
 *
 */
#include <exception>
#include <string>
#include <stdexcept>

#include "core/utils.h"
#include "core/layer.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Layer::Context implementation
//
yann::Layer::Context::Context(
    const MatrixSize & output_size,
    const MatrixSize & batch_size)
{
  YANN_CHECK_GT(output_size, 0);
  YANN_CHECK_GT(batch_size, 0);

  resize_batch(_output_buffer, batch_size, output_size);
  _output = _output_buffer;
}

yann::Layer::Context::Context(
    const RefVectorBatch & output) :
    _output(output)
{
  YANN_CHECK_GT(yann::get_batch_item_size(output), 0);
  YANN_CHECK_GT(yann::get_batch_size(output), 0);
}

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

string yann::Layer::get_info() const
{
  ostringstream oss;

  oss << get_name()
    << "[" << get_input_size() << " -> " << get_output_size() << "]"
  ;
  return oss.str();
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

