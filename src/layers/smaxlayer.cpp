/*
 * smaxlayer.cpp
 *
 * This is SOFTMAX layer
 *
 *  Feed-forward:
 *    a(l, i) = softmax(a(l), i) = exp(a(l-1, i) / sum(exp(a(l-1)))
 *
 *  Back propagation:
 *    gradient(C, a(l)) = gradient(C, a(l+1)) * M(a(l + 1))
 *    where M(a(l), i, j) = softmax(a(l + 1), i) * (sigma(i, j) - softmax(a(l + 1), j)) where sigma(i, j) = 1 for i=j and 0 otherwise
 */
#include <boost/assert.hpp>

#include "utils.h"
#include "contlayer.h"
#include "smaxlayer.h"

using namespace std;
using namespace boost;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// SoftmaxLayer_Context and SoftmaxLayer_TrainingContext implementations
//
namespace yann {

typedef Layer::Context SoftmaxLayer_Context;

class SoftmaxLayer_TrainingContext :
    public SoftmaxLayer_Context
{
  typedef SoftmaxLayer_Context Base;

  friend class SoftmaxLayer;

public:
  SoftmaxLayer_TrainingContext(const MatrixSize & output_size,
                               const MatrixSize & batch_size) :
    Base(output_size, batch_size),
    _softmax_derivative(output_size, output_size)
  {
  }

  SoftmaxLayer_TrainingContext(const RefVectorBatch & output) :
    Base(output),
    _softmax_derivative(get_batch_item_size(output), get_batch_item_size(output))
  {
  }

protected:
  Matrix _softmax_derivative;
}; // class SoftmaxLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::SoftmaxLayer implementation
//
void yann::SoftmaxLayer::softmax_plus_equal(const RefConstVector & input, RefVector output, const Value & beta)
{
  BOOST_VERIFY(is_same_size(input, output));
  BOOST_VERIFY(input.rows() == 1); // RowMajor layout, breaks for ColMajor

  Value max = input.maxCoeff(); // adjust the computations to avoid overflowing
  Value sum = exp((input.array() - max) * beta).sum();
  output.array() += (exp((input.array() - max) * beta)) / sum;
}

void yann::SoftmaxLayer::softmax_derivative(const RefConstVector & input, RefMatrix output, const Value & beta)
{
  BOOST_VERIFY(input.size() > 0);
  BOOST_VERIFY(input.size() == output.rows());
  BOOST_VERIFY(input.size() == output.cols());

  // We are going to do a trick here by using the first row of the output matrix
  // as a temporary buffer. This depends on matrix layout (RowMajor) and will
  // break if it is switched to ColMajor
  output.row(0).setZero();
  softmax_plus_equal(input, output.row(0), beta);

  // The softmax derivative:
  // out(ii, jj) = - softmax(ii) * softmax(jj) if ii != jj
  // out(ii, jj) = softmax(ii) * (1 - softmax(ii)) if ii == jj
  MatrixSize size = input.size();

  // Notice that output for row ii uses common out0(ii) factor
  auto calculate_one_row = [beta, size, output](MatrixSize ii, Value out_0_ii) mutable {
    for(MatrixSize jj = 0; jj < size; ++jj) {
      if(ii != jj) {
        output(ii, jj) = - beta * out_0_ii * output(0, jj);
      } else {
        output(ii, jj) = beta *  out_0_ii * (1 - out_0_ii);
      }
    }
  };

  // skip row(0) since this is our temp buffer
  for(MatrixSize ii = 1; ii < size; ++ii) {
    calculate_one_row(ii, output(0, ii));
  }

  // now handle row(0)
  calculate_one_row(0, output(0, 0));
}

yann::SoftmaxLayer::SoftmaxLayer(const MatrixSize & size, const Value & beta) :
    _size(size),
    _beta(beta)
{
  BOOST_VERIFY(size > 0);
}

yann::SoftmaxLayer::~SoftmaxLayer()
{
}

// Layer overwrites
std::string yann::SoftmaxLayer::get_name() const
{
  return "SoftmaxLayer";
}

bool yann::SoftmaxLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const SoftmaxLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  if(_size != the_other->_size) {
    return false;
  }
  return true;
}

MatrixSize yann::SoftmaxLayer::get_input_size() const
{
  return _size;
}

MatrixSize yann::SoftmaxLayer::get_output_size() const
{
  return _size;
}

unique_ptr<Layer::Context> yann::SoftmaxLayer::create_context(const MatrixSize & batch_size) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<SoftmaxLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_context(const RefVectorBatch & output) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<SoftmaxLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_training_context(const MatrixSize & batch_size) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<SoftmaxLayer_TrainingContext>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_training_context(const RefVectorBatch & output) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<SoftmaxLayer_TrainingContext>(output);
}

void yann::SoftmaxLayer::feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode) const
{
  auto ctx = dynamic_cast<SoftmaxLayer_Context *>(context);
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(input) > 0);
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(get_batch_size(ctx->get_output()) == get_batch_size(input));
  BOOST_VERIFY(get_batch_item_size(ctx->get_output()) == get_output_size());

  RefVectorBatch output = ctx->get_output();
  switch(mode) {
  case Operation_Assign:
    output.setZero();
    break;
  case Operation_PlusEqual:
    // do nothing
    break;
  }
  const auto batch_size = get_batch_size(input);
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    softmax_plus_equal(get_batch(input, ii), get_batch(output, ii), _beta);
  }
}

void yann::SoftmaxLayer::backprop(const RefConstVectorBatch & gradient_output,
                                         const RefConstVectorBatch & input,
                                         optional<RefVectorBatch> gradient_input,
                                         Context * context) const
{
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(gradient_output) > 0);
  BOOST_VERIFY(get_batch_item_size(gradient_output) == get_output_size());
  BOOST_VERIFY(get_batch_size(input) == get_batch_size(gradient_output));
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(!gradient_input || is_same_size(input, *gradient_input));

  auto ctx = static_cast<SoftmaxLayer_TrainingContext*>(context);
  BOOST_VERIFY(ctx);

  // nothing to do for the softmax layer itself

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  if(gradient_input) {
    const auto batch_size = get_batch_size(input);
    for(MatrixSize ii = 0; ii < batch_size; ++ii) {
      softmax_derivative(get_batch(input, ii), ctx->_softmax_derivative, _beta);
      get_batch(*gradient_input, ii).noalias() = YANN_FAST_MATRIX_PRODUCT(get_batch(gradient_output, ii), ctx->_softmax_derivative);
    }
  }
}

void yann::SoftmaxLayer::init(enum InitMode /* mode */)
{
  // nothing to do
}

void yann::SoftmaxLayer::update(Context * /* context */,
                                double /* learning_factor */,
                                double /* decay_factor */)
{
  // auto ctx = dynamic_cast<SoftmaxLayer_Context *>(context);
  // BOOST_VERIFY(ctx);
  // nothing to do
}

void yann::SoftmaxLayer::read(std::istream & is)
{
  Base::read(is);
  // nothing to do
}

void yann::SoftmaxLayer::write(std::ostream & os) const
{
  Base::write(os);
  // nothing to do
}

