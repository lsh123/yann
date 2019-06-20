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

#include "core/utils.h"
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
                               const MatrixSize & batch_size,
                               const MatrixSize & input_size) :
    Base(output_size, batch_size),
    _tmp(input_size)
  {
  }

  SoftmaxLayer_TrainingContext(const RefVectorBatch & output,
                               const MatrixSize & input_size) :
    Base(output),
    _tmp(input_size)
  {
  }

protected:
  Vector _tmp;
}; // class SoftmaxLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::SoftmaxLayer implementation
//
void yann::SoftmaxLayer::softmax_plus_equal(const RefConstVector & input, RefVector output, const Value & beta)
{
  YANN_CHECK(is_same_size(input, output));
  YANN_CHECK_EQ(input.rows(), 1); // RowMajor layout, breaks for ColMajor

  Value max = input.maxCoeff(); // adjust the computations to avoid overflowing
  Value sum = exp((input.array() - max) * beta).sum();
  output.array() += (exp((input.array() - max) * beta)) / sum;
}


// gradient_input = softmax_derivative(input) * gradient_output
//
// where
//    softmax_derivative(input, ii, jj) = - softmax(input, ii) * softmax(input, jj) if ii != jj
//    softmax_derivative(input, ii, jj) = softmax(ii) * (1 - softmax(ii)) if ii == jj
//
// gradient_input(ii) = sum(softmax_derivative(input, ii, jj) * gradient_output(jj))
//
// gradient_input(ii) = sum(ii != jj, - softmax(input, ii) * softmax(input, jj) * gradient_output(jj)) +
//                      softmax(ii) * (1 - softmax(ii)) * * gradient_output(ii)
//
// gradient_input(ii) = sum(ii != jj, - softmax(input, ii) * softmax(input, jj) * gradient_output(jj)) +
//                      softmax(ii) * gradient_output(ii) - softmax(ii) * softmax(ii) * gradient_output(ii)
//
// gradient_input(ii) = softmax(input, ii) * (sum(-softmax(input, jj) * gradient_output(jj)) + gradient_output(ii))
//
void yann::SoftmaxLayer::softmax_gradient(
    const RefConstVector & input,
    const RefConstVector & gradient_output,
    RefVector tmp,
    RefVector gradient_input,
    const Value & beta)
{
  YANN_CHECK(is_same_size(input, gradient_output));
  YANN_CHECK(is_same_size(input, tmp));
  YANN_CHECK(is_same_size(input, gradient_input));

  tmp.setZero();
  softmax_plus_equal(input, tmp, beta);

  Value sum = (tmp.array() * gradient_output.array()).sum();
  gradient_input = beta * tmp.array() * (gradient_output.array() - sum);
}

yann::SoftmaxLayer::SoftmaxLayer(const MatrixSize & size, const Value & beta) :
    _size(size),
    _beta(beta)
{
  YANN_CHECK_GT(size, 0);
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
  YANN_CHECK(is_valid());
  return make_unique<SoftmaxLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  return make_unique<SoftmaxLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_training_context(
    const MatrixSize & batch_size,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  return make_unique<SoftmaxLayer_TrainingContext>(get_output_size(), batch_size, get_input_size());
}
unique_ptr<Layer::Context> yann::SoftmaxLayer::create_training_context(
    const RefVectorBatch & output,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  return make_unique<SoftmaxLayer_TrainingContext>(output, get_input_size());
}

void yann::SoftmaxLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<SoftmaxLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(input), 0);
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_LE(get_batch_size(input), get_batch_size(ctx->get_output()));
  YANN_CHECK_EQ(get_batch_item_size(ctx->get_output()), get_output_size());

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

void yann::SoftmaxLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  throw runtime_error("SoftmaxLayer::feedforward() is not implemented for sparse vectors");
}

void yann::SoftmaxLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  YANN_CHECK(is_valid());
  YANN_SLOW_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_SLOW_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_SLOW_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_SLOW_CHECK_EQ(get_batch_item_size(input), get_input_size());

  auto ctx = static_cast<SoftmaxLayer_TrainingContext*>(context);
  YANN_CHECK(ctx);

  // nothing to do for the softmax layer itself

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  if(gradient_input) {
    YANN_SLOW_CHECK_EQ(get_batch_item_size(input), get_batch_item_size(*gradient_input));
    YANN_SLOW_CHECK_EQ(get_batch_size(input), get_batch_size(*gradient_input));

    const auto batch_size = get_batch_size(input);
    for(MatrixSize ii = 0; ii < batch_size; ++ii) {
      softmax_gradient(
          get_batch(input, ii),
          get_batch(gradient_output, ii),
          ctx->_tmp,
          get_batch(*gradient_input, ii),
          _beta);
    }
  }
}

void yann::SoftmaxLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  throw runtime_error("SoftmaxLayer::backprop() is not implemented for sparse vectors");
}

void yann::SoftmaxLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  // nothing to do
}

void yann::SoftmaxLayer::update(Context * context, const size_t & batch_size)
{
  // auto ctx = dynamic_cast<SoftmaxLayer_Context *>(context);
  // YANN_CHECK(ctx);
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

