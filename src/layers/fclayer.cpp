/*
 * fclayer.cpp
 *
 * Feed-forward:
 *    z(l) = a(l-1) * w(l) + b(l) // where * (vector, matrix) multiplication
 *    a(l) = activation(z(l))
 *
 * Back propagation:
 *    gradient(C, a(l)) = transp(w(l+1)) * gradient(C, a(l+1))
 *    delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
 *    dC/db(l) = delta(l)
 *    dC/dw(l) = a(l-1) * delta(l)
 */
#include <boost/assert.hpp>

#include "utils.h"
#include "random.h"
#include "functions.h"
#include "fclayer.h"

using namespace std;
using namespace boost;
using namespace yann;


namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// FullyConnectedLayer_Context implementation
//
class FullyConnectedLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class FullyConnectedLayer;

public:
  FullyConnectedLayer_Context(const MatrixSize & output_size,
                              const MatrixSize & batch_size) :
    Base(output_size, batch_size)
  {
    YANN_CHECK_GT(output_size, 0);
    YANN_CHECK_GT(batch_size, 0);
    _zz.resize(get_batch_size(), get_output_size());
  }

  FullyConnectedLayer_Context(const RefVectorBatch & output) :
    Base(output)
  {
    YANN_CHECK_GT(yann::get_batch_size(output), 0);
    YANN_CHECK_GT(yann::get_batch_item_size(output), 0);

    _zz.resize(get_batch_size(), get_output_size());
  }

protected:
  VectorBatch              _zz;
}; // class FullyConnectedLayer_Context

////////////////////////////////////////////////////////////////////////////////////////////////
//
// FullyConnectedLayer_TrainingContext implementation
//
class FullyConnectedLayer_TrainingContext :
    public FullyConnectedLayer_Context
{
  typedef FullyConnectedLayer_Context Base;

  friend class FullyConnectedLayer;

public:
  FullyConnectedLayer_TrainingContext(
      const unique_ptr<Layer::Updater> & updater,
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const MatrixSize & batch_size) :
    Base(output_size, batch_size),
    _ww_updater(updater->copy()),
    _bb_updater(updater->copy())
  {
    YANN_CHECK_GT(input_size, 0);

    _delta_ww.resize(input_size, get_output_size());
    _delta_bb.resize(get_output_size());
    _delta.resizeLike(_zz);

    _sigma_derivative_zz.resizeLike(_zz);
    _collapse_vector = Vector::Ones(get_batch_size());
  }

  FullyConnectedLayer_TrainingContext(
      const unique_ptr<Layer::Updater> & updater,
      const MatrixSize & input_size,
      const RefVectorBatch & output) :
    Base(output),
    _ww_updater(updater->copy()),
    _bb_updater(updater->copy())
  {
    YANN_CHECK_GT(input_size, 0);

    _delta_ww.resize(input_size, get_output_size());
    _delta_bb.resize(get_output_size());
    _delta.resizeLike(_zz);

    _sigma_derivative_zz.resizeLike(_zz);
    _collapse_vector = Vector::Ones(get_batch_size());
  }

  // Layer::Context overwrites
  virtual void reset_state()
  {
    _delta_ww.setZero();
    _delta_bb.setZero();

    _ww_updater->reset(_delta_ww);
    _bb_updater->reset(_delta_bb);
  }

private:
  Matrix _delta_ww;
  Vector _delta_bb;
  VectorBatch _delta;

  VectorBatch _sigma_derivative_zz;
  Vector _collapse_vector;

  unique_ptr<Layer::Updater> _ww_updater;
  unique_ptr<Layer::Updater> _bb_updater;
}; // class FullyConnectedLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::FullyConnectedLayer implementation
//
yann::FullyConnectedLayer::FullyConnectedLayer(const MatrixSize & input_size,
                                               const MatrixSize & output_size) :
    _ww(input_size, output_size),
    _bb(output_size),
    _activation_function(new SigmoidFunction())
{
}

yann::FullyConnectedLayer::~FullyConnectedLayer()
{
}

void yann::FullyConnectedLayer::set_activation_function(const unique_ptr<ActivationFunction> & activation_function)
{
  _activation_function = activation_function->copy();
}

void yann::FullyConnectedLayer::set_values(const Matrix & ww, const Vector & bb)
{
  YANN_CHECK(is_same_size(ww, _ww));
  YANN_CHECK(is_same_size(bb, _bb));
  _ww = ww;
  _bb = bb;
}

// Layer overwrites
bool yann::FullyConnectedLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_activation_function) {
    return false;
  }
  return true;
}

std::string yann::FullyConnectedLayer::get_name() const
{
  return "FullyConnectedLayer";
}

void yann::FullyConnectedLayer::print_info(std::ostream & os) const
{
  YANN_CHECK(is_valid());

  Base::print_info(os);
  os << " activation: " << _activation_function->get_name();
}

bool yann::FullyConnectedLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const FullyConnectedLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  // TOOD: add deep compare
  if(_activation_function->get_name() != the_other->_activation_function->get_name()) {
    return false;
  }
  if(!_ww.isApprox(the_other->_ww, tolerance)) {
    return false;
  }
  if(!_bb.isApprox(the_other->_bb, tolerance)) {
    return false;
  }
  return true;
}

MatrixSize yann::FullyConnectedLayer::get_input_size() const
{
  return _ww.rows();
}

MatrixSize yann::FullyConnectedLayer::get_output_size() const
{
  return _ww.cols();
}

unique_ptr<Layer::Context> yann::FullyConnectedLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  return make_unique<FullyConnectedLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::FullyConnectedLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  return make_unique<FullyConnectedLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::FullyConnectedLayer::create_training_context(
    const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<FullyConnectedLayer_TrainingContext>(
      updater, get_input_size(), get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::FullyConnectedLayer::create_training_context(
    const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<FullyConnectedLayer_TrainingContext>(
      updater, get_input_size(), output);
}

void yann::FullyConnectedLayer::feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode) const
{
  auto ctx = dynamic_cast<FullyConnectedLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_EQ(get_batch_size(input), ctx->get_batch_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  // z(l) = a(l-1)*w(l) + b(l)
  ctx->_zz.noalias() = YANN_FAST_MATRIX_PRODUCT(input, _ww);
  plus_batches(ctx->_zz, _bb);

  // a(l) = activation(z(l))
  YANN_CHECK(is_same_size(ctx->_zz, ctx->get_output()));
  _activation_function->f(ctx->_zz, ctx->get_output(), mode);
}

void yann::FullyConnectedLayer::backprop(const RefConstVectorBatch & gradient_output,
                                         const RefConstVectorBatch & input,
                                         optional<RefVectorBatch> gradient_input,
                                         Context * context) const
{
  auto ctx = dynamic_cast<FullyConnectedLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));

  // just to make it easier to read
  const auto & zz = ctx->_zz;
  auto & sigma_derivative_zz = ctx->_sigma_derivative_zz;
  auto & delta = ctx->_delta;
  const auto & collapse_vector = ctx->_collapse_vector;
  auto & delta_ww = ctx->_delta_ww;
  auto & delta_bb = ctx->_delta_bb;
  const auto batch_size = get_batch_size(input);

  // delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
  YANN_CHECK(is_same_size(zz, sigma_derivative_zz));
  YANN_CHECK(is_same_size(zz, gradient_output));
  _activation_function->derivative(zz, sigma_derivative_zz);
  delta.array() = gradient_output.array() * sigma_derivative_zz.array();

  // update deltas
  // dC/db(l) = delta(l)
  // dC/dw(l) = a(l-1) * delta(l)
  YANN_CHECK_EQ(collapse_vector.size(), get_batch_size(delta));
  YANN_CHECK_EQ(delta_bb.size(), get_output_size());
  YANN_CHECK_EQ(batch_size,  get_batch_size(gradient_output));
  YANN_CHECK(is_same_size(delta_ww, _ww));
  YANN_CHECK(is_same_size(delta_bb, _bb));
  delta_ww.noalias() += YANN_FAST_MATRIX_PRODUCT(input.transpose(), delta);
  delta_bb.noalias() += YANN_FAST_MATRIX_PRODUCT(collapse_vector, delta);

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  // gradient(C, a(l)) = transp(w(l+1)) * gradient(C, a(l+1))
  if(gradient_input) {
    (*gradient_input).noalias() = YANN_FAST_MATRIX_PRODUCT(delta,  _ww.transpose());
  }
}

void yann::FullyConnectedLayer::init(enum InitMode mode)
{
  switch (mode) {
  case InitMode_Zeros:
    _ww.setZero();
    _bb.setZero();
    break;
  case InitMode_Random_01:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1);
      gen01->generate(_ww);
      gen01->generate(_bb);
    }
    break;
  case InitMode_Random_SqrtInputs:
    {
      unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, sqrt((Value)get_input_size()));
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1);
      gen->generate(_ww);
      gen01->generate(_bb);
    }
    break;
  }
}

void yann::FullyConnectedLayer::update(Context * context, const size_t & batch_size)
{
  auto ctx = dynamic_cast<FullyConnectedLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->_ww_updater);
  YANN_CHECK(ctx->_bb_updater);
  YANN_CHECK(is_same_size(_ww, ctx->_delta_ww));
  YANN_CHECK(is_same_size(_bb, ctx->_delta_bb));

  ctx->_ww_updater->update(ctx->_delta_ww, batch_size, _ww);
  ctx->_bb_updater->update(ctx->_delta_bb, batch_size, _bb);
}

// the format is (w:<weights>,b:<biases>)
void yann::FullyConnectedLayer::read(std::istream & is)
{
  Base::read(is);

  char ch;
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  if(is >> ch && ch != 'w') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  if(is >> ch && ch != ':') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  is >> _ww;
  if(is.fail()) {
    return;
  }
  if(is >> ch && ch != ',') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  if(is >> ch && ch != 'b') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  if(is >> ch && ch != ':') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
  is >> _bb;
  if(is.fail()) {
    return;
  }
  if(is >> ch && ch != ')') {
    is.putback(ch);
    is.setstate(std::ios_base::failbit);
    return;
  }
}

// the format is (w:<weights>,b:<biases>)
void yann::FullyConnectedLayer::write(std::ostream & os) const
{
  Base::write(os);
  os << "(w:" << _ww << ",b:" << _bb << ")";
}

