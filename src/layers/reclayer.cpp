/*
 * reclayer.cpp
 *
 * Feed-forward:
 *  state:
 *    zz_h(t) = ww_hh * hh(t-1) + ww_xh*x(t) + b_h
 *    hh(t)   = state_activation_function(zz_h(t))
 *  output:
 *    zz_a(t) = ww_ha * hh(t) + b_a
 *    a(t)    = output_activation_function(zz_a(t))
 *
 * Back propagation:
 *   output:
 *    delta_a(t)  = elem_prod(gradient(C, a(t)), activation_derivative(zz_a(t)))
 *    dC/d b_a(t)   = delta_a(t)
 *    dC/d ww_ha(t) = hh(t) * delta_a(t)
 *    total_gradient(C, hh(t)) = gradient(C, hh(t)) + transp(ww_ha) * delta_a(t)
 *  state:
 *    delta_h(t) = elem_prod(total_gradient(C, hh(t)), activation_derivative(zz_h(t)))
 *    dC/d b_h(t) = delta_h(t)
 *    dC/d ww_xh(t) = x(t) * delta_h(t)
 *    dC/d ww_hh(t) = hh(t-1) * delta_h(t)
 *    gradient(C, hh(t-1)) = transp(ww_hh) * delta_h(t)
 *    gradient(C, x(t)) = transp(ww_xh) * delta_h(t)
 */
#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/random.h"
#include "core/functions.h"
#include "reclayer.h"

using namespace std;
using namespace boost;
using namespace yann;


namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// RecurrentLayer_Context implementation
//
class RecurrentLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class RecurrentLayer;

public:
  RecurrentLayer_Context(
      const MatrixSize & state_size,
      const MatrixSize & output_size,
      const MatrixSize & max_batch_size) :
    Base(output_size, max_batch_size),
    _pos(0)
  {
    init(state_size);
  }

  RecurrentLayer_Context(
      const MatrixSize & state_size,
      const RefVectorBatch & output) :
    Base(output),
    _pos(0)
  {
    init(state_size);
  }

  inline MatrixSize get_pos() const
  {
    return _pos;
  }

  // Layer::Context overwrites
  virtual void reset_state()
  {
    Base::reset_state();

    _pos = 0;
  }

private:
  void init(const MatrixSize & state_size)
  {
    YANN_CHECK_GT(state_size, 0);

    _hh.resize(get_batch_size(), state_size);
    _zz_h.resize(get_batch_size(), state_size);
    _zz_a.resize(get_batch_size(), get_output_size());
  }

protected:
  MatrixSize _pos;
  VectorBatch  _hh;       // state (save it for each step)
  VectorBatch  _zz_h;
  VectorBatch  _zz_a;
}; // class RecurrentLayer_Context

////////////////////////////////////////////////////////////////////////////////////////////////
//
// RecurrentLayer_TrainingContext implementation
//
class RecurrentLayer_TrainingContext :
    public RecurrentLayer_Context
{
  typedef RecurrentLayer_Context Base;
  friend class RecurrentLayer;

public:
  RecurrentLayer_TrainingContext(
      const MatrixSize & input_size,
      const MatrixSize & state_size,
      const MatrixSize & output_size,
      const MatrixSize & batch_size,
      const unique_ptr<Layer::Updater> & updater) :
    Base(state_size, output_size, batch_size),
    _ww_hh_updater(updater->copy()),
    _ww_xh_updater(updater->copy()),
    _bb_h_updater(updater->copy()),
    _ww_ha_updater(updater->copy()),
    _bb_a_updater(updater->copy())
  {
    init(input_size, state_size);
  }

  RecurrentLayer_TrainingContext(
      const MatrixSize & input_size,
      const MatrixSize & state_size,
      const RefVectorBatch & output,
      const unique_ptr<Layer::Updater> & updater) :
    Base(state_size, output),
    _ww_hh_updater(updater->copy()),
    _ww_xh_updater(updater->copy()),
    _bb_h_updater(updater->copy()),
    _ww_ha_updater(updater->copy()),
    _bb_a_updater(updater->copy())
  {
    init(input_size, state_size);
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    YANN_SLOW_CHECK(_ww_hh_updater);
    YANN_SLOW_CHECK(_ww_xh_updater);
    YANN_SLOW_CHECK(_bb_h_updater);
    YANN_SLOW_CHECK(_ww_ha_updater);
    YANN_SLOW_CHECK(_bb_a_updater);

    Base::start_epoch();

    _ww_hh_updater->start_epoch();
    _ww_xh_updater->start_epoch();
    _bb_h_updater->start_epoch();
    _ww_ha_updater->start_epoch();
    _bb_a_updater->start_epoch();
  }

  virtual void reset_state()
  {
    YANN_SLOW_CHECK(_ww_hh_updater);
    YANN_SLOW_CHECK(_ww_xh_updater);
    YANN_SLOW_CHECK(_bb_h_updater);
    YANN_SLOW_CHECK(_ww_ha_updater);
    YANN_SLOW_CHECK(_bb_a_updater);

    Base::reset_state();

    _gradient_h.setZero();

    _delta_ww_hh.setZero();
    _delta_ww_xh.setZero();
    _delta_bb_h.setZero();
    _delta_ww_ha.setZero();
    _delta_bb_a.setZero();

    _ww_hh_updater->reset();
    _ww_xh_updater->reset();
    _bb_h_updater->reset();
    _ww_ha_updater->reset();
    _bb_a_updater->reset();
  }

private:
  void init(const MatrixSize & input_size, const MatrixSize & state_size)
  {
    YANN_CHECK_GT(input_size, 0);
    YANN_CHECK_GT(state_size, 0);

    _delta_ww_hh.resize(state_size, state_size);
    _delta_ww_xh.resize(input_size, state_size);
    _delta_bb_h.resize(state_size);
    _delta_ww_ha.resize(state_size, get_output_size());
    _delta_bb_a.resize(get_output_size());

    _gradient_h.resize(state_size);
    _delta_h.resize(state_size);
    _delta_a.resize(get_output_size());

    _sigma_derivative_zz_h.resize(state_size);
    _sigma_derivative_zz_a.resize(get_output_size());

    _ww_hh_updater->init(_delta_ww_hh.rows(), _delta_ww_hh.cols());
    _ww_xh_updater->init(_delta_ww_xh.rows(), _delta_ww_xh.cols());
    _bb_h_updater->init(1, _delta_bb_h.size()); // RowMajor
    _ww_ha_updater->init(_delta_ww_ha.rows(), _delta_ww_ha.cols());
    _bb_a_updater->init(1, _delta_bb_a.size()); // RowMajor
  }

private:
  Matrix _delta_ww_hh;
  Matrix _delta_ww_xh;
  Vector _delta_bb_h;
  Matrix _delta_ww_ha;
  Vector _delta_bb_a;

  Vector _gradient_h;
  Vector _delta_h;
  Vector _delta_a;

  Vector _sigma_derivative_zz_h;
  Vector _sigma_derivative_zz_a;

  unique_ptr<Layer::Updater> _ww_hh_updater;
  unique_ptr<Layer::Updater> _ww_xh_updater;
  unique_ptr<Layer::Updater> _bb_h_updater;
  unique_ptr<Layer::Updater> _ww_ha_updater;
  unique_ptr<Layer::Updater> _bb_a_updater;
}; // class RecurrentLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RecurrentLayer implementation
//
yann::RecurrentLayer::RecurrentLayer(
    const MatrixSize & input_size,
    const MatrixSize & state_size,
    const MatrixSize & output_size) :
    _state_activation_function(new SigmoidFunction()),
    _output_activation_function(new SigmoidFunction())
{
  YANN_CHECK_GT(input_size, 0);
  YANN_CHECK_GT(state_size, 0);
  YANN_CHECK_GT(output_size, 0);

  _ww_hh.resize(state_size, state_size);
  _ww_xh.resize(input_size, state_size);
  _bb_h.resize(state_size);

  _ww_ha.resize(state_size, output_size);
  _bb_a.resize(output_size);
}

yann::RecurrentLayer::~RecurrentLayer()
{
}

void yann::RecurrentLayer::set_values(
    const Matrix & ww_hh, const Matrix & ww_xh, const Vector & bb_h,
    const Matrix & ww_ha, const Vector & bb_a)
{
  YANN_CHECK(is_same_size(ww_hh, _ww_hh));
  YANN_CHECK(is_same_size(ww_xh, _ww_xh));
  YANN_CHECK(is_same_size(bb_h, _bb_h));
  YANN_CHECK(is_same_size(ww_ha, _ww_ha));
  YANN_CHECK(is_same_size(bb_a, _bb_a));
  _ww_hh = ww_hh;
  _ww_xh = ww_xh;
  _bb_h  = bb_h;
  _ww_ha = ww_ha;
  _bb_a  = bb_a;
}

void yann::RecurrentLayer::set_activation_functions(
    const std::unique_ptr<ActivationFunction> & state_activation_function,
    const std::unique_ptr<ActivationFunction> & output_activation_function)
{
  YANN_CHECK(state_activation_function);
  YANN_CHECK(output_activation_function);
  _state_activation_function = state_activation_function->copy();
  _output_activation_function = output_activation_function->copy();
}

// Layer overwrites
bool yann::RecurrentLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_state_activation_function || !_output_activation_function) {
    return false;
  }
  if(_ww_hh.rows() != _ww_hh.cols()) {
    return false;
  }
  if(_ww_xh.cols() != _ww_hh.cols()) {
    return false;
  }
  if(_ww_hh.cols() != _bb_h.size()) {
    return false;
  }
  if(_ww_hh.cols() != _ww_ha.rows()) {
    return false;
  }
  if(_ww_ha.cols() != _bb_a.size()) {
    return false;
  }

  return true;
}

std::string yann::RecurrentLayer::get_name() const
{
  return "RecurrentLayer";
}

string yann::RecurrentLayer::get_info() const
{
  YANN_CHECK(is_valid());

  ostringstream oss;
  oss << Base::get_info()
      << " state size: " << get_state_size()
      << ", state activation: " << _state_activation_function->get_info()
      << ", output activation: " << _output_activation_function->get_info()
      ;
  return oss.str();
}

bool yann::RecurrentLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const RecurrentLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  // TOOD: add deep compare
  if(_state_activation_function->get_info() != the_other->_state_activation_function->get_info()) {
    return false;
  }
  if(_output_activation_function->get_info() != the_other->_output_activation_function->get_info()) {
    return false;
  }

  if(!_ww_hh.isApprox(the_other->_ww_hh, tolerance)) {
    return false;
  }
  if(!_ww_xh.isApprox(the_other->_ww_xh, tolerance)) {
    return false;
  }
  if(!_bb_h.isApprox(the_other->_bb_h, tolerance)) {
    return false;
  }
  if(!_ww_ha.isApprox(the_other->_ww_ha, tolerance)) {
    return false;
  }
  if(!_bb_a.isApprox(the_other->_bb_a, tolerance)) {
    return false;
  }
  return true;
}

MatrixSize yann::RecurrentLayer::get_input_size() const
{
  YANN_SLOW_CHECK(is_valid());
  return _ww_xh.rows();
}

MatrixSize yann::RecurrentLayer::get_state_size() const
{
  YANN_SLOW_CHECK(is_valid());
  return _ww_xh.cols();
}

MatrixSize yann::RecurrentLayer::get_output_size() const
{
  YANN_SLOW_CHECK(is_valid());
  return _ww_ha.cols();
}

unique_ptr<Layer::Context> yann::RecurrentLayer::create_context(const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  return make_unique<RecurrentLayer_Context>(get_state_size(), get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::RecurrentLayer::create_context(const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  return make_unique<RecurrentLayer_Context>(get_state_size(), output);
}
unique_ptr<Layer::Context> yann::RecurrentLayer::create_training_context(
    const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<RecurrentLayer_TrainingContext>(
      get_input_size(), get_state_size(), get_output_size(), batch_size, updater);
}
unique_ptr<Layer::Context> yann::RecurrentLayer::create_training_context(
    const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<RecurrentLayer_TrainingContext>(
      get_input_size(), get_state_size(), output, updater);
}

template<typename InputType>
void yann::RecurrentLayer::feedforward_internal(
    const InputType & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<RecurrentLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_LE(get_batch_size(input) + ctx->_pos, ctx->get_batch_size());
  YANN_CHECK_LE(get_batch_size(input), get_batch_size(ctx->get_output()));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_SLOW_CHECK(is_same_size(ctx->_zz_h, ctx->_hh));
  YANN_SLOW_CHECK(is_same_size(ctx->_zz_a, ctx->get_output()));

  for(MatrixSize ii = 0; ii < get_batch_size(input); ++ii, ++ctx->_pos) {
    // hh(t) = state_activation_function(ww_hh * hh(t-1) + ww_xh*x(t) + b_h)  // state
    auto in = get_batch(input, ii);
    auto out = get_batch(ctx->get_output(), ii);
    auto hh = get_batch(ctx->_hh, ctx->_pos);
    auto zz_h = get_batch(ctx->_zz_h, ctx->_pos);
    auto zz_a = get_batch(ctx->_zz_a, ctx->_pos);

    if(ctx->_pos > 0) {
      auto hh_prev = get_batch(ctx->_hh, ctx->_pos - 1);
      zz_h = MatrixFunctions<InputType>::product(hh_prev, _ww_hh) +
             MatrixFunctions<InputType>::product(in, _ww_xh) +
             _bb_h;
    } else {
      zz_h = MatrixFunctions<InputType>::product(in, _ww_xh) +
            _bb_h;
    }
    _state_activation_function->f(zz_h, hh, Operation_Assign);

    // a(t) = output_activation_function(ww_ha * hh(t) + b_a) // output
    zz_a = MatrixFunctions<InputType>::product(hh, _ww_ha) +
        _bb_a;
    _state_activation_function->f(zz_a, out, mode);
  }
}

void yann::RecurrentLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::RecurrentLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

template<typename InputType>
void yann::RecurrentLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<RecurrentLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_LE(get_batch_size(gradient_output), ctx->_pos);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));

  for(MatrixSize ii = get_batch_size(gradient_output) - 1; ii >= 0 && (--ctx->_pos) >= 0; --ii) {
    auto gradient_out = get_batch(gradient_output, ii);
    auto in = get_batch(input, ii);
    auto hh = get_batch(ctx->_hh, ctx->_pos);
    auto zz_h = get_batch(ctx->_zz_h, ctx->_pos);
    auto zz_a = get_batch(ctx->_zz_a, ctx->_pos);

    //
    // output:
    //
    auto & sigma_derivative_zz_a = ctx->_sigma_derivative_zz_a;
    auto & delta_a = ctx->_delta_a;
    auto & delta_ww_ha = ctx->_delta_ww_ha;
    auto & delta_bb_a = ctx->_delta_bb_a;
    auto & gradient_h = ctx->_gradient_h;

    // delta_a(t)    = elem_prod(gradient(C, a(t)), activation_derivative(zz_a(t)))
    YANN_SLOW_CHECK(is_same_size(zz_a, sigma_derivative_zz_a));
    YANN_SLOW_CHECK(is_same_size(zz_a, gradient_out));
    YANN_SLOW_CHECK(is_same_size(zz_a, delta_a));
    _output_activation_function->derivative(zz_a, sigma_derivative_zz_a);
    delta_a.array() = gradient_out.array() * sigma_derivative_zz_a.array();

    // dC/d b_a(t)   = delta_a(t)
    YANN_SLOW_CHECK(is_same_size(delta_bb_a, delta_a));
    delta_bb_a.noalias()  += delta_a;

    // dC/d ww_ha(t) = hh(t) * delta_a(t)
    YANN_SLOW_CHECK(is_same_size(delta_ww_ha, _ww_ha));
    delta_ww_ha.noalias() += MatrixFunctions<InputType>::product(hh.transpose(), delta_a);

    // total_gradient(C, hh(t)) = gradient(C, hh(t)) + transp(ww_ha) * delta_a(t)
    gradient_h.noalias()  += MatrixFunctions<InputType>::product(delta_a, _ww_ha.transpose());

    //
    // state:
    //
    auto & sigma_derivative_zz_h = ctx->_sigma_derivative_zz_h;
    auto & delta_h = ctx->_delta_h;
    auto & delta_ww_xh = ctx->_delta_ww_xh;
    auto & delta_ww_hh = ctx->_delta_ww_hh;
    auto & delta_bb_h = ctx->_delta_bb_h;

    // delta_h(t) = elem_prod(total_gradient(C, hh(t)), activation_derivative(zz_h(t)))
    YANN_CHECK(is_same_size(zz_h, sigma_derivative_zz_h));
    YANN_CHECK(is_same_size(zz_h, gradient_h));
    _state_activation_function->derivative(zz_h, sigma_derivative_zz_h);
    delta_h.array() = gradient_h.array() * sigma_derivative_zz_h.array();

    // dC/d b_h(t) = delta_h(t)
    YANN_CHECK(is_same_size(delta_bb_h, delta_h));
    delta_bb_h.noalias() += delta_h;

    // dC/d ww_xh(t) = x(t) * delta_h(t)
    YANN_SLOW_CHECK(is_same_size(delta_ww_xh, _ww_xh));
    delta_ww_xh += MatrixFunctions<InputType>::product(in.transpose(), delta_h);

    // dC/d ww_hh(t) = hh(t-1) * delta_h(t)
    if(ctx->_pos > 0) {
      auto prev_hh = get_batch(ctx->_hh, ctx->_pos - 1);
      YANN_SLOW_CHECK(is_same_size(delta_ww_hh, _ww_hh));
      delta_ww_hh.noalias() += MatrixFunctions<InputType>::product(prev_hh.transpose(), delta_h);
    }

    // gradient(C, hh(t-1)) = transp(ww_hh) * delta_h(t)
    if(ctx->_pos > 0) {
      gradient_h.noalias() = MatrixFunctions<InputType>::product(delta_h, _ww_hh.transpose());
    }

    // gradient(C, x(t)) = transp(ww_xh) * delta_h(t)
    if(gradient_input) {
      auto gradient_in = get_batch(*gradient_input, ii);
      gradient_in.noalias() = MatrixFunctions<InputType>::product(delta_h, _ww_xh.transpose());
    }
  }
}

void yann::RecurrentLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  backprop_internal(gradient_output, input, gradient_input, context);
}

void yann::RecurrentLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  // <RefConstSparseMatrix> is required for MatrixFunctions<>::product
  backprop_internal<RefConstSparseMatrix>(gradient_output, input, gradient_input, context);
}

void yann::RecurrentLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  switch (mode) {
  case InitMode_Zeros:
    _ww_hh.setZero();
    _ww_xh.setZero();
    _bb_h.setZero();
    _ww_ha.setZero();
    _bb_a.setZero();
    break;
  case InitMode_Random:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1,
          init_context ? optional<Value>(init_context->seed()) : boost::none);
      gen01->generate(_ww_hh);
      gen01->generate(_ww_xh);
      gen01->generate(_bb_h);
      gen01->generate(_ww_ha);
      gen01->generate(_bb_a);
    }
    break;
  }
}

void yann::RecurrentLayer::update(Context * context, const size_t & batch_size)
{
  auto ctx = dynamic_cast<RecurrentLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_SLOW_CHECK(ctx->_ww_hh_updater);
  YANN_SLOW_CHECK(ctx->_ww_xh_updater);
  YANN_SLOW_CHECK(ctx->_bb_h_updater);
  YANN_SLOW_CHECK(ctx->_ww_ha_updater);
  YANN_SLOW_CHECK(ctx->_bb_a_updater);
  YANN_SLOW_CHECK(is_same_size(_ww_hh, ctx->_delta_ww_hh));
  YANN_SLOW_CHECK(is_same_size(_ww_xh, ctx->_delta_ww_xh));
  YANN_SLOW_CHECK(is_same_size(_bb_h, ctx->_delta_bb_h));
  YANN_SLOW_CHECK(is_same_size(_ww_ha, ctx->_delta_ww_ha));
  YANN_SLOW_CHECK(is_same_size(_bb_a, ctx->_delta_bb_a));

  // set batch_size to 1 since we don't really operate on batches
  ctx->_ww_hh_updater->update(ctx->_delta_ww_hh, 1, _ww_hh);
  ctx->_ww_xh_updater->update(ctx->_delta_ww_xh, 1, _ww_xh);
  ctx->_bb_h_updater->update(ctx->_delta_bb_h, 1, _bb_h);
  ctx->_ww_ha_updater->update(ctx->_delta_ww_ha, 1, _ww_ha);
  ctx->_bb_a_updater->update(ctx->_delta_bb_a, 1, _bb_a);
}

// the format is (wh:<_ww_hh>,wx:<_ww_xh>,bh:<_bb_h>,wa:<_ww_ha>,ba:<_bb_a>)
void yann::RecurrentLayer::read(std::istream & is)
{
  Base::read(is);

  read_char(is, '(');
  read_object(is, "wh", _ww_hh);
  read_char(is, ',');
  read_object(is, "wx", _ww_xh);
  read_char(is, ',');
  read_object(is, "bh", _bb_h);
  read_char(is, ',');
  read_object(is, "wa", _ww_ha);
  read_char(is, ',');
  read_object(is, "ba", _bb_a);
  read_char(is, ')');
}

// the format is (wh:<_ww_hh>,wx:<_ww_xh>,bh:<_bb_h>,wa:<_ww_ha>,ba:<_bb_a>)
void yann::RecurrentLayer::write(std::ostream & os) const
{
  Base::write(os);

  os << "(";
  write_object(os, "wh", _ww_hh);
  os << ",";
  write_object(os, "wx", _ww_xh);
  os << ",";
  write_object(os, "bh", _bb_h);
  os << ",";
  write_object(os, "wa", _ww_ha);
  os << ",";
  write_object(os, "ba", _bb_a);
  os << ")";
}

