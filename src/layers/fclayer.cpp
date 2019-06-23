/*
 * fclayer.cpp
 *
 * Feed-forward:
 *    z(l) = a(l-1) * w + b // where * (vector, matrix) multiplication
 *    a(l) = activation(z(l))
 *
 * Back propagation:
 *    delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
 *    dC/db(l) = delta(l)
 *    dC/dw(l) = a(l-1) * delta(l)
 *    gradient(C, a(l - 1)) = transp(w) * delta(l)
 */
#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/random.h"
#include "core/functions.h"
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

  typedef vector<pair<MatrixSize, Value>> SamplingSortingVector;
  typedef vector<MatrixSize> SamplingCounterVector;

  friend class FullyConnectedLayer;

public:
  FullyConnectedLayer_TrainingContext(
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const MatrixSize & batch_size,
      const unique_ptr<Layer::Updater> & updater,
      bool is_sampled) :
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

    _ww_updater->init(input_size, get_output_size());
    _bb_updater->init(1, get_output_size()); // RowMajor

    if(is_sampled) {
      _sampling_gradient_out_sorting = SamplingSortingVector(get_output_size());
      _sampling_counter = SamplingCounterVector(get_output_size(), 0);
    }
  }

  FullyConnectedLayer_TrainingContext(
      const MatrixSize & input_size,
      const RefVectorBatch & output,
      const unique_ptr<Layer::Updater> & updater,
      bool is_sampled) :
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

    _ww_updater->init(input_size, get_output_size());
    _bb_updater->init(1, get_output_size()); // RowMajor

    if(is_sampled) {
      _sampling_gradient_out_sorting = SamplingSortingVector(get_output_size());
      _sampling_counter = SamplingCounterVector(get_output_size(), 0);
    }
  }

  // Layer::Context overwrites
  virtual void start_epoch()
  {
    YANN_SLOW_CHECK(_ww_updater);
    YANN_SLOW_CHECK(_bb_updater);

    Base::start_epoch();

    _ww_updater->start_epoch();
    _bb_updater->start_epoch();
  }

  virtual void reset_state()
  {
    YANN_SLOW_CHECK(_ww_updater);
    YANN_SLOW_CHECK(_bb_updater);

    Base::reset_state();

    _delta_ww.setZero();
    _delta_bb.setZero();

    _ww_updater->reset();
    _bb_updater->reset();

    if(_sampling_counter) {
      fill(_sampling_counter->begin(), _sampling_counter->end(), 0);
    }
  }

private:
  Matrix _delta_ww;
  Vector _delta_bb;
  VectorBatch _delta;

  VectorBatch _sigma_derivative_zz;
  Vector _collapse_vector;

  unique_ptr<Layer::Updater> _ww_updater;
  unique_ptr<Layer::Updater> _bb_updater;

  // sampling support
  optional<SamplingSortingVector> _sampling_gradient_out_sorting;
  optional<SamplingCounterVector> _sampling_counter;
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
    _fixed_bias(false),
    _sampling_rate(2.0), // bigger than 1.0
    _activation_function(new SigmoidFunction())
{
}

yann::FullyConnectedLayer::~FullyConnectedLayer()
{
}

void yann::FullyConnectedLayer::set_activation_function(
    const unique_ptr<ActivationFunction> & activation_function)
{
  YANN_CHECK(activation_function);
  _activation_function = activation_function->copy();
}

void yann::FullyConnectedLayer::set_values(const Matrix & ww, const Vector & bb)
{
  YANN_CHECK(is_same_size(ww, _ww));
  YANN_CHECK(is_same_size(bb, _bb));
  _ww = ww;
  _bb = bb;
}

void yann::FullyConnectedLayer::set_fixed_bias(const Value & val)
{
  _bb.setConstant(val);
  _fixed_bias = true;
}

void yann::FullyConnectedLayer::set_sampling_rate(const double & sampling_rate)
{
  YANN_CHECK_GE(sampling_rate, 0.0);
  YANN_CHECK_LE(sampling_rate, 1.0);
  _sampling_rate = sampling_rate;
}

bool yann::FullyConnectedLayer::is_sampled() const
{
  return 0 < _sampling_rate && _sampling_rate < 1.0;
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
  if(_ww.cols() != _bb.size()) {
    return false;
  }
  return true;
}

std::string yann::FullyConnectedLayer::get_name() const
{
  return "FullyConnectedLayer";
}

string yann::FullyConnectedLayer::get_info() const
{
  YANN_CHECK(is_valid());

  ostringstream oss;
  oss << Base::get_info()
      << " activation: " << _activation_function->get_info()
      ;
  return oss.str();
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
  if(_activation_function->get_info() != the_other->_activation_function->get_info()) {
    return false;
  }
  if(_fixed_bias != the_other->_fixed_bias) {
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
      get_input_size(), get_output_size(), batch_size, updater, is_sampled());
}
unique_ptr<Layer::Context> yann::FullyConnectedLayer::create_training_context(
    const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  YANN_CHECK(updater);
  return make_unique<FullyConnectedLayer_TrainingContext>(
      get_input_size(), output, updater, is_sampled());
}

template<typename InputType>
void yann::FullyConnectedLayer::feedforward_internal(
    const InputType & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<FullyConnectedLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_EQ(get_batch_size(input), ctx->get_batch_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  // z(l) = a(l-1)*w + b
  ctx->_zz.noalias() = MatrixFunctions<InputType>::product(input, _ww);
  plus_batches(ctx->_zz, _bb);

  // a(l) = activation(z(l))
  YANN_CHECK(is_same_size(ctx->_zz, ctx->get_output()));
  _activation_function->f(ctx->_zz, ctx->get_output(), mode);
}

void yann::FullyConnectedLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

void yann::FullyConnectedLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  feedforward_internal(input, context, mode);
}

template<typename InputType>
void yann::FullyConnectedLayer::backprop_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
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
  delta_ww.noalias() += MatrixFunctions<InputType>::product(input.transpose(), delta);
  if(!_fixed_bias) {
    delta_bb.noalias() += MatrixFunctions<InputType>::product(collapse_vector, delta);
  }

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  // gradient(C, a(l - 1)) = transp(w) * delta(l)
  if(gradient_input) {
    (*gradient_input).noalias() = MatrixFunctions<InputType>::product(delta, _ww.transpose());
  }
}


template<typename InputType>
void yann::FullyConnectedLayer::backprop_with_sampling_internal(
    const RefConstVectorBatch & gradient_output,
    const InputType & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<FullyConnectedLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->_sampling_gradient_out_sorting);
  YANN_CHECK(ctx->_sampling_counter);
  YANN_CHECK(is_valid());
  YANN_CHECK(is_sampled());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK(!gradient_input || is_same_size(input, *gradient_input));

  // one row at a time
  const auto batch_size = get_batch_size(input);
  const auto output_size = get_output_size();
  const MatrixSize sampled_num = get_output_size() * _sampling_rate;
  auto & delta_ww = ctx->_delta_ww;
  auto & delta_bb = ctx->_delta_bb;
  YANN_SLOW_CHECK_EQ(delta_bb.size(), get_output_size());
  auto & sampling_gradient_out_sorting = *(ctx->_sampling_gradient_out_sorting);
  auto & sampling_counter = *(ctx->_sampling_counter);

  YANN_SLOW_CHECK(is_same_size(delta_ww, _ww));
  YANN_SLOW_CHECK(is_same_size(delta_bb, _bb));

  if(gradient_input) {
    gradient_input->setZero();
  }

  for(MatrixSize batch_pos = 0; batch_pos < batch_size; ++batch_pos) {
    // just to make it easier to read
    const auto in = get_batch(input, batch_pos);
    const auto gradient_out = get_batch(gradient_output, batch_pos);
    const auto zz = get_batch(ctx->_zz, batch_pos);
    const auto sigma_derivative_zz = get_batch(ctx->_sigma_derivative_zz, batch_pos);

    YANN_CHECK(is_same_size(zz, sigma_derivative_zz));
    YANN_CHECK(is_same_size(zz, gradient_out));

    // prepare sampling
    for(MatrixSize ii = 0, out_size = output_size; ii < out_size; ++ii) {
      sampling_gradient_out_sorting[ii].first = ii;
      sampling_gradient_out_sorting[ii].second = std::abs(gradient_out(ii));
    }
    sort(sampling_gradient_out_sorting.begin(), sampling_gradient_out_sorting.end(),
        [](const auto & left, const auto & right) -> bool
        {
          return left.second > right.second; // sort in descending order
        }
    );
    // TODO: we don't need to calculate it for all values
    _activation_function->derivative(zz, sigma_derivative_zz);

    for(MatrixSize ii = 0; ii < sampled_num; ++ii) {
      const auto out_pos = sampling_gradient_out_sorting[ii].first;
      YANN_SLOW_CHECK_GE(out_pos, 0);
      YANN_SLOW_CHECK_LT(out_pos, output_size);
      ++sampling_counter[out_pos];

      // delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
      const auto delta = gradient_out(out_pos) * sigma_derivative_zz(out_pos);

      // update deltas
      // dC/db(l) = delta(l)
      // dC/dw(l) = a(l-1) * delta(l)
      delta_ww.col(out_pos) += in * delta;
      if(!_fixed_bias) {
        delta_bb(out_pos) += delta;
      }
      // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
      // // gradient(C, a(l - 1)) = transp(w) * delta(l)
      if(gradient_input) {
        auto gradient_in = get_batch(*gradient_input, batch_pos);
        gradient_in.noalias() += delta * _ww.col(out_pos);
      }
    }
  }
}

void yann::FullyConnectedLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  if(is_sampled()) {
    backprop_with_sampling_internal(gradient_output, input, gradient_input, context);
  } else {
    backprop_internal(gradient_output, input, gradient_input, context);
  }
}

void yann::FullyConnectedLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  // <RefConstSparseMatrix> is required for MatrixFunctions<>::product
  if(is_sampled()) {
    backprop_with_sampling_internal<RefConstSparseMatrix>(gradient_output, input, gradient_input, context);
  } else {
    backprop_internal<RefConstSparseMatrix>(gradient_output, input, gradient_input, context);
  }
}

void yann::FullyConnectedLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  switch (mode) {
  case InitMode_Zeros:
    _ww.setZero();
    if(!_fixed_bias) {
      _bb.setZero();
    }
    break;
  case InitMode_Random:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1,
          init_context ? optional<Value>(init_context->seed()) : boost::none);
      gen01->generate(_ww);
      if(!_fixed_bias) {
        gen01->generate(_bb);
      }
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

  if(!is_sampled()) {
    ctx->_ww_updater->update(ctx->_delta_ww, batch_size, _ww);
    if(!_fixed_bias) {
      ctx->_bb_updater->update(ctx->_delta_bb, batch_size, _bb);
    }
  } else {
    const auto & sampling_counter = *(ctx->_sampling_counter);
    const auto & delta_ww = ctx->_delta_ww;
    const auto & delta_bb = ctx->_delta_bb;
    const auto rows = delta_ww.rows();

    YANN_CHECK_EQ((MatrixSize)sampling_counter.size(), delta_ww.cols());
    for(MatrixSize ii = 0; ii < delta_ww.cols(); ++ii) {
      if(sampling_counter[ii] <= 0) {
        continue;
      }
      // can't use delta_ww.col(ii) and _ww.col(ii) here
      ctx->_ww_updater->update(
          delta_ww.block(0, ii, rows, 1),
          sampling_counter[ii],
          _ww.block(0, ii, rows, 1));
      if(!_fixed_bias) {
        ctx->_bb_updater->update(delta_bb(ii), sampling_counter[ii], _bb(ii));
      }
    }
  }
}

// the format is (w:<weights>,b:<biases>)
void yann::FullyConnectedLayer::read(std::istream & is)
{
  Base::read(is);

  read_char(is, '(');
  read_object(is, "w", _ww);
  read_char(is, ',');
  read_object(is, "b", _bb);
  read_char(is, ')');
}

// the format is (w:<weights>,b:<biases>)
void yann::FullyConnectedLayer::write(std::ostream & os) const
{
  Base::write(os);

  os << "(";
  write_object(os, "w", _ww);
  os << ",";
  write_object(os, "b", _bb);
  os << ")";
}

