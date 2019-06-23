/*
 * polllayer.cpp
 *
 * This is polling layer
 *
 * MAX mode:
 *  Feed-forward:
 *    poll_result(l, i, j) = max(a(l-1, k, l) where k,l in (i, i + filter_size), (j, j + filter_size)
 *
 *  Back propagation:
 *    gradient(C, a(l)) = gradient(C, a(l+1)) for the position of max() for feedforward
 *
 * AVG mode:
 *  Feed-forward:
 *    poll_result(l, i, j) = avg(a(l-1, k, l) where k,l in (i, i + filter_size), (j, j + filter_size)
 *  Back propagation:
 *    gradient(C, a(l)) = gradient(C, a(l+1)) for each cell in submatrix
 *
 * ALL:
 *  Feed-forward:
 *    z(l) = w * poll_result(l) + b
 *    a(l) = activation(z(l))
 *
 * Back propagation (see FullyConnected layer):
 *    delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
 *    dC/db(l) = elem_sum(delta(l))
 *    dC/dw(l) = elem_sum(elem_prod(poll_result(a(l-1) * delta(l)))
 */
#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/random.h"
#include "core/functions.h"
#include "contlayer.h"
#include "polllayer.h"

using namespace std;
using namespace boost;
using namespace yann;

namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// PollingLayer_Context implementation
//
class PollingLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class PollingLayer;

public:
  PollingLayer_Context(const MatrixSize & output_size, const MatrixSize & batch_size) :
    Base(output_size, batch_size)
  {
    _poll_zz.resizeLike(get_output());
    _zz.resizeLike(get_output());
  }
  PollingLayer_Context(const RefVectorBatch & output) :
    Base(output)
  {
    _poll_zz.resizeLike(get_output());
    _zz.resizeLike(get_output());
  }

protected:
  VectorBatch _poll_zz;
  VectorBatch _zz;
}; // class PollingLayer_Context

////////////////////////////////////////////////////////////////////////////////////////////////
//
// PollingLayer_TrainingContext implementation
//
class PollingLayer_TrainingContext :
  public PollingLayer_Context
{
  typedef PollingLayer_Context Base;

  friend class PollingLayer;

public:
  PollingLayer_TrainingContext(
      const MatrixSize & output_size,
      const MatrixSize & batch_size,
      const unique_ptr<Layer::Updater> & updater) :
    Base(output_size, batch_size),
    _delta_ww(0),
    _delta_bb(0),
    _ww_updater(updater->copy()),
    _bb_updater(updater->copy())
  {
    _delta.resizeLike(_zz);
    _sigma_derivative_zz.resizeLike(_zz);

    _ww_updater->init(1, 1);
    _bb_updater->init(1, 1);
  }

  PollingLayer_TrainingContext(
     const RefVectorBatch & output,
     const unique_ptr<Layer::Updater> & updater) :
    Base(output),
    _delta_ww(0),
    _delta_bb(0),
    _ww_updater(updater->copy()),
    _bb_updater(updater->copy())
  {
    _delta.resizeLike(_zz);
    _sigma_derivative_zz.resizeLike(_zz);

    _ww_updater->init(1, 1);
    _bb_updater->init(1, 1);
  }

  // Layer::Context  overwrites
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

    _delta_ww = 0;
    _delta_bb = 0;

    _ww_updater->reset();
    _bb_updater->reset();
  }

private:
  Value _delta_ww;
  Value _delta_bb;

  VectorBatch _delta;
  VectorBatch _sigma_derivative_zz;

  unique_ptr<Layer::Updater> _ww_updater;
  unique_ptr<Layer::Updater> _bb_updater;
}; // class PollingLayer_TrainingContext

}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::PollingLayer implementation
//
MatrixSize yann::PollingLayer::get_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_size)
{
  return input_rows / filter_size;
}
MatrixSize yann::PollingLayer::get_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_size)
{
  return input_cols / filter_size;
}
MatrixSize yann::PollingLayer::get_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                  const MatrixSize & filter_size)
{
  return get_output_rows(input_rows, filter_size) * get_output_cols(input_cols, filter_size);
}

// Polling layer: each conv layer sends output to the corresponding poll layer
unique_ptr<ParallelLayer> yann::PollingLayer::create_poll_parallel_layer(
    const size_t & frames_num,
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const MatrixSize & filter_size,
    enum Mode mode,
    const std::unique_ptr<ActivationFunction> & activation_function)
{
  YANN_CHECK_GT(frames_num, 0);

  auto poll_container_layer = make_unique<ParallelLayer>(frames_num);
  YANN_CHECK(poll_container_layer);
  for(auto ii = frames_num; ii > 0; --ii) {
    auto poll_layer = make_unique<PollingLayer>(
        input_rows,
        input_cols,
        filter_size,
        mode);
    YANN_CHECK(poll_layer);
    if(activation_function) {
      poll_layer->set_activation_function(activation_function);
    }

    poll_container_layer->append_layer(std::move(poll_layer));
  }

  return poll_container_layer;
}

yann::PollingLayer::PollingLayer(const MatrixSize & input_rows,
                                 const MatrixSize & input_cols,
                                 const MatrixSize & filter_size,
                                 enum Mode mode) :
    _input_rows(input_rows),
    _input_cols(input_cols),
    _filter_size(filter_size),
    _mode(mode),
    _ww(0),
    _bb(0),
    _activation_function(new SigmoidFunction())
{
  YANN_CHECK_GE(input_rows, filter_size);
  YANN_CHECK_GE(input_cols, filter_size);
}

yann::PollingLayer::~PollingLayer()
{
}

void yann::PollingLayer::set_activation_function(
    const unique_ptr<ActivationFunction> & activation_function)
{
  YANN_CHECK(activation_function);
  _activation_function = activation_function->copy();
}

void yann::PollingLayer::set_values(
    const Value & ww, const Value & bb)
{
  _ww = ww;
  _bb = bb;
}

MatrixSize yann::PollingLayer::get_output_rows() const
{
  return get_output_rows(_input_rows, _filter_size);
}

MatrixSize yann::PollingLayer::get_output_cols() const
{
  return get_output_cols(_input_cols, _filter_size);
}

// Layer overwrites
std::string yann::PollingLayer::get_name() const
{
  return "PollingLayer";
}

string yann::PollingLayer::get_info() const
{
  ostringstream oss;

  oss << Base::get_info();

  // assume that all layers are the same, print only one
  oss << " activation: " << _activation_function->get_info()
      << ", input rows: " << _input_rows
      << ", input cols: " << _input_cols
      << ", filter: " << _filter_size
      << ", output rows: " << get_output_rows()
      << ", output cols: " << get_output_cols()
  ;
  switch(_mode) {
  case PollMode_Max:
    oss << ", mode: Max";
    break;
  case PollMode_Avg:
    oss << ", mode: Avg";
    break;
  }
  return oss.str();
}

bool yann::PollingLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_activation_function) {
    return false;
  }
  return true;
}

bool yann::PollingLayer::is_equal(const Layer & other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto the_other = dynamic_cast<const PollingLayer*>(&other);
  if(the_other == nullptr) {
    return false;
  }
  if(_input_rows != the_other->_input_rows) {
    return false;
  }
  if(_input_cols != the_other->_input_cols) {
    return false;
  }
  if(_filter_size != the_other->_filter_size) {
    return false;
  }
  if(_mode != the_other->_mode) {
    return false;
  }
  // TOOD: add deep compare
  if(_activation_function->get_info() != the_other->_activation_function->get_info()) {
    return false;
  }
  if(fabs(_ww - the_other->_ww) >= tolerance) {
    return false;
  }
  if(fabs(_bb - the_other->_bb) >= tolerance) {
    return false;
  }
  return true;
}

MatrixSize yann::PollingLayer::get_input_size() const
{
  return _input_rows * _input_cols;
}

MatrixSize yann::PollingLayer::get_output_size() const
{
  return get_output_size(_input_rows, _input_cols, _filter_size);
}

unique_ptr<Layer::Context> yann::PollingLayer::create_context(
    const MatrixSize & batch_size) const
{
  YANN_CHECK(is_valid());
  return make_unique<PollingLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_context(
    const RefVectorBatch & output) const
{
  YANN_CHECK(is_valid());
  return make_unique<PollingLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_training_context(
    const MatrixSize & batch_size,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  return make_unique<PollingLayer_TrainingContext>(
      get_output_size(), batch_size, updater);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_training_context(
    const RefVectorBatch & output,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  YANN_CHECK(is_valid());
  return make_unique<PollingLayer_TrainingContext>(output, updater);
}

void yann::PollingLayer::poll(
    const RefConstMatrix & input,
    const MatrixSize & filter_size,
    enum Mode mode,
    RefMatrix output)
{
  YANN_CHECK_GT(filter_size, 0);
  YANN_CHECK_EQ(output.rows(), get_output_rows(input.rows(), filter_size));
  YANN_CHECK_EQ(output.cols(), get_output_cols(input.cols(), filter_size));

  // Rows/cols order should change if matrix layout is change from RowMajor to ColMajor for perf reasons
  for(MatrixSize in_ii = 0, out_ii = 0; out_ii < output.rows(); in_ii += filter_size, ++out_ii) {
    for(MatrixSize in_jj = 0, out_jj = 0; out_jj < output.cols(); in_jj += filter_size, ++out_jj) {
      switch(mode) {
        case PollMode_Max:
          output(out_ii, out_jj) = input.block(in_ii, in_jj, filter_size, filter_size).maxCoeff();
          break;
        case PollMode_Avg:
          output(out_ii, out_jj) = input.block(in_ii, in_jj, filter_size, filter_size).sum() / (filter_size * filter_size);
          break;
      }
    }
  }
}

void yann::PollingLayer::poll(
    const RefConstVectorBatch & input,
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const MatrixSize & filter_size,
    enum Mode mode,
    RefVectorBatch output)
{
  YANN_CHECK_GT(filter_size, 0);
  YANN_CHECK_LE(filter_size, input_rows);
  YANN_CHECK_LE(filter_size, input_cols);
  YANN_CHECK_GT(get_batch_size(input), 0);
  YANN_CHECK_EQ(get_batch_item_size(input), input_rows * input_cols);
  YANN_CHECK_EQ(get_batch_size(output), get_batch_size(input));

  const auto batch_size = get_batch_size(input);
  const auto output_rows = get_output_rows(input_rows, filter_size);
  const auto output_cols = get_output_cols(input_cols, filter_size);
  YANN_CHECK_EQ(get_batch_item_size(output), output_rows * output_cols);

  // ATTENTION: this code operates on raw Matrix.data() and might be broken
  // if Matrix.data() layout changes
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    MapConstMatrix in(get_batch(input, ii).data(), input_rows, input_cols);
    MapMatrix out(get_batch(output, ii).data(), output_rows, output_cols);
    poll(in, filter_size, mode, out);
  }
}

//
// MAX mode:
//   gradient(C, a(l)) = gradient(C, a(l+1)) for the position of max() for feedforward
//
// AVG mode:
//   gradient(C, a(l)) = gradient(C, a(l+1)) for each cell in sub-matrix
void yann::PollingLayer::poll_gradient_backprop(
    const RefConstMatrix & gradient_output,
    const RefConstVectorBatch & input,
    const MatrixSize & filter_size,
    enum Mode mode,
    RefMatrix gradient_input)
{
  YANN_CHECK_EQ(gradient_output.rows(), get_output_rows(input.rows(), filter_size));
  YANN_CHECK_EQ(gradient_output.cols(), get_output_cols(input.cols(), filter_size));
  YANN_CHECK(is_same_size(gradient_input, input));

  // Rows/cols order should change if matrix layout is change from RowMajor to ColMajor for perf reasons
  MatrixSize ii, jj;
  for(MatrixSize in_ii = 0, out_ii = 0; out_ii < gradient_output.rows(); in_ii += filter_size, ++out_ii) {
    for(MatrixSize in_jj = 0, out_jj = 0; out_jj < gradient_output.cols(); in_jj += filter_size, ++out_jj) {
      Value val = gradient_output(out_ii, out_jj);
      switch(mode) {
        case PollMode_Max:
          input.block(in_ii, in_jj, filter_size, filter_size).maxCoeff(&ii, &jj);
          gradient_input(in_ii + ii, in_jj + jj) += val;
          break;
        case PollMode_Avg:
          gradient_input.block(in_ii, in_jj, filter_size, filter_size).array() += val;
          break;
      }
    }
  }
}

void yann::PollingLayer::poll_gradient_backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const MatrixSize & filter_size,
    enum Mode mode,
    RefVectorBatch gradient_input)
{
  YANN_CHECK(is_same_size(input, gradient_input));
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(gradient_output));

  const auto batch_size = get_batch_size(input);
  const auto output_rows = get_output_rows(input_rows, filter_size);
  const auto output_cols = get_output_cols(input_cols, filter_size);

  gradient_input.setZero();
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    MapMatrix gradient_in(get_batch(gradient_input, ii).data(), input_rows, input_cols);
    MapConstMatrix in(get_batch(input, ii).data(), input_rows, input_cols);
    MapConstMatrix gradient_out(get_batch(gradient_output, ii).data(), output_rows, output_cols);
    poll_gradient_backprop(gradient_out, in, filter_size, mode, gradient_in);
  }
}

void yann::PollingLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  auto ctx = dynamic_cast<PollingLayer_Context *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  BOOST_VERIFY(is_same_size(ctx->_zz, ctx->get_output()));

  // z(l) = poll(a(l-1))*w(l) + b(l)
  poll(input, _input_rows, _input_cols, _filter_size, _mode,  ctx->_poll_zz);
  ctx->_zz.array() = _ww *  ctx->_poll_zz.array() + _bb;

  // a(l) = activation(z(l))
  _activation_function->f(ctx->_zz, ctx->get_output(), mode);
}

void yann::PollingLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  throw runtime_error("PollingLayer::feedforward() is not implemented for sparse vectors");
}

void yann::PollingLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  auto ctx = dynamic_cast<PollingLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(is_valid());
  YANN_CHECK_GT(get_batch_size(gradient_output), 0);
  YANN_CHECK_EQ(get_batch_item_size(gradient_output), get_output_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());
  YANN_CHECK_EQ(get_batch_item_size(input), get_input_size());

  // just to make it easier to read
  const auto & poll_zz = ctx->_poll_zz;
  const auto & zz = ctx->_zz;
  auto & sigma_derivative_zz = ctx->_sigma_derivative_zz;
  auto & delta = ctx->_delta;
  auto & delta_ww = ctx->_delta_ww;
  auto & delta_bb = ctx->_delta_bb;

  // delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
  YANN_CHECK(is_same_size(zz, sigma_derivative_zz));
  YANN_CHECK(is_same_size(zz, gradient_output));
  YANN_CHECK(is_same_size(zz, delta));
  _activation_function->derivative(zz, sigma_derivative_zz);
  delta.array() = gradient_output.array() * sigma_derivative_zz.array();

  // update deltas
  // dC/dw(l) = elem_sum(elem_prod(poll_result(a(l-1)) * delta(l)))
  // dC/db(l) = elem_sum(delta(l))
  YANN_CHECK(is_same_size(poll_zz, delta));
  delta_ww += (poll_zz.array() * delta.array()).sum();
  delta_bb += delta.array().sum();

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  if(gradient_input) {
    poll_gradient_backprop(
        gradient_output, input, _input_rows, _input_cols,
        _filter_size, _mode, *gradient_input);
  }
}

void yann::PollingLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  throw runtime_error("PollingLayer::backprop() is not implemented for sparse vectors");
}

void yann::PollingLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  switch (mode) {
  case InitMode_Zeros:
    _ww = 0;
    _bb = 0;
    break;
  case InitMode_Random:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1,
          init_context ? optional<Value>(init_context->seed()) : boost::none);
      gen01->generate(_ww);
      gen01->generate(_bb);
    }
    break;
  }
}

void yann::PollingLayer::update(Context * context, const size_t & batch_size)
{
  auto ctx = dynamic_cast<PollingLayer_TrainingContext *>(context);
  YANN_CHECK(ctx);
  YANN_CHECK(ctx->_ww_updater);
  YANN_CHECK(ctx->_bb_updater);

  ctx->_ww_updater->update(ctx->_delta_ww, batch_size, _ww);
  ctx->_bb_updater->update(ctx->_delta_bb, batch_size, _bb);
}

// the format is (w:<weight>,b:<bias>)
void yann::PollingLayer::read(std::istream & is)
{
  Base::read(is);

  read_char(is, '(');
  read_object(is, "w", _ww);
  read_char(is, ',');
  read_object(is, "b", _bb);
  read_char(is, ')');
}

// the format is (w:<weight>,b:<bias>)
void yann::PollingLayer::write(std::ostream & os) const
{
  Base::write(os);

  os << "(";
  write_object(os, "w", _ww);
  os << ",";
  write_object(os, "b", _bb);
  os << ")";
}

