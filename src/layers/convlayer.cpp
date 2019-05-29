/*
 * convlayer.cpp
 *
 * Feed-forward:
 *    z(l) = a(l-1) * w(l) + b(l) // where * is conv operation and b(l) is a scalar
 *    a(l) = activation(z(l))
 *
 * Back propagation:
 *    gradient(C, a(L)) = gradient(cost(a(L), a(expected)), a(L))
 *    gradient(C, a(l-1)) = full_conv(delta(l), rotate180(w(l)))
 *
 *    delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
 *    dC/dw(l) = conv(a(l), delta(l + 1))
 *    dC/db(l) = sum_elem(delta(l + 1))
 */
#include <boost/assert.hpp>

#include "utils.h"
#include "random.h"
#include "functions.h"
#include "contlayer.h"
#include "convlayer.h"

using namespace std;
using namespace boost;
using namespace yann;

namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// ConvolutionalLayerContext implementation
//
class ConvolutionalLayer_Context :
    public Layer::Context
{
  typedef Layer::Context Base;

  friend class ConvolutionalLayer;

public:
  ConvolutionalLayer_Context(const MatrixSize & output_size, const MatrixSize & batch_size) :
    Base(output_size, batch_size)
  {
    _zz.resizeLike(get_output());
  }
  ConvolutionalLayer_Context(const RefVectorBatch & output) :
    Base(output)
  {
    _zz.resizeLike(get_output());
  }

protected:
  VectorBatch _zz;
}; // class ConvolutionalLayer_Context


////////////////////////////////////////////////////////////////////////////////////////////////
//
// ConvolutionalLayer_TrainingContext implementation
//
class ConvolutionalLayer_TrainingContext :
  public ConvolutionalLayer_Context
{
  typedef ConvolutionalLayer_Context Base;

  friend class ConvolutionalLayer;

public:
  ConvolutionalLayer_TrainingContext(const MatrixSize & filter_size,
                                     const MatrixSize & output_size,
                                     const MatrixSize & batch_size) :
    Base(output_size, batch_size),
    _ww_rotated(filter_size, filter_size),
    _delta_ww(filter_size, filter_size),
    _delta_bb(0)
  {
    BOOST_VERIFY(filter_size > 0);
    BOOST_VERIFY(output_size > 0);
    BOOST_VERIFY(batch_size > 0);

    _delta.resizeLike(_zz);
    _sigma_derivative_zz.resizeLike(_zz);
  }
  ConvolutionalLayer_TrainingContext(const MatrixSize & filter_size,
                                     const RefVectorBatch & output) :
    Base(output),
    _ww_rotated(filter_size, filter_size),
    _delta_ww(filter_size, filter_size),
    _delta_bb(0)
  {
    BOOST_VERIFY(filter_size > 0);
    BOOST_VERIFY(yann::get_batch_size(output) > 0);
    BOOST_VERIFY(yann::get_batch_item_size(output) > 0);

    _delta.resizeLike(_zz);
    _sigma_derivative_zz.resizeLike(_zz);
  }

  // Layer::Context  overwrites
  virtual void reset_state()
  {
    _delta_ww.setZero();
    _delta_bb = 0;
  }

private:
  Matrix _ww_rotated;
  Matrix _delta_ww;
  Value  _delta_bb;

  VectorBatch _delta;
  VectorBatch _sigma_derivative_zz;
}; // class ConvolutionalLayer_TrainingContext


}; // namespace yann

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::ConvolutionalLayer implementation
//
MatrixSize yann::ConvolutionalLayer::get_input_size(const MatrixSize & input_rows, const MatrixSize & input_cols)
{
  return input_rows * input_cols;
}

MatrixSize yann::ConvolutionalLayer::get_conv_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_rows)
{
  return (input_rows - filter_rows + 1);
}
MatrixSize yann::ConvolutionalLayer::get_conv_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_cols)
{
  return (input_cols - filter_cols + 1);
}
MatrixSize yann::ConvolutionalLayer::get_conv_output_size(
    const MatrixSize & input_rows, const MatrixSize & input_cols,
    const MatrixSize & filter_rows, const MatrixSize & filter_cols)
{
  return get_conv_output_rows(input_rows, filter_rows)
      * get_conv_output_cols(input_cols, filter_cols);
}
MatrixSize yann::ConvolutionalLayer::get_conv_output_size(
    const MatrixSize & input_rows, const MatrixSize & input_cols,
    const MatrixSize & filter_size)
{
  return get_conv_output_size(input_rows, input_cols, filter_size, filter_size);
}

MatrixSize yann::ConvolutionalLayer::get_full_conv_output_rows(
    const MatrixSize & input_rows, const MatrixSize & filter_rows)
{
  return (input_rows + filter_rows - 1);
}
MatrixSize yann::ConvolutionalLayer::get_full_conv_output_cols(
    const MatrixSize & input_cols, const MatrixSize & filter_cols)
{
  return (input_cols + filter_cols - 1);
}
MatrixSize yann::ConvolutionalLayer::get_full_conv_output_size(
    const MatrixSize & input_rows, const MatrixSize & input_cols,
    const MatrixSize & filter_rows, const MatrixSize & filter_cols)
{
  return get_full_conv_output_rows(input_rows, filter_rows)
      * get_full_conv_output_cols(input_cols, filter_cols);
}
MatrixSize yann::ConvolutionalLayer::get_full_conv_output_size(
    const MatrixSize & input_rows, const MatrixSize & input_cols,
    const MatrixSize & filter_size)
{
  return get_full_conv_output_size(input_rows, input_cols, filter_size,
                                   filter_size);
}

void yann::ConvolutionalLayer::plus_conv(const RefConstMatrix & input,
                                         const RefConstMatrix & filter,
                                         RefMatrix output,
                                         bool clear_output)
{
  BOOST_VERIFY(filter.rows() > 0);
  BOOST_VERIFY(filter.cols() > 0);
  BOOST_VERIFY(filter.rows() <= input.rows());
  BOOST_VERIFY(filter.cols() <= input.cols());

  const auto input_rows = input.rows();
  const auto filter_rows = filter.rows();
  const auto filter_cols = filter.cols();
  const auto output_rows = get_conv_output_rows(input.rows(), filter_rows);
  const auto output_cols = get_conv_output_cols(input.cols(), filter_cols);

  if(clear_output) {
    output.setZero();
  }

  BOOST_VERIFY(output.rows() == output_rows);
  BOOST_VERIFY(output.cols() == output_cols);

  // iterate through the input by rows
  MatrixSize filter_start_row = 0, filter_max_rows = 0;
  for(MatrixSize ii = 0; ii < input_rows; ++ii) {
    const auto in_row = input.row(ii);
    if(ii < filter_rows) {
      // filter at the top of the input; bottom filter rows
      // don't contribute to the output
      ++filter_max_rows;
    }
    if(ii >= output_rows) {
      // filter at the bottom of the input; top filter rows
      // don't contribute to the output
      ++filter_start_row;
    }

    // and now through each row
    for(MatrixSize jj = 0; jj < output_cols; ++jj) {
      const auto in_block = in_row.segment(jj, filter_cols);

      // then through the filter by rows, account for "cutoff" at bottom rows
      for(MatrixSize kk = filter_start_row; kk < filter_max_rows; ++kk) {
        output(ii - kk, jj) += (in_block.array() * filter.row(kk).array()).sum();
      }
    }
  }
}

void yann::ConvolutionalLayer::plus_conv(const RefConstVectorBatch & input,
                                    const MatrixSize & input_rows,
                                    const MatrixSize & input_cols,
                                    const RefConstMatrix & filter,
                                    RefVectorBatch output,
                                    bool clear_output)
{
  BOOST_VERIFY(filter.rows() > 0);
  BOOST_VERIFY(filter.cols() > 0);
  BOOST_VERIFY(filter.rows() <= input_rows);
  BOOST_VERIFY(filter.cols() <= input_cols);
  BOOST_VERIFY(get_batch_size(input) > 0);
  BOOST_VERIFY(get_batch_item_size(input) == input_rows * input_cols);
  BOOST_VERIFY(get_batch_size(output) == get_batch_size(input));

  const auto filter_rows = filter.rows();
  const auto filter_cols = filter.cols();
  const auto output_rows = get_conv_output_rows(input_rows, filter_rows);
  const auto output_cols = get_conv_output_cols(input_cols, filter_cols);
  const auto batch_size = get_batch_size(input);
  BOOST_VERIFY(get_batch_item_size(output) == output_rows * output_cols);

  if(clear_output) {
    output.setZero();
  }

  // ATTENTION: this code operates on raw Matrix.data() and might be broken
  // if Matrix.data() layout changes
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    const auto in_batch = get_batch(input, ii);
    auto out_batch = get_batch(output, ii);
    MapConstMatrix in(in_batch.data(), input_rows, input_cols);
    MapMatrix out(out_batch.data(), output_rows, output_cols);
    plus_conv(in, filter, out, false); // we already cleared output if needed
  }
}

void yann::ConvolutionalLayer::full_conv(const RefConstMatrix & input, const RefConstMatrix & filter, RefMatrix output)
{
  BOOST_VERIFY(filter.rows() > 0);
  BOOST_VERIFY(filter.cols() > 0);
  BOOST_VERIFY(filter.rows() <= input.rows());
  BOOST_VERIFY(filter.cols() <= input.cols());

  const auto input_rows = input.rows();
  const auto input_cols = input.cols();
  const auto filter_rows = filter.rows();
  const auto filter_cols = filter.cols();
  const auto output_rows = get_full_conv_output_rows(input_rows, filter_rows);
  const auto output_cols = get_full_conv_output_cols(input_cols, filter_cols);

  BOOST_VERIFY(output.rows() == output_rows);
  BOOST_VERIFY(output.cols() == output_cols);

  // clear output
  output.setZero();

  MatrixSize input_row, output_col, output_last_row;
  auto apply_filter = [&](const auto & in_block, const auto & filter_block) mutable {
    // iterate through the filter by rows, account for "cutoff" at bottom rows
    for(MatrixSize kk = 0; kk < filter_rows; ++kk) {
      output(output_last_row - kk, output_col) += (in_block.array() * filter_block.row(kk).array()).sum();
    }
  };

  // iterate through the input by rows
  for(input_row = 0, output_last_row = filter_rows - 1;
      input_row < input_rows;
      ++input_row, ++output_last_row)
  {
    auto in_row = input.row(input_row);

    // iterate through output columns: we have 3 stages:
    // - filter on the left side of input with partial overlap
    // - filter over input with full overlap
    // - filter on the right side of input with partial overlap

    // on the left side from input
    MatrixSize in_start = 0, filter_start = filter_cols - 1, col_size = 1;
    for(output_col = 0; col_size < filter_cols; ++output_col, ++col_size, --filter_start) {
      // in_start == 0
      // col_size == output_col + 1
      // filter_start == filter_cols - col_size;
      auto in_block = in_row.segment(0, col_size);
      apply_filter(in_block, filter.rightCols(col_size));
    }

    // full overlap by columns
    for(; output_col < input_cols; ++output_col, ++in_start) {
      // in_start == output_col - filter_cols + 1;
      // col_size == filter_cols
      // filter_start == 0
      auto in_block = in_row.segment(in_start, col_size);
      // using here just filter instead of filter.rightCols() affects perf in a bad way
      apply_filter(in_block, filter.leftCols(col_size));
    }

    // on the right side from input
    for(--col_size; output_col < output_cols; ++output_col, ++in_start, --col_size) {
      // in_start == output_col - filter_cols + 1;
      // col_size == input_cols - in_start;
      // filter_start == 0
      auto in_block = in_row.segment(in_start, col_size);
      apply_filter(in_block, filter.leftCols(col_size));
    }
  }
}

void yann::ConvolutionalLayer::full_conv(const RefConstVectorBatch & input,
                                         const MatrixSize & input_rows,
                                         const MatrixSize & input_cols,
                                         const RefConstMatrix & filter,
                                         RefVectorBatch output)
{
  BOOST_VERIFY(filter.rows() > 0);
  BOOST_VERIFY(filter.cols() > 0);
  BOOST_VERIFY(filter.rows() <= input_rows);
  BOOST_VERIFY(filter.cols() <= input_cols);
  BOOST_VERIFY(get_batch_size(input) > 0);
  BOOST_VERIFY(get_batch_item_size(input) == input_rows * input_cols);
  BOOST_VERIFY(get_batch_size(output) == get_batch_size(input));

  const auto filter_rows = filter.rows();
  const auto filter_cols = filter.cols();
  const auto output_rows = get_full_conv_output_rows(input_rows, filter_rows);
  const auto output_cols = get_full_conv_output_cols(input_cols, filter_cols);
  const auto batch_size = get_batch_size(input);
  BOOST_VERIFY(get_batch_item_size(output) == output_rows * output_cols);

  // ATTENTION: this code operates on raw Matrix.data() and might be broken
  // if Matrix.data() layout changes
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    auto in_batch = get_batch(input, ii);
    auto out_batch = get_batch(output, ii);
    MapConstMatrix in(in_batch.data(), input_rows, input_cols);
    MapMatrix out(out_batch.data(), output_rows, output_cols);
    full_conv(in, filter, out);
  }
}

void yann::ConvolutionalLayer::rotate180(const Matrix & input, Matrix & output)
{
  output.resize(input.rows(), input.cols());
  for (MatrixSize ii = 0, size1 = input.rows(), size2 = input.cols(); ii < size1;
      ++ii) {
    for (MatrixSize jj = 0; jj < size2; ++jj) {
      output(ii, jj) = input(size1 - ii - 1, size2 - jj - 1);
    }
  }
}

// Convolutional layer: broadcast from input to all the conv layers
unique_ptr<BroadcastLayer> yann::ConvolutionalLayer::create_conv_bcast_layer(
    const size_t & output_frames_num,
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const MatrixSize & filter_size,
    const unique_ptr<ActivationFunction> & activation_function)
{
  BOOST_VERIFY(output_frames_num > 0);

  auto bcast_layer = make_unique<BroadcastLayer>();
  BOOST_VERIFY(bcast_layer);
  for(auto ii = output_frames_num; ii > 0; --ii) {
    auto conv_layer = make_unique<ConvolutionalLayer>(
        input_rows,
        input_cols,
        filter_size);
    BOOST_VERIFY(conv_layer);

    if(activation_function) {
      conv_layer->set_activation_function(activation_function);
    }
    bcast_layer->append_layer(std::move(conv_layer));
  }

  return bcast_layer;
}

yann::ConvolutionalLayer::ConvolutionalLayer(const MatrixSize & input_rows,
                                             const MatrixSize & input_cols,
                                             const MatrixSize & filter_size) :
  _input_rows(input_rows),
  _input_cols(input_cols),
  _filter_size(filter_size),
  _ww(filter_size, filter_size),
  _bb(0),
  _activation_function(new SigmoidFunction())
{
  BOOST_VERIFY(input_rows >= filter_size);
  BOOST_VERIFY(input_cols >= filter_size);
  BOOST_VERIFY(filter_size > 0);
}

yann::ConvolutionalLayer::~ConvolutionalLayer()
{
}

void yann::ConvolutionalLayer::set_activation_function(const unique_ptr<ActivationFunction> & activation_function)
{
  _activation_function = activation_function->copy();
}

void yann::ConvolutionalLayer::set_values(const Matrix & ww, const Value & bb)
{
  BOOST_VERIFY(is_same_size(ww, _ww));
  _ww = ww;
  _bb = bb;
}

// Layer overwrites
bool yann::ConvolutionalLayer::is_valid() const
{
  if(!Base::is_valid()) {
    return false;
  }
  if(!_activation_function) {
    return false;
  }
  return true;
}

std::string yann::ConvolutionalLayer::get_name() const
{
  return "ConvolutionalLayer";
}

void yann::ConvolutionalLayer::print_info(std::ostream & os) const
{
  BOOST_VERIFY(is_valid());

  Base::print_info(os);

  os << " activation: " << _activation_function->get_name()
     << ", input rows: " << _input_rows
     << ", input cols: " << _input_cols
     << ", filter: " << _filter_size
     << ", output rows: " << get_output_rows()
     << ", output cols: " << get_output_cols()
  ;
}

bool yann::ConvolutionalLayer::is_equal(const Layer& other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
    return false;
  }
  auto * the_other = dynamic_cast<const ConvolutionalLayer*>(&other);
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
  if(abs(_bb - the_other->_bb) > tolerance) {
    return false;
  }
  return true;
}

MatrixSize yann::ConvolutionalLayer::get_input_size() const
{
  return get_input_size(_input_rows, _input_cols);
}

MatrixSize yann::ConvolutionalLayer::get_output_size() const
{
  return get_conv_output_size(_input_rows, _input_cols, _filter_size);
}

MatrixSize yann::ConvolutionalLayer::get_output_rows() const
{
  return get_conv_output_rows(_input_rows, _filter_size);
}

MatrixSize yann::ConvolutionalLayer::get_output_cols() const
{
  return get_conv_output_cols(_input_cols, _filter_size);
}

unique_ptr<Layer::Context> yann::ConvolutionalLayer::create_context(const MatrixSize & batch_size) const
{
  return make_unique<ConvolutionalLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::ConvolutionalLayer::create_context(const RefVectorBatch & output) const
{
  return make_unique<ConvolutionalLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::ConvolutionalLayer::create_training_context(const MatrixSize & batch_size) const
{
  return make_unique<ConvolutionalLayer_TrainingContext>(_filter_size, get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::ConvolutionalLayer::create_training_context(const RefVectorBatch & output) const
{
  return make_unique<ConvolutionalLayer_TrainingContext>(_filter_size, output);
}

void yann::ConvolutionalLayer::feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode) const
{
  auto ctx = dynamic_cast<ConvolutionalLayer_Context *>(context);
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(input) == ctx->get_batch_size());
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());

  // z(l) = a(l-1)*w(l) + b(l)
  plus_conv(input, _input_rows, _input_cols, _ww, ctx->_zz);
  ctx->_zz.array() += _bb;

  // a(l) = activation(z(l))
  BOOST_VERIFY(is_same_size(ctx->_zz, ctx->get_output()));
  _activation_function->f(ctx->_zz, ctx->get_output(), mode);
}

void yann::ConvolutionalLayer::backprop(const RefConstVectorBatch & gradient_output,
                                        const RefConstVectorBatch & input,
                                        optional<RefVectorBatch> gradient_input,
                                        Context * context) const
{
  auto ctx = dynamic_cast<ConvolutionalLayer_TrainingContext*>(context);
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(gradient_output) > 0);
  BOOST_VERIFY(get_batch_item_size(gradient_output) == get_output_size());
  BOOST_VERIFY(get_batch_size(input) == get_batch_size(gradient_output));
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(!gradient_input || is_same_size(input, *gradient_input));

  // just to make it easier to read
  const auto & zz = ctx->_zz;
  auto & sigma_derivative_zz = ctx->_sigma_derivative_zz;
  auto & delta = ctx->_delta;
  auto & delta_ww = ctx->_delta_ww;
  auto & delta_bb = ctx->_delta_bb;
  const auto batch_size = get_batch_size(input);

  // delta(l) = elem_prod(gradient(C, a(l)), activation_derivative(z(l)))
  BOOST_VERIFY(is_same_size(zz, sigma_derivative_zz));
  BOOST_VERIFY(is_same_size(zz, gradient_output));
  _activation_function->derivative(zz, sigma_derivative_zz);
  delta.array() = gradient_output.array() * sigma_derivative_zz.array();

  // update deltas
  // dC/dw(l) = conv(a(l), delta(l + 1))
  // dC/db(l) = sum_elem(delta(l + 1))
  // ATTENTION: this code operates on raw Matrix.data() and might be broken
  // if Matrix.data() layout changes
  BOOST_VERIFY(batch_size == get_batch_size(delta));
  for (MatrixSize ii = 0; ii < batch_size; ++ii) {
    auto input_batch = get_batch(input, ii);
    auto delta_batch = get_batch(delta, ii);
    MapConstMatrix input_row(input_batch.data(), _input_rows, _input_cols);
    MapConstMatrix delta_row(delta_batch.data(), get_output_rows(), get_output_cols());
    plus_conv(input_row, delta_row, delta_ww, false); // delta_ww += conv()
  }
  delta_bb += delta.array().sum();

  // we don't need to calculate the first gradient(C, a(l)) for the actual inputs
  if(gradient_input) {
    // gradient(C, a(l-1)) = full_conv(delta(l), rotate180(w(l)))
    rotate180(_ww, ctx->_ww_rotated); // TODO: we can cache the rotation if _ww doesn't change
    full_conv(delta, get_output_rows(), get_output_cols(), ctx->_ww_rotated, *gradient_input);
  }
}

void yann::ConvolutionalLayer::init(enum InitMode mode)
{
  switch (mode) {
  case InitMode_Zeros:
    _ww.setZero();
    _bb = 0;
    break;
  case InitMode_Random_01:
    {
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1);
      gen01->generate(_ww);
      _bb = gen01->next();
    }
    break;
  case InitMode_Random_SqrtInputs:
    {
      unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, sqrt((Value)get_input_size()));
      unique_ptr<RandomGenerator> gen01 = RandomGenerator::normal_distribution(0, 1);
      gen->generate(_ww);
      _bb = gen01->next();
    }
    break;
  }
}

void yann::ConvolutionalLayer::update(Context * context,
                                      double learning_factor,
                                      double decay_factor)
{
  auto ctx = dynamic_cast<ConvolutionalLayer_TrainingContext *>(context);
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_same_size(_ww, ctx->_delta_ww));

  _ww = decay_factor * _ww - learning_factor * ctx->_delta_ww;
  _bb -= learning_factor * ctx->_delta_bb;
}

// the format is (w:<weights>,b:<biases>)
void yann::ConvolutionalLayer::read(std::istream & is)
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
void yann::ConvolutionalLayer::write(std::ostream & os) const
{
  Base::write(os);
  os << "(w:" << _ww << ",b:" << _bb << ")";
}

