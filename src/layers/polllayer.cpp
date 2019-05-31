/*
 * polllayer.cpp
 *
 * This is polling layer
 *
 * MAX mode:
 *  Feed-forward:
 *    a(l, i, j) = max(a(l-1, k, l) where k,l in (i, i + filter_size), (j, j + filter_size)
 *  Back propagation:
 *    gradient(C, a(l)) = gradient(C, a(l+1)) for the position of max() for feedforward
 *
 * AVG mode:
 *  Feed-forward:
 *    a(l, i, j) = avg(a(l-1, k, l) where k,l in (i, i + filter_size), (j, j + filter_size)
 *  Back propagation:
 *    gradient(C, a(l)) = gradient(C, a(l+1)) for each cell in submatrix
 */
#include <boost/assert.hpp>

#include "utils.h"
#include "contlayer.h"
#include "polllayer.h"

using namespace std;
using namespace boost;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// PollingLayer_Context / PollingLayer_TrainingContext implementation
//
namespace yann {

typedef Layer::Context PollingLayer_Context;
typedef PollingLayer_Context PollingLayer_TrainingContext;

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
    enum Mode mode)
{
  BOOST_VERIFY(frames_num > 0);

  auto poll_container_layer = make_unique<ParallelLayer>(frames_num);
  BOOST_VERIFY(poll_container_layer);
  for(auto ii = frames_num; ii > 0; --ii) {
    auto poll_layer = make_unique<PollingLayer>(
        input_rows,
        input_cols,
        filter_size,
        mode);
    BOOST_VERIFY(poll_layer);

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
    _mode(mode)
{
  BOOST_VERIFY(input_rows >= filter_size);
  BOOST_VERIFY(input_cols >= filter_size);
}

yann::PollingLayer::~PollingLayer()
{
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

void yann::PollingLayer::print_info(std::ostream & os) const
{
  Base::print_info(os);

  // assume that all layers are the same, print only one
  os << " input rows: " << _input_rows
     << ", input cols: " << _input_cols
     << ", filter: " << _filter_size
     << ", output rows: " << get_output_rows()
     << ", output cols: " << get_output_cols()
  ;
  switch(_mode) {
  case PollMode_Max:
    os << ", mode: Max";
    break;
  case PollMode_Avg:
    os << ", mode: Avg";
    break;
  }
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

unique_ptr<Layer::Context> yann::PollingLayer::create_context(const MatrixSize & batch_size) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<PollingLayer_Context>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_context(const RefVectorBatch & output) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<PollingLayer_Context>(output);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<PollingLayer_TrainingContext>(get_output_size(), batch_size);
}
unique_ptr<Layer::Context> yann::PollingLayer::create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const
{
  BOOST_VERIFY(is_valid());
  return make_unique<PollingLayer_TrainingContext>(output);
}

void yann::PollingLayer::poll_plus_equal(const RefConstMatrix & input, const MatrixSize & filter_size, enum Mode mode, RefMatrix output)
{
  BOOST_VERIFY(output.rows() == get_output_rows(input.rows(), filter_size));
  BOOST_VERIFY(output.cols() == get_output_cols(input.cols(), filter_size));

  // Rows/cols order should change if matrix layout is change from RowMajor to ColMajor for perf reasons
  for(MatrixSize in_ii = 0, out_ii = 0; out_ii < output.rows(); in_ii += filter_size, ++out_ii) {
    for(MatrixSize in_jj = 0, out_jj = 0; out_jj < output.cols(); in_jj += filter_size, ++out_jj) {
      switch(mode) {
        case PollMode_Max:
          output(out_ii, out_jj) += input.block(in_ii, in_jj, filter_size, filter_size).maxCoeff();
          break;
        case PollMode_Avg:
          output(out_ii, out_jj) += input.block(in_ii, in_jj, filter_size, filter_size).sum() / (filter_size * filter_size);
          break;
      }
    }
  }
}


void yann::PollingLayer::backprop(const RefConstMatrix & gradient_output, const RefConstVectorBatch & input,
              const MatrixSize & filter_size, enum Mode mode, RefMatrix gradient_input)
{
  BOOST_VERIFY(gradient_output.rows() == get_output_rows(input.rows(), filter_size));
  BOOST_VERIFY(gradient_output.cols() == get_output_cols(input.cols(), filter_size));
  BOOST_VERIFY(is_same_size(gradient_input, input));

  // Rows/cols order should change if matrix layout is change from RowMajor to ColMajor for perf reasons
  MatrixSize ii, jj;
  for(MatrixSize in_ii = 0, out_ii = 0; out_ii < gradient_output.rows(); in_ii += filter_size, ++out_ii) {
    for(MatrixSize in_jj = 0, out_jj = 0; out_jj < gradient_output.cols(); in_jj += filter_size, ++out_jj) {
      Value val = gradient_output(out_ii, out_jj);
      switch(mode) {
        case PollMode_Max:
          input.block(in_ii, in_jj, filter_size, filter_size).maxCoeff(&ii, &jj);
          gradient_input(in_ii + ii, in_jj + jj) = val;
          break;
        case PollMode_Avg:
          gradient_input.block(in_ii, in_jj, filter_size, filter_size).array() += val;
          break;
      }
    }
  }
}

void yann::PollingLayer::feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode) const
{
  auto ctx = dynamic_cast<PollingLayer_Context *>(context);
  BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(input) == ctx->get_batch_size());
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());

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
  const auto output_rows = get_output_rows();
  const auto output_cols = get_output_cols();
  for(MatrixSize ii = 0; ii < batch_size; ++ii) {
    MapConstMatrix in(get_batch(input, ii).data(), _input_rows, _input_cols);
    MapMatrix out(get_batch(output, ii).data(), output_rows, output_cols);

    poll_plus_equal(in, _filter_size, _mode, out);
  }
}

void yann::PollingLayer::backprop(const RefConstVectorBatch & gradient_output,
                                         const RefConstVectorBatch & input,
                                         optional<RefVectorBatch> gradient_input,
                                         Context * /* context */) const
{
  // auto ctx = dynamic_cast<PollingLayer_TrainingContext *>(context);
  // BOOST_VERIFY(ctx);
  BOOST_VERIFY(is_valid());
  BOOST_VERIFY(get_batch_size(gradient_output) > 0);
  BOOST_VERIFY(get_batch_item_size(gradient_output) == get_output_size());
  BOOST_VERIFY(get_batch_size(input) == get_batch_size(gradient_output));
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(get_batch_item_size(input) == get_input_size());
  BOOST_VERIFY(!gradient_input || is_same_size(input, *gradient_input));

  // nothing to do for the polling layer itself

  // we don't need to calculate the gradient(C, a(l)) for the "first" layer (actual inputs)
  if(gradient_input) {
    const auto batch_size = get_batch_size(input);
    const auto output_rows = get_output_rows();
    const auto output_cols = get_output_cols();

    gradient_input->setZero();
    for(MatrixSize ii = 0; ii < batch_size; ++ii) {
      MapMatrix gradient_in(get_batch(*gradient_input, ii).data(), _input_rows, _input_cols);
      MapConstMatrix in(get_batch(input, ii).data(), _input_rows, _input_cols);
      MapConstMatrix gradient_out(get_batch(gradient_output, ii).data(), output_rows, output_cols);

      backprop(gradient_out, in, _filter_size, _mode, gradient_in);
    }

  }
}

void yann::PollingLayer::init(enum InitMode mode)
{
  // nothing to do
}

void yann::PollingLayer::update(Context * context, const size_t & batch_size)
{
  // auto ctx = dynamic_cast<PollingLayer_TrainingContext *>(context);
  // BOOST_VERIFY(ctx);

  // nothing to do
}

void yann::PollingLayer::read(std::istream & is)
{
  Base::read(is);
  // nothing to do
}

void yann::PollingLayer::write(std::ostream & os) const
{
  Base::write(os);
  // nothing to do
}

