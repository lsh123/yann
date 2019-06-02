/*
 * nntraining.cpp
 *
 */
#include <algorithm>
#include <random>
#include <string>

#include "core/utils.h"
#include "core/layer.h"

#include "core/training.h"

using namespace std;
using namespace yann;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_GradientDescent implementation
//
yann::Updater_GradientDescent::Updater_GradientDescent(
    double learning_rate,
    double regularization_parameter) :
    _learning_rate(learning_rate),
    _regularization_parameter(regularization_parameter)
{

}

std::string yann::Updater_GradientDescent::get_info() const
{
  ostringstream oss;
  oss << "GradientDescent["
      << "learning_rate=" << _learning_rate
      << ", regularization_parameter=" << _regularization_parameter
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescent::copy() const
{
  return make_unique<Updater_GradientDescent>(*this);
}

void yann::Updater_GradientDescent::init(const MatrixSize & rows, const MatrixSize & cols)
{
  // do nothing
}

void yann::Updater_GradientDescent::reset()
{
  // do nothing
}

double yann::Updater_GradientDescent::learning_factor(const size_t & batch_size) const
{
  YANN_CHECK_GT(batch_size, 0);
  return _learning_rate / (double) batch_size;
}

double yann::Updater_GradientDescent::decay_factor(const size_t & batch_size) const
{
  YANN_CHECK_GT(batch_size, 0);
  YANN_CHECK_GT(batch_size, _regularization_parameter * _learning_rate);
  return 1 - _regularization_parameter * _learning_rate / (double) batch_size;
}

void yann::Updater_GradientDescent::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_CHECK(is_same_size(delta, value));
  value = decay_factor(batch_size) * value - learning_factor(batch_size) * delta;
}

void yann::Updater_GradientDescent::update(const Value & delta, const size_t & batch_size, Value & value)
{
  value = decay_factor(batch_size) * value - learning_factor(batch_size) * delta;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_GradientDescentWithMomentum implementation
//
yann::Updater_GradientDescentWithMomentum::Updater_GradientDescentWithMomentum(
    double learning_rate,
    double regularization_parameter) :
    _learning_rate(learning_rate),
    _regularization_parameter(regularization_parameter)
{
}

std::string yann::Updater_GradientDescentWithMomentum::get_info() const
{
  ostringstream oss;
  oss << "GradientDescentWithMomentum["
      << "learning_rate=" << _learning_rate
      << ", regularization_parameter=" << _regularization_parameter
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescentWithMomentum::copy() const
{
  return make_unique<Updater_GradientDescentWithMomentum>(*this);
}

void yann::Updater_GradientDescentWithMomentum::init(const MatrixSize & rows, const MatrixSize & cols)
{
  _velocity.resize(rows, cols);
}

void yann::Updater_GradientDescentWithMomentum::reset()
{
  _velocity.setZero();
}

double yann::Updater_GradientDescentWithMomentum::learning_factor(const size_t & batch_size) const
{
  YANN_CHECK_GT(batch_size, 0);
  return _learning_rate / (double) batch_size;
}

double yann::Updater_GradientDescentWithMomentum::decay_factor(const size_t & batch_size) const
{
  YANN_CHECK_GT(batch_size, 0);
  YANN_CHECK_GT(batch_size, _regularization_parameter * _learning_rate);
  return 1 - _regularization_parameter * _learning_rate / (double) batch_size;
}

void yann::Updater_GradientDescentWithMomentum::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_CHECK(is_same_size(delta, value));
  YANN_CHECK(is_same_size(_velocity, value));

  _velocity = decay_factor(batch_size) * _velocity + learning_factor(batch_size) * delta;
  value -= _velocity;
}

void yann::Updater_GradientDescentWithMomentum::update(const Value & delta, const size_t & batch_size, Value & value)
{
  YANN_CHECK_EQ(_velocity.size(), 1);

  _velocity(0,0) = decay_factor(batch_size) * _velocity(0,0) + learning_factor(batch_size) * delta;
  value -= _velocity(0,0);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Trainer implementation
//
yann::Trainer::Trainer(const unique_ptr<Layer::Updater> & updater) :
  _updater(updater->copy()),
  _batch_progress_callback(nullptr)
{
  YANN_CHECK(_updater);
}

yann::Trainer::~Trainer()
{
}

string yann::Trainer::get_info() const
{
  ostringstream oss;
  oss << "yann::Trainer(updater=" << _updater->get_info() << ")";
  return oss.str();
}

Value yann::Trainer::train(Network & nn, DataSource & data_source) const
{
  // prepare training
  auto num_batches = data_source.get_num_batches();
  auto batch_size = data_source.get_batch_size();
  auto ctx = nn.create_training_context(batch_size, _updater);
  YANN_CHECK(ctx);

  data_source.start_epoch();

  Value total_cost = 0;
  for (size_t ii = 0; ; ++ii) {
    auto batch = data_source.get_next_batch();
    if(!batch || !batch->is_valid()) {
      break;
    }
    if(_batch_progress_callback != nullptr) {
      _batch_progress_callback(ii, num_batches);
    }

    ctx->reset_state();
    if(batch->_inputs) {
      const auto inputs = *(batch->_inputs);
      const auto outputs = batch->_outputs;
      YANN_CHECK_EQ(get_batch_size(inputs), batch_size);
      YANN_CHECK_EQ(get_batch_size(outputs), batch_size);

      total_cost += nn.train(inputs, outputs, ctx.get());
    } else if(batch->_sparse_inputs) {
      const auto inputs = *(batch->_sparse_inputs);
      const auto outputs = batch->_outputs;
      YANN_CHECK_EQ(get_batch_size(inputs), batch_size);
      YANN_CHECK_EQ(get_batch_size(outputs), batch_size);

      total_cost += nn.train(inputs, outputs, ctx.get());
    } else {
      YANN_CHECK("we can't be here" == nullptr);
    }
    nn.update(ctx.get(), batch_size);
  }

  data_source.end_epoch();

  return total_cost;
}


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Apply training data in mini batches
//
yann::DataSource_Stochastic::DataSource_Stochastic(
    const RefConstVectorBatch & inputs,
    const RefConstVectorBatch & outputs,
    enum Mode mode,
    const MatrixSize & batch_size) :
  _inputs(inputs),
  _outputs(outputs),
  _mode(mode),
  _batch_size(batch_size),
  _cur_batch(0),
  _shuffled_pos(yann::get_batch_size(inputs))
{
  YANN_CHECK_GT(_batch_size, 0);
  YANN_CHECK_LE(_batch_size, yann::get_batch_size(_inputs));
  YANN_CHECK_EQ(yann::get_batch_size(_inputs), yann::get_batch_size(_outputs));

  resize_batch(_inputs_batch, _batch_size, get_batch_item_size(_inputs));
  resize_batch(_outputs_batch, _batch_size, get_batch_item_size(_outputs));
}

// Trainer::DataSource overwrites
string yann::DataSource_Stochastic::get_info() const
{
  ostringstream oss;
  oss << "yann::DataSource_Stochastic("
      << " mode=" << _mode
      << ", batch_size=" << _batch_size
      << ")";
  return oss.str();
}

MatrixSize yann::DataSource_Stochastic::get_batch_size() const
{
  YANN_CHECK_GT(_batch_size, 0);
  return _batch_size;
}

MatrixSize yann::DataSource_Stochastic::get_num_batches() const
{
  YANN_CHECK_GT(_batch_size, 0);
  YANN_CHECK_EQ(yann::get_batch_size(_inputs), yann::get_batch_size(_outputs));
  YANN_CHECK_LE(_batch_size, yann::get_batch_size(_inputs));
  return yann::get_batch_size(_inputs) / _batch_size;
}

void yann::DataSource_Stochastic::start_epoch()
{
  _cur_batch = 0;

  // prepare shuffled positions
  for (size_t ii = 0; ii < _shuffled_pos.size(); ++ii) {
    _shuffled_pos[ii] = ii;
  }
  switch (_mode) {
    case Sequential:
      // do nothing
      break;
    case Random:
      // shuffle the positions
      auto rng = default_random_engine { };
      shuffle(begin(_shuffled_pos), end(_shuffled_pos), rng);
      break;
  }
}

boost::optional<yann::Trainer::DataSource::Batch> yann::DataSource_Stochastic::get_next_batch()
{
  if(_cur_batch >= get_num_batches()) {
    return boost::none;
  }

  // prepare inputs/outputs
  MatrixSize start_pos = _cur_batch * _batch_size;
  for (MatrixSize ii = 0; ii < _batch_size; ++ii) {
    YANN_SLOW_CHECK_LE(start_pos + ii, (MatrixSize)_shuffled_pos.size());
    MatrixSize pos = _shuffled_pos[start_pos + ii];
    YANN_SLOW_CHECK_LT(pos, yann::get_batch_size(_inputs));

    get_batch(_inputs_batch, ii)  = get_batch_const(_inputs, pos);
    get_batch(_outputs_batch, ii) = get_batch_const(_outputs, pos);
  }
  ++_cur_batch;

  return Batch(_inputs_batch,_outputs_batch);
}

void yann::DataSource_Stochastic::end_epoch()
{
  // do nothing
}


