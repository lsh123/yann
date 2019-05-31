/*
 * nntraining.cpp
 *
 */
#include <algorithm>
#include <random>
#include <string>

#include "utils.h"
#include "layer.h"

#include "training.h"

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
  return "GradientDescent"; // TODO: print other params
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescent::copy() const
{
  return make_unique<Updater_GradientDescent>(*this);
}

void yann::Updater_GradientDescent::reset(const RefConstMatrix & delta)
{
  // do nothing
}

void yann::Updater_GradientDescent::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  BOOST_VERIFY(is_same_size(delta, value));
  BOOST_VERIFY(batch_size > 0);
  BOOST_VERIFY(_regularization_parameter * _learning_rate < batch_size);

  double learning_factor = _learning_rate / (double) batch_size;
  double decay_factor = 1 - _regularization_parameter * _learning_rate;
  value = decay_factor * value - learning_factor * delta;
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
  return "GradientDescentWithMomentum"; // TODO: print other params
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescentWithMomentum::copy() const
{
  return make_unique<Updater_GradientDescentWithMomentum>(*this);
}

void yann::Updater_GradientDescentWithMomentum::reset(const RefConstMatrix & delta)
{
  // TODO: create init method for resize
  if(!is_same_size(_velocity, delta)) {
    _velocity.resizeLike(delta);
    _velocity.setZero();
  }
}

void yann::Updater_GradientDescentWithMomentum::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  BOOST_VERIFY(is_same_size(delta, value));
  BOOST_VERIFY(batch_size > 0);
  BOOST_VERIFY(_regularization_parameter * _learning_rate < batch_size);

  double learning_factor = _learning_rate / (double) batch_size;
  double decay_factor = 1 - _regularization_parameter * _learning_rate;

  _velocity = decay_factor * _velocity + learning_factor * delta;
  value -= _velocity;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Trainer implementation
//
yann::Trainer::Trainer(const MatrixSize & batch_size) :
  _batch_size(batch_size),
  _progress_callback(nullptr)
{
  BOOST_VERIFY(batch_size > 0);
}

yann::Trainer::~Trainer()
{
}

std::string yann::Trainer::get_info() const
{
  ostringstream oss;
  print_info(oss);
  return oss.str();
}

void yann::Trainer::prepare_shuffled_pos(enum InputSelectionMode select_mode, std::vector<MatrixSize> & shuffled_pos)
{
  for (size_t ii = 0; ii < shuffled_pos.size(); ++ii) {
    shuffled_pos[ii] = ii;
  }
  switch (select_mode) {
    case Sequential:
      // do nothing
      break;
    case Random:
      // shuffle the positions
      auto rng = default_random_engine { };
      shuffle(begin(shuffled_pos), end(shuffled_pos), rng);
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Apply training data in batches but don't apply deltas until all
// the training data is processed.
//
yann::Trainer_Batch::Trainer_Batch(
    const unique_ptr<Layer::Updater> & updater,
    enum InputSelectionMode select_mode,
    const MatrixSize & batch_size) :
    Trainer(batch_size),
    _updater(updater->copy()),
    _select_mode(select_mode)
{
}

// Trainer
void yann::Trainer_Batch::print_info(ostream & os) const
{
  os << "yann::Trainer_Batch("
     << " select_mode=" << _select_mode << " batch_size=" << _batch_size
     << ", updater=" << _updater->get_info()
     << ")";
}

void yann::Trainer_Batch::train(Network & nn,
                                const VectorBatch & inputs,
                                const VectorBatch & outputs) const
{
  BOOST_VERIFY(get_batch_size(inputs) == get_batch_size(outputs));
  BOOST_VERIFY(_batch_size > 0);
  BOOST_VERIFY(_batch_size <= get_batch_size(inputs));
  BOOST_VERIFY(get_batch_item_size(inputs) == nn.get_input_size());
  BOOST_VERIFY(get_batch_item_size(outputs) == nn.get_output_size());

  // prepare training
  auto ctx = nn.create_training_context(1, _updater);
  BOOST_VERIFY(ctx);
  ctx->reset_state();

  // prepare positions vector
  vector<MatrixSize> shuffled_pos(get_batch_size(inputs));
  prepare_shuffled_pos(_select_mode, shuffled_pos);

  Vector in(get_batch_item_size(inputs));
  Vector out(get_batch_item_size(outputs));
  auto last_batch_pos = shuffled_pos.size() - _batch_size;
  for (size_t ii = 0; ii <= last_batch_pos; ii += _batch_size) {
    if(_progress_callback != nullptr) {
      _progress_callback(ii, _batch_size, last_batch_pos);
    }

    for (MatrixSize jj = 0; jj < _batch_size; ++jj) {
      MatrixSize pos = shuffled_pos[ii + jj];
      BOOST_VERIFY(pos < get_batch_size(inputs));
      in = get_batch_const(inputs, pos);
      out = get_batch_const(outputs, pos);
      nn.train(in, out, ctx.get());
    }
  }

  // update deltas at the end of the training set
  nn.update(ctx.get(), get_batch_size(inputs));
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// After each batch of training data, apply deltas to the network.
//
yann::Trainer_Stochastic::Trainer_Stochastic(
    const unique_ptr<Layer::Updater> & updater,
    enum InputSelectionMode select_mode,
    const MatrixSize & batch_size) :
    Trainer(batch_size),
    _updater(updater->copy()),
    _select_mode(select_mode)
{
}

// Trainer
void yann::Trainer_Stochastic::print_info(ostream & os) const
{
  os << "yann::Trainer_Stochastic("
     << " select_mode=" << _select_mode << " batch_size=" << _batch_size
     << ", updater=" << _updater->get_info()
     << ")";
}

void yann::Trainer_Stochastic::train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const
{
  BOOST_VERIFY(_batch_size > 0);
  BOOST_VERIFY(_batch_size <= get_batch_size(inputs));
  BOOST_VERIFY(get_batch_size(inputs) == get_batch_size(outputs));
  BOOST_VERIFY(get_batch_item_size(inputs) == nn.get_input_size());
  BOOST_VERIFY(get_batch_item_size(outputs) == nn.get_output_size());

  // prepare positions vector
  vector<MatrixSize> shuffled_pos(get_batch_size(inputs));
  prepare_shuffled_pos(_select_mode, shuffled_pos);

  auto ctx = nn.create_training_context(_batch_size, _updater);
  BOOST_VERIFY(ctx);

  VectorBatch ins, outs;
  resize_batch(ins, _batch_size, nn.get_input_size());
  resize_batch(outs, _batch_size, nn.get_output_size());
  auto last_batch_pos = shuffled_pos.size() - _batch_size;
  for (size_t ii = 0; ii <= last_batch_pos; ii += _batch_size) {
    if(_progress_callback != nullptr) {
      _progress_callback(ii, _batch_size, last_batch_pos);
    }

    // prepare inputs/outputs
    for (MatrixSize jj = 0; jj < _batch_size; ++jj) {
      MatrixSize pos = shuffled_pos[ii + jj];
      BOOST_VERIFY(pos < get_batch_size(inputs));

      get_batch(ins, jj)  = get_batch_const(inputs, pos);
      get_batch(outs, jj) = get_batch(outputs, pos);
    }

    // train and apply deltas after each "mini-batch"
    ctx->reset_state();
    nn.train(ins, outs, ctx.get());
    nn.update(ctx.get(), _batch_size);
  }
}

