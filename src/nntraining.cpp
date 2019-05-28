/*
 * nntraining.cpp
 *
 */
#include <algorithm>
#include <random>
#include <string>

#include "utils.h"
#include "nn.h"
#include "nnlayer.h"
#include "nntraining.h"

using namespace std;
using namespace yann;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// NN Trainer implementation
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
// Incremental gradient descent: train one row at a time, apply deltas in batches after 1 or more trainings
//
yann::Trainer_Incremental_GD::Trainer_Incremental_GD(
    double learning_rate, double regularization_parameter,
    enum InputSelectionMode select_mode,
    const MatrixSize & batch_size) :
    Trainer(batch_size),
    _learning_rate(learning_rate),
    _regularization_parameter(regularization_parameter),
    _select_mode(select_mode)
{
  BOOST_VERIFY(regularization_parameter * learning_rate < _batch_size);
}

// Trainer
void yann::Trainer_Incremental_GD::print_info(ostream & os) const
{
  os << "yann::Trainer_Incremental_GD(" << "learning_rate=" << _learning_rate
     << " regularization_parameter=" << _regularization_parameter
     << " select_mode=" << _select_mode << " batch_size=" << _batch_size
     << ")";
}

void yann::Trainer_Incremental_GD::train(Network & nn,
                                         const VectorBatch & inputs,
                                         const VectorBatch & outputs) const
{
  BOOST_VERIFY(get_batch_size(inputs) == get_batch_size(outputs));
  BOOST_VERIFY(_batch_size > 0);
  BOOST_VERIFY(_batch_size <= get_batch_size(inputs));
  BOOST_VERIFY(get_batch_item_size(inputs) == nn.get_input_size());
  BOOST_VERIFY(get_batch_item_size(outputs) == nn.get_output_size());

  // prepare positions vector
  vector<MatrixSize> shuffled_pos(get_batch_size(inputs));
  prepare_shuffled_pos(_select_mode, shuffled_pos);

  double learning_factor = _learning_rate / (double) _batch_size;
  double decay_factor = 1 - _regularization_parameter * _learning_rate;
  std::unique_ptr<TrainingContext> ctx(nn.create_training_context());
  Vector in(get_batch_item_size(inputs));
  Vector out(get_batch_item_size(outputs));
  auto last_batch_pos = shuffled_pos.size() - _batch_size;
  for (size_t ii = 0; ii <= last_batch_pos; ii += _batch_size) {
    if(_progress_callback != nullptr) {
      _progress_callback(ii, _batch_size, last_batch_pos);
    }

    // train
    ctx->reset_state();

    for (MatrixSize jj = 0; jj < _batch_size; ++jj) {
      MatrixSize pos = shuffled_pos[ii + jj];
      BOOST_VERIFY(pos < get_batch_size(inputs));
      in = get_batch_const(inputs, pos);
      out = get_batch_const(outputs, pos);
      nn.train(in, out, ctx.get());
    }
    nn.update(ctx.get(), learning_factor, decay_factor);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Mini-batch gradient descent: train and update in mini batches
//
yann::Trainer_MiniBatch_GD::Trainer_MiniBatch_GD(
    double learning_rate, double regularization_parameter,
    enum InputSelectionMode select_mode,
    const MatrixSize & batch_size) :
    Trainer(batch_size),
    _learning_rate(learning_rate),
    _regularization_parameter(regularization_parameter),
    _select_mode(select_mode)
{
  BOOST_VERIFY(regularization_parameter * learning_rate < _batch_size);
}

// Trainer
void yann::Trainer_MiniBatch_GD::print_info(ostream & os) const
{
  os << "yann::Trainer_MiniBatch_GD(" << "learning_rate=" << _learning_rate
     << " regularization_parameter=" << _regularization_parameter
     << " select_mode=" << _select_mode << " batch_size=" << _batch_size
     << ")";
}

void yann::Trainer_MiniBatch_GD::train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const
{
  BOOST_VERIFY(_batch_size > 0);
  BOOST_VERIFY(_batch_size <= get_batch_size(inputs));
  BOOST_VERIFY(get_batch_size(inputs) == get_batch_size(outputs));
  BOOST_VERIFY(get_batch_item_size(inputs) == nn.get_input_size());
  BOOST_VERIFY(get_batch_item_size(outputs) == nn.get_output_size());

  // prepare positions vector
  vector<MatrixSize> shuffled_pos(get_batch_size(inputs));
  prepare_shuffled_pos(_select_mode, shuffled_pos);

  // TODO: move this to TrainingContext?
  double learning_factor = _learning_rate / (double) _batch_size;
  double decay_factor = 1 - _regularization_parameter * _learning_rate;
  std::unique_ptr<TrainingContext> ctx(nn.create_training_context(_batch_size));
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

    // train
    ctx->reset_state();
    nn.train(ins, outs, ctx.get());
    nn.update(ctx.get(), learning_factor, decay_factor);
  }
}

