/*
 * training.cpp
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
// Trainer implementation
//
yann::Trainer::Trainer(const unique_ptr<Layer::Updater> & updater) :
  _updater(updater->copy()),
  _batch_progress_callback(nullptr),
  _epochs_progress_callback(nullptr)
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

Value yann::Trainer::train(Network & nn, DataSource & data_source, TrainingContext * ctx) const
{
  YANN_CHECK(ctx);

  // prepare training
  auto num_batches = data_source.get_num_batches();
  auto batch_size = data_source.get_batch_size();

  ctx->start_epoch();
  data_source.start_epoch();

  Value total_cost = 0;
  for (size_t ii = 0; ; ++ii) {
    auto batch = data_source.get_next_batch();
    if(!batch || !batch->is_valid()) {
      break;
    }
    if(_batch_progress_callback != nullptr) {
      _batch_progress_callback(ii, num_batches, "");
    }

    ctx->reset_state();
    if(batch->_inputs) {
      const auto inputs = *(batch->_inputs);
      const auto outputs = batch->_outputs;
      YANN_CHECK_LE(get_batch_size(inputs), batch_size);
      YANN_CHECK_LE(get_batch_size(outputs), batch_size);

      total_cost += nn.train(inputs, outputs, ctx);
    } else if(batch->_sparse_inputs) {
      const auto inputs = *(batch->_sparse_inputs);
      const auto outputs = batch->_outputs;
      YANN_CHECK_LE(get_batch_size(inputs), batch_size);
      YANN_CHECK_LE(get_batch_size(outputs), batch_size);

      total_cost += nn.train(inputs, outputs, ctx);
    } else {
      YANN_CHECK("we can't be here" == nullptr);
    }
    nn.update(ctx, batch_size);
  }

  data_source.end_epoch();

  return total_cost;
}

Value yann::Trainer::train(Network & nn, DataSource & data_source, const size_t & epochs) const
{
  auto ctx = nn.create_training_context(data_source.get_batch_size(), _updater);
  YANN_CHECK(ctx);

  auto tests_num = data_source.get_tests_num();
  Value cost = 0;
  for(size_t ii = 0; ii < epochs; ++ii) {
    cost = train(nn, data_source, ctx.get());
    if(_epochs_progress_callback != nullptr) {
      ostringstream oss;
      oss << "cost per test: " << cost / tests_num;
      _epochs_progress_callback(ii + 1, epochs, oss.str());
    }
  }
  return cost / tests_num;
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

yann::DataSource_Stochastic::~DataSource_Stochastic()
{
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

MatrixSize yann::DataSource_Stochastic::get_tests_num() const
{
  return get_num_batches() * get_batch_size();
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


