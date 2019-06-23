/*
 * training.h
 *
 */
#ifndef TRAINING_H_
#define TRAINING_H_

#include <iostream>
#include <functional>
#include <memory>
#include <vector>

#include "core/nn.h"
#include "core/layer.h"

namespace yann {

class Trainer
{
public:
  typedef std::function<void(const MatrixSize & cur, const MatrixSize & total, const std::string & message)> ProgressCallback;

  class DataSource {
  public:
    class Batch {
    public:
      Batch(
          const RefConstVectorBatch inputs,
          const RefConstVectorBatch outputs) :
        _inputs(inputs),
        _outputs(outputs)
      {
      }
      Batch(
          const RefConstSparseVectorBatch sparse_inputs,
          const RefConstVectorBatch outputs) :
        _sparse_inputs(sparse_inputs),
        _outputs(outputs)
      {
      }
      inline bool is_valid() const
      {
        return _inputs || _sparse_inputs;
      }
    public:
      boost::optional<RefConstVectorBatch> _inputs;
      boost::optional<RefConstSparseVectorBatch> _sparse_inputs;
      RefConstVectorBatch _outputs;
    }; // Batch;

  public:
    virtual std::string get_info() const = 0;
    virtual MatrixSize get_tests_num() const = 0;
    virtual MatrixSize get_batch_size() const = 0;
    virtual MatrixSize get_num_batches() const = 0;
    virtual void start_epoch() = 0;
    virtual boost::optional<Batch> get_next_batch() = 0;
    virtual void end_epoch() = 0;
  }; // DataSource

public:
  Trainer(const std::unique_ptr<Layer::Updater> & updater);
  virtual ~Trainer();

  std::string get_info() const;
  Value train(Network & nn, DataSource & data_source, const size_t & epochs) const;

  void set_batch_progress_callback(ProgressCallback callback) {  _batch_progress_callback = callback; }
  void set_epochs_progress_callback(ProgressCallback callback) {  _epochs_progress_callback = callback; }

private:
  Value train(Network & nn, DataSource & data_source, TrainingContext * ctx) const;

protected:
  std::unique_ptr<Layer::Updater> _updater;
  ProgressCallback _batch_progress_callback;
  ProgressCallback _epochs_progress_callback;
}; // class Trainer

// After each batch of training data, apply deltas to the network.
class DataSource_Stochastic : public Trainer::DataSource
{
public:
  enum Mode
  {
    Sequential = 0,
    Random,
  };

public:
  DataSource_Stochastic(
      const RefConstVectorBatch & inputs,
      const RefConstVectorBatch & outputs,
      enum  Mode mode,
      const MatrixSize & batch_size);
  virtual ~DataSource_Stochastic();

  // Trainer::DataSource overwrites
  virtual std::string get_info() const;
  virtual MatrixSize get_batch_size() const;
  virtual MatrixSize get_num_batches() const;
  virtual MatrixSize get_tests_num() const;
  virtual void start_epoch();
  virtual boost::optional<Batch> get_next_batch();
  virtual void end_epoch();

private:
  RefConstVectorBatch _inputs;
  RefConstVectorBatch _outputs;
  enum Mode _mode;
  MatrixSize _batch_size;
  MatrixSize _cur_batch;
  std::vector<MatrixSize> _shuffled_pos;
  VectorBatch _inputs_batch;
  VectorBatch _outputs_batch;
}; // class Trainer_Stochastic

}; // namespace yann

#endif /* TRAINING_H_ */
