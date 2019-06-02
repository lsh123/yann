/*
 * nntraining.h
 *
 */
#ifndef NNTRAINING_H_
#define NNTRAINING_H_

#include <iostream>
#include <memory>
#include <vector>

#include "core/nn.h"
#include "core/layer.h"

namespace yann {

// Updates the values according to the gradient descent with weight decay
class Updater_GradientDescent :
    public Layer::Updater
{
public:
  Updater_GradientDescent(double learning_rate = 1.0, double regularization_parameter = 0.0);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value);
  virtual void update(const Value & delta, const size_t & batch_size, Value & value);

private:
  double learning_factor(const size_t & batch_size) const;
  double decay_factor(const size_t & batch_size) const;

private:
  double _learning_rate;
  double _regularization_parameter;
}; // Updater_GradientDescent

// Updates the values according to the gradient descent with momentum and weight decay:
//
class Updater_GradientDescentWithMomentum :
    public Layer::Updater
{
public:
  Updater_GradientDescentWithMomentum(double learning_rate = 1.0, double regularization_parameter = 0.0);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value);
  virtual void update(const Value & delta, const size_t & batch_size, Value & value);

private:
  double learning_factor(const size_t & batch_size) const;
  double decay_factor(const size_t & batch_size) const;

private:
  double _learning_rate;
  double _regularization_parameter;
  Matrix _velocity;
}; // Updater_GradientDescentWithMomentum


class Trainer
{
public:
  typedef void (*ProgressCallback)(const MatrixSize & cur, const MatrixSize & total);

  class DataSource {
  public:
    class Batch {
    public:
      Batch(
          const VectorBatch & inputs,
          const VectorBatch & outputs) :
        _inputs(inputs),
        _outputs(outputs)
      {
      }
      Batch(
          const SparseVectorBatch & sparse_inputs,
          const VectorBatch & outputs) :
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
  Value train(Network & nn, DataSource & data_source) const;
  void set_batch_progress_callback(ProgressCallback callback) {  _batch_progress_callback = callback; }

protected:
  std::unique_ptr<Layer::Updater> _updater;
  ProgressCallback _batch_progress_callback;
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

  // Trainer::DataSource overwrites
  virtual std::string get_info() const;
  virtual MatrixSize get_batch_size() const;
  virtual MatrixSize get_num_batches() const;
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

#endif /* NNTRAINING_H_ */
