/*
 * nntraining.h
 *
 */
#ifndef NNTRAINING_H_
#define NNTRAINING_H_

#include <iostream>
#include <memory>
#include <vector>

#include "nn.h"
#include "layer.h"

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

  virtual void reset(const RefConstMatrix & delta);
  virtual void update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value);

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

  virtual void reset(const RefConstMatrix & delta);
  virtual void update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value);

private:
  double _learning_rate;
  double _regularization_parameter;
  Matrix _velocity;
}; // Updater_GradientDescentWithMomentum


class Trainer
{
public:
  enum InputSelectionMode
  {
    Sequential = 0,
    Random,
  };

  typedef void (*ProgressCallback)(const MatrixSize & cur_pos, const MatrixSize & step, const MatrixSize & total);


public:
  Trainer(const MatrixSize & batch_size);
  virtual ~Trainer();

  std::string get_info() const;
  virtual void print_info(std::ostream & os) const = 0;
  virtual void train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const = 0;

  void set_progress_callback(ProgressCallback callback) {  _progress_callback = callback; }

protected:
  static void prepare_shuffled_pos(enum InputSelectionMode select_mode, std::vector<MatrixSize> & shuffled_pos);

protected:
  MatrixSize _batch_size;
  ProgressCallback _progress_callback;
}; // class Trainer

// Apply training data in batches but don't apply deltas until all
// the training data is processed.
class Trainer_Batch : public Trainer
{
public:
  Trainer_Batch(
      const std::unique_ptr<Layer::Updater> & updater,
      enum InputSelectionMode select_mode,
      const MatrixSize & batch_size);

  virtual void print_info(std::ostream & os) const;
  void train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const;

private:
  std::unique_ptr<Layer::Updater> _updater;
  enum InputSelectionMode _select_mode;
}; // class Trainer_Batch

// After each batch of training data, apply deltas to the network.
class Trainer_Stochastic : public Trainer
{
public:
  Trainer_Stochastic(
      const std::unique_ptr<Layer::Updater> & updater,
      enum InputSelectionMode select_mode,
      const MatrixSize & batch_size);

  virtual void print_info(std::ostream & os) const;
  void train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const;

private:
  std::unique_ptr<Layer::Updater> _updater;
  enum InputSelectionMode _select_mode;
}; // class Trainer_Stochastic

}; // namespace yann

#endif /* NNTRAINING_H_ */
