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
#include "nnlayer.h"

namespace yann {

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

class Trainer_Incremental_GD : public Trainer
{
public:
  Trainer_Incremental_GD(double learning_rate, double regularization_parameter,
                         enum InputSelectionMode select_mode,
                         const MatrixSize & batch_size);

  virtual void print_info(std::ostream & os) const;
  void train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const;

private:
  double _learning_rate;
  double _regularization_parameter;
  enum InputSelectionMode _select_mode;
}; // class Trainer_Incremental_GD

class Trainer_MiniBatch_GD : public Trainer
{
public:
  Trainer_MiniBatch_GD(double learning_rate,
                       double regularization_parameter,
                       enum InputSelectionMode select_mode,
                       const MatrixSize & batch_size);

  virtual void print_info(std::ostream & os) const;
  void train(Network & nn, const VectorBatch & inputs, const VectorBatch & outputs) const;

private:
  double _learning_rate;
  double _regularization_parameter;
  enum InputSelectionMode _select_mode;
}; // class Trainer_MiniBatch_GD

}; // namespace yann

#endif /* NNTRAINING_H_ */
