/*
 * smaxlayer.h
 *
 * Polls a submatrix into a single output value
 */

#ifndef SMAXLAYER_H_
#define SMAXLAYER_H_

#include "nnlayer.h"

namespace yann {

class SoftmaxLayer : public Layer
{
  typedef Layer Base;

public:
  static void softmax_plus_equal(const RefConstVector & input, RefVector output, const Value & beta);
  static void softmax_derivative(const RefConstVector & input, RefMatrix output, const Value & beta);

public:
  SoftmaxLayer(const MatrixSize & size, const Value & beta = 1.0);
  virtual ~SoftmaxLayer();

public:
  // Layer overwrites
  virtual std::string get_name() const;
  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const;
  virtual std::unique_ptr<Context> create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const;

  virtual void feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode = Operation_Assign) const;
  virtual void backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input, Context * context) const;

  virtual void init(enum InitMode mode);
  virtual void update(Context * context, const size_t & batch_size);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

private:
  MatrixSize _size;
  const Value _beta;
}; // class SoftmaxLayer

}; // namespace yann

#endif /* SMAXLAYER_H_ */
