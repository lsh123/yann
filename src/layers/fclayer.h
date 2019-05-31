/*
 * fclayer.h
 *
 */

#ifndef FCLAYER_H_
#define FCLAYER_H_

#include "layer.h"

namespace yann {

class FullyConnectedLayer : public Layer
{
  typedef Layer Base;

public:
  FullyConnectedLayer(const MatrixSize & input_size, const MatrixSize & output_size);
  virtual ~FullyConnectedLayer();

  void set_activation_function(const std::unique_ptr<ActivationFunction> & sactivation_function);
  void set_values(const Matrix & ww, const Vector & bb);

public:
  // Layer overwrites
  virtual bool is_valid() const;
  virtual std::string get_name() const;
  virtual void print_info(std::ostream & os) const;
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
  Matrix _ww;
  Vector _bb;
  std::unique_ptr<ActivationFunction> _activation_function;
}; // class FullyConnectedLayer

}; // namespace yann

#endif /* FCLAYER_H_ */
