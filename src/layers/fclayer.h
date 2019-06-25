/*
 * fclayer.h
 *
 */

#ifndef FCLAYER_H_
#define FCLAYER_H_

#include "core/layer.h"

namespace yann {

class FullyConnectedLayer : public Layer
{
  typedef Layer Base;

public:
  FullyConnectedLayer(const MatrixSize & input_size, const MatrixSize & output_size);
  virtual ~FullyConnectedLayer();

  void set_activation_function(const std::unique_ptr<ActivationFunction> & activation_function);
  void set_values(const Matrix & ww, const Vector & bb);
  void set_fixed_bias(const Value & val);

  void set_sampling_rate(const double & sampling_rate);
  bool is_sampled() const;

  const Matrix & get_weights() const { return _ww; }
  const Vector & get_bias() const { return _bb; }

public:
  // Layer overwrites
  virtual bool is_valid() const;
  virtual std::string get_name() const;
  virtual std::string get_info() const;
  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(
      const MatrixSize & batch_size,
      const std::unique_ptr<Layer::Updater> & updater) const;
  virtual std::unique_ptr<Context> create_training_context(
      const RefVectorBatch & output,
      const std::unique_ptr<Layer::Updater> & updater) const;

  virtual void feedforward(
      const RefConstVectorBatch & input,
      Context * context,
      enum OperationMode mode) const;
  virtual void feedforward(
      const RefConstSparseVectorBatch & input,
      Context * context,
      enum OperationMode mode) const;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;
  virtual void backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstSparseVectorBatch & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;

  virtual void init(enum InitMode mode, boost::optional<InitContext> init_context);
  virtual void update(Context * context, const size_t & tests_num);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

private:
  template<typename InputType>
  void feedforward_internal(
        const InputType & input,
        Context * context,
        enum OperationMode mode) const;

  template<typename InputType>
  void backprop_internal(
      const RefConstVectorBatch & gradient_output,
      const InputType & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;

  template<typename InputType>
  void backprop_with_sampling_internal(
      const RefConstVectorBatch & gradient_output,
      const InputType & input,
      boost::optional<RefVectorBatch> gradient_input,
      Context * context) const;

private:
  Matrix _ww;
  Vector _bb;
  bool   _fixed_bias;
  double _sampling_rate;
  std::unique_ptr<ActivationFunction> _activation_function;
}; // class FullyConnectedLayer

}; // namespace yann

#endif /* FCLAYER_H_ */
