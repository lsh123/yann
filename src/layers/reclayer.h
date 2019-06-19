/*
 * reclayer.h
 *
 *  Forward propagation:
 *    hh(t) = state_activation_function(ww_hh * hh(t-1) + ww_xh*x(t) + b_h)  // state
 *    a(t) = output_activation_function(ww_ha * hh(t) + b_a) // output
 */

#ifndef RECLAYER_H_
#define RECLAYER_H_

#include "core/layer.h"

namespace yann {

class RecurrentLayer : public Layer
{
  typedef Layer Base;

public:
  RecurrentLayer(
      const MatrixSize & input_size,
      const MatrixSize & state_size,
      const MatrixSize & output_size);
  virtual ~RecurrentLayer();

  MatrixSize get_state_size() const;

  void set_activation_functions(
      const std::unique_ptr<ActivationFunction> & state_activation_function,
      const std::unique_ptr<ActivationFunction> & output_activation_function);

  void set_values(
      const Matrix & ww_hh, const Matrix & ww_xh, const Vector & bb_h,
      const Matrix & ww_ha, const Vector & bb_a);

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
  virtual void update(Context * context, const size_t & batch_size);

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

private:
  template<typename OutputType>
  bool read(std::istream & is, const char ch1, const char ch2, OutputType & out);

private:
  // state: hh(t) = state_activation_function(ww_hh * hh(t-1) + ww_xh*x(t) + b_h)
  Matrix _ww_hh;
  Matrix _ww_xh;
  Vector _bb_h;

  // output: a(t) = output_activation_function(ww_ha * hh(t) + b_a)
  Matrix _ww_ha;
  Vector _bb_a;

  std::unique_ptr<ActivationFunction> _state_activation_function;
  std::unique_ptr<ActivationFunction> _output_activation_function;
}; // class RecurrentLayer

}; // namespace yann

#endif /* RECLAYER_H_ */
