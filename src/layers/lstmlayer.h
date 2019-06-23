/*
 * lstmlayer.h
 *
 *  Forward propagation:
 *    x(t) -- inputs
 *    h(t-1) -- hidden state, aka outputs on prev timestep
 *    c(t) -- cell state
 *
 *    Phase 1: I/O gates
 *      zz_a(t) = ww_xa*x(t) + ww_ha * h(t-1) + bb_a  -- input
 *      zz_i(t) = ww_xi*x(t) + ww_hi * h(t-1) + bb_i  -- input gate
 *      zz_f(t) = ww_xf*x(t) + ww_hf * h(t-1) + bb_g  -- forget gate
 *      zz_o(t) = ww_xo*x(t) + ww_ho * h(t-1) + bb_o  -- output gate
 *
 *      a(t) = activ1 (zz_a(t))
 *      i(t) = activ2 (zz_i(t))
 *      f(t) = activ2 (zz_f(t))
 *      o(t) = activ2 (zz_o(t))
 *
 *    Phase 2: State
 *      c(t) = elem_prod(i(t), a(t)) + elem_prod(f(t), c(t-1))
 *
 *    Phase 3: Output
 *      h(t) = elem_prod(o(t), activ1(c(t))
 */
#ifndef LSTMLAYER_H_
#define LSTMLAYER_H_

#include "core/layer.h"

namespace yann {

class LstmLayer : public Layer
{
  typedef Layer Base;

public:
  // use this enum to access gate's data in the arrays
  enum IOGates {
    Gate_A = 0,
    Gate_I = 1,
    Gate_F = 2,
    Gate_O = 3,
    Gate_Max = 4, // last value for arrays size
  }; // enum IOGates

public:
  LstmLayer(
      const MatrixSize & input_size,
      const MatrixSize & output_size);
  virtual ~LstmLayer();

  void set_activation_functions(
      const std::unique_ptr<ActivationFunction> & gate_activation_function,
      const std::unique_ptr<ActivationFunction> & io_activation_function);

  void set_values(
      const Matrix (& ww_x)[Gate_Max],
      const Matrix (& ww_h)[Gate_Max],
      const Vector (& bb)[Gate_Max]);

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
  template<
    typename MainInputType,
    typename InputType,
    typename GradientInputType,
    typename ContextType>
  void backprop_gate(
      enum IOGates gate,
      const std::unique_ptr<ActivationFunction> & activation_function,
      const RefConstVector & gradient_gate,
      const InputType & input,
      boost::optional<GradientInputType> gradient_input,
      ContextType * ctx) const;

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
  Matrix _ww_x[Gate_Max] ;
  Matrix _ww_h[Gate_Max];
  Vector _bb[Gate_Max];

  std::unique_ptr<ActivationFunction> _gate_activation_function;
  std::unique_ptr<ActivationFunction> _io_activation_function;
}; // class LstmLayer

}; // namespace yann

#endif /* LSTMLAYER_H_ */
