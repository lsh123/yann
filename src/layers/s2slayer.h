/*
 * s2slayer.h
 *
 * Sepcial layer consisting of encoder and decoder (usually LSTM).
 */

#ifndef s2slayer_H_
#define s2slayer_H_

#include "core/layer.h"

namespace yann {

class Seq2SeqLayer : public Layer
{
  typedef Layer Base;

public:
  Seq2SeqLayer(std::unique_ptr<Layer> encoder, std::unique_ptr<Layer> decoder);
  virtual ~Seq2SeqLayer();

  Layer * get_encoder() { return _encoder.get(); }
  Layer * get_decoder() { return _decoder.get(); }

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


  static std::unique_ptr<Seq2SeqLayer> create_lstm(
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & activation_function);

  static std::unique_ptr<Seq2SeqLayer> create_lstm(
      const MatrixSize & input_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & gate_activation_function,
      const std::unique_ptr<ActivationFunction> & io_activation_function);

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
  std::unique_ptr<Layer> _encoder;
  std::unique_ptr<Layer> _decoder;
}; // class Seq2SeqLayer

}; // namespace yann

#endif /* s2slayer_H_ */
