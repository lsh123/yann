/*
 * polllayer.h
 *
 * Polls a sub-matrix into a single output value
 */

#ifndef POLLLAYER_H_
#define POLLLAYER_H_

#include "core/layer.h"

namespace yann {

class ParallelLayer;

class PollingLayer : public Layer
{
  typedef Layer Base;

public:
  enum Mode {
    PollMode_Max,
    PollMode_Avg,
  }; // enum Mode

public:
  static MatrixSize get_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_size);
  static MatrixSize get_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_size);
  static MatrixSize get_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                    const MatrixSize & filter_size);

  // Polling layer: each conv layer sends output to the corresponding poll layer
  static std::unique_ptr<ParallelLayer> create_poll_parallel_layer(
      const size_t & frames_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      enum Mode mode,
      const std::unique_ptr<ActivationFunction> & activation_function);

public:
  PollingLayer(const MatrixSize & input_rows, const MatrixSize & input_cols,
               const MatrixSize & filter_size, enum Mode mode);
  virtual ~PollingLayer();

public:
  void set_activation_function(const std::unique_ptr<ActivationFunction> & activation_function);
  void set_values(const Value & ww, const Value & bb);

  // Layer overwrites
  virtual std::string get_name() const;
  virtual std::string get_info() const;
  virtual bool is_valid() const;
  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(const MatrixSize & batch_size, const std::unique_ptr<Layer::Updater> & updater) const;
  virtual std::unique_ptr<Context> create_training_context(const RefVectorBatch & output, const std::unique_ptr<Layer::Updater> & updater) const;

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

  //
  MatrixSize get_output_rows() const;
  MatrixSize get_output_cols() const;

public:
  static void poll(
      const RefConstMatrix & input,
      const MatrixSize & filter_size,
      enum Mode mode,
      RefMatrix output);
  static void poll(
      const RefConstVectorBatch & input,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      enum Mode mode,
      RefVectorBatch output);

  static void poll_gradient_backprop(
      const RefConstMatrix & gradient_output,
      const RefConstVectorBatch & input,
      const MatrixSize & filter_size,
      enum Mode mode,
      RefMatrix gradient_input);
  static void poll_gradient_backprop(
      const RefConstVectorBatch & gradient_output,
      const RefConstVectorBatch & input,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      enum Mode mode,
      RefVectorBatch gradient_input);

private:
  MatrixSize _input_rows;
  MatrixSize _input_cols;
  MatrixSize _filter_size;
  enum Mode  _mode;

  Value _ww;
  Value _bb;
  std::unique_ptr<ActivationFunction> _activation_function;
}; // class PollingLayer

}; // namespace yann

#endif /* POLLLAYER_H_ */
