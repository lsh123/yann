/*
 * convlayer.h
 *
 */

#ifndef CONVLAYER_H_
#define CONVLAYER_H_

#include "nnlayer.h"

namespace yann {

class BroadcastLayer;

class ConvolutionalLayer : public Layer
{
  typedef Layer Base;

public:
  // helpers
  static MatrixSize get_input_size(const MatrixSize & input_rows, const MatrixSize & input_cols);

  static MatrixSize get_conv_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_rows);
  static MatrixSize get_conv_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_cols);
  static MatrixSize get_conv_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                         const MatrixSize & filter_rows, const MatrixSize & filter_cols);
  static MatrixSize get_conv_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                         const MatrixSize & filter_size);

  static MatrixSize get_full_conv_output_rows(const MatrixSize & input_rows, const MatrixSize & filter_rows);
  static MatrixSize get_full_conv_output_cols(const MatrixSize & input_cols, const MatrixSize & filter_cols);
  static MatrixSize get_full_conv_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                              const MatrixSize & filter_rows, const MatrixSize & filter_cols);
  static MatrixSize get_full_conv_output_size(const MatrixSize & input_rows, const MatrixSize & input_cols,
                                              const MatrixSize & filter_size);

  static void plus_conv(const RefConstMatrix & input, const RefConstMatrix & filter, RefMatrix output, bool clear_output = true);
  static void plus_conv(const RefConstVectorBatch & input, const MatrixSize & input_rows, const MatrixSize & input_cols,
                        const RefConstMatrix & filter, RefVectorBatch output, bool clear_output = true);
  static void full_conv(const RefConstMatrix & input, const RefConstMatrix & filter, RefMatrix output);
  static void full_conv(const RefConstVectorBatch & input, const MatrixSize & input_rows, const MatrixSize & input_cols,
                        const RefConstMatrix & filter, RefVectorBatch output);

  static void rotate180(const Matrix & input, Matrix & output);

  // Convolutional layer: broadcast from input to all the conv layers
  static std::unique_ptr<BroadcastLayer> create_conv_bcast_layer(
      const size_t & output_frames_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & filter_size,
      const std::unique_ptr<ActivationFunction> & activation_function);

public:
  ConvolutionalLayer(const MatrixSize & input_rows, const MatrixSize & input_cols,
                     const MatrixSize & filter_size);
  virtual ~ConvolutionalLayer();

  void set_activation_function(const std::unique_ptr<ActivationFunction> & activation_function);
  void set_values(const Matrix & ww, const Value & bb);


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
  virtual void backprop(const RefConstVectorBatch & gradient_output,
                        const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input,
                        Context * context) const;

  virtual void init(enum InitMode mode);
  virtual void update(Context * context, const size_t & batch_size);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

  //
  MatrixSize get_output_rows() const;
  MatrixSize get_output_cols() const;

private:
  MatrixSize _input_rows;
  MatrixSize _input_cols;
  MatrixSize _filter_size;

  Matrix _ww;
  Matrix _bb;
  std::unique_ptr<ActivationFunction> _activation_function;
}; // class ConvolutionalLayer

}; // namespace yann

#endif /* CONVLAYER_H_ */
