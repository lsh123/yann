/*
 * convnn.h
 *
 */

#ifndef CONVNN_H_
#define CONVNN_H_

#include "core/nn.h"
#include "layers/polllayer.h"
#include "layers/contlayer.h"

namespace yann {

// Convolutional neural networks
class ConvolutionalNetwork
{
public:
  // Parameters for a single Conv+Poll layers combination
  class ConvPollParams {
  public:
    ConvPollParams();
  public:
    size_t                                   _output_frames_num;
    std::vector<MappingLayer::InputsMapping> _mappings; // must be set if _input_frames_num > 1
    MatrixSize                               _conv_filter_size;
    std::unique_ptr<ActivationFunction>      _conv_activation_funtion;
    PollingLayer::Mode                       _polling_mode;
    MatrixSize                               _polling_filter_size;
    std::unique_ptr<ActivationFunction>      _polling_activation_funtion;
  }; // class ConvPollParams

  // create network with one conv+poll layer
  static std::unique_ptr<SequentialLayer> create(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const ConvPollParams & params,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
      const MatrixSize & output_size);

  // create network with two conv+poll layers
  static std::unique_ptr<SequentialLayer> create(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const ConvPollParams & params1,
      const ConvPollParams & params2,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
      const MatrixSize & output_size);

  static std::unique_ptr<SequentialLayer> create_lenet1(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      PollingLayer::Mode polling_mode,
      const MatrixSize & fc_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
      const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion);

  static std::unique_ptr<SequentialLayer> create_boosted_lenet1(
      const MatrixSize & paths_num,
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      PollingLayer::Mode polling_mode,
      const MatrixSize & fc_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
      const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion);

  static std::unique_ptr<SequentialLayer> create_lenet5(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      PollingLayer::Mode polling_mode,
      const MatrixSize & fc1_size,
      const MatrixSize & fc2_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
      const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion);

private:
  // Appends one conv+poll layer to the sequential network
  static void append(std::unique_ptr<SequentialLayer> & container,
                     const MatrixSize & input_rows,
                     const MatrixSize & input_cols,
                     const MatrixSize & input_frames_num,
                     const ConvPollParams & params);
}; // class ConvolutionalNetwork

}; // namespace yann

#endif /* CONVNN_H_ */
