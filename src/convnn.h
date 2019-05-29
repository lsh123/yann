/*
 * convnn.h
 *
 */

#ifndef CONVNN_H_
#define CONVNN_H_

#include "nn.h"
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
    MatrixSize                              _polling_filter_size;
  }; // class ConvPollParams

  // create network with one conv+poll layer
  static std::unique_ptr<Network> create(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const ConvPollParams & params,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
      const MatrixSize & output_size);

  // create network with two conv+poll layers
  static std::unique_ptr<Network> create(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const ConvPollParams & params1,
      const ConvPollParams & params2,
      const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
      const MatrixSize & output_size);

  static std::unique_ptr<Network> create_lenet4(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & fc_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & activation_funtion,
      const std::unique_ptr<CostFunction> & cost_funtion);

  static std::unique_ptr<Network> create_lenet5(
      const MatrixSize & input_rows,
      const MatrixSize & input_cols,
      const MatrixSize & fc1_size,
      const MatrixSize & fc2_size,
      const MatrixSize & output_size,
      const std::unique_ptr<ActivationFunction> & activation_funtion,
      const std::unique_ptr<CostFunction> & cost_funtion);

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
