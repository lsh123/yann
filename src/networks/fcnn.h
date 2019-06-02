/*
 * fcnn.h
 *
 */

#ifndef FCNN_H_
#define FCNN_H_

#include <vector>

#include "core/nn.h"

namespace yann {

// Fully Connected FeedForward Network
class FullyConnectedNetwork
{
public:
  class Params {
  public:
    Params();

  public:
    MatrixSize _input_size;
    MatrixSize _output_size;
    std::unique_ptr<ActivationFunction> _activation_function;
  }; // class Params

public:
  static std::unique_ptr<Network> create(const std::vector<MatrixSize> & layer_sizes);
  static std::unique_ptr<Network> create(const std::vector<Params> & params);
}; // class FullyConnected

}; // namespace yann

#endif /* FCNN_H_ */
