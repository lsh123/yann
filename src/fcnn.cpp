/*
 * fcnn.cpp
 *
 */
#include <string>

#include <boost/assert.hpp>

#include "utils.h"
#include "layers/fclayer.h"
#include "fcnn.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// FullyConnectedNetwork implementation
//
yann::FullyConnectedNetwork::Params::Params() :
    _input_size(0),
    _output_size(0)
{
}

unique_ptr<Network>yann::FullyConnectedNetwork::create(const vector<MatrixSize> & layer_sizes)
{
  YANN_CHECK_GE(layer_sizes.size(), 2); // should have at least 2 layers: input and output

  auto fcnn = make_unique<Network>();
  YANN_CHECK(fcnn);

  for (size_t ii = 1; ii < layer_sizes.size(); ++ii) {
    YANN_CHECK_GT(layer_sizes[ii - 1], 0);
    YANN_CHECK_GT(layer_sizes[ii], 0);
    auto layer = make_unique<FullyConnectedLayer>(layer_sizes[ii - 1], layer_sizes[ii]);
    YANN_CHECK(layer);
    fcnn->append_layer(std::move(layer));
  }

  return fcnn;
}

unique_ptr<Network>yann::FullyConnectedNetwork::create(const vector<Params> & params)
{
  YANN_CHECK_GE(params.size(), 1); // should have at least 1 layer for input/output

  auto fcnn = make_unique<Network>();
  YANN_CHECK(fcnn);

  for (const auto & param : params) {
    YANN_CHECK_GT(param._input_size, 0);
    YANN_CHECK_GT(param._output_size, 0);
    auto layer = make_unique<FullyConnectedLayer>(param._input_size, param._output_size);
    if(param._activation_function) {
      layer->set_activation_function(param._activation_function);
    }
    fcnn->append_layer(std::move(layer));
  }

  return fcnn;
}
