/*
 * convnn.cpp
 *
 */
#include <string>

#include <boost/assert.hpp>

#include "core/utils.h"
#include "core/functions.h"
#include "layers/contlayer.h"
#include "layers/convlayer.h"
#include "layers/fclayer.h"
#include "layers/polllayer.h"
#include "layers/smaxlayer.h"

#include "cnn.h"

using namespace std;
using namespace yann;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// ConvolutionalNetwork implementation
//
yann::ConvolutionalNetwork::ConvPollParams::ConvPollParams() :
    _output_frames_num(0),
    _conv_filter_size(0),
    _polling_mode(PollingLayer::PollMode_Avg),
    _polling_filter_size(0)
{
}

void yann::ConvolutionalNetwork::append(unique_ptr<SequentialLayer> & container,
                   const MatrixSize & input_rows,
                   const MatrixSize & input_cols,
                   const MatrixSize & input_frames_num,
                   const ConvPollParams & params)
{
  YANN_CHECK(container);
  YANN_CHECK_GT(input_rows, 0);
  YANN_CHECK_GT(input_cols, 0);
  YANN_CHECK_GT(input_frames_num, 0);
  YANN_CHECK(input_frames_num == 1 || !params._mappings.empty());
  YANN_CHECK_GT(params._output_frames_num, 0 || !params._mappings.empty());
  YANN_CHECK_GT(params._conv_filter_size, 0);
  YANN_CHECK_GT(params._polling_filter_size, 0);

  // ConvolutionalLayer
  if(input_frames_num == 1) {
    auto conv_bcast_layer = ConvolutionalLayer::create_conv_bcast_layer(
        params._output_frames_num,   // N output frames
        input_rows,
        input_cols,
        params._conv_filter_size,
        params._conv_activation_funtion
    );
    YANN_CHECK(conv_bcast_layer);
    container->append_layer(std::move(conv_bcast_layer));
  } else {
    YANN_CHECK(!params._mappings.empty());
    YANN_CHECK(params._output_frames_num == 0 || params._output_frames_num == params._mappings.size());

    auto conv_mapping_layer = make_unique<MappingLayer>(input_frames_num);
    YANN_CHECK(conv_mapping_layer);
    for(auto & layer_mappings : params._mappings) {
      auto conv_layer = make_unique<ConvolutionalLayer>(
          input_rows,
          input_cols,
          params._conv_filter_size);
      YANN_CHECK(conv_layer);

      if(params._conv_activation_funtion) {
        conv_layer->set_activation_function(params._conv_activation_funtion);
      }
      conv_mapping_layer->append_layer(std::move(conv_layer), layer_mappings);
    }
    container->append_layer(std::move(conv_mapping_layer));
  }

  // PollingLayer
  MatrixSize poll_input_rows = ConvolutionalLayer::get_conv_output_rows(input_rows, params._conv_filter_size);
  MatrixSize poll_input_cols = ConvolutionalLayer::get_conv_output_cols(input_cols, params._conv_filter_size);
  auto poll_layer = PollingLayer::create_poll_parallel_layer(
     params._output_frames_num,
     poll_input_rows,
     poll_input_cols,
     params._polling_filter_size,
     params._polling_mode,
     params._polling_activation_funtion);
  YANN_CHECK(poll_layer);
  container->append_layer(std::move(poll_layer));
}

unique_ptr<SequentialLayer> yann::ConvolutionalNetwork::create(
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const ConvPollParams & params,
    const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
    const MatrixSize & output_size)
{
  YANN_CHECK_GT(output_size, 0);

  auto container = make_unique<SequentialLayer>();
  YANN_CHECK(container);

  // append one layer: we have 1 input frame
  append(container, input_rows, input_cols, 1, params);

  // FullyConnectedLayer
  auto fc_layer = make_unique<FullyConnectedLayer>(
      container->get_output_size(),
      output_size);
  YANN_CHECK(fc_layer);
  if(fc_activation_funtion) {
    fc_layer->set_activation_function(fc_activation_funtion);
  }
  container->append_layer(std::move(fc_layer));

  // done
  return container;
}

std::unique_ptr<SequentialLayer> yann::ConvolutionalNetwork::create(
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    const ConvPollParams & params1,
    const ConvPollParams & params2,
    const std::unique_ptr<ActivationFunction> & fc_activation_funtion,
    const MatrixSize & output_size)
{
  YANN_CHECK_GT(output_size, 0);

  auto container = make_unique<SequentialLayer>();
  YANN_CHECK(container);

  // append one layer: we have 1 input frame
  append(container, input_rows, input_cols, 1, params1);

  // append second layer: the output frames from first layer are our inputs
  MatrixSize poll1_input_rows = ConvolutionalLayer::get_conv_output_rows(input_rows, params1._conv_filter_size);
  MatrixSize poll1_input_cols = ConvolutionalLayer::get_conv_output_cols(input_cols, params1._conv_filter_size);
  MatrixSize conv2_input_rows = PollingLayer::get_output_rows(poll1_input_rows, params1._polling_filter_size);
  MatrixSize conv2_input_cols = PollingLayer::get_output_cols(poll1_input_cols, params1._polling_filter_size);
  append(container, conv2_input_rows, conv2_input_cols, params1._output_frames_num, params2);

  // FullyConnectedLayer
  auto fc_layer = make_unique<FullyConnectedLayer>(
      container->get_output_size(),
      output_size);
  YANN_CHECK(fc_layer);
  if(fc_activation_funtion) {
    fc_layer->set_activation_function(fc_activation_funtion);
  }
  container->append_layer(std::move(fc_layer));

  // done
  return container;
}

std::unique_ptr<SequentialLayer> yann::ConvolutionalNetwork::create_lenet1(
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    PollingLayer::Mode polling_mode,
    const MatrixSize & fc_size,
    const MatrixSize & output_size,
    const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
    const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
    const std::unique_ptr<ActivationFunction> & fc_activation_funtion)
{
  // see http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf
  ConvPollParams params1;
  params1._output_frames_num = 4;
  params1._conv_filter_size  = 5;
  if(conv_activation_funtion) {
    params1._conv_activation_funtion = conv_activation_funtion->copy();
  }
  params1._polling_mode = polling_mode;
  params1._polling_filter_size = 2;
  if(poll_activation_funtion) {
    params1._polling_activation_funtion = poll_activation_funtion->copy();
  }

  ConvPollParams params2;
  params2._output_frames_num = 12;
  params2._conv_filter_size  = 5;
  if(conv_activation_funtion) {
    params2._conv_activation_funtion = conv_activation_funtion->copy();
  }
  params2._polling_mode = polling_mode;
  params2._polling_filter_size = 2;
  if(poll_activation_funtion) {
    params2._polling_activation_funtion = poll_activation_funtion->copy();
  }
  params2._mappings.push_back({ 0 });
  params2._mappings.push_back({ 0, 1 });
  params2._mappings.push_back({ 0, 1 });
  params2._mappings.push_back({ 1 });
  params2._mappings.push_back({ 0, 1 });
  params2._mappings.push_back({ 0, 1 });
  params2._mappings.push_back({ 2 });
  params2._mappings.push_back({ 2, 3 });
  params2._mappings.push_back({ 2, 3 });
  params2._mappings.push_back({ 3 });
  params2._mappings.push_back({ 2, 3 });
  params2._mappings.push_back({ 2, 3 });

  auto container = create(
      input_rows, input_cols, params1, params2,
      fc_activation_funtion, fc_size > 0 ? fc_size : output_size);
  YANN_CHECK(container);

  // add one more FC layer if needed
  if(fc_size > 0) {
    auto fc_layer = make_unique<FullyConnectedLayer>(fc_size, output_size);
    YANN_CHECK(fc_layer);
    if(fc_activation_funtion) {
      fc_layer->set_activation_function(fc_activation_funtion);
    }
    container->append_layer(std::move(fc_layer));
  }

  // done
  return container;
}

std::unique_ptr<SequentialLayer> yann::ConvolutionalNetwork::create_boosted_lenet1(
    const MatrixSize & paths_num,
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    PollingLayer::Mode polling_mode,
    const MatrixSize & fc_size,
    const MatrixSize & output_size,
    const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
    const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
    const std::unique_ptr<ActivationFunction> & fc_activation_funtion)
{
  // create N paths
  auto bcast_layer = make_unique<BroadcastLayer>();
  YANN_CHECK(bcast_layer);
  for(auto ii = paths_num; ii > 0; --ii) {
    auto path = create_lenet1(
        input_rows, input_cols, polling_mode,
        0, fc_size, // don't create fc layer, we will add one for the merge
        conv_activation_funtion, poll_activation_funtion,
        fc_activation_funtion);
    YANN_CHECK(path);
    bcast_layer->append_layer(std::move(path));
  }

  // merge them together
  auto fc_layer = make_unique<FullyConnectedLayer>(paths_num * fc_size, output_size);
  YANN_CHECK(fc_layer);
  fc_layer->set_activation_function(fc_activation_funtion);

  // and finally create container
  auto container = make_unique<SequentialLayer>();
  YANN_CHECK(container);
  container->append_layer(std::move(bcast_layer));
  container->append_layer(std::move(fc_layer));

  // done
  return container;
}

std::unique_ptr<SequentialLayer> yann::ConvolutionalNetwork::create_lenet5(
    const MatrixSize & input_rows,
    const MatrixSize & input_cols,
    PollingLayer::Mode polling_mode,
    const MatrixSize & fc1_size,
    const MatrixSize & fc2_size,
    const MatrixSize & output_size,
    const std::unique_ptr<ActivationFunction> & conv_activation_funtion,
    const std::unique_ptr<ActivationFunction> & poll_activation_funtion,
    const std::unique_ptr<ActivationFunction> & fc_activation_funtion)
{
  // see http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
  ConvPollParams params1;
  params1._output_frames_num = 6;
  params1._conv_filter_size  = 5;
  if(conv_activation_funtion) {
    params1._conv_activation_funtion = conv_activation_funtion->copy();
  }
  params1._polling_mode = polling_mode;
  params1._polling_filter_size = 2;
  if(poll_activation_funtion) {
    params1._polling_activation_funtion = poll_activation_funtion->copy();
  }

  ConvPollParams params2;
  params2._output_frames_num = 16;
  params2._conv_filter_size  = 5;
  if(conv_activation_funtion) {
    params2._conv_activation_funtion = conv_activation_funtion->copy();
  }
  params2._polling_mode = polling_mode;
  params2._polling_filter_size = 2;
  if(poll_activation_funtion) {
    params2._polling_activation_funtion = poll_activation_funtion->copy();
  }
  params2._mappings.push_back({ 0, 1, 2 }); // 0
  params2._mappings.push_back({ 1, 2, 3 });
  params2._mappings.push_back({ 2, 3, 4 });
  params2._mappings.push_back({ 3, 4, 5 });
  params2._mappings.push_back({ 0, 4, 5 }); // 4
  params2._mappings.push_back({ 0, 1, 5 });
  params2._mappings.push_back({ 0, 1, 2, 3 });
  params2._mappings.push_back({ 1, 2, 3, 4 });
  params2._mappings.push_back({ 2, 3, 4, 5 }); // 8
  params2._mappings.push_back({ 0, 3, 4, 5 });
  params2._mappings.push_back({ 0, 1, 4, 5 });
  params2._mappings.push_back({ 0, 1, 2, 5 });
  params2._mappings.push_back({ 0, 1, 3, 4 }); // 12
  params2._mappings.push_back({ 1, 2, 4, 5 });
  params2._mappings.push_back({ 0, 2, 3, 5 });
  params2._mappings.push_back({ 0, 1, 2, 3, 4, 5 });

  auto container = create(input_rows, input_cols, params1, params2, fc_activation_funtion, fc1_size);
  YANN_CHECK(container);

  // add one more FC layer
  auto fc_layer1 = make_unique<FullyConnectedLayer>(fc1_size, fc2_size);
  YANN_CHECK(fc_layer1);
  if(fc_activation_funtion) {
    fc_layer1->set_activation_function(fc_activation_funtion);
  }
  container->append_layer(std::move(fc_layer1));

  // and one more FC layer
  auto fc_layer2 = make_unique<FullyConnectedLayer>(fc2_size, output_size);
  YANN_CHECK(fc_layer2);
  if(fc_activation_funtion) {
    fc_layer2->set_activation_function(fc_activation_funtion);
  }
  container->append_layer(std::move(fc_layer2));

  // done
  return container;
}


