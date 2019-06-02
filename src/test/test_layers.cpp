//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Layers Tests"

#include <boost/test/unit_test.hpp>

#include "core/utils.h"
#include "core/functions.h"
#include "core/training.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// AvgLayer implementation
//
size_t yann::test::AvgLayer::g_counter = 0;

yann::test::AvgLayer::AvgLayer(MatrixSize input_size) :
  _input_size(input_size),
  _value(++g_counter)
{
}
yann::test::AvgLayer:: ~AvgLayer()
{
}

// Layer overwrites
string yann::test::AvgLayer::get_name() const
{
  return "AvgLayer" ;
}
bool yann::test::AvgLayer::is_equal(const Layer& other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
      return false;
    }
    auto the_other = dynamic_cast<const AvgLayer*>(&other);
    if(the_other == nullptr) {
      return false;
    }
    if(_value != the_other->_value) {
      return false;
    }
    return true;
}

MatrixSize yann::test::AvgLayer::get_input_size() const
{
  return _input_size;
}
MatrixSize yann::test::AvgLayer::get_output_size() const
{
  return _output_size;
}

unique_ptr<Layer::Context> yann::test::AvgLayer::create_context(const MatrixSize & batch_size ) const
{
  return make_unique<Layer::Context>(_output_size, batch_size);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_context(const RefVectorBatch & output) const
{
  return make_unique<Context>(output);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_training_context(
    const MatrixSize & batch_size,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  return make_unique<Context>(_output_size, batch_size);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_training_context(
    const RefVectorBatch & output,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  return make_unique<Context>(output);
}

void yann::test::AvgLayer::feedforward(
    const RefConstVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  YANN_CHECK(context);
  YANN_CHECK_EQ(get_batch_item_size(input), _input_size);
  YANN_CHECK_EQ(get_batch_item_size(context->get_output()), _output_size);

  RefVectorBatch output = context->get_output();
  switch(mode) {
  case Operation_Assign:
    output.setZero();
    break;
  case Operation_PlusEqual:
    // do nothing
    break;
  }
  const auto batch_size = get_batch_size(input);
  for (MatrixSize ii = 0; ii < batch_size; ++ii) {
    get_batch(output, ii)(0) += (get_batch(input, ii).sum() / ((double)_input_size));
  }
}

void yann::test::AvgLayer::feedforward(
    const RefConstSparseVectorBatch & input,
    Context * context,
    enum OperationMode mode) const
{
  throw runtime_error("test::AvgLayer::feedforward() is not implemented for sparse vectors");
}

void yann::test::AvgLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstVectorBatch & input,
    boost::optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  if(gradient_input) {
    const auto batch_size = get_batch_size(input);
    for (MatrixSize ii = 0; ii < batch_size; ++ii) {
      get_batch(*gradient_input, ii).array() = get_batch(gradient_output, ii)(0);
    }
  }
}

void yann::test::AvgLayer::backprop(
    const RefConstVectorBatch & gradient_output,
    const RefConstSparseVectorBatch & input,
    boost::optional<RefVectorBatch> gradient_input,
    Context * context) const
{
  throw runtime_error("test::AvgLayer::backprop() is not implemented for sparse vectors");
}

void yann::test::AvgLayer::init(enum InitMode mode, boost::optional<InitContext> init_context)
{
  // do nothing
}
void yann::test::AvgLayer::update(Context * context, const size_t & batch_size)
{
  // do nothing
}

void yann::test::AvgLayer::read(istream & is)
{
  Base::read(is);

  char ch;
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(ios_base::failbit);
    return;
  }
  if(is >> _value >> ch && ch != ')') {
    is.putback(ch);
    is.setstate(ios_base::failbit);
    return;
  }
}
void yann::test::AvgLayer::write(ostream & os) const
{
  Base::write(os);
  os << "(" << _value << ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Helpers for testing layers
//

void yann::test::test_layer_feedforward(
    Layer & layer,
    const RefConstVectorBatch & input,
    const RefConstVectorBatch & expected_output)
{
  YANN_CHECK_EQ(get_batch_size(input), get_batch_size(expected_output));

  // Test writing output to the internal buffer
  {
    std::unique_ptr<Layer::Context> ctx = layer.create_context(get_batch_size(input));
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected_output.isApprox(ctx->get_output(), TEST_TOLERANCE));
    }
  }

  // Test writing output to an external buffer
  {
    VectorBatch output;
    resize_batch(output, get_batch_size(expected_output), get_batch_item_size(expected_output));
    std::unique_ptr<Layer::Context> ctx = layer.create_context(output);
    YANN_CHECK (ctx);
    {
      // ensure we don't do allocations in eigen
      BlockAllocations block;

      layer.feedforward(input, ctx.get());
      BOOST_CHECK(expected_output.isApprox(output, TEST_TOLERANCE));
    }
  }
}


// TODO: add backprop tests for all layers
void yann::test::test_layer_backprop(
    Layer & layer,
    const RefConstVectorBatch & input,
    boost::optional<RefConstVectorBatch> expected_input,
    const RefConstVectorBatch & expected_output,
    std::unique_ptr<CostFunction> cost,
    const double learning_rate,
    const size_t & epochs)
{
  auto ctx = layer.create_training_context(
      get_batch_size(input),
      make_unique<Updater_GradientDescent>(0.0, 0.0)); // learning_rate doesn't matter
  BOOST_CHECK(ctx);
  ctx->reset_state();

  VectorBatch in, gradient_input, gradient_output;
  gradient_input.resizeLike(input);
  gradient_output.resizeLike(expected_output);

  in = input;
  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    for(size_t ii = 0; ii < epochs; ++ii) {
      // feed forward
      layer.feedforward(in, ctx.get());

      // backprop
      cost->derivative(ctx->get_output(), expected_output, gradient_output);
      YANN_CHECK(ii > 0 || gradient_output.squaredNorm() > 0); // we shouldn't have 0 gradient on first try
      layer.backprop(gradient_output, in, optional<RefVectorBatch>(gradient_input), ctx.get());

      // update input
      in -= learning_rate * gradient_input;
    }

    // one more time
    layer.feedforward(in, ctx.get());
  }

  BOOST_TEST_MESSAGE("actual_input=" << in);
  if(expected_input) {
    BOOST_TEST_MESSAGE("expected_input=" << *expected_input);
    BOOST_CHECK(expected_input->isApprox(in, TEST_TOLERANCE));
  }

  BOOST_TEST_MESSAGE("expected_output=" << expected_output);
  BOOST_TEST_MESSAGE("actual_output=" << ctx->get_output());
  BOOST_CHECK(expected_output.isApprox(ctx->get_output(), TEST_TOLERANCE));

}

void yann::test::test_layer_training(
    Layer & layer,
    const RefConstVectorBatch & input,
    const RefConstVectorBatch & expected_output,
    std::unique_ptr<CostFunction> cost,
    const double learning_rate,
    const size_t & epochs)
{
  auto ctx = layer.create_training_context(
      get_batch_size(input),
      make_unique<Updater_GradientDescent>(learning_rate, 0.0));
  BOOST_CHECK(ctx);
  ctx->reset_state();

  VectorBatch gradient_input, gradient_output;
  gradient_input.resizeLike(input);
  gradient_output.resizeLike(expected_output);

  {
    // ensure we don't do allocations in eigen
    BlockAllocations block;

    for(auto ii = epochs; ii > 0; --ii) {
      // reset
      ctx->reset_state();

      // feed forward
      layer.feedforward(input, ctx.get());

      // backprop
      cost->derivative(ctx->get_output(), expected_output, gradient_output);
      YANN_CHECK(ii > 0 || gradient_output.squaredNorm() > 0); // we shouldn't have 0 gradient on first try
      layer.backprop(gradient_output, input,
                      optional<RefVectorBatch>(gradient_input),
                      ctx.get());

      // update
      layer.update(ctx.get(), get_batch_size(input));
    }

    // one more time
    layer.feedforward(input, ctx.get());
  }

  BOOST_TEST_MESSAGE("expected_output=" << expected_output);
  BOOST_TEST_MESSAGE("actual_output=" << ctx->get_output());
  BOOST_CHECK(expected_output.isApprox(ctx->get_output(), TEST_TOLERANCE));
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Test
//
struct LayersTestFixture
{
  LayersTestFixture()
  {

  }
  ~LayersTestFixture()
  {

  }
};
// struct LayersTestFixture

BOOST_FIXTURE_TEST_SUITE(LayersTest, LayersTestFixture);

BOOST_AUTO_TEST_CASE(Layer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** Layer IO test ...");

  const MatrixSize input_size = 2;
  auto one = make_unique<AvgLayer>(input_size);
  one->init(Layer::InitMode_Random, boost::none);

  BOOST_TEST_MESSAGE("AvgLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  auto two = make_unique<AvgLayer>(input_size);
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("AvgLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_SUITE_END()

